"""
Train HiFuse-Small or MedViT-Small from scratch on the lesion-level split.

Identical protocol for both models: AdamW (lr 1e-4), 5-epoch linear warm-up + cosine decay,
cross-entropy, batch 16, 224x224 inputs. Defaults below are those of the first lesion-level
protocol; run.py passes the final-protocol values (30 epochs, patience 7, class-weighted loss,
stochastic depth / dropout / weight decay).
  * lesion-level split (no lesion in two partitions)
  * early stopping and checkpoint selection on validation loss (threshold-free)
  * dihedral augmentation (H/V flips + 90-degree rotations): lesions have no canonical orientation
  * no weight decay on norm/bias parameters, as in the official HiFuse training script
Class imbalance is handled by minority oversampling or a class-weighted loss, never by
undersampling.

Example:
  python train.py --model medvit_small --split splits/lesion_split_seed42.csv --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import HamDataset, MinorityOversampler, build_transforms, class_weights, load_split
from metrics import binary_metrics, to_lesion_level
from models import build_model, count_parameters

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **_):
        return x

HIGHER_IS_BETTER = {"loss": False, "auprc": True, "auroc": True, "balanced_accuracy": True}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["hifuse_small", "medvit_small"])
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--epochs", type=int, default=50, help="maximum epochs (cosine schedule length)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (VRAM fallback)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--select", default="loss", choices=list(HIGHER_IS_BETTER),
                    help="validation metric for checkpoint selection AND early stopping")
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience in epochs (0 = off)")
    ap.add_argument("--imbalance", default="oversample", choices=["oversample", "weighted_loss", "none"])
    ap.add_argument("--aug", default="strong", choices=["strong", "light"])
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deterministic", action="store_true", help="slower, bit-reproducible cuDNN")
    ap.add_argument("--limit_lesions", type=int, default=0, help="pilot: keep N lesions per class per split")
    ap.add_argument("--no_resume", action="store_true", help="ignore last.pt and start this run from scratch")
    # ---- regularisation / overfitting controls (defaults = the main experiment's behaviour)
    ap.add_argument("--drop_path", type=float, default=None, help="stochastic depth rate (None = official default)")
    ap.add_argument("--dropout", type=float, default=None, help="dropout rate (None = official default)")
    ap.add_argument("--class_weight", type=float, default=None,
                    help="melanoma weight in weighted_loss (nevus = 1); None = inverse class frequency")
    ap.add_argument("--eval_train", action="store_true",
                    help="each epoch, also score the training set with the validation transform (clean curves)")
    return ap.parse_args()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def param_groups(model: nn.Module, wd: float):
    """No weight decay on biases and normalisation weights (1-D tensors)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def resolve_amp(mode: str, device: torch.device):
    if device.type != "cuda" or mode == "off":
        return None
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[mode]


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    probs, losses, labels = [], [], []
    ce = nn.CrossEntropyLoss(reduction="none")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        logits = logits.float()
        losses.append(ce(logits, y).cpu())
        probs.append(logits.softmax(1)[:, 1].cpu())
        labels.append(y.cpu())
    return torch.cat(probs).numpy(), torch.cat(losses).mean().item(), torch.cat(labels).numpy()


class Logger:
    """print() that is also appended to <run_dir>/train.log."""

    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        text = " ".join(str(m) for m in msg)
        print(text, flush=True)
        self.f.write(text + "\n"); self.f.flush()


def atomic_save(obj, path: Path) -> None:
    """Write to a temp file then rename, so a crash mid-save never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp(args.amp, device)
    run_dir = args.out / args.model / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    if args.no_resume and last_path.exists():
        last_path.unlink()
    log = Logger(run_dir / "train.log")
    log(f"\n===== {args.model} seed {args.seed} | {time.strftime('%Y-%m-%d %H:%M:%S')} =====")

    df = load_split(args.split, args.limit_lesions or None)
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    for s, d in parts.items():
        log(f"{s:5s}: {len(d):5d} images | {d['lesion_id'].nunique():5d} lesions | "
            f"mel {int(d['label'].sum()):4d} ({d['label'].mean():.3f})")

    train_ds = HamDataset(parts["train"], build_transforms(True, args.aug, args.img_size))
    eval_tf = build_transforms(False, img_size=args.img_size)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    g = torch.Generator().manual_seed(args.seed)

    sampler = None
    if args.imbalance == "oversample":
        sampler = MinorityOversampler(parts["train"]["label"], parts["train"]["lesion_id"], seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=sampler is None, drop_last=True, generator=g, **loader_kw)
    # "train_clean" = the training images scored exactly like validation (no augmentation, no resampling)
    eval_loaders = {s: DataLoader(HamDataset(parts[s if s != "train_clean" else "train"], eval_tf),
                                  batch_size=args.batch_size * 2, shuffle=False, **loader_kw)
                    for s in ("val", "test", "train_clean")}

    model = build_model(args.model, num_classes=2, drop_path=args.drop_path, dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log(f"{args.model}: {n_params:.2f}M params | {gpu} | amp {amp_dtype} | "
        f"{len(train_loader) * args.batch_size} training samples/epoch | drop_path {args.drop_path} "
        f"dropout {args.dropout} wd {args.wd} | imbalance {args.imbalance}"
        + (f" (melanoma weight {args.class_weight})" if args.imbalance == "weighted_loss" else ""))

    weight = None
    if args.imbalance == "weighted_loss":
        weight = (torch.tensor([1.0, args.class_weight]) if args.class_weight
                  else class_weights(parts["train"]["label"])).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum)
    scheduler = warmup_cosine(optimizer, args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    (run_dir / "config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "params_M": n_params,
         "train_samples_per_epoch": len(train_loader) * args.batch_size}, indent=2), encoding="utf-8")

    higher = HIGHER_IS_BETTER[args.select]
    best, best_epoch, bad_epochs, history = -math.inf if higher else math.inf, -1, 0, []
    start_epoch, finished, prev_minutes = 1, False, 0.0

    # ---- resume an interrupted run from its last completed epoch
    if last_path.exists():
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"]); scaler.load_state_dict(st["scaler"])
        best, best_epoch, bad_epochs, history = st["best"], st["best_epoch"], st["bad_epochs"], st["history"]
        start_epoch, finished, prev_minutes = st["epoch"] + 1, st["finished"], st["minutes"]
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"])
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        log(f"resumed after epoch {st['epoch']} (best so far: epoch {best_epoch})")

    t0 = time.time() - prev_minutes * 60
    for epoch in range(start_epoch, args.epochs + 1):
        if finished:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        run_loss, run_correct, seen = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}", leave=False)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            loss = criterion(logits.float(), y)
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or i + 1 == len(train_loader):
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0

        p_val, val_loss, y_val = predict(model, eval_loaders["val"], device, amp_dtype)
        m = binary_metrics(y_val, p_val)
        les = to_lesion_level(parts["val"].assign(prob_mel=p_val))
        m_les = binary_metrics(les["label"], les["prob_mel"])
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": run_loss / seen, "train_acc": run_correct / seen,
               "val_loss": val_loss, "val_acc": m["accuracy"], "val_balanced_accuracy": m["balanced_accuracy"],
               "val_recall_mel": m["recall"], "val_precision_mel": m["precision"],
               "val_auroc": m["auroc"], "val_auprc": m["auprc"],
               "val_lesion_auroc": m_les["auroc"], "val_lesion_auprc": m_les["auprc"],
               "peak_gpu_gb": peak_gb}
        if args.eval_train:  # same transform + same unweighted loss as validation -> comparable curves
            p_tr, tr_loss, y_tr = predict(model, eval_loaders["train_clean"], device, amp_dtype)
            m_tr = binary_metrics(y_tr, p_tr)
            row.update({"train_eval_loss": tr_loss, "train_eval_acc": m_tr["accuracy"],
                        "train_eval_balanced_accuracy": m_tr["balanced_accuracy"],
                        "train_eval_auroc": m_tr["auroc"], "train_eval_auprc": m_tr["auprc"]})
        row["elapsed_min"] = (time.time() - t0) / 60
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        score = val_loss if args.select == "loss" else row[f"val_{args.select}"]
        improved = score > best if higher else score < best
        if improved:
            best, best_epoch, bad_epochs = score, epoch, 0
            atomic_save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, run_dir / "best.pt")
        else:
            bad_epochs += 1
        clean = (f"clean train loss {row['train_eval_loss']:.4f} bacc {row['train_eval_balanced_accuracy']:.4f} | "
                 if args.eval_train else "")
        ref = row["train_eval_loss"] if args.eval_train else row["train_loss"]
        log(f"ep {epoch:3d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | {clean}"
            f"val loss {val_loss:.4f} bacc {m['balanced_accuracy']:.4f} auprc {m['auprc']:.4f} "
            f"auroc {m['auroc']:.4f} | gap {val_loss - ref:+.3f} | "
            f"{peak_gb:.1f}GB {row['elapsed_min']:.0f}min{'  *best' if improved else ''}")
        if args.patience and bad_epochs >= args.patience:
            log(f"early stop: no val {args.select} improvement for {args.patience} epochs")
            finished = True
        finished = finished or epoch == args.epochs
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "history": history,
                     "epoch": epoch, "finished": finished, "minutes": (time.time() - t0) / 60,
                     "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                    last_path)

    # ---- reload the selected checkpoint and save val/test probabilities for evaluate.py
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    out = {}
    for s in ("val", "test"):
        p, loss, _ = predict(model, eval_loaders[s], device, amp_dtype)
        cols = ["image_id", "lesion_id", "label"]
        parts[s][cols].assign(prob_mel=p).to_csv(run_dir / f"preds_{s}.csv", index=False)
        out[s] = {"loss": loss, **binary_metrics(parts[s]["label"], p)}
    p, loss, _ = predict(model, eval_loaders["train_clean"], device, amp_dtype)  # clean train score at the
    out["train"] = {"loss": loss, **binary_metrics(parts["train"]["label"], p)}   # selected checkpoint
    summary = {"model": args.model, "seed": args.seed, "best_epoch": best_epoch,
               "epochs_run": len(history), "select": args.select, "best_val_score": best,
               "train_minutes": (time.time() - t0) / 60, "params_M": n_params,
               "peak_gpu_gb_train": max((h["peak_gpu_gb"] for h in history), default=0.0),
               **{f"{k}@0.5": v for k, v in out.items()}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    last_path.unlink(missing_ok=True)  # resume state no longer needed (it is ~3x the model size)
    log(f"best epoch {best_epoch} | test@0.5 bacc {out['test']['balanced_accuracy']:.4f} "
        f"auprc {out['test']['auprc']:.4f} | {summary['train_minutes']:.0f} min | saved to {run_dir}")


if __name__ == "__main__":  # required for DataLoader workers on Windows
    main()
