"""
Evaluate the 10 final models (runs_reg) on an external holdout and repeat the efficiency
timing. No training happens here.

Holdout = official ISIC 2018 Task 3 test set (melanoma + nevus images only). It was never
used for training, validation, tuning, or any protocol decision. Lesions that also occur in
the HAM10000 training data (same lesion_id) are removed, so the holdout is lesion-disjoint.

Usage (from the repository root):
    python external_test.py --gt data/ISIC2018_Task3_Test_GroundTruth.csv --images data/ISIC2018_Task3_Test_Images

Writes results_reg/external_isic2018/.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, binomtest
from torch.utils.data import DataLoader

from data import HamDataset, build_transforms
from metrics import binary_metrics, to_lesion_level, youden_threshold
from models import build_model

MODELS = {"hifuse_small": "HiFuse-Small", "medvit_small": "MedViT-Small"}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def load_ground_truth(path: Path) -> pd.DataFrame:
    gt = read_table(path)
    gt.columns = [c.strip().strip('"') for c in gt.columns]
    if {"image_id", "dx"} <= set(gt.columns):              # HAM10000-style metadata
        df = gt.copy()
    elif "image" in gt.columns and {"MEL", "NV"} <= set(gt.columns):   # ISIC one-hot format
        classes = [c for c in gt.columns if c != "image"]
        df = pd.DataFrame({"image_id": gt["image"],
                           "dx": gt[classes].astype(float).idxmax(axis=1).str.lower()})
    else:
        raise SystemExit(f"Unrecognised ground-truth columns in {path}: {list(gt.columns)}")
    df["image_id"] = df["image_id"].astype(str).str.strip().str.replace(".jpg", "", regex=False)
    df["dx"] = df["dx"].astype(str).str.strip().str.lower()
    df = df[df["dx"].isin(["mel", "nv"])].copy()
    df["label"] = (df["dx"] == "mel").astype(int)
    if "lesion_id" not in df.columns:
        print("NOTE: ground truth has no lesion_id; each image is treated as its own lesion.")
        df["lesion_id"] = "IMG_" + df["image_id"]
    return df.reset_index(drop=True)


def find_images(images: Path, data_dir: Path) -> dict:
    """Image files by name. Accepts a folder (searched recursively) or a .zip, and falls back to
    any 'ISIC2018_Task3_Test*' folder or zip next to the ground-truth file."""
    exts = {".jpg", ".jpeg", ".png"}
    cands = [images] + sorted(data_dir.glob("ISIC2018_Task3_Test*"))
    for c in cands:
        if c.is_file() and c.suffix.lower() == ".zip":
            out = c.with_suffix("")
            if not out.exists():
                print(f"extracting {c} -> {out}")
                zipfile.ZipFile(c).extractall(out)
            c = out
        if c.is_dir():
            found = {p.stem: p for p in c.rglob("*") if p.suffix.lower() in exts}
            if found:
                print(f"using images from {c} ({len(found)} files)")
                return found
    print("NO IMAGES FOUND. Looked in:", *[str(c) for c in cands], sep="\n  ")
    print("Contents of", data_dir, ":", *sorted(p.name for p in data_dir.iterdir()), sep="\n  ")
    return {}


@torch.inference_mode()
def predict(model, loader, device, amp_dtype):
    probs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        probs.append(logits.float().softmax(1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def load_run(run_dir: Path, name: str, device):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dp = float(cfg["drop_path"]) if cfg.get("drop_path") not in (None, "None") else None
    do = float(cfg["dropout"]) if cfg.get("dropout") not in (None, "None") else None
    model = build_model(name, 2, drop_path=dp, dropout=do)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
    return model.to(device).eval()


def mcnemar(ca, cb) -> dict:
    b = int((ca & ~cb).sum()); c = int((~ca & cb).sum())
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    from scipy.stats import chi2 as chi2d
    return {"only_A_correct": b, "only_B_correct": c, "chi2_cc": chi2,
            "p_chi2": float(chi2d.sf(chi2, 1)) if b + c else 1.0,
            "p_exact": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def time_forward(model, x, device, warmup, iters):
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        t = __import__("time").perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        out.append((__import__("time").perf_counter() - t) * 1000)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True, help="ISIC 2018 Task 3 test ground-truth file (.csv/.tab)")
    ap.add_argument("--images", type=Path, required=True, help="folder with the ISIC 2018 Task 3 test images")
    ap.add_argument("--runs", type=Path, default=Path("runs_reg"))
    ap.add_argument("--split", type=Path, default=Path("splits/lesion_split_seed42.csv"))
    ap.add_argument("--out", type=Path, default=Path("results_reg/external_isic2018"))
    ap.add_argument("--metadata", type=Path, default=Path("data/HAM10000_metadata"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eff_repeats", type=int, default=3)
    args = ap.parse_args()
    (args.out / "preds").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    torch.backends.cudnn.benchmark = True

    # ---------- holdout data
    gt = load_ground_truth(args.gt)
    paths = find_images(args.images, args.gt.parent)
    gt["path"] = gt["image_id"].map(lambda i: str(paths.get(i, "")))
    missing = int((gt["path"] == "").sum())
    gt = gt[gt["path"] != ""].copy()
    split = pd.read_csv(args.split)
    overlap = sorted(set(gt["lesion_id"]) & set(split["lesion_id"]))
    gt_ext = gt[~gt["lesion_id"].isin(overlap)].reset_index(drop=True)
    info = {"gt_file": str(args.gt), "mel_nv_images_in_gt": int(len(gt) + missing), "images_not_found": missing,
            "lesions_overlapping_HAM10000_removed": len(overlap),
            "images_removed_for_overlap": int(len(gt) - len(gt_ext)),
            "holdout_images": int(len(gt_ext)), "holdout_mel_images": int(gt_ext["label"].sum()),
            "holdout_lesions": int(gt_ext["lesion_id"].nunique()),
            "holdout_mel_lesions": int(gt_ext.loc[gt_ext.label == 1, "lesion_id"].nunique()),
            "has_lesion_ids": not gt_ext["lesion_id"].str.startswith("IMG_").all(),
            "dx_type_counts": ({f"{a}|{b}": int(n) for (a, b), n in gt_ext.groupby(["dx", "dx_type"]).size().items()}
                               if "dx_type" in gt_ext.columns else {}),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()}
    print(json.dumps(info, indent=2))
    if len(gt_ext) == 0:
        raise SystemExit("No holdout images found - check --images and --gt.")
    gt_ext.drop(columns=["path"]).to_csv(args.out / "holdout_images.csv", index=False)
    loader = DataLoader(HamDataset(gt_ext, build_transforms(False)), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    # internal test images for a pipeline check (must reproduce the stored predictions)
    first_stored = pd.read_csv(sorted((args.runs / "hifuse_small").glob("seed*"))[0] / "preds_test.csv")
    internal = (split[(split["split"] == "test") & split["image_id"].isin(first_stored["image_id"])]
                .reset_index(drop=True).head(64))
    check_loader = DataLoader(HamDataset(internal, build_transforms(False)), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)

    rows, preds, checks = [], {}, []
    for name in MODELS:
        for run_dir in sorted((args.runs / name).glob("seed*")):
            seed = int(run_dir.name.replace("seed", ""))
            model = load_run(run_dir, name, device)
            stored = pd.read_csv(run_dir / "preds_test.csv").set_index("image_id")["prob_mel"]
            p_chk = predict(model, check_loader, device, amp_dtype)
            checks.append({"model": name, "seed": seed,
                           "max_abs_diff_vs_stored": float(np.abs(p_chk - stored.loc[internal["image_id"]].to_numpy()).max())})
            p = predict(model, loader, device, amp_dtype)
            df = gt_ext[["image_id", "lesion_id", "label"]].assign(prob_mel=p)
            df.to_csv(args.out / "preds" / f"{name}_seed{seed}.csv", index=False)
            preds[(name, seed)] = df
            val = pd.read_csv(run_dir / "preds_val.csv")
            thr = youden_threshold(val["label"], val["prob_mel"])
            for level, d in (("image", df), ("lesion", to_lesion_level(df))):
                for mode, t in (("0.5", 0.5), ("val", thr)):
                    rows.append({"model": name, "seed": seed, "level": level, "threshold_mode": mode,
                                 **binary_metrics(d["label"], d["prob_mel"], t)})
            print(f"{name} seed {seed}: AUROC {rows[-4]['auroc']:.4f} MCC@0.5 {rows[-4]['mcc']:.4f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    per_run = pd.DataFrame(rows)
    per_run.to_csv(args.out / "per_run_metrics.csv", index=False)
    pd.DataFrame(checks).to_csv(args.out / "pipeline_check.csv", index=False)

    # ---------- summary
    md = ["# Untouched holdout: ISIC 2018 Task 3 test set (melanoma vs nevus)\n",
          "```", json.dumps(info, indent=2), "```\n",
          f"Pipeline check (internal test images re-predicted, max |diff| vs stored): "
          f"{pd.DataFrame(checks)['max_abs_diff_vs_stored'].max():.4f}\n"]
    keys = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc", "auroc", "auprc"]
    for (level, mode), g in per_run.groupby(["level", "threshold_mode"]):
        md.append(f"\n## {level} level, threshold {mode}\n")
        md.append("| model | " + " | ".join(keys) + " |"); md.append("|---" * (len(keys) + 1) + "|")
        for name, gm in g.groupby("model"):
            md.append(f"| {MODELS[name]} | " + " | ".join(f"{gm[k].mean():.4f} ± {gm[k].std(ddof=1):.4f}" for k in keys) + " |")
        a = g[g.model == "hifuse_small"].sort_values("seed"); b = g[g.model == "medvit_small"].sort_values("seed")
        if len(a) == len(b) > 1:
            md.append("\n| paired t-test (HiFuse − MedViT) | diff | t | p |"); md.append("|---|---|---|---|")
            for k in ("mcc", "auroc", "recall", "balanced_accuracy"):
                t, pv = ttest_rel(a[k].to_numpy(), b[k].to_numpy())
                md.append(f"| {k} | {a[k].mean() - b[k].mean():+.4f} | {t:.3f} | {pv:.4f} |")
    # consensus McNemar (majority vote over seeds), image and lesion level, threshold 0.5
    for level in ("image", "lesion"):
        cons = {}
        for name in MODELS:
            ds = [preds[(name, s)] if level == "image" else to_lesion_level(preds[(name, s)])
                  for s in sorted(s for (n, s) in preds if n == name)]
            votes = np.mean([(d["prob_mel"].to_numpy() >= 0.5) for d in ds], axis=0) > 0.5
            cons[name] = (votes == ds[0]["label"].to_numpy().astype(bool))
        r = mcnemar(cons["hifuse_small"], cons["medvit_small"])
        md.append(f"\nConsensus McNemar ({level} level, threshold 0.5): " + json.dumps(r))
    (args.out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------- repeated efficiency timing
    eff = []
    for rep in range(1, args.eff_repeats + 1):
        for name in MODELS:
            ck = sorted((args.runs / name).glob("seed*"))[0]
            model = load_run(ck, name, device)
            x1 = torch.randn(1, 3, 224, 224, device=device)
            xb = torch.randn(32 if device.type == "cuda" else 2, 3, 224, 224, device=device)
            with torch.inference_mode():
                if device.type == "cuda":
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                w, n = (30, 200) if device.type == "cuda" else (1, 3)
                lat = time_forward(model, x1, device, w, n)
                mem = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else float("nan")
                tb = time_forward(model, xb, device, max(1, w // 5), max(2, n // 5))
            eff.append({"repeat": rep, "model": MODELS[name], "latency_ms_mean": statistics.mean(lat),
                        "latency_ms_sd": statistics.stdev(lat), "latency_ms_median": statistics.median(lat),
                        "throughput_img_s": xb.shape[0] / (statistics.mean(tb) / 1000), "peak_mem_MB": mem})
            del model
    pd.DataFrame(eff).to_csv(args.out / "efficiency_repeats.csv", index=False)
    print(pd.DataFrame(eff).to_string(index=False))

    print(f"\nDone. Results in {args.out}/")


if __name__ == "__main__":
    main()
