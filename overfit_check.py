"""
Overfitting check at the SELECTED checkpoint of every run (no training).

Training loss logged during training is computed on augmented (and, in the main experiment,
oversampled) batches, so it can't be compared directly with validation loss. Here every
partition is scored the same way: validation transform, no resampling, unweighted
cross-entropy, threshold 0.5. A small train–test gap means the selected model is not
overfitting.

Runs whose summary.json already has "train@0.5" (trained with the newer train.py) are read
directly; older runs are re-scored from best.pt (needs the GPU/CPU, ~30 s per run on a GPU).

Usage:
  python overfit_check.py --exp Main=runs --exp Regularized=runs_reg \
      --split splits/lesion_split_seed42.csv --out results_reg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"hifuse_small": "HiFuse-Small", "medvit_small": "MedViT-Small"}
METRICS = [("loss", "Loss"), ("auroc", "ROC-AUC"), ("balanced_accuracy", "Balanced acc."), ("mcc", "MCC")]


def none_or_float(v):
    return None if v in (None, "None", "") else float(v)


def rescore(run_dir: Path, split_csv: Path, workers: int) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from data import HamDataset, build_transforms, load_split
    from metrics import binary_metrics
    from models import build_model
    from train import predict, resolve_amp

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = resolve_amp(cfg.get("amp", "auto"), device)
    limit = int(cfg.get("limit_lesions", 0) or 0)
    df = load_split(split_csv, limit or None)
    model = build_model(cfg["model"], 2, none_or_float(cfg.get("drop_path")), none_or_float(cfg.get("dropout")))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
    model = model.to(device).eval()
    tf = build_transforms(False, img_size=int(cfg.get("img_size", 224)))
    out = {}
    for s in ("train", "val", "test"):
        part = df[df["split"] == s].reset_index(drop=True)
        loader = DataLoader(HamDataset(part, tf), batch_size=32, shuffle=False, num_workers=workers)
        p, loss, y = predict(model, loader, device, amp)
        out[f"{s}@0.5"] = {"loss": loss, **binary_metrics(y, p)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", action="append", required=True, help="Label=runs_dir (repeatable)")
    ap.add_argument("--split", type=Path, default=Path("splits/lesion_split_seed42.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.exp:
        label, runs = spec.split("=", 1)
        for summ in sorted(Path(runs).glob("*/seed*/summary.json")):
            s = json.loads(summ.read_text(encoding="utf-8"))
            if "train@0.5" not in s:  # older run: score the training set now and cache the result
                print(f"re-scoring {summ.parent} ...", flush=True)
                s.update(rescore(summ.parent, args.split, args.workers))
                summ.write_text(json.dumps(s, indent=2), encoding="utf-8")
            row = {"experiment": label, "model": LABELS.get(s["model"], s["model"]), "seed": s["seed"],
                   "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"]}
            for part in ("train", "val", "test"):
                for k, _ in METRICS:
                    row[f"{part}_{k}"] = s[f"{part}@0.5"][k]
            row["gap_loss"] = row["test_loss"] - row["train_loss"]
            row["gap_auroc"] = row["train_auroc"] - row["test_auroc"]
            row["gap_balanced_accuracy"] = row["train_balanced_accuracy"] - row["test_balanced_accuracy"]
            rows.append(row)
    if not rows:
        raise SystemExit("no finished runs found")
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "overfitting_check_per_run.csv", index=False)

    # ---- mean ± SD per experiment and model
    cols = [("best_epoch", "Best epoch")] + [(f"{p}_{k}", f"{n} ({p})") for k, n in METRICS for p in
                                             ("train", "val", "test")] + \
           [("gap_loss", "Loss gap (test − train)"), ("gap_auroc", "ROC-AUC gap (train − test)"),
            ("gap_balanced_accuracy", "Bal. acc. gap (train − test)")]
    g = df.groupby(["experiment", "model"], sort=False)
    summary = []
    for (exp, model), sub in g:
        r = {"Experiment": exp, "Model": model, "Runs": len(sub)}
        for c, name in cols:
            r[name] = f"{sub[c].mean():.4f} ± {sub[c].std(ddof=1) if len(sub) > 1 else 0:.4f}" \
                if c != "best_epoch" else f"{sub[c].mean():.1f} ± {sub[c].std(ddof=1) if len(sub) > 1 else 0:.1f}"
        summary.append(r)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out / "overfitting_check.csv", index=False)

    md = ["# Overfitting check at the selected checkpoint",
          "",
          "All partitions scored identically (validation transform, no resampling, unweighted cross-entropy, "
          "threshold 0.5). Mean ± SD over seeds.", ""]
    show = ["Experiment", "Model", "Runs", "Best epoch", "Loss (train)", "Loss (val)", "Loss (test)",
            "Loss gap (test − train)", "ROC-AUC (train)", "ROC-AUC (test)", "ROC-AUC gap (train − test)",
            "Balanced acc. (train)", "Balanced acc. (test)", "Bal. acc. gap (train − test)"]
    md.append("| " + " | ".join(show) + " |")
    md.append("|---" * len(show) + "|")
    for _, r in summary.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in show) + " |")
    (args.out / "overfitting_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- skimmable PNG table
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    head = ["Experiment", "Model", "Best epoch", "Loss\ntrain", "Loss\nval", "Loss\ntest", "Loss gap\n(test−train)",
            "ROC-AUC\ntrain", "ROC-AUC\ntest", "Bal. acc.\ntrain", "Bal. acc.\ntest"]
    src = ["Experiment", "Model", "Best epoch", "Loss (train)", "Loss (val)", "Loss (test)", "Loss gap (test − train)",
           "ROC-AUC (train)", "ROC-AUC (test)", "Balanced acc. (train)", "Balanced acc. (test)"]
    cells = [[str(r[c]).replace(" ± ", "\n± ") for c in src] for _, r in summary.iterrows()]
    fig, ax = plt.subplots(figsize=(16, 1.0 + 0.62 * len(cells)))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=head, cellLoc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False); t.set_fontsize(8.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_facecolor("#dfe8f3"); cell.set_text_props(weight="bold")
        elif c == 6:
            cell.set_facecolor("#fff4e0"); cell.set_text_props(weight="bold")
    fig.suptitle("Overfitting check at the selected checkpoint (all partitions scored the same way)",
                 fontsize=12, weight="bold")
    fig.text(0.5, 0.01, "Validation transform, no resampling, unweighted cross-entropy, threshold 0.5; "
             "mean ± SD over seeds. Smaller loss gap = less overfitting.", ha="center", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.12)
    fig.savefig(args.out / "overfitting_check.png", dpi=200); plt.close(fig)

    print("\n".join(md))
    print(f"\nwrote {args.out}/overfitting_check.md, .csv, .png and overfitting_check_per_run.csv")


if __name__ == "__main__":
    main()
