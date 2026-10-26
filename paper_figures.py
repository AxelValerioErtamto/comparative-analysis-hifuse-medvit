"""
Figures for the manuscript, generated from the final runs (runs_reg).

    python paper_figures.py                      # reads runs_reg/, writes results_reg/paper_figures/
    python paper_figures.py --runs runs_reg --out results_reg/paper_figures

Output (numbered as in the manuscript):
    fig1_workflow.png            Fig. 1  research workflow
    fig2_training_curves.png     Fig. 2  training vs validation curves (training set scored like validation)
    fig3_val_loss_runs.png       Fig. 3  validation loss of every run, outlier runs highlighted
    fig4_confusion_consensus.png Fig. 4  confusion matrices summed over the 5 seeds
    fig5_pr_roc.png              Fig. 5  precision-recall and ROC curves of all runs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, default=Path("runs_reg"))
ap.add_argument("--out", type=Path, default=Path("results_reg/paper_figures"))
args = ap.parse_args()
RUNS, OUT = args.runs, args.out
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"hifuse_small": ("HiFuse-Small", "#1f77b4"), "medvit_small": ("MedViT-Small", "#d62728")}
# non-colour cues so the figures stay readable in black-and-white print and for colour-blind readers
MARK = {"hifuse_small": "o", "medvit_small": "^"}            # Fig. 2 markers per model
VLINE = {"hifuse_small": ":", "medvit_small": "-."}          # Fig. 2 selected-epoch lines
CURVE = {"hifuse_small": "-", "medvit_small": (0, (4, 1.5))}  # Fig. 5 line style per model
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "font.family": "DejaVu Serif"})
runs = {m: sorted((RUNS / m).glob("seed*")) for m in MODELS}
summ = {m: [json.loads((d / "summary.json").read_text()) for d in runs[m]] for m in MODELS}

# ---------------- Fig. 1 workflow
steps = [
    ("1. Data collection", "HAM10000 binary subset\nmelanoma + nevus\n7,818 images, 6,017 lesions"),
    ("2. Preprocessing", "integrity screening,\nresize to 224 × 224,\nnormalization (0.5, 0.5)"),
    ("3. Lesion-level split", "70/15/15 by lesion_id,\nstratified; no lesion or\nidentical file shared"),
    ("4. Model construction", "official HiFuse-Small and\nMedViT-Small, 2 outputs,\nrandom initialization"),
    ("5. Validation tuning", "class weight {2, 3, 4, 6} ×\nregularization level;\nselect by validation MCC"),
    ("6. Final training", "5 seeds per model, chosen\nsetting, early stopping\non validation loss"),
    ("7. Evaluation", "per-seed metrics, confusion\nmatrices, PR/ROC curves,\nthreshold, overfitting check"),
    ("8. Statistical analysis", "bootstrap CIs, McNemar,\npaired t-test; computational\nefficiency"),
]
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
w, h = 0.86, 0.78
pos = [(i + 0.07, 1.12) for i in range(4)] + [(3 - i + 0.07, 0.1) for i in range(4)]
for (title, body), (x, y) in zip(steps, pos):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
                                fc="#eef3fa", ec="#2f5597", lw=1.2))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="center", fontsize=10, weight="bold", color="#1f3864")
    ax.text(x + w / 2, y + 0.33, body, ha="center", va="center", fontsize=8.5, color="#222222", linespacing=1.25)
arr = lambda a, b: ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#2f5597"))
for i in range(3):
    arr((pos[i][0] + w, 1.12 + h / 2), (pos[i + 1][0], 1.12 + h / 2))
arr((pos[3][0] + w / 2, 1.12), (pos[4][0] + w / 2, 0.1 + h))
for i in range(4, 7):
    arr((pos[i][0], 0.1 + h / 2), (pos[i + 1][0] + w, 0.1 + h / 2))
fig.tight_layout(pad=0.2); fig.savefig(OUT / "fig1_workflow.png", dpi=300); plt.close(fig)

# ---------------- Fig. 2 training curves (training set scored like validation)
fig, axes4 = plt.subplots(1, 4, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1, 1, 0.62]})
axes, axn = axes4[:3], axes4[3]
panels = [("train_eval_loss", "val_loss", "Loss"), ("train_eval_balanced_accuracy", "val_balanced_accuracy",
          "Balanced accuracy"), ("train_eval_auroc", "val_auroc", "ROC-AUC")]
for m, (lab, col) in MODELS.items():
    hs = [pd.read_csv(d / "history.csv") for d in runs[m]]
    n = max(len(hh) for hh in hs)
    active = np.array([sum(len(hh) > e for hh in hs) for e in range(n)])
    keep = int(np.max(np.nonzero(active >= 3)[0])) + 1
    best = np.mean([s["best_epoch"] for s in summ[m]])
    for ax, (tr, va, title) in zip(axes, panels):
        for c, ls, tag in [(tr, "--", "train"), (va, "-", "val")]:
            a = np.full((len(hs), n), np.nan)
            for i, hh in enumerate(hs):
                a[i, :len(hh)] = hh[c].to_numpy()
            mu, sd = np.nanmean(a, 0)[:keep], np.nanstd(a, 0)[:keep]
            x = np.arange(1, keep + 1)
            ax.plot(x, mu, ls, color=col, lw=1.1, label=f"{lab} {tag}", marker=MARK[m], markersize=2.6,
                    markevery=3, markerfacecolor="white" if tag == "train" else col, markeredgewidth=0.7)
            ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.axvline(best, color=col, ls=VLINE[m], lw=1)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3, lw=0.5)
    # runs contributing to the mean at each epoch (curves are drawn while >= 3 runs are active)
    axn.step(np.arange(1, n + 1), active, where="mid", color=col, ls=CURVE[m], lw=1.1, label=lab.split("-")[0])
axn.axhline(3, color="gray", ls=":", lw=0.8)
axn.set(title="Runs active", xlabel="Epoch", ylim=(0, 5.4), yticks=range(0, 6))
axn.grid(alpha=0.3, lw=0.5); axn.legend(frameon=False, loc="lower left", fontsize=6)
axes[0].legend(frameon=False, loc="upper right")
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig2_training_curves.png", dpi=400); plt.close(fig)

# ---------------- Fig. 3 aggregated confusion matrices (sum over seeds)
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.45))
for ax, (m, (lab, col)) in zip(axes, MODELS.items()):
    ts = [pd.read_csv(d / "preds_test.csv") for d in runs[m]]
    y = ts[0]["label"].to_numpy()
    p = (np.mean([t["prob_mel"].to_numpy() >= 0.5 for t in ts], axis=0) > 0.5).astype(int)  # majority of 5 seeds
    cm = np.array([[((y == 0) & (p == 0)).sum(), ((y == 0) & (p == 1)).sum()],
                   [((y == 1) & (p == 0)).sum(), ((y == 1) & (p == 1)).sum()]])
    frac = cm / cm.sum(1, keepdims=True)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for a in range(2):
        for b in range(2):
            ax.text(b, a, f"{cm[a, b]:,}\n({frac[a, b]:.1%})", ha="center", va="center", fontsize=8,
                    weight="bold", color="white" if frac[a, b] > 0.5 else "black")
    ax.set_xticks([0, 1], ["Nevus", "Melanoma"]); ax.set_yticks([0, 1], ["Nevus", "Melanoma"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{lab} (5-seed vote)")
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig4_confusion_consensus.png", dpi=400); plt.close(fig)

# ---------------- Fig. 4 PR and ROC
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.0, 2.45))
edges = np.linspace(0, 1, 11)
prev = None
for m, (lab, col) in MODELS.items():
    for k, d in enumerate(runs[m]):
        t = pd.read_csv(d / "preds_test.csv"); prev = t["label"].mean()
        p, r, _ = precision_recall_curve(t["label"], t["prob_mel"]); f, tp, _ = roc_curve(t["label"], t["prob_mel"])
        a1.plot(r, p, color=col, ls=CURVE[m], lw=0.8, alpha=0.75, label=lab if k == 0 else None)
        a2.plot(f, tp, color=col, ls=CURVE[m], lw=0.8, alpha=0.75, label=lab if k == 0 else None)
    yy = np.concatenate([pd.read_csv(d / "preds_test.csv")["label"].to_numpy() for d in runs[m]])
    pp = np.concatenate([pd.read_csv(d / "preds_test.csv")["prob_mel"].to_numpy() for d in runs[m]])
    idx = np.clip(np.digitize(pp, edges[1:-1]), 0, 9)
    mp = [pp[idx == b].mean() for b in range(10) if (idx == b).any()]
    ob = [yy[idx == b].mean() for b in range(10) if (idx == b).any()]
    a3.plot(mp, ob, color=col, ls=CURVE[m], marker=MARK[m], ms=3, lw=1, label=lab)
a3.plot([0, 1], [0, 1], ":", color="gray", lw=0.9, label="Perfect calibration")
a3.set(xlabel="Mean predicted P(melanoma)", ylabel="Observed melanoma fraction", title="Calibration")
a1.axhline(prev, ls=":", color="gray", lw=0.9, label=f"Chance ({prev:.3f})")
a2.plot([0, 1], [0, 1], ":", color="gray", lw=0.9, label="Chance")
a1.set(xlabel="Recall (melanoma)", ylabel="Precision", title="Precision-recall")
a2.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
for a in (a1, a2, a3):
    a.grid(alpha=0.3, lw=0.5); a.legend(frameon=False, fontsize=6)
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig5_pr_roc.png", dpi=400); plt.close(fig)

# ---------------- Fig. 5 validation loss of every run (early stopping and outlier runs)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True)
outlier = {("hifuse_small", 3), ("medvit_small", 4)}
for ax, (m, (lab, col)) in zip(axes, MODELS.items()):
    for d, s in zip(runs[m], summ[m]):
        hh = pd.read_csv(d / "history.csv"); seed = s["seed"]
        hi = (m, seed) in outlier
        ax.plot(hh["epoch"], hh["val_loss"], color=col if hi else "0.55", lw=1.4 if hi else 0.8,
                alpha=1 if hi else 0.8, label=f"seed {seed} (stopped at {len(hh)})" if hi else None)
        b = s["best_epoch"]
        ax.plot(b, hh.loc[hh.epoch == b, "val_loss"].iloc[0], "o", ms=3.5, color=col if hi else "0.35")
    ax.set_title(lab); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3, lw=0.5); ax.set_ylim(0.2, 0.45)
    ax.legend(frameon=False, loc="upper right")
axes[0].set_ylabel("Validation loss")
fig.tight_layout(pad=0.4); fig.savefig(OUT / "fig3_val_loss_runs.png", dpi=400); plt.close(fig)
print(f"wrote 5 figures to {OUT}/")
