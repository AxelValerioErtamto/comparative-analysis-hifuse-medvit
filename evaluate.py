"""
Evaluate all runs in runs/<model>/seed<k>/ and compare the two architectures.

Everything is reported at two levels:
  image  – one prediction per image
  lesion – images of the same lesion averaged into one prediction. Lesions are the
           independent units of the partition.
and at two operating points:
  0.5    – the default threshold
  val    – Youden-optimal threshold chosen on that run's VALIDATION predictions,
           then applied unchanged to test (no test information used)

Confidence intervals: cluster bootstrap over test LESIONS (images of one lesion are
resampled together), applied to the across-seed mean.
Tests: per-seed McNemar (exact, Holm-corrected) at lesion level, consensus McNemar,
and paired t-test across seeds.

Usage:  python evaluate.py --runs runs --out results
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from metrics import binary_metrics, to_lesion_level, youden_threshold

KEY_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc", "auroc", "auprc"]
CI_METRICS = ["accuracy", "recall", "f1", "balanced_accuracy", "mcc", "auroc", "auprc"]
LABELS = {"hifuse_small": "HiFuse-Small", "medvit_small": "MedViT-Small"}
COLORS = {"hifuse_small": "tab:blue", "medvit_small": "tab:red"}
SWEEP_THRESHOLDS = [round(t, 1) for t in np.arange(0.1, 0.91, 0.1)]
SWEEP_METRICS = ["precision", "recall", "specificity", "f1", "balanced_accuracy", "mcc"]


def classification_report_df(y, p, thr) -> pd.DataFrame:
    """sklearn-style report: nv, mel, accuracy, macro avg, weighted avg."""
    from sklearn.metrics import precision_recall_fscore_support
    pred = (p >= thr).astype(int)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    df = pd.DataFrame({"precision": pr, "recall": rc, "f1-score": f1, "support": sup}, index=["nv", "mel"])
    acc = float((pred == y).mean())
    w = sup / sup.sum()
    df.loc["accuracy"] = [np.nan, np.nan, acc, sup.sum()]
    df.loc["macro avg"] = [pr.mean(), rc.mean(), f1.mean(), sup.sum()]
    df.loc["weighted avg"] = [(pr * w).sum(), (rc * w).sum(), (f1 * w).sum(), sup.sum()]
    return df


def discover(runs: Path) -> dict[str, dict[int, Path]]:
    found: dict[str, dict[int, Path]] = {}
    for p in sorted(runs.glob("*/seed*/preds_test.csv")):
        model, seed = p.parent.parent.name, int(p.parent.name.removeprefix("seed"))
        found.setdefault(model, {})[seed] = p.parent
    if not found:
        raise SystemExit(f"no runs with preds_test.csv under {runs}")
    # paired tests and the shared bootstrap need every run to have the SAME test cases in the SAME order
    ref = None
    for model, seeds in found.items():
        for seed, d in seeds.items():
            ids = pd.read_csv(d / "preds_test.csv", usecols=["image_id"])["image_id"].tolist()
            if ref is None:
                ref = (ids, f"{model}/seed{seed}")
            elif ids != ref[0]:
                raise SystemExit(f"{model}/seed{seed} has a different test set from {ref[1]}. "
                                 "All runs must use the same split CSV (and the same --limit_lesions).")
    return found


def level_frames(run_dir: Path, level: str):
    val, test = pd.read_csv(run_dir / "preds_val.csv"), pd.read_csv(run_dir / "preds_test.csv")
    if level == "lesion":
        val, test = to_lesion_level(val), to_lesion_level(test)
    return val, test


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def to_md(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|---" * len(df.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def mcnemar(correct_a, correct_b) -> dict:
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())  # only B right, only A right
    n = n01 + n10
    exact = stats.binomtest(min(n01, n10), n, 0.5).pvalue if n else 1.0
    chi2 = (abs(n10 - n01) - 1) ** 2 / n if n else 0.0
    return {"both_correct": int((a & b).sum()), "both_wrong": int((~a & ~b).sum()),
            "only_A_correct": n10, "only_B_correct": n01,
            "chi2_cc": chi2, "p_chi2": float(stats.chi2.sf(chi2, 1)), "p_exact": float(exact)}


def bootstrap_mean_ci(frames: list[pd.DataFrame], thresholds: list[float], n_boot: int, seed: int,
                      level: str) -> dict:
    """CI of the across-seed mean metric, resampling test lesions (all runs share the test set)."""
    rng = np.random.default_rng(seed)
    base = frames[0]
    lesion_codes, lesions = pd.factorize(base["lesion_id"])
    members = [np.flatnonzero(lesion_codes == k) for k in range(len(lesions))]
    probs = [f["prob_mel"].to_numpy() for f in frames]
    y = base["label"].to_numpy()
    out = {m: [] for m in CI_METRICS}
    warnings.filterwarnings("ignore", message="Mean of empty slice")  # resample without a melanoma -> NaN, skipped
    for _ in range(n_boot):
        pick = rng.integers(0, len(lesions), len(lesions))
        idx = pick if level == "lesion" else np.concatenate([members[k] for k in pick])
        vals = [binary_metrics(y[idx], p[idx], t) for p, t in zip(probs, thresholds)]
        for m in CI_METRICS:
            out[m].append(np.nanmean([v[m] for v in vals]))
    return {m: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for m, v in out.items()}


def plot_curves(found, out: Path):
    """Mean ± SD over seeds. Runs stop at different epochs (early stopping), so an epoch is only
    drawn while at least half of the runs are still training; otherwise the tail is 1–2 runs."""
    import matplotlib.pyplot as plt
    colors = {"hifuse_small": "tab:blue", "medvit_small": "tab:red"}
    all_h = {m: [pd.read_csv(d / "history.csv") for d in runs.values()] for m, runs in found.items()}
    clean = all(("train_eval_loss" in h.columns) for hs in all_h.values() for h in hs)
    if clean:  # training set scored like validation: directly comparable
        panels = [("train_eval_loss", "val_loss", "Loss (train scored like validation)"),
                  ("train_eval_balanced_accuracy", "val_balanced_accuracy", "Balanced accuracy"),
                  ("train_eval_auroc", "val_auroc", "ROC-AUC")]
        tags = ("train", "val")
    else:      # older runs: training loss/acc are on augmented, oversampled batches
        panels = [("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy"),
                  (None, "val_auprc", "Val AUPRC (image)")]
        tags = ("train (augmented batches)", "val")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for model, hs in all_h.items():
        n = max(len(h) for h in hs)
        active = np.array([sum(len(h) > e for h in hs) for e in range(n)])
        keep = int(np.max(np.nonzero(active >= int(np.ceil(len(hs) / 2)))[0])) + 1
        best = [int(json.loads((d / "summary.json").read_text(encoding="utf-8"))["best_epoch"])
                for d in found[model].values()]
        for ax, (tr, va, title) in zip(axes, panels):
            for col, ls, tag in [(tr, "--", tags[0]), (va, "-", tags[1])]:
                if col is None:
                    continue
                arr = np.full((len(hs), n), np.nan)
                for i, h in enumerate(hs):
                    arr[i, :len(h)] = h[col].to_numpy()
                mu, sd = np.nanmean(arr, 0)[:keep], np.nanstd(arr, 0)[:keep]
                x = np.arange(1, keep + 1)
                ax.plot(x, mu, ls, color=colors.get(model), label=f"{LABELS.get(model, model)} {tag}")
                ax.fill_between(x, mu - sd, mu + sd, color=colors.get(model), alpha=0.15)
            ax.axvline(np.mean(best), color=colors.get(model), ls=":", lw=1.2)
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.text(0.5, 0.005, "Mean ± SD over seeds, drawn while at least half of the runs are still training "
             "(early stopping). Dotted line = mean selected epoch per model.",
             ha="center", va="bottom", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "fig_training_curves.png", dpi=200); plt.close(fig)


def plot_pr_roc(found, out: Path, level: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    colors = {"hifuse_small": "tab:blue", "medvit_small": "tab:red"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    prevalence = None
    for model, runs in found.items():
        for i, d in enumerate(runs.values()):
            _, t = level_frames(d, level)
            prevalence = t["label"].mean()
            pr, rc, _ = precision_recall_curve(t["label"], t["prob_mel"])
            fpr, tpr, _ = roc_curve(t["label"], t["prob_mel"])
            lab = LABELS.get(model, model) if i == 0 else None
            a1.plot(rc, pr, color=colors.get(model), alpha=0.6, label=lab)
            a2.plot(fpr, tpr, color=colors.get(model), alpha=0.6, label=lab)
    a1.axhline(prevalence, ls=":", color="gray", label=f"chance ({prevalence:.3f})")
    a2.plot([0, 1], [0, 1], ":", color="gray")
    a1.set(xlabel="Recall (melanoma)", ylabel="Precision", title=f"Precision-recall ({level} level)")
    a2.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC ({level} level)")
    for a in (a1, a2):
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"fig_pr_roc_{level}.png", dpi=200); plt.close(fig)


def plot_threshold_sweep(found, cache, out: Path, level: str):
    import matplotlib.pyplot as plt
    grid = np.round(np.arange(0.02, 0.981, 0.01), 2)
    panels = [("precision", "Precision (melanoma)"), ("recall", "Recall / sensitivity (melanoma)"),
              ("specificity", "Specificity (nevus recall)"), ("f1", "F1 (melanoma)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for model, runs in found.items():
        frames = [cache[(model, s, level, "0.5")][0] for s in sorted(runs)]
        vals = {k: np.array([[binary_metrics(f["label"], f["prob_mel"], t)[k] for t in grid] for f in frames])
                for k, _ in panels}
        val_thr = np.mean([cache[(model, s, level, "val")][1] for s in sorted(runs)])
        for ax, (k, title) in zip(axes, panels):
            mu, sd = vals[k].mean(0), vals[k].std(0)
            ax.plot(grid, mu, color=COLORS.get(model), label=LABELS.get(model, model))
            ax.fill_between(grid, mu - sd, mu + sd, color=COLORS.get(model), alpha=0.15)
            ax.axvline(val_thr, color=COLORS.get(model), ls=":", lw=1)
            ax.set(title=title, xlabel="Decision threshold on P(melanoma)", ylim=(0, 1.02))
            ax.grid(alpha=0.3)
    for ax in axes:
        ax.axvline(0.5, color="gray", ls="--", lw=1)
    axes[0].legend(fontsize=8, title="dotted = val-tuned threshold", title_fontsize=7)
    fig.suptitle(f"Threshold sweep on the test set ({level} level, mean ± SD over seeds)")
    fig.tight_layout(); fig.savefig(out / f"fig_threshold_sweep_{level}.png", dpi=200); plt.close(fig)


def plot_confusion(per_run, found, out: Path, level: str, mode: str):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(found), figsize=(5.2 * len(found), 4.4))
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, found):
        sub = per_run[(per_run.model == model) & (per_run.level == level) & (per_run.threshold_mode == mode)]
        # rows = true (nv, mel), cols = predicted (nv, mel)
        cells = np.array([[sub.tn, sub.fp], [sub.fn, sub.tp]], dtype=object)
        mean = np.array([[c.mean() for c in row] for row in cells])
        sd = np.array([[c.std(ddof=1) if len(c) > 1 else 0.0 for c in row] for row in cells])
        ax.imshow(mean / mean.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                frac = mean[i, j] / mean[i].sum()
                ax.text(j, i, f"{mean[i, j]:.1f} ± {sd[i, j]:.1f}\n({frac:.1%} of row)", ha="center", va="center",
                        color="white" if frac > 0.5 else "black", fontsize=10)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["nevus", "melanoma"], yticklabels=["nevus", "melanoma"],
               xlabel="Predicted", ylabel="True", title=f"{LABELS.get(model, model)} (n = {len(sub)} seeds)")
    thr = "threshold tuned on validation" if mode == "val" else "threshold 0.5"
    fig.suptitle(f"Test-set confusion matrix, mean ± SD of per-seed counts ({level} level, {thr})", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"fig_confusion_{level}_{'valthr' if mode == 'val' else 'thr0.5'}.png",
                                    dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    found = discover(args.runs)
    print({m: sorted(r) for m, r in found.items()})

    # ---------------- per-run metrics
    per_run, cache = [], {}
    for model, runs in found.items():
        for seed, d in runs.items():
            for level in ("image", "lesion"):
                val, test = level_frames(d, level)
                for mode in ("0.5", "val"):
                    thr = 0.5 if mode == "0.5" else youden_threshold(val["label"], val["prob_mel"])
                    m = binary_metrics(test["label"], test["prob_mel"], thr)
                    per_run.append({"model": model, "seed": seed, "level": level, "threshold_mode": mode, **m})
                    cache[(model, seed, level, mode)] = (test, thr)
    per_run = pd.DataFrame(per_run)
    for model, runs in found.items():  # attach best epoch for the record
        for seed, d in runs.items():
            h = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            per_run.loc[(per_run.model == model) & (per_run.seed == seed), "best_epoch"] = h["best_epoch"]
    per_run.to_csv(args.out / "per_run_metrics.csv", index=False)

    # ---------------- summary: mean ± sd [95% cluster-bootstrap CI]
    rows, md = [], []
    for level in ("lesion", "image"):
        for mode in ("val", "0.5"):
            md.append(f"\n### Test set — {level} level — threshold: "
                      f"{'Youden on validation' if mode == 'val' else '0.5'}\n")
            md.append("| Model | " + " | ".join(KEY_METRICS) + " |")
            md.append("|---" * (len(KEY_METRICS) + 1) + "|")
            for model, runs in found.items():
                seeds = sorted(runs)
                sub = per_run[(per_run.model == model) & (per_run.level == level) & (per_run.threshold_mode == mode)]
                frames = [cache[(model, s, level, mode)][0] for s in seeds]
                thrs = [cache[(model, s, level, mode)][1] for s in seeds]
                ci = bootstrap_mean_ci(frames, thrs, args.n_boot, seed=0, level=level)
                cells = []
                for k in KEY_METRICS:
                    mu, sd = sub[k].mean(), sub[k].std(ddof=1) if len(sub) > 1 else 0.0
                    cell = f"{mu:.4f} ± {sd:.4f}"
                    if k in ci:
                        cell += f" [{ci[k][0]:.4f}, {ci[k][1]:.4f}]"
                    cells.append(cell)
                    rows.append({"model": model, "level": level, "threshold_mode": mode, "metric": k,
                                 "mean": mu, "sd": sd, "ci_low": ci.get(k, (np.nan,))[0],
                                 "ci_high": ci.get(k, (np.nan, np.nan))[1], "n_seeds": len(seeds)})
                md.append(f"| {LABELS.get(model, model)} | " + " | ".join(cells) + " |")
            # confusion matrices: every seed separately + the mean, so the source is unambiguous
            md.append("\nConfusion matrices on the test set (positive class = melanoma). Each seed's row reproduces "
                      "that seed's metrics exactly; the table above averages the per-seed metrics, the last row "
                      "of each model averages the per-seed counts.\n")
            md.append("| Model | seed | threshold | TP | FN | FP | TN |")
            md.append("|---|---|---|---|---|---|---|")
            for model in found:
                sub = per_run[(per_run.model == model) & (per_run.level == level)
                              & (per_run.threshold_mode == mode)].sort_values("seed")
                for r in sub.itertuples():
                    md.append(f"| {LABELS.get(model, model)} | {r.seed} | {r.threshold:.3f} | "
                              f"{r.tp} | {r.fn} | {r.fp} | {r.tn} |")
                md.append(f"| **{LABELS.get(model, model)}** | **mean ± SD** | {sub.threshold.mean():.3f} | "
                          + " | ".join(f"{sub[c].mean():.1f} ± {sub[c].std(ddof=1) if len(sub) > 1 else 0:.1f}"
                                       for c in ("tp", "fn", "fp", "tn")) + " |")
    pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False)

    # ---------------- classification report: both classes + macro / weighted averages
    md.append("\n## Classification report (test set, mean ± SD over seeds)\n")
    cr_rows = []
    for level in ("lesion", "image"):
        for mode in ("val", "0.5"):
            md.append(f"\n### {level} level — threshold: {'Youden on validation' if mode == 'val' else '0.5'}\n")
            md.append("| Model | class | precision | recall | f1-score | support |")
            md.append("|---|---|---|---|---|---|")
            for model, runs in found.items():
                per_seed = []
                for s in sorted(runs):
                    test, thr = cache[(model, s, level, mode)]
                    per_seed.append(classification_report_df(test["label"].to_numpy(), test["prob_mel"].to_numpy(), thr))
                stacked = pd.concat(per_seed, keys=sorted(runs), names=["seed", "class"])
                for cls in per_seed[0].index:
                    vals = stacked.xs(cls, level="class")
                    cells = []
                    for col in ("precision", "recall", "f1-score"):
                        mu, sd = vals[col].mean(), vals[col].std(ddof=1) if len(vals) > 1 else 0.0
                        if np.isnan(mu):  # accuracy row only has a single value, like sklearn's report
                            cells.append("")
                            continue
                        cells.append(f"{mu:.4f} ± {sd:.4f}")
                        cr_rows.append({"model": model, "level": level, "threshold_mode": mode, "class": cls,
                                        "metric": col, "mean": mu, "sd": sd})
                    md.append(f"| {LABELS.get(model, model)} | {cls} | " + " | ".join(cells)
                              + f" | {int(vals['support'].iloc[0])} |")
    pd.DataFrame(cr_rows).to_csv(args.out / "classification_report.csv", index=False)

    # ---------------- statistics (needs both models)
    if len(found) == 2:
        A, B = list(found)
        common = sorted(set(found[A]) & set(found[B]))
        md.append(f"\n## Statistical comparison: {LABELS.get(A, A)} (A) vs {LABELS.get(B, B)} (B), seeds {common}\n")
        for level in ("lesion", "image"):
            for mode in ("val", "0.5"):
                md.append(f"\n### McNemar — {level} level, threshold {mode}"
                          + (" (images within a lesion are NOT independent; treat as descriptive)" if level == "image" else ""))
                res = []
                correct = {}
                for s in common:
                    (ta, tha), (tb, thb) = cache[(A, s, level, mode)], cache[(B, s, level, mode)]
                    key = "lesion_id" if level == "lesion" else "image_id"
                    ta = ta.set_index(key)
                    tb = tb.set_index(key).loc[ta.index]  # align the two models on the same cases
                    ca = (ta["prob_mel"] >= tha).astype(int) == ta["label"]
                    cb = (tb["prob_mel"] >= thb).astype(int) == tb["label"]
                    correct[s] = (ca.to_numpy(), cb.to_numpy())
                    res.append({"seed": s, **mcnemar(ca, cb)})
                res = pd.DataFrame(res)
                res["p_exact_holm"] = holm(res["p_exact"])
                # consensus: majority vote of seeds per model
                va = np.mean([c[0] for c in correct.values()], 0) > 0.5
                vb = np.mean([c[1] for c in correct.values()], 0) > 0.5
                cons = mcnemar(va, vb)
                md.append(to_md(res))
                md.append(f"\nConsensus (majority over seeds): only A right {cons['only_A_correct']}, only B right "
                          f"{cons['only_B_correct']}, chi2 {cons['chi2_cc']:.3f}, p_chi2 {cons['p_chi2']:.4f}, "
                          f"p_exact {cons['p_exact']:.4f}")
                res.assign(level=level, threshold_mode=mode).to_csv(
                    args.out / f"mcnemar_{level}_{mode}.csv", index=False)

        paired_rows = []
        for level in ("lesion", "image"):
            for mode in ("val", "0.5"):
                md.append(f"\n### Paired tests across seeds (seed-matched) — {level} level, threshold {mode}\n")
                md.append("| metric | mean A | mean B | diff (A−B) | t | p (paired t) | p (Wilcoxon) | Cohen's dz |")
                md.append("|---|---|---|---|---|---|---|---|")
                sub = per_run[(per_run.level == level) & (per_run.threshold_mode == mode)]
                for k in KEY_METRICS:
                    a = sub[sub.model == A].set_index("seed").loc[common, k].to_numpy()
                    b = sub[sub.model == B].set_index("seed").loc[common, k].to_numpy()
                    d = a - b
                    t = p = dz = pw = float("nan")
                    if len(common) > 1 and d.std(ddof=1) > 0:
                        t, p = stats.ttest_rel(a, b)
                        dz = d.mean() / d.std(ddof=1)
                    if len(common) > 1 and np.any(d != 0):
                        pw = stats.wilcoxon(a, b).pvalue
                    paired_rows.append({"level": level, "threshold_mode": mode, "metric": k, "mean_A": a.mean(),
                                        "mean_B": b.mean(), "diff": d.mean(), "t": t, "p_ttest": p,
                                        "p_wilcoxon": pw, "cohens_dz": dz, "n_seeds": len(common)})
                    md.append(f"| {k} | {a.mean():.4f} | {b.mean():.4f} | {d.mean():+.4f} | {t:.3f} | {p:.4f} "
                              f"| {pw:.4f} | {dz:.3f} |")
        pd.DataFrame(paired_rows).to_csv(args.out / "paired_tests.csv", index=False)
        md.append(f"\nNote: with n = {len(common)} seeds the smallest possible two-sided Wilcoxon p is "
                  f"{2 / 2 ** len(common):.4f}, so the paired t-test is the primary across-seed test "
                  "and Wilcoxon is reported for completeness.")

    # ---------------- threshold sweep (post-hoc, no retraining)
    md.append("\n## Threshold sweep (test set, mean ± SD over seeds)\n")
    md.append("Same trained models, only the decision threshold on P(melanoma) changes. "
              "For reporting, the threshold is chosen on validation (Youden); this sweep shows the "
              "precision–recall trade-off around it.\n")
    sweep_rows = []
    for level in ("lesion", "image"):
        for model, runs in found.items():
            md.append(f"\n### {LABELS.get(model, model)} — {level} level\n")
            md.append("| threshold | " + " | ".join(SWEEP_METRICS) + " | FP | FN |")
            md.append("|---" * (len(SWEEP_METRICS) + 3) + "|")
            for thr in SWEEP_THRESHOLDS:
                ms = [binary_metrics(cache[(model, s, level, "0.5")][0]["label"],
                                     cache[(model, s, level, "0.5")][0]["prob_mel"], thr) for s in sorted(runs)]
                cells = []
                for k in SWEEP_METRICS + ["fp", "fn"]:
                    v = np.array([m[k] for m in ms], float)
                    sd = v.std(ddof=1) if len(v) > 1 else 0.0
                    cells.append(f"{v.mean():.1f}" if k in ("fp", "fn") else f"{v.mean():.4f} ± {sd:.4f}")
                    sweep_rows.append({"model": model, "level": level, "threshold": thr, "metric": k,
                                       "mean": v.mean(), "sd": sd})
                md.append(f"| {thr:.1f} | " + " | ".join(cells) + " |")
    pd.DataFrame(sweep_rows).to_csv(args.out / "threshold_sweep.csv", index=False)

    (args.out / "report.md").write_text("# Results (lesion-level split)\n" + "\n".join(md) + "\n",
                                        encoding="utf-8")  # Windows default (cp1252) can't encode − / ±
    if not args.no_plots:
        plot_curves(found, args.out)
        for level in ("lesion", "image"):
            plot_pr_roc(found, args.out, level)
            plot_threshold_sweep(found, cache, args.out, level)
            for mode in ("val", "0.5"):
                plot_confusion(per_run, found, args.out, level, mode)
    print("\n".join(md))
    print(f"\nwrote {args.out}/report.md, summary.csv, per_run_metrics.csv and figures")


if __name__ == "__main__":
    main()
