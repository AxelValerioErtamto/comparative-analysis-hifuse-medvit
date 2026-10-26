"""
Statistical analysis of the 10 final runs (runs_reg). Uses only the saved
validation/test predictions (no training, no GPU).

    python stat_analysis.py                     # writes results_reg/analysis/
    python stat_analysis.py --split splits/lesion_split_seed42.csv   # + subgroup/error analysis

Outputs (results_reg/analysis/):
  differences.csv        paired HiFuse - MedViT differences, three kinds of 95% CI:
                         lesion-sampling (cluster bootstrap, models fixed), seed (t over 5 pairs),
                         combined (resample seed pairs AND lesions)
  per_seed_pairs.csv     the 5 seed-matched differences for each metric
  lesion_level.csv       lesion-level metrics (mean probability of a lesion's images)
  consensus_confusion.csv  one prediction per test image/lesion (majority vote of 5 seeds)
  calibration.csv        Brier score and ECE per run; reliability.csv for the diagram
  thresholds.csv         per-seed validation Youden thresholds and test metrics at them
  tuning_by_model.csv    validation MCC of each model under every tuning setting
  curve_counts.csv       number of runs still training at each epoch (Fig. 2)
  subgroups.csv          (with --split) recall/FN rate by lesion attributes
  report.md              everything above in readable form
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from metrics import binary_metrics, to_lesion_level, youden_threshold

A, B = "hifuse_small", "medvit_small"
LABEL = {A: "HiFuse-Small", B: "MedViT-Small"}
METRICS = ["mcc", "auroc", "recall", "specificity", "precision", "f1", "balanced_accuracy", "accuracy", "auprc"]


def load(runs: Path):
    out = {}
    for m in (A, B):
        for d in sorted((runs / m).glob("seed*")):
            s = int(d.name[4:])
            out[(m, s)] = {"val": pd.read_csv(d / "preds_val.csv"), "test": pd.read_csv(d / "preds_test.csv"),
                           "hist": pd.read_csv(d / "history.csv"),
                           "summary": json.loads((d / "summary.json").read_text(encoding="utf-8"))}
    return out


def frame(run, level, part="test"):
    df = run[part]
    return to_lesion_level(df) if level == "lesion" else df


def boot_diffs(runs, seeds, level, thr_mode, metrics, n_boot, rng):
    """Cluster bootstrap over test lesions of the across-seed mean difference A - B.
    Returns {metric: (lesion-sampling CI, combined CI)}; 'combined' also resamples the seed pairs."""
    base = frame(runs[(A, seeds[0])], level)
    codes, les = pd.factorize(base["lesion_id"])
    members = [np.flatnonzero(codes == k) for k in range(len(les))]
    y = base["label"].to_numpy()
    P = {k: frame(runs[k], level)["prob_mel"].to_numpy() for k in runs}
    T = {k: (0.5 if thr_mode == "0.5" else youden_threshold(frame(runs[k], level, "val")["label"],
                                                             frame(runs[k], level, "val")["prob_mel"])) for k in runs}
    fixed = {k: [] for k in metrics}; comb = {k: [] for k in metrics}
    for _ in range(n_boot):
        pick = rng.integers(0, len(les), len(les))
        idx = pick if level == "lesion" else np.concatenate([members[i] for i in pick])
        if y[idx].sum() == 0:
            continue
        d = {}
        for s in seeds:
            ma = binary_metrics(y[idx], P[(A, s)][idx], T[(A, s)]); mb = binary_metrics(y[idx], P[(B, s)][idx], T[(B, s)])
            d[s] = {k: ma[k] - mb[k] for k in metrics}
        ss = rng.choice(seeds, len(seeds), replace=True)
        for k in metrics:
            fixed[k].append(np.mean([d[s][k] for s in seeds]))
            comb[k].append(np.mean([d[s][k] for s in ss]))
    return {k: (np.percentile(fixed[k], [2.5, 97.5]), np.percentile(comb[k], [2.5, 97.5])) for k in metrics}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_reg"))
    ap.add_argument("--tuning", type=Path, default=Path("results_reg/tuning_per_model.csv"))
    ap.add_argument("--split", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results_reg/analysis"))
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    runs = load(args.runs)
    seeds = sorted(s for (m, s) in runs if m == A)
    md = ["# Statistical analysis (final protocol, runs_reg)\n"]

    # ---- per-run metrics at both levels, both threshold modes
    rows = []
    for (m, s), r in runs.items():
        for level in ("image", "lesion"):
            te, va = frame(r, level), frame(r, level, "val")
            for mode in ("0.5", "val"):
                t = 0.5 if mode == "0.5" else youden_threshold(va["label"], va["prob_mel"])
                rows.append({"model": m, "seed": s, "level": level, "threshold_mode": mode,
                             **binary_metrics(te["label"], te["prob_mel"], t)})
    per = pd.DataFrame(rows)
    per.to_csv(args.out / "per_run_both_levels.csv", index=False)
    lev = per[(per.level == "lesion") & (per.threshold_mode == "0.5")]
    lev.groupby("model")[METRICS].agg(["mean", "std"]).to_csv(args.out / "lesion_level.csv")
    te0 = runs[(A, seeds[0])]["test"]; tl = to_lesion_level(te0)
    md.append(f"Test set: {len(te0)} images ({te0.label.sum()} melanoma, prevalence {te0.label.mean():.3f}); "
              f"{len(tl)} lesions ({tl.label.sum()} melanoma, prevalence {tl.label.mean():.3f}). "
              f"Lesion prediction = mean P(melanoma) over the lesion's images.\n")
    for level in ("image", "lesion"):
        for mode in ("0.5", "val"):
            g = per[(per.level == level) & (per.threshold_mode == mode)]
            md.append(f"\n## {level} level, threshold {mode}: mean ± SD over 5 seeds\n")
            md.append("| model | " + " | ".join(METRICS) + " |"); md.append("|---" * (len(METRICS) + 1) + "|")
            for m in (A, B):
                gm = g[g.model == m]
                md.append(f"| {LABEL[m]} | " + " | ".join(f"{gm[k].mean():.4f} ± {gm[k].std(ddof=1):.4f}" for k in METRICS) + " |")

    # ---- 95% cluster-bootstrap CI of each model's across-seed mean (all metrics, both levels)
    crow = []
    for m in (A, B):
        for level in ("image", "lesion"):
            for mode in ("0.5", "val"):
                fr = [frame(runs[(m, s)], level) for s in seeds]
                th = [0.5 if mode == "0.5" else youden_threshold(frame(runs[(m, s)], level, "val")["label"],
                                                                  frame(runs[(m, s)], level, "val")["prob_mel"]) for s in seeds]
                codes, les = pd.factorize(fr[0]["lesion_id"]); mem = [np.flatnonzero(codes == k) for k in range(len(les))]
                y = fr[0]["label"].to_numpy(); P = [f["prob_mel"].to_numpy() for f in fr]
                vals = {k: [] for k in METRICS}
                for _ in range(args.n_boot):
                    pick = rng.integers(0, len(les), len(les))
                    idx = pick if level == "lesion" else np.concatenate([mem[i] for i in pick])
                    if y[idx].sum() == 0:
                        continue
                    r = [binary_metrics(y[idx], p[idx], t) for p, t in zip(P, th)]
                    for k in METRICS:
                        vals[k].append(np.mean([x[k] for x in r]))
                for k in METRICS:
                    g = per[(per.model == m) & (per.level == level) & (per.threshold_mode == mode)][k]
                    crow.append({"model": m, "level": level, "threshold_mode": mode, "metric": k, "mean": g.mean(),
                                 "sd": g.std(ddof=1), "ci_lo": np.percentile(vals[k], 2.5), "ci_hi": np.percentile(vals[k], 97.5)})
    pd.DataFrame(crow).to_csv(args.out / "model_ci.csv", index=False)

    # ---- paired differences with three kinds of uncertainty
    drows, prow = [], []
    DM = ("mcc", "auroc", "recall", "specificity", "balanced_accuracy", "auprc", "f1")
    for level in ("image", "lesion"):
        for mode in ("0.5", "val"):
            g = per[(per.level == level) & (per.threshold_mode == mode)]
            cis = boot_diffs(runs, seeds, level, mode, DM, args.n_boot, rng)
            for k in DM:
                a = g[g.model == A].sort_values("seed")[k].to_numpy(); b = g[g.model == B].sort_values("seed")[k].to_numpy()
                d = a - b
                tci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
                lci, cci = cis[k]
                tt = stats.ttest_rel(a, b)
                drows.append({"level": level, "threshold_mode": mode, "metric": k, "diff_mean": d.mean(),
                              "seed_ci_lo": tci[0], "seed_ci_hi": tci[1], "lesion_ci_lo": lci[0], "lesion_ci_hi": lci[1],
                              "combined_ci_lo": cci[0], "combined_ci_hi": cci[1], "t": tt.statistic, "p_t": tt.pvalue,
                              "dz": d.mean() / d.std(ddof=1), "pairs_favouring_A": int((d > 0).sum()),
                              "pairs_favouring_B": int((d < 0).sum())})
                for s, v in zip(seeds, d):
                    prow.append({"level": level, "threshold_mode": mode, "metric": k, "seed": s, "diff": v})
    dif = pd.DataFrame(drows); dif.to_csv(args.out / "differences.csv", index=False)
    pd.DataFrame(prow).to_csv(args.out / "per_seed_pairs.csv", index=False)
    md.append("\n## Paired differences HiFuse − MedViT (95% CIs)\n")
    md.append("| level | thr | metric | diff | seed CI (t, 5 pairs) | lesion CI (cluster bootstrap) | combined CI | pairs A>B / B>A |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in dif.itertuples():
        md.append(f"| {r.level} | {r.threshold_mode} | {r.metric} | {r.diff_mean:+.4f} | [{r.seed_ci_lo:+.4f}, {r.seed_ci_hi:+.4f}] | "
                  f"[{r.lesion_ci_lo:+.4f}, {r.lesion_ci_hi:+.4f}] | [{r.combined_ci_lo:+.4f}, {r.combined_ci_hi:+.4f}] | "
                  f"{r.pairs_favouring_A}/{r.pairs_favouring_B} |")

    # ---- consensus confusion matrices (one prediction per image / lesion)
    crows = []
    for level in ("image", "lesion"):
        for m in (A, B):
            fs = [frame(runs[(m, s)], level) for s in seeds]
            y = fs[0]["label"].to_numpy().astype(bool)
            vote = np.mean([f["prob_mel"].to_numpy() >= 0.5 for f in fs], axis=0) > 0.5
            mean_p = np.mean([f["prob_mel"].to_numpy() for f in fs], axis=0)
            crows.append({"level": level, "model": m, "rule": "majority vote",
                          **{k: v for k, v in binary_metrics(y, vote.astype(float), 0.5).items() if k in ("tp", "fn", "fp", "tn", "recall", "specificity", "precision", "mcc")}})
            crows.append({"level": level, "model": m, "rule": "mean probability",
                          **{k: v for k, v in binary_metrics(y, mean_p, 0.5).items() if k in ("tp", "fn", "fp", "tn", "recall", "specificity", "precision", "mcc", "auroc")}})
    con = pd.DataFrame(crows); con.to_csv(args.out / "consensus_confusion.csv", index=False)
    md.append("\n## Consensus predictions (one per test image / lesion)\n\n" + con.to_markdown(index=False, floatfmt=".4f"))

    # ---- calibration
    cal, rel = [], []
    for (m, s), r in runs.items():
        y, p = r["test"]["label"].to_numpy(), r["test"]["prob_mel"].to_numpy()
        cal.append({"model": m, "seed": s, "brier": float(np.mean((p - y) ** 2)), "ece_10bin": ece(y, p),
                    "mean_pred": p.mean(), "prevalence": y.mean()})
    calib = pd.DataFrame(cal); calib.to_csv(args.out / "calibration.csv", index=False)
    edges = np.linspace(0, 1, 11)
    for m in (A, B):
        y = np.concatenate([runs[(m, s)]["test"]["label"].to_numpy() for s in seeds])
        p = np.concatenate([runs[(m, s)]["test"]["prob_mel"].to_numpy() for s in seeds])
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        for b in range(10):
            k = idx == b
            if k.any():
                rel.append({"model": m, "bin": b, "mean_pred": p[k].mean(), "observed": y[k].mean(), "n": int(k.sum())})
    pd.DataFrame(rel).to_csv(args.out / "reliability.csv", index=False)
    md.append("\n## Calibration (test, image level): mean ± SD over seeds\n")
    for m in (A, B):
        g = calib[calib.model == m]
        md.append(f"- {LABEL[m]}: Brier {g.brier.mean():.4f} ± {g.brier.std(ddof=1):.4f}, "
                  f"ECE {g.ece_10bin.mean():.4f} ± {g.ece_10bin.std(ddof=1):.4f}, mean predicted P {g.mean_pred.mean():.3f} "
                  f"vs prevalence {g.prevalence.mean():.3f}")

    # ---- thresholds per seed
    th = per[(per.threshold_mode == "val")][["model", "seed", "level", "threshold", "recall", "specificity", "precision",
                                             "balanced_accuracy", "mcc"]]
    th.to_csv(args.out / "thresholds.csv", index=False)
    md.append("\n## Validation-Youden thresholds per seed (image level)\n\n"
              + th[th.level == "image"].to_markdown(index=False, floatfmt=".3f"))

    # ---- tuning by model
    if args.tuning.exists():
        tu = pd.read_csv(args.tuning)
        piv = tu.pivot_table(index=["regularisation", "melanoma_weight"], columns="model", values="val_mcc").reset_index()
        piv["mean"] = piv[[A, B]].mean(axis=1)
        piv["rank_hifuse"] = piv[A].rank(ascending=False).astype(int); piv["rank_medvit"] = piv[B].rank(ascending=False).astype(int)
        piv.to_csv(args.out / "tuning_by_model.csv", index=False)
        md.append("\n## Validation MCC of each model under every tuning setting (seed 0)\n\n"
                  + piv.sort_values("mean", ascending=False).to_markdown(index=False, floatfmt=".4f"))

    # ---- runs contributing to the averaged curves (Fig. 2)
    cc = []
    for m in (A, B):
        n = max(len(runs[(m, s)]["hist"]) for s in seeds)
        for e in range(1, n + 1):
            cc.append({"model": m, "epoch": e, "runs_active": sum(len(runs[(m, s)]["hist"]) >= e for s in seeds)})
    pd.DataFrame(cc).to_csv(args.out / "curve_counts.csv", index=False)

    # ---- subgroup / error analysis (needs the split file with metadata columns)
    if args.split is not None and args.split.exists():
        sp = pd.read_csv(args.split)
        sp = sp[sp.split == "test"].drop_duplicates("image_id").set_index("image_id")
        srows = []
        for m in (A, B):
            fs = [runs[(m, s)]["test"].set_index("image_id") for s in seeds]
            df = fs[0][["lesion_id", "label"]].copy()
            df["vote"] = (np.mean([f["prob_mel"].to_numpy() >= 0.5 for f in fs], axis=0) > 0.5).astype(int)
            df = df.join(sp[[c for c in ("dx_type", "n_images_in_lesion", "localization", "age", "sex", "dataset") if c in sp]])
            if "age" in df:
                df["age_group"] = pd.cut(df["age"], [-1, 39, 59, 200], labels=["<40", "40-59", ">=60"])
            if "n_images_in_lesion" in df:
                df["images_per_lesion"] = np.where(df["n_images_in_lesion"] > 1, "multi-image", "single image")
            for col in ("dx_type", "images_per_lesion", "localization", "age_group", "sex", "dataset"):
                if col not in df:
                    continue
                for val, g in df.groupby(col, observed=True):
                    mel, nv = g[g.label == 1], g[g.label == 0]
                    srows.append({"model": m, "attribute": col, "value": val, "mel_images": len(mel), "nv_images": len(nv),
                                  "mel_recall": mel["vote"].mean() if len(mel) else np.nan,
                                  "nv_specificity": 1 - nv["vote"].mean() if len(nv) else np.nan})
        sg = pd.DataFrame(srows); sg.to_csv(args.out / "subgroups.csv", index=False)
        md.append("\n## Subgroups (consensus prediction, threshold 0.5)\n\n" + sg.to_markdown(index=False, floatfmt=".3f"))

    (args.out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.out}/report.md")


if __name__ == "__main__":
    main()
