"""Metric helpers shared by train.py and evaluate.py (melanoma = positive class)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_curve


def _counts(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp = int((y & pred).sum()); tn = int((~y & ~pred).sum())
    fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    return tp, tn, fp, fn


def auroc(y, p) -> float:
    """Mann-Whitney AUROC (fast, tie-aware)."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y, p) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def binary_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    tp, tn, fp, fn = _counts(y, p >= threshold)
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"threshold": float(threshold), "n": n,
            "accuracy": (tp + tn) / n if n else float("nan"),
            "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
            "balanced_accuracy": (rec + spec) / 2, "mcc": float(mcc),
            "auroc": auroc(y, p), "auprc": auprc(y, p),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def youden_threshold(y, p) -> float:
    """Threshold maximising sensitivity + specificity - 1 (pick it on VALIDATION only)."""
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0.0, 1.0)
    return float(thr[int(np.argmax(tpr - fpr))])


def to_lesion_level(df: pd.DataFrame, prob_col: str = "prob_mel") -> pd.DataFrame:
    """One row per lesion: label of the lesion, mean melanoma probability over its images."""
    return (df.groupby("lesion_id", sort=True)
              .agg(label=("label", "first"), **{prob_col: (prob_col, "mean")}, n_images=("label", "size"))
              .reset_index())
