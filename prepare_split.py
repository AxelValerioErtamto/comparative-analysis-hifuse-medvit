"""
Lesion-level 70/15/15 split of the HAM10000 melanoma-vs-nevus subset.

Why: HAM10000 often contains several dermoscopic images of the SAME lesion
(same `lesion_id`, different `image_id`). Splitting by image lets photos of one
lesion land in both train and test, so the test score partly measures memorisation.
Here the unit of splitting is the lesion: every image of a lesion goes to the
same partition.

How:
  1. keep dx in {mel, nv}; label 1 = melanoma, 0 = nevus
  2. build a lesion table (one row per lesion_id)
  3. split LESIONS 70/15/15, stratified by (diagnosis x number-of-images bucket),
     so class ratio AND duplicate structure are similar in every partition
  4. every image inherits its lesion's partition
  5. hard checks: no lesion_id / image_id / identical file appears in two partitions

Usage:
  python prepare_split.py --metadata data/HAM10000_metadata.csv \
      --images data/HAM10000_images_part_1 data/HAM10000_images_part_2 \
      --out splits/lesion_split_seed42.csv --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = {"nv": 0, "mel": 1}
IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- io
def read_metadata(path: Path) -> pd.DataFrame:
    """Reads the metadata whether it is comma- or tab-separated (Dataverse's .tab export),
    with or without a file extension."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    missing = {"lesion_id", "image_id", "dx"} - set(df.columns)
    if missing:
        sys.exit(f"Metadata is missing columns {missing}. Found: {list(df.columns)}")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.strip('"')
    return df


def index_images(roots: list[Path]) -> dict[str, Path]:
    """Maps image_id (file stem) -> path, searching every root recursively."""
    found: dict[str, Path] = {}
    dupes = 0
    for root in roots:
        if not root.exists():
            sys.exit(f"Image folder not found: {root}")
        for p in root.rglob("*"):
            if p.suffix.lower() in IMG_EXT and p.is_file():
                if p.stem in found:
                    dupes += 1
                    continue
                found[p.stem] = p.resolve()
    if dupes:
        print(f"[warn] {dupes} image files found more than once across folders; kept the first copy.")
    return found


def is_readable(path: Path) -> bool:
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() leaves the file unusable, so reopen and decode
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------ split
def split_lesions(lesions: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> pd.Series:
    """Returns a Series lesion_id -> split, stratified on label x image-count bucket."""
    bucket = lesions["n_images"].clip(upper=3)  # 1, 2, 3+ images per lesion
    strat = lesions["label"].astype(str) + "_" + bucket.astype(str)
    # every stratum must be large enough to appear in the smallest partition
    min_needed = int(np.ceil(2 / min(val_frac, test_frac)))
    if strat.value_counts().min() < min_needed:
        print("[info] some (label, n_images) strata are tiny; stratifying on label only.")
        strat = lesions["label"].astype(str)

    holdout = val_frac + test_frac
    tr, rest = train_test_split(lesions.index, test_size=holdout, stratify=strat, random_state=seed)
    va, te = train_test_split(rest, test_size=test_frac / holdout, stratify=strat.loc[rest],
                              random_state=seed)
    out = pd.Series(index=lesions.index, dtype=object)
    out.loc[tr], out.loc[va], out.loc[te] = "train", "val", "test"
    return out


def simulate_image_level_leak(df: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> dict:
    """Quantifies the leak of the OLD approach: a stratified split by image."""
    holdout = val_frac + test_frac
    tr, rest = train_test_split(df.index, test_size=holdout, stratify=df["label"], random_state=seed)
    _, te = train_test_split(rest, test_size=test_frac / holdout, stratify=df.loc[rest, "label"],
                             random_state=seed)
    train_lesions = set(df.loc[tr, "lesion_id"])
    test = df.loc[te]
    leaked = test["lesion_id"].isin(train_lesions)
    mel = test["label"] == 1
    return {
        "test_images": int(len(test)),
        "test_images_whose_lesion_is_also_in_train": int(leaked.sum()),
        "pct_test_images_leaked": round(100 * leaked.mean(), 2),
        "pct_test_melanoma_images_leaked": round(100 * leaked[mel].mean(), 2),
    }


def assert_no_leakage(df: pd.DataFrame) -> None:
    for key in ("lesion_id", "image_id"):
        per = df.groupby(key)["split"].nunique()
        bad = per[per > 1]
        if len(bad):
            raise AssertionError(f"{len(bad)} {key}s appear in more than one split, e.g. {bad.index[:5].tolist()}")


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, required=True, help="HAM10000_metadata.csv or .tab")
    ap.add_argument("--images", type=Path, nargs="+", required=True, help="folder(s) containing ISIC_*.jpg")
    ap.add_argument("--out", type=Path, default=Path("splits/lesion_split_seed42.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--no_verify", action="store_true", help="skip decoding every image to screen corrupt files")
    ap.add_argument("--no_hash_check", action="store_true", help="skip the byte-identical-file check")
    args = ap.parse_args()

    meta = read_metadata(args.metadata)
    print(f"metadata rows: {len(meta)} | dx counts: {meta['dx'].value_counts().to_dict()}")

    df = meta[meta["dx"].isin(CLASSES)].copy()
    if df["image_id"].duplicated().any():
        sys.exit(f"Duplicate image_id rows in metadata: {df.loc[df['image_id'].duplicated(), 'image_id'].head().tolist()}")
    multi_dx = df.groupby("lesion_id")["dx"].nunique()
    if (multi_dx > 1).any():
        sys.exit(f"Lesions with conflicting diagnoses: {multi_dx[multi_dx > 1].index[:5].tolist()}")
    df["label"] = df["dx"].map(CLASSES).astype(int)

    # ---- attach file paths, screen missing / corrupt images
    index = index_images(args.images)
    df["path"] = df["image_id"].map(index)
    missing = df["path"].isna()
    if missing.any():
        print(f"[warn] {missing.sum()} images listed in metadata were not found on disk and are dropped, "
              f"e.g. {df.loc[missing, 'image_id'].head(3).tolist()}")
        df = df[~missing].copy()
    if not args.no_verify:
        ok = df["path"].map(is_readable)
        if (~ok).any():
            print(f"[warn] {(~ok).sum()} unreadable images dropped: {df.loc[~ok, 'image_id'].tolist()[:10]}")
            df = df[ok].copy()
    df["path"] = df["path"].astype(str)

    # ---- lesion table and split
    lesions = (df.groupby("lesion_id")
                 .agg(label=("label", "first"), n_images=("image_id", "size")))
    lesion_split = split_lesions(lesions, args.seed, args.val_frac, args.test_frac)
    df["split"] = df["lesion_id"].map(lesion_split)
    df["n_images_in_lesion"] = df["lesion_id"].map(lesions["n_images"]).astype(int)
    assert_no_leakage(df)

    # ---- byte-identical files must not cross splits either
    hash_report = {"checked": False}
    if not args.no_hash_check:
        df["md5"] = df["path"].map(lambda p: md5(Path(p)))
        groups = df.groupby("md5")
        identical = groups.filter(lambda g: len(g) > 1)
        crossing = identical.groupby("md5")["split"].nunique()
        crossing = crossing[crossing > 1]
        hash_report = {"checked": True,
                       "identical_file_groups": int(identical["md5"].nunique()),
                       "groups_crossing_splits": int(len(crossing))}
        if len(crossing):
            ex = identical[identical["md5"].isin(crossing.index)][["image_id", "lesion_id", "split"]]
            print("[ERROR] byte-identical images under DIFFERENT lesion_ids ended up in different splits:")
            print(ex.head(20).to_string(index=False))
            sys.exit("Fix: merge those lesion_ids before splitting (see the rows above).")
        df = df.drop(columns="md5")

    # ---- write split
    order = ["image_id", "lesion_id", "dx", "label", "split", "n_images_in_lesion", "path"]
    extra = [c for c in ("dx_type", "age", "sex", "localization", "dataset") if c in df.columns]
    df = df[order + extra].sort_values(["split", "lesion_id", "image_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # ---- partition report
    rows = []
    for s in SPLITS:
        d = df[df["split"] == s]
        rows.append({
            "split": s,
            "nv_images": int((d.label == 0).sum()), "mel_images": int((d.label == 1).sum()),
            "total_images": len(d),
            "nv_lesions": int(d.loc[d.label == 0, "lesion_id"].nunique()),
            "mel_lesions": int(d.loc[d.label == 1, "lesion_id"].nunique()),
            "total_lesions": int(d["lesion_id"].nunique()),
            "pct_images": round(100 * len(d) / len(df), 1),
            "mel_prevalence_images": round((d.label == 1).mean(), 4),
        })
    table = pd.DataFrame(rows)

    multi = lesions[lesions.n_images > 1]
    summary = {
        "seed": args.seed,
        "n_images": len(df), "n_lesions": int(len(lesions)),
        "lesions_with_multiple_images": int(len(multi)),
        "images_in_multi_image_lesions": int(multi.n_images.sum()),
        "max_images_per_lesion": int(lesions.n_images.max()),
        "per_class": {
            dx: {"images": int((df.label == y).sum()),
                 "lesions": int((lesions.label == y).sum()),
                 "lesions_with_multiple_images": int(((lesions.label == y) & (lesions.n_images > 1)).sum())}
            for dx, y in CLASSES.items()},
        "partition": rows,
        "old_image_level_split_leak": simulate_image_level_leak(df.reset_index(drop=True), args.seed,
                                                                 args.val_frac, args.test_frac),
        "identical_file_check": hash_report,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Lesion-level partition ===")
    print(table.to_string(index=False))
    print(f"\n{summary['n_images']} images from {summary['n_lesions']} lesions; "
          f"{summary['lesions_with_multiple_images']} lesions have >1 image "
          f"({summary['images_in_multi_image_lesions']} images).")
    leak = summary["old_image_level_split_leak"]
    print(f"Old image-level split (same seed): {leak['pct_test_images_leaked']}% of test images "
          f"({leak['pct_test_melanoma_images_leaked']}% of test melanoma) had the same lesion in train.")
    print("Leakage check passed: no lesion_id / image_id overlap between splits.")
    print(f"\nwrote {args.out}\nwrote {summary_path}")


if __name__ == "__main__":
    main()
