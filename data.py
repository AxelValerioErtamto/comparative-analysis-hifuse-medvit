"""Dataset, transforms and the oversampling sampler (no undersampling anywhere)."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
CLASS_NAMES = ("nv", "mel")  # label 0 = nevus, 1 = melanoma (positive class)


class RandomRot90:
    """Rotates by a random multiple of 90 degrees. Lossless on square crops (no black corners)."""

    _ops = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)

    def __call__(self, img: Image.Image) -> Image.Image:
        op = random.choice(self._ops)
        return img if op is None else img.transpose(op)


def build_transforms(train: bool, aug: str = "strong", img_size: int = 224) -> T.Compose:
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if not train:
        return T.Compose([T.Resize((img_size, img_size)), *norm])
    if aug == "light":  # lighter alternative (one flip, ±15° rotation, brightness/contrast)
        ops = [T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
               T.RandomHorizontalFlip(),
               T.RandomRotation(15),
               T.ColorJitter(brightness=0.1, contrast=0.1)]
    elif aug == "strong":
        # Lesions have no canonical orientation, so use the full flip/rotation group.
        # Crop scale stays conservative so lesion borders are kept.
        ops = [T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
               T.RandomHorizontalFlip(),
               T.RandomVerticalFlip(),
               RandomRot90(),
               T.RandomRotation(15),
               T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)]
    else:
        raise ValueError(f"unknown aug {aug!r}")
    return T.Compose([*ops, *norm])


class HamDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), int(self.labels[i])


class MinorityOversampler(Sampler[int]):
    """
    Pure oversampling: every training image appears exactly once per epoch, then extra
    melanoma draws (with replacement) are added until both classes have equal counts.
    No nevus image is ever skipped, so nothing is undersampled.

    Extra draws are weighted 1 / (#images of that lesion), so each melanoma LESION is
    equally likely to be repeated; a lesion photographed 4 times is not repeated
    4x as often as a lesion photographed once.
    """

    def __init__(self, labels, lesion_ids, seed: int = 0):
        labels = np.asarray(labels)
        counts = np.bincount(labels, minlength=2)
        self.n = len(labels)
        self.minority = int(np.argmin(counts))
        self.n_extra = int(counts.max() - counts.min())
        self.min_idx = np.flatnonzero(labels == self.minority)
        lesions = pd.Series(np.asarray(lesion_ids)[self.min_idx])
        w = 1.0 / lesions.map(lesions.value_counts()).to_numpy(dtype=float)
        self.p = w / w.sum()
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n + self.n_extra

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 100_003 + self.epoch)
        extra = rng.choice(self.min_idx, size=self.n_extra, replace=True, p=self.p)
        idx = np.concatenate([np.arange(self.n), extra])
        rng.shuffle(idx)
        return iter(idx.tolist())


def class_weights(labels) -> torch.Tensor:
    """Inverse-frequency weights for weighted cross-entropy (the no-resampling alternative)."""
    counts = np.bincount(np.asarray(labels), minlength=2).astype(float)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def load_split(csv_path, limit_lesions_per_class: int | None = None) -> pd.DataFrame:
    """Reads the split CSV and re-checks that no lesion crosses partitions.
    `limit_lesions_per_class` keeps a small lesion-level subset per split (pilot/smoke runs);
    the subset is fixed (not seed dependent) so every run sees the same test cases."""
    df = pd.read_csv(csv_path)
    crossing = df.groupby("lesion_id")["split"].nunique()
    if (crossing > 1).any():
        raise AssertionError(f"lesion leakage in {csv_path}: {crossing[crossing > 1].index[:5].tolist()}")
    if limit_lesions_per_class:
        keep = []
        for (_, _), g in df.groupby(["split", "label"]):
            les = g["lesion_id"].drop_duplicates()
            keep.extend(les.sample(min(len(les), limit_lesions_per_class), random_state=0))
        df = df[df["lesion_id"].isin(set(keep))]
    return df.reset_index(drop=True)
