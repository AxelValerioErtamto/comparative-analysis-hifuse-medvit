"""
Grad-CAM on test images the models misclassified (qualitative interpretability, no training).

For one seed (default: the first seed both models have), each model uses its own
validation-tuned threshold, and test images are grouped as:
  missed_melanoma_both   melanoma that both models called nevus      (false negatives)
  false_alarm_both       nevus that both models called melanoma      (false positives)
  wrong_only_hifuse      HiFuse wrong, MedViT right
  wrong_only_medvit      MedViT wrong, HiFuse right
  correct_melanoma_both  melanoma both detected (reference for what "correct" attention looks like)
Within each group the most confident errors are shown, one image per lesion.

Each figure row: image | HiFuse-Small Grad-CAM | MedViT-Small Grad-CAM, titled with P(melanoma).
Grad-CAM is taken for the melanoma logit at the last 7x7 feature map before global pooling:
  HiFuse-Small: `fu4` (final hierarchical feature-fusion block)
  MedViT-Small: `norm` (BatchNorm after the last stage)

Usage:  python gradcam.py --runs runs --split splits/lesion_split_seed42.csv --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from data import build_transforms
from metrics import youden_threshold
from models import build_model

MODELS = {"hifuse_small": ("HiFuse-Small", "fu4"), "medvit_small": ("MedViT-Small", "norm")}
GROUPS = ["missed_melanoma_both", "false_alarm_both", "wrong_only_hifuse", "wrong_only_medvit",
          "correct_melanoma_both"]
TITLES = {"missed_melanoma_both": "Melanoma missed by both models (false negatives)",
          "false_alarm_both": "Nevi flagged as melanoma by both models (false positives)",
          "wrong_only_hifuse": "Misclassified by HiFuse-Small only",
          "wrong_only_medvit": "Misclassified by MedViT-Small only",
          "correct_melanoma_both": "Melanoma correctly detected by both (reference)"}


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor, target: int = 1):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            logits[:, target].sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)              # channel importance
        cam = F.relu((w * self.acts).sum(1, keepdim=True))           # B,1,h,w
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam.detach().cpu().numpy(), logits.softmax(1)[:, 1].detach().cpu().numpy()


def load_preds(run_dir: Path) -> tuple[pd.DataFrame, float]:
    val, test = pd.read_csv(run_dir / "preds_val.csv"), pd.read_csv(run_dir / "preds_test.csv")
    return test, youden_threshold(val["label"], val["prob_mel"])


def select(merged: pd.DataFrame, per_group: int) -> dict[str, pd.DataFrame]:
    y, ph, pm = merged["label"], merged["p_hifuse"], merged["p_medvit"]
    wh, wm = merged["pred_hifuse"] != y, merged["pred_medvit"] != y
    groups = {
        "missed_melanoma_both": (merged[(y == 1) & wh & wm], (ph + pm), True),          # lowest P(mel) first
        "false_alarm_both": (merged[(y == 0) & wh & wm], (ph + pm), False),             # highest P(mel) first
        "wrong_only_hifuse": (merged[wh & ~wm], (ph - y).abs(), False),                 # most confident HiFuse error
        "wrong_only_medvit": (merged[wm & ~wh], (pm - y).abs(), False),
        "correct_melanoma_both": (merged[(y == 1) & ~wh & ~wm], (ph + pm), False),      # most confident hits
    }
    out = {}
    for g, (df, key, ascending) in groups.items():
        df = df.assign(_k=key.loc[df.index]).sort_values("_k", ascending=ascending)
        out[g] = df.drop_duplicates("lesion_id").head(per_group).drop(columns="_k")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--split", type=Path, default=Path("splits/lesion_split_seed42.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=None, help="which training seed's models to explain")
    ap.add_argument("--per_group", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()
    out = args.out / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [sorted(int(p.parent.name.removeprefix("seed")) for p in args.runs.glob(f"{m}/seed*/best.pt")) for m in MODELS]
    common = sorted(set(seeds[0]) & set(seeds[1]))
    if not common:
        raise SystemExit("need a finished run of both models with the same seed")
    seed = args.seed if args.seed is not None else common[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # predictions + each model's validation-tuned threshold for this seed
    merged, thr = None, {}
    for name in MODELS:
        test, t = load_preds(args.runs / name / f"seed{seed}")
        short = name.split("_")[0]
        thr[name] = t
        test = test.rename(columns={"prob_mel": f"p_{short}"})
        test[f"pred_{short}"] = (test[f"p_{short}"] >= t).astype(int)
        merged = test if merged is None else merged.merge(test[["image_id", f"p_{short}", f"pred_{short}"]], on="image_id")
    paths = pd.read_csv(args.split).set_index("image_id")["path"]
    merged["path"] = merged["image_id"].map(paths)
    chosen = select(merged, args.per_group)

    # load models + Grad-CAM hooks
    cams = {}
    for name, (_, layer) in MODELS.items():
        model = build_model(name, 2)
        model.load_state_dict(torch.load(args.runs / name / f"seed{seed}" / "best.pt", map_location="cpu",
                                         weights_only=False)["model"])
        model = model.to(device).eval()
        cams[name] = GradCAM(model, getattr(model, layer))

    tf = build_transforms(False, img_size=args.img_size)
    listing = []
    for group in GROUPS:
        df = chosen[group]
        if df.empty:
            print(f"{group}: no images in this group for seed {seed}")
            continue
        fig, axes = plt.subplots(len(df), 3, figsize=(9, 3.1 * len(df)), squeeze=False)
        for row, r in enumerate(df.itertuples()):
            img = Image.open(r.path).convert("RGB")
            shown = np.asarray(img.resize((args.img_size, args.img_size))) / 255.0
            x = tf(img).unsqueeze(0).to(device)
            axes[row, 0].imshow(shown)
            axes[row, 0].set_title(f"{r.image_id}\ntrue: {'melanoma' if r.label else 'nevus'}", fontsize=9)
            for col, name in enumerate(MODELS, start=1):
                cam, p = cams[name](x)
                pred = int(p[0] >= thr[name])
                verdict = "correct" if pred == r.label else "WRONG"
                axes[row, col].imshow(shown)
                axes[row, col].imshow(cam[0], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[row, col].set_title(f"{MODELS[name][0]}\nP(mel) = {p[0]:.3f} (thr {thr[name]:.3f}) {verdict}",
                                         fontsize=9, color="black" if verdict == "correct" else "crimson")
                listing.append({"group": group, "image_id": r.image_id, "lesion_id": r.lesion_id,
                                "true": "mel" if r.label else "nv", "model": name, "p_mel": float(p[0]),
                                "threshold": thr[name], "predicted": "mel" if pred else "nv", "seed": seed})
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle(f"{TITLES[group]} — Grad-CAM for the melanoma output (seed {seed})", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"gradcam_{group}.png", dpi=150)
        plt.close(fig)
        print(f"{group}: {len(df)} images -> {out / f'gradcam_{group}.png'}")

    pd.DataFrame(listing).to_csv(out / "gradcam_selected.csv", index=False)
    counts = {g: int(len(chosen[g])) for g in GROUPS}
    print(f"\nseed {seed}, thresholds {{'hifuse': {thr['hifuse_small']:.3f}, 'medvit': {thr['medvit_small']:.3f}}}, "
          f"shown per group: {counts}\nwrote {out}/")


if __name__ == "__main__":
    main()
