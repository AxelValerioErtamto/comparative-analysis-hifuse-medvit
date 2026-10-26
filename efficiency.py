"""
Computational efficiency of the trained models (no training involved):
  * parameters
  * GMACs and GFLOPs for one 224x224 image (PyTorch's built-in counter; GFLOPs = 2 x GMACs).
    fvcore/thop report MACs under the name "FLOPs".
  * inference latency per image at batch 1 (mean ± SD, median) and throughput at batch 32
  * peak GPU memory during batch-1 inference (includes the weights)
  * training cost per run, read from each run's summary.json

Usage:  python efficiency.py --runs runs --out results
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from models import build_model, count_parameters

LABELS = {"hifuse_small": "HiFuse-Small", "medvit_small": "MedViT-Small"}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def time_forward(model, x, device, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        model(x)
    sync(device)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        model(x)
        sync(device)
        times.append((time.perf_counter() - t) * 1000)
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--models", nargs="+", default=["hifuse_small", "medvit_small"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--throughput_batch", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":  # keep CPU runs short; numbers are only meaningful on the GPU
        args.warmup, args.iters, args.throughput_batch = 2, 5, 4
    torch.backends.cudnn.benchmark = True
    hw = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU"

    rows = []
    for name in args.models:
        model = build_model(name, 2)
        ckpts = sorted(args.runs.glob(f"{name}/seed*/best.pt"))
        if ckpts:  # cost doesn't depend on the weights, but measure the trained model anyway
            model.load_state_dict(torch.load(ckpts[0], map_location="cpu", weights_only=False)["model"])
        model = model.to(device).eval()

        x1 = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        fc = FlopCounterMode(display=False)
        with torch.inference_mode(), fc:
            model(x1)
        flops = fc.get_total_flops()

        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        lat = time_forward(model, x1, device, args.warmup, args.iters)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else float("nan")

        xb = torch.randn(args.throughput_batch, 3, args.img_size, args.img_size, device=device)
        tb = time_forward(model, xb, device, max(3, args.warmup // 5), max(5, args.iters // 5))
        throughput = args.throughput_batch / (statistics.mean(tb) / 1000)

        # training cost from the finished runs
        sums = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.runs.glob(f"{name}/seed*/summary.json"))]
        tr_min = np.array([s["train_minutes"] for s in sums], float)
        ep = np.array([s["epochs_run"] for s in sums], float)
        peak_tr = np.array([s.get("peak_gpu_gb_train", np.nan) for s in sums], float)
        sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0

        rows.append({
            "model": LABELS.get(name, name), "params_M": round(count_parameters(model), 2),
            "GMACs": round(flops / 2e9, 2), "GFLOPs": round(flops / 1e9, 2),
            "latency_bs1_ms_mean": round(statistics.mean(lat), 2), "latency_bs1_ms_sd": round(statistics.stdev(lat), 2),
            "latency_bs1_ms_median": round(statistics.median(lat), 2),
            f"throughput_bs{args.throughput_batch}_img_per_s": round(throughput, 1),
            "peak_inference_mem_MB_bs1": round(peak_mb, 1),
            "train_runs": len(sums),
            "train_minutes_mean": round(tr_min.mean(), 1) if len(sums) else np.nan,
            "train_minutes_sd": round(sd(tr_min), 1) if len(sums) else np.nan,
            "epochs_run_mean": round(ep.mean(), 1) if len(sums) else np.nan,
            "minutes_per_epoch": round((tr_min / ep).mean(), 2) if len(sums) else np.nan,
            "peak_train_mem_GB": round(np.nanmean(peak_tr), 2) if len(sums) and not np.isnan(peak_tr).all() else np.nan,
        })
        print(f"{name}: done")
        del model; torch.cuda.empty_cache() if device.type == "cuda" else None

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "efficiency.csv", index=False)
    a, b = rows[0], rows[-1]
    md = [f"# Computational efficiency\n",
          f"Hardware: {hw} | PyTorch {torch.__version__} | fp32 inference, {args.img_size}×{args.img_size} input, "
          f"{args.iters} timed iterations after {args.warmup} warm-up.\n",
          "| Model | Params (M) | GMACs | GFLOPs | Latency bs=1 (ms, mean ± SD) | median (ms) | "
          f"Throughput bs={args.throughput_batch} (img/s) | Peak inference mem (MB) | "
          "Training time / run (min) | Epochs run | min / epoch | Peak training mem (GB) |",
          "|---" * 12 + "|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['params_M']} | {r['GMACs']} | {r['GFLOPs']} | "
                  f"{r['latency_bs1_ms_mean']} ± {r['latency_bs1_ms_sd']} | {r['latency_bs1_ms_median']} | "
                  f"{r[f'throughput_bs{args.throughput_batch}_img_per_s']} | {r['peak_inference_mem_MB_bs1']} | "
                  f"{r['train_minutes_mean']} ± {r['train_minutes_sd']} | {r['epochs_run_mean']} | "
                  f"{r['minutes_per_epoch']} | {r['peak_train_mem_GB']} |")
    if len(rows) == 2:
        md.append(f"\nRatios {a['model']} / {b['model']}: params {a['params_M'] / b['params_M']:.2f}×, "
                  f"MACs {a['GMACs'] / b['GMACs']:.2f}×, latency {a['latency_bs1_ms_mean'] / b['latency_bs1_ms_mean']:.2f}×, "
                  f"throughput {a[f'throughput_bs{args.throughput_batch}_img_per_s'] / b[f'throughput_bs{args.throughput_batch}_img_per_s']:.2f}×.")
    md.append("\nGMACs = multiply–accumulate operations (the quantity fvcore/thop call \"FLOPs\"); "
              "GFLOPs = 2 × GMACs.")
    if device.type == "cpu":
        md.append("\n**Measured on CPU (no GPU found): timing and memory numbers are not representative.**")
    (args.out / "efficiency.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {args.out}/efficiency.md and efficiency.csv")


if __name__ == "__main__":
    main()
