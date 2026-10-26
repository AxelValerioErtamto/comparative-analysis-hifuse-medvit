# Computational efficiency

Hardware: NVIDIA GeForce RTX 5060 | PyTorch 2.11.0+cu128 | fp32 inference, 224×224 input, 200 timed iterations after 30 warm-up.

| Model | Params (M) | GMACs | GFLOPs | Latency bs=1 (ms, mean ± SD) | median (ms) | Throughput bs=32 (img/s) | Peak inference mem (MB) | Training time / run (min) | Epochs run | min / epoch | Peak training mem (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HiFuse-Small | 123.26 | 18.25 | 36.49 | 26.57 ± 0.28 | 26.51 | 223.6 | 677.8 | 57.0 ± 13.9 | 25.2 | 2.27 | 4.44 |
| MedViT-Small | 31.14 | 5.86 | 11.71 | 27.62 ± 0.56 | 27.55 | 260.7 | 165.0 | 45.4 ± 15.6 | 30.6 | 1.5 | 3.52 |

Ratios HiFuse-Small / MedViT-Small: params 3.96×, MACs 3.11×, latency 0.96×, throughput 0.86×.

GMACs = multiply–accumulate operations (the quantity fvcore/thop call "FLOPs"); GFLOPs = 2 × GMACs. 
