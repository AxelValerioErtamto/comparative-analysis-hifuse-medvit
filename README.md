# HiFuse-Small vs MedViT-Small for melanoma vs nevus classification (HAM10000, lesion-level split)

Code, data partition, run logs, predictions and statistical outputs for the paper

> A. V. Ertamto and D. M. Wonohadidjojo, "Comparative Analysis of HiFuse and MedViT for Melanoma and Nevi
> Binary Classification on HAM10000," *bit-Tech*, under review, 2026.

Both official architectures are trained from scratch on a binary melanoma/nevus subset of HAM10000
(7,818 images, 6,017 lesions) split 70/15/15 **by lesion**, with five random seeds each, and are then
evaluated unchanged on the melanoma/nevus images of the official ISIC 2018 Task 3 test set.

## Model weights

The ten final checkpoints (`runs_reg/<model>/seed<k>/best.pt`) are too large for GitHub and are available at
https://drive.google.com/drive/folders/1mUfynLdbwKctu6RrwPODDGv0rXbDsypl?usp=sharing. Place each `best.pt` in the matching `runs_reg/<model>/seed<k>/` folder of this repository.

## Data

Download from the HAM10000 Harvard Dataverse record, https://doi.org/10.7910/DVN/DBW86T, into `data/`:

* `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`, `HAM10000_metadata` (training data)
* `ISIC2018_Task3_Test_Images.zip`, `ISIC2018_Task3_Test_GroundTruth.csv` (external holdout)

## Official model code

Clone the two official repositories into `third_party/`:

```
git clone https://github.com/huoxiangzuo/HiFuse.git third_party/HiFuse     # commit 9070020449dcfe6cb677c47d37e5c0cb9d257eb9
git clone https://github.com/Omid-Nejati/MedViT.git third_party/MedViT     # commit 1abcaf6054b9f94835d3e97994e9984e2b0ce390
```

The network definitions are used unmodified; only the number of output classes and the stochastic depth and
dropout rates exposed by the official constructors are set.

## Environment

Python 3.13, PyTorch 2.11.0 with CUDA 12.8 (Windows 10, NVIDIA GeForce RTX 5060 8 GB).

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Reproduce

| Step | Command | Output |
|---|---|---|
| Lesion-level split + leakage checks | `python run.py split` | `splits/lesion_split_seed42.csv` |
| Tuning (16 runs, validation only) and final training (2 models x 5 seeds) | `python run.py reg` | `runs_reg_tune/`, `runs_reg/`, `results_reg/` |
| Efficiency (parameters, FLOPs, latency, memory) | `python run.py analysis` | `results/efficiency.md` |
| Paper analyses (both levels, bootstrap CIs, paired differences, McNemar, calibration, subgroups) | `python stat_analysis.py --split splits/lesion_split_seed42.csv` | `results_reg/analysis/` |
| External holdout + repeated timing | `python external_test.py --gt data/ISIC2018_Task3_Test_GroundTruth.csv --images data/ISIC2018_Task3_Test_Images.zip` | `results_reg/external_isic2018/` |
| Figures | `python paper_figures.py` | `results_reg/paper_figures/` |

Training is not bit-exact across GPUs and driver versions; the stored predictions let every table be
recomputed without retraining.

## Where each result in the paper comes from

| Paper item | File |
|---|---|
| Table 1 (partition) | `splits/lesion_split_seed42.csv`, `splits/*.summary.json` |
| Tuning (Section III.E) | `results_reg/tuning.md`, `results_reg/analysis/tuning_by_model.csv` |
| Table 3 (image and lesion level, bootstrap CIs) | `results_reg/analysis/model_ci.csv`, `per_run_both_levels.csv` |
| Table 4 (paired differences, three CIs) | `results_reg/analysis/differences.csv`, `per_seed_pairs.csv` |
| Figure 4, McNemar tests | `results_reg/analysis/consensus_confusion.csv`, `report.md` |
| Per-seed confusion matrices | `results_reg/analysis/per_run_both_levels.csv` (tp, tn, fp, fn) |
| Thresholds, calibration (Figure 5) | `results_reg/analysis/thresholds.csv`, `calibration.csv`, `reliability.csv` |
| Error analysis (Section IV.F) | `results_reg/analysis/subgroups.csv` |
| Table 5 (external holdout) | `results_reg/external_isic2018/per_run_metrics.csv`, `summary.md`, `preds/` |
| Table 6 (efficiency) | parameters, FLOPs, latency, throughput, inference memory: `results/efficiency.md`; repeated timing sessions: `results_reg/external_isic2018/efficiency_repeats.csv`; training time, epochs and training memory of the final runs: `runs_reg/<model>/seed<k>/summary.json` |
| Training curves (Figures 2 and 3) | `runs_reg/<model>/seed<k>/history.csv` |
| First lesion-level protocol (Section III.E) | `runs/`, `results/` |

Each run folder holds its full configuration (`config.json`), log, per-epoch history, and per-image
validation and test predictions (`preds_*.csv`: image_id, lesion_id, label, melanoma probability).
