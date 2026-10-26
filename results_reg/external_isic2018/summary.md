# Untouched holdout: ISIC 2018 Task 3 test set (melanoma vs nevus)

```
{
  "gt_file": "data\\ISIC2018_Task3_Test_GroundTruth.csv",
  "mel_nv_images_in_gt": 1080,
  "images_not_found": 1,
  "lesions_overlapping_HAM10000_removed": 0,
  "images_removed_for_overlap": 0,
  "holdout_images": 1079,
  "holdout_mel_images": 171,
  "holdout_lesions": 879,
  "holdout_mel_lesions": 109,
  "has_lesion_ids": true,
  "dx_type_counts": {
    "mel|histo": 171,
    "nv|consensus": 121,
    "nv|follow_up": 458,
    "nv|histo": 329
  },
  "torch": "2.11.0+cu128",
  "cuda": "12.8",
  "cudnn": 91900,
  "device": "NVIDIA GeForce RTX 5060"
}
```

Pipeline check (internal test images re-predicted, max |diff| vs stored): 0.0012


## image level, threshold 0.5

| model | accuracy | precision | recall | specificity | f1 | balanced_accuracy | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| HiFuse-Small | 0.8945 ± 0.0089 | 0.6948 ± 0.0369 | 0.6000 ± 0.0441 | 0.9500 ± 0.0099 | 0.6429 ± 0.0316 | 0.7750 ± 0.0209 | 0.5843 ± 0.0349 | 0.9162 ± 0.0063 | 0.7132 ± 0.0514 |
| MedViT-Small | 0.8795 ± 0.0099 | 0.6171 ± 0.0310 | 0.6327 ± 0.0682 | 0.9260 ± 0.0105 | 0.6235 ± 0.0429 | 0.7794 ± 0.0318 | 0.5529 ± 0.0470 | 0.9216 ± 0.0080 | 0.7135 ± 0.0297 |

| paired t-test (HiFuse − MedViT) | diff | t | p |
|---|---|---|---|
| mcc | +0.0314 | 1.188 | 0.3007 |
| auroc | -0.0054 | -1.668 | 0.1706 |
| recall | -0.0327 | -0.970 | 0.3872 |
| balanced_accuracy | -0.0044 | -0.272 | 0.7994 |

## image level, threshold val

| model | accuracy | precision | recall | specificity | f1 | balanced_accuracy | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| HiFuse-Small | 0.8411 ± 0.0268 | 0.5071 ± 0.0609 | 0.7930 ± 0.0454 | 0.8502 ± 0.0404 | 0.6150 ± 0.0287 | 0.8216 ± 0.0036 | 0.5453 ± 0.0295 | 0.9162 ± 0.0063 | 0.7132 ± 0.0514 |
| MedViT-Small | 0.7943 ± 0.0318 | 0.4329 ± 0.0446 | 0.8947 ± 0.0289 | 0.7753 ± 0.0429 | 0.5817 ± 0.0329 | 0.8350 ± 0.0089 | 0.5227 ± 0.0319 | 0.9216 ± 0.0080 | 0.7135 ± 0.0297 |

| paired t-test (HiFuse − MedViT) | diff | t | p |
|---|---|---|---|
| mcc | +0.0226 | 1.144 | 0.3165 |
| auroc | -0.0054 | -1.668 | 0.1706 |
| recall | -0.1018 | -3.914 | 0.0173 |
| balanced_accuracy | -0.0134 | -3.889 | 0.0177 |

## lesion level, threshold 0.5

| model | accuracy | precision | recall | specificity | f1 | balanced_accuracy | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| HiFuse-Small | 0.9147 ± 0.0064 | 0.6706 ± 0.0347 | 0.6183 ± 0.0359 | 0.9566 ± 0.0084 | 0.6424 ± 0.0230 | 0.7875 ± 0.0162 | 0.5954 ± 0.0255 | 0.9346 ± 0.0052 | 0.6915 ± 0.0556 |
| MedViT-Small | 0.8990 ± 0.0074 | 0.5825 ± 0.0281 | 0.6569 ± 0.0586 | 0.9332 ± 0.0081 | 0.6165 ± 0.0352 | 0.7951 ± 0.0276 | 0.5606 ± 0.0395 | 0.9378 ± 0.0069 | 0.6869 ± 0.0341 |

| paired t-test (HiFuse − MedViT) | diff | t | p |
|---|---|---|---|
| mcc | +0.0348 | 1.825 | 0.1421 |
| auroc | -0.0032 | -0.930 | 0.4049 |
| recall | -0.0385 | -1.300 | 0.2635 |
| balanced_accuracy | -0.0076 | -0.564 | 0.6027 |

## lesion level, threshold val

| model | accuracy | precision | recall | specificity | f1 | balanced_accuracy | mcc | auroc | auprc |
|---|---|---|---|---|---|---|---|---|---|
| HiFuse-Small | 0.8680 ± 0.0235 | 0.4885 ± 0.0595 | 0.8110 ± 0.0403 | 0.8761 ± 0.0326 | 0.6064 ± 0.0326 | 0.8436 ± 0.0041 | 0.5607 ± 0.0296 | 0.9346 ± 0.0052 | 0.6915 ± 0.0556 |
| MedViT-Small | 0.8225 ± 0.0276 | 0.4090 ± 0.0393 | 0.9174 ± 0.0395 | 0.8091 ± 0.0369 | 0.5639 ± 0.0290 | 0.8633 ± 0.0043 | 0.5353 ± 0.0220 | 0.9378 ± 0.0069 | 0.6869 ± 0.0341 |

| paired t-test (HiFuse − MedViT) | diff | t | p |
|---|---|---|---|
| mcc | +0.0254 | 1.296 | 0.2648 |
| auroc | -0.0032 | -0.930 | 0.4049 |
| recall | -0.1064 | -4.247 | 0.0132 |
| balanced_accuracy | -0.0197 | -37.749 | 0.0000 |

Consensus McNemar (image level, threshold 0.5): {"only_A_correct": 32, "only_B_correct": 10, "chi2_cc": 10.5, "p_chi2": 0.0011937454448720044, "p_exact": 0.0009406740282429382}

Consensus McNemar (lesion level, threshold 0.5): {"only_A_correct": 23, "only_B_correct": 8, "chi2_cc": 6.32258064516129, "p_chi2": 0.011921001496914012, "p_exact": 0.010673840530216694}
