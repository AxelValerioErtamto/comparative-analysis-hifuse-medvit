# Statistical analysis (final protocol, runs_reg)

Test set: 1173 images (165 melanoma, prevalence 0.141); 903 lesions (92 melanoma, prevalence 0.102). Lesion prediction = mean P(melanoma) over the lesion's images.


## image level, threshold 0.5: mean ± SD over 5 seeds

| model | mcc | auroc | recall | specificity | precision | f1 | balanced_accuracy | accuracy | auprc |
|---|---|---|---|---|---|---|---|---|---|
| HiFuse-Small | 0.4848 ± 0.0413 | 0.9041 ± 0.0089 | 0.5188 ± 0.0545 | 0.9403 ± 0.0147 | 0.5906 ± 0.0469 | 0.5504 ± 0.0375 | 0.7295 ± 0.0248 | 0.8810 ± 0.0115 | 0.6050 ± 0.0385 |
| MedViT-Small | 0.5240 ± 0.0565 | 0.9140 ± 0.0067 | 0.5806 ± 0.0851 | 0.9369 ± 0.0080 | 0.6000 ± 0.0253 | 0.5881 ± 0.0557 | 0.7588 ± 0.0397 | 0.8868 ± 0.0084 | 0.6400 ± 0.0459 |

## image level, threshold val: mean ± SD over 5 seeds

| model | mcc | auroc | recall | specificity | precision | f1 | balanced_accuracy | accuracy | auprc |
|---|---|---|---|---|---|---|---|---|---|
| HiFuse-Small | 0.5204 ± 0.0144 | 0.9041 ± 0.0089 | 0.7879 ± 0.0664 | 0.8484 ± 0.0368 | 0.4661 ± 0.0474 | 0.5819 ± 0.0172 | 0.8181 ± 0.0159 | 0.8399 ± 0.0227 | 0.6050 ± 0.0385 |
| MedViT-Small | 0.5145 ± 0.0207 | 0.9140 ± 0.0067 | 0.8994 ± 0.0399 | 0.7835 ± 0.0398 | 0.4082 ± 0.0339 | 0.5600 ± 0.0254 | 0.8415 ± 0.0070 | 0.7998 ± 0.0290 | 0.6400 ± 0.0459 |

## lesion level, threshold 0.5: mean ± SD over 5 seeds

| model | mcc | auroc | recall | specificity | precision | f1 | balanced_accuracy | accuracy | auprc |
|---|---|---|---|---|---|---|---|---|---|
| HiFuse-Small | 0.5235 ± 0.0562 | 0.9432 ± 0.0050 | 0.5326 ± 0.0753 | 0.9610 ± 0.0082 | 0.6090 ± 0.0452 | 0.5662 ± 0.0546 | 0.7468 ± 0.0365 | 0.9174 ± 0.0087 | 0.6700 ± 0.0303 |
| MedViT-Small | 0.5732 ± 0.0490 | 0.9477 ± 0.0047 | 0.6109 ± 0.0789 | 0.9581 ± 0.0057 | 0.6228 ± 0.0248 | 0.6150 ± 0.0473 | 0.7845 ± 0.0375 | 0.9227 ± 0.0057 | 0.7076 ± 0.0293 |

## lesion level, threshold val: mean ± SD over 5 seeds

| model | mcc | auroc | recall | specificity | precision | f1 | balanced_accuracy | accuracy | auprc |
|---|---|---|---|---|---|---|---|---|---|
| HiFuse-Small | 0.5247 ± 0.0379 | 0.9432 ± 0.0050 | 0.9283 ± 0.0557 | 0.8202 ± 0.0524 | 0.3787 ± 0.0574 | 0.5341 ± 0.0488 | 0.8742 ± 0.0113 | 0.8312 ± 0.0419 | 0.6700 ± 0.0303 |
| MedViT-Small | 0.5456 ± 0.0220 | 0.9477 ± 0.0047 | 0.9261 ± 0.0424 | 0.8397 ± 0.0284 | 0.3990 ± 0.0323 | 0.5564 ± 0.0270 | 0.8829 ± 0.0123 | 0.8485 ± 0.0221 | 0.7076 ± 0.0293 |

## Paired differences HiFuse − MedViT (95% CIs)

| level | thr | metric | diff | seed CI (t, 5 pairs) | lesion CI (cluster bootstrap) | combined CI | pairs A>B / B>A |
|---|---|---|---|---|---|---|---|
| image | 0.5 | mcc | -0.0392 | [-0.1229, +0.0445] | [-0.0831, +0.0013] | [-0.1101, +0.0297] | 2/3 |
| image | 0.5 | auroc | -0.0099 | [-0.0230, +0.0032] | [-0.0186, -0.0009] | [-0.0232, +0.0030] | 1/4 |
| image | 0.5 | recall | -0.0618 | [-0.1614, +0.0378] | [-0.1112, -0.0187] | [-0.1445, +0.0160] | 1/4 |
| image | 0.5 | specificity | +0.0034 | [-0.0122, +0.0190] | [-0.0042, +0.0111] | [-0.0109, +0.0163] | 4/1 |
| image | 0.5 | balanced_accuracy | -0.0292 | [-0.0786, +0.0201] | [-0.0541, -0.0074] | [-0.0716, +0.0101] | 1/4 |
| image | 0.5 | auprc | -0.0350 | [-0.0945, +0.0245] | [-0.0686, -0.0049] | [-0.0873, +0.0168] | 1/4 |
| image | 0.5 | f1 | -0.0377 | [-0.1149, +0.0395] | [-0.0776, -0.0011] | [-0.1039, +0.0247] | 2/3 |
| image | val | mcc | +0.0060 | [-0.0296, +0.0415] | [-0.0328, +0.0411] | [-0.0397, +0.0546] | 3/2 |
| image | val | auroc | -0.0099 | [-0.0230, +0.0032] | [-0.0187, -0.0009] | [-0.0232, +0.0028] | 1/4 |
| image | val | recall | -0.1115 | [-0.2250, +0.0019] | [-0.1548, -0.0706] | [-0.2075, -0.0402] | 0/5 |
| image | val | specificity | +0.0649 | [-0.0172, +0.1469] | [+0.0507, +0.0791] | [+0.0241, +0.1280] | 5/0 |
| image | val | balanced_accuracy | -0.0233 | [-0.0429, -0.0037] | [-0.0473, -0.0011] | [-0.0526, +0.0034] | 0/5 |
| image | val | auprc | -0.0350 | [-0.0945, +0.0245] | [-0.0663, -0.0042] | [-0.0883, +0.0188] | 1/4 |
| image | val | f1 | +0.0219 | [-0.0213, +0.0651] | [-0.0108, +0.0513] | [-0.0179, +0.0678] | 4/1 |
| lesion | 0.5 | mcc | -0.0497 | [-0.1148, +0.0153] | [-0.0909, -0.0103] | [-0.1175, +0.0069] | 0/5 |
| lesion | 0.5 | auroc | -0.0045 | [-0.0118, +0.0027] | [-0.0107, +0.0013] | [-0.0143, +0.0035] | 1/4 |
| lesion | 0.5 | recall | -0.0783 | [-0.1548, -0.0017] | [-0.1241, -0.0341] | [-0.1591, -0.0122] | 0/5 |
| lesion | 0.5 | specificity | +0.0030 | [-0.0096, +0.0155] | [-0.0035, +0.0093] | [-0.0087, +0.0137] | 4/1 |
| lesion | 0.5 | balanced_accuracy | -0.0377 | [-0.0750, -0.0003] | [-0.0608, -0.0162] | [-0.0780, -0.0054] | 0/5 |
| lesion | 0.5 | auprc | -0.0376 | [-0.0847, +0.0095] | [-0.0713, -0.0059] | [-0.0894, +0.0082] | 1/4 |
| lesion | 0.5 | f1 | -0.0488 | [-0.1111, +0.0135] | [-0.0876, -0.0124] | [-0.1138, +0.0042] | 0/5 |
| lesion | val | mcc | -0.0209 | [-0.0505, +0.0087] | [-0.0447, +0.0030] | [-0.0547, +0.0156] | 0/5 |
| lesion | val | auroc | -0.0045 | [-0.0118, +0.0027] | [-0.0104, +0.0013] | [-0.0132, +0.0038] | 1/4 |
| lesion | val | recall | +0.0022 | [-0.0669, +0.0713] | [-0.0272, +0.0329] | [-0.0619, +0.0572] | 3/1 |
| lesion | val | specificity | -0.0195 | [-0.0578, +0.0189] | [-0.0307, -0.0089] | [-0.0462, +0.0098] | 1/4 |
| lesion | val | balanced_accuracy | -0.0087 | [-0.0329, +0.0156] | [-0.0243, +0.0075] | [-0.0338, +0.0163] | 2/3 |
| lesion | val | auprc | -0.0376 | [-0.0847, +0.0095] | [-0.0738, -0.0055] | [-0.0852, +0.0076] | 1/4 |
| lesion | val | f1 | -0.0223 | [-0.0550, +0.0105] | [-0.0418, -0.0025] | [-0.0537, +0.0105] | 1/4 |

## Consensus predictions (one per test image / lesion)

| level   | model        | rule             |   precision |   recall |   specificity |    mcc |   tp |   tn |   fp |   fn |    auroc |
|:--------|:-------------|:-----------------|------------:|---------:|--------------:|-------:|-----:|-----:|-----:|-----:|---------:|
| image   | hifuse_small | majority vote    |      0.6043 |   0.5091 |        0.9454 | 0.4890 |   84 |  953 |   55 |   81 | nan      |
| image   | hifuse_small | mean probability |      0.5985 |   0.4970 |        0.9454 | 0.4789 |   82 |  953 |   55 |   83 |   0.9108 |
| image   | medvit_small | majority vote    |      0.6447 |   0.5939 |        0.9464 | 0.5594 |   98 |  954 |   54 |   67 | nan      |
| image   | medvit_small | mean probability |      0.6340 |   0.5879 |        0.9444 | 0.5495 |   97 |  952 |   56 |   68 |   0.9220 |
| lesion  | hifuse_small | majority vote    |      0.6154 |   0.5217 |        0.9630 | 0.5220 |   48 |  781 |   30 |   44 | nan      |
| lesion  | hifuse_small | mean probability |      0.6410 |   0.5435 |        0.9655 | 0.5480 |   50 |  783 |   28 |   42 |   0.9486 |
| lesion  | medvit_small | majority vote    |      0.6705 |   0.6413 |        0.9642 | 0.6176 |   59 |  782 |   29 |   33 | nan      |
| lesion  | medvit_small | mean probability |      0.6437 |   0.6087 |        0.9618 | 0.5848 |   56 |  780 |   31 |   36 |   0.9528 |

## Calibration (test, image level): mean ± SD over seeds

- HiFuse-Small: Brier 0.0830 ± 0.0057, ECE 0.0396 ± 0.0124, mean predicted P 0.178 vs prevalence 0.141
- MedViT-Small: Brier 0.0789 ± 0.0024, ECE 0.0415 ± 0.0097, mean predicted P 0.180 vs prevalence 0.141

## Validation-Youden thresholds per seed (image level)

| model        |   seed | level   |   threshold |   recall |   specificity |   precision |   balanced_accuracy |   mcc |
|:-------------|-------:|:--------|------------:|---------:|--------------:|------------:|--------------------:|------:|
| hifuse_small |      0 | image   |       0.309 |    0.776 |         0.865 |       0.485 |               0.820 | 0.534 |
| hifuse_small |      1 | image   |       0.304 |    0.824 |         0.828 |       0.440 |               0.826 | 0.515 |
| hifuse_small |      2 | image   |       0.227 |    0.848 |         0.820 |       0.436 |               0.834 | 0.522 |
| hifuse_small |      3 | image   |       0.296 |    0.812 |         0.822 |       0.428 |               0.817 | 0.499 |
| hifuse_small |      4 | image   |       0.334 |    0.679 |         0.906 |       0.541 |               0.792 | 0.533 |
| medvit_small |      0 | image   |       0.332 |    0.842 |         0.834 |       0.454 |               0.838 | 0.536 |
| medvit_small |      1 | image   |       0.196 |    0.885 |         0.784 |       0.401 |               0.834 | 0.502 |
| medvit_small |      2 | image   |       0.160 |    0.909 |         0.783 |       0.407 |               0.846 | 0.518 |
| medvit_small |      3 | image   |       0.182 |    0.909 |         0.794 |       0.419 |               0.851 | 0.531 |
| medvit_small |      4 | image   |       0.145 |    0.952 |         0.723 |       0.360 |               0.837 | 0.485 |

## Validation MCC of each model under every tuning setting (seed 0)

| regularisation   |   melanoma_weight |   hifuse_small |   medvit_small |   mean |   rank_hifuse |   rank_medvit |
|:-----------------|------------------:|---------------:|---------------:|-------:|--------------:|--------------:|
| strong           |                 2 |         0.6621 |         0.6139 | 0.6380 |             1 |             2 |
| strong           |                 3 |         0.6374 |         0.6015 | 0.6194 |             2 |             3 |
| official         |                 2 |         0.6263 |         0.5995 | 0.6129 |             3 |             4 |
| official         |                 4 |         0.6245 |         0.5905 | 0.6075 |             4 |             5 |
| official         |                 3 |         0.6101 |         0.5893 | 0.5997 |             5 |             6 |
| official         |                 6 |         0.5971 |         0.5710 | 0.5840 |             6 |             7 |
| strong           |                 4 |         0.4968 |         0.6241 | 0.5604 |             7 |             1 |
| strong           |                 6 |         0.4607 |         0.5200 | 0.4904 |             8 |             8 |

## Subgroups (consensus prediction, threshold 0.5)

| model        | attribute         | value           |   mel_images |   nv_images |   mel_recall |   nv_specificity |
|:-------------|:------------------|:----------------|-------------:|------------:|-------------:|-----------------:|
| hifuse_small | dx_type           | consensus       |            0 |          88 |      nan     |            0.955 |
| hifuse_small | dx_type           | follow_up       |            0 |         556 |      nan     |            1.000 |
| hifuse_small | dx_type           | histo           |          165 |         364 |        0.509 |            0.860 |
| hifuse_small | images_per_lesion | multi-image     |          130 |         345 |        0.469 |            0.872 |
| hifuse_small | images_per_lesion | single image    |           35 |         663 |        0.657 |            0.983 |
| hifuse_small | localization      | abdomen         |           22 |         128 |        0.318 |            0.984 |
| hifuse_small | localization      | back            |           38 |         202 |        0.579 |            0.896 |
| hifuse_small | localization      | chest           |           18 |          23 |        0.500 |            0.957 |
| hifuse_small | localization      | ear             |            3 |           6 |        0.333 |            0.500 |
| hifuse_small | localization      | face            |           14 |          13 |        0.714 |            0.846 |
| hifuse_small | localization      | foot            |            4 |          50 |        0.000 |            0.960 |
| hifuse_small | localization      | genital         |            0 |           9 |      nan     |            0.778 |
| hifuse_small | localization      | hand            |            0 |          13 |      nan     |            1.000 |
| hifuse_small | localization      | lower extremity |           22 |         211 |        0.545 |            0.972 |
| hifuse_small | localization      | neck            |            3 |          20 |        0.667 |            0.800 |
| hifuse_small | localization      | scalp           |            1 |           6 |        0.000 |            1.000 |
| hifuse_small | localization      | trunk           |            8 |         191 |        0.625 |            1.000 |
| hifuse_small | localization      | unknown         |            0 |          29 |      nan     |            0.931 |
| hifuse_small | localization      | upper extremity |           32 |         107 |        0.500 |            0.907 |
| hifuse_small | age_group         | <40             |           13 |         259 |        0.154 |            0.934 |
| hifuse_small | age_group         | 40-59           |           46 |         525 |        0.435 |            0.968 |
| hifuse_small | age_group         | >=60            |          106 |         217 |        0.585 |            0.903 |
| hifuse_small | sex               | female          |           55 |         496 |        0.382 |            0.946 |
| hifuse_small | sex               | male            |          110 |         503 |        0.573 |            0.944 |
| hifuse_small | sex               | unknown         |            0 |           9 |      nan     |            1.000 |
| hifuse_small | dataset           | rosendahl       |           52 |         117 |        0.442 |            0.855 |
| hifuse_small | dataset           | vidir_modern    |           98 |         286 |        0.571 |            0.878 |
| hifuse_small | dataset           | vidir_molemax   |            3 |         557 |        0.000 |            1.000 |
| hifuse_small | dataset           | vienna_dias     |           12 |          48 |        0.417 |            0.938 |
| medvit_small | dx_type           | consensus       |            0 |          88 |      nan     |            0.943 |
| medvit_small | dx_type           | follow_up       |            0 |         556 |      nan     |            1.000 |
| medvit_small | dx_type           | histo           |          165 |         364 |        0.594 |            0.865 |
| medvit_small | images_per_lesion | multi-image     |          130 |         345 |        0.562 |            0.878 |
| medvit_small | images_per_lesion | single image    |           35 |         663 |        0.714 |            0.982 |
| medvit_small | localization      | abdomen         |           22 |         128 |        0.500 |            0.969 |
| medvit_small | localization      | back            |           38 |         202 |        0.632 |            0.896 |
| medvit_small | localization      | chest           |           18 |          23 |        0.500 |            0.957 |
| medvit_small | localization      | ear             |            3 |           6 |        0.667 |            0.833 |
| medvit_small | localization      | face            |           14 |          13 |        0.643 |            0.846 |
| medvit_small | localization      | foot            |            4 |          50 |        0.000 |            0.960 |
| medvit_small | localization      | genital         |            0 |           9 |      nan     |            0.778 |
| medvit_small | localization      | hand            |            0 |          13 |      nan     |            1.000 |
| medvit_small | localization      | lower extremity |           22 |         211 |        0.591 |            0.986 |
| medvit_small | localization      | neck            |            3 |          20 |        0.667 |            0.800 |
| medvit_small | localization      | scalp           |            1 |           6 |        0.000 |            0.833 |
| medvit_small | localization      | trunk           |            8 |         191 |        0.875 |            1.000 |
| medvit_small | localization      | unknown         |            0 |          29 |      nan     |            0.931 |
| medvit_small | localization      | upper extremity |           32 |         107 |        0.656 |            0.897 |
| medvit_small | age_group         | <40             |           13 |         259 |        0.308 |            0.946 |
| medvit_small | age_group         | 40-59           |           46 |         525 |        0.543 |            0.966 |
| medvit_small | age_group         | >=60            |          106 |         217 |        0.651 |            0.899 |
| medvit_small | sex               | female          |           55 |         496 |        0.455 |            0.958 |
| medvit_small | sex               | male            |          110 |         503 |        0.664 |            0.934 |
| medvit_small | sex               | unknown         |            0 |           9 |      nan     |            1.000 |
| medvit_small | dataset           | rosendahl       |           52 |         117 |        0.615 |            0.872 |
| medvit_small | dataset           | vidir_modern    |           98 |         286 |        0.653 |            0.881 |
| medvit_small | dataset           | vidir_molemax   |            3 |         557 |        0.000 |            1.000 |
| medvit_small | dataset           | vienna_dias     |           12 |          48 |        0.167 |            0.896 |
