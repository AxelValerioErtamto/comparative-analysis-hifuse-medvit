# Overfitting check at the selected checkpoint

All partitions scored identically (validation transform, no resampling, unweighted cross-entropy, threshold 0.5). Mean ± SD over seeds.

| Experiment | Model | Runs | Best epoch | Loss (train) | Loss (val) | Loss (test) | Loss gap (test − train) | ROC-AUC (train) | ROC-AUC (test) | ROC-AUC gap (train − test) | Balanced acc. (train) | Balanced acc. (test) | Bal. acc. gap (train − test) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Main | HiFuse-Small | 5 | 15.2 ± 6.8 | 0.2509 ± 0.0358 | 0.2688 ± 0.0186 | 0.3017 ± 0.0213 | 0.0508 ± 0.0192 | 0.9327 ± 0.0190 | 0.8980 ± 0.0098 | 0.0347 ± 0.0118 | 0.8231 ± 0.0410 | 0.7755 ± 0.0386 | 0.0475 ± 0.0183 |
| Main | MedViT-Small | 5 | 20.6 ± 11.2 | 0.2266 ± 0.0381 | 0.2794 ± 0.0153 | 0.2951 ± 0.0143 | 0.0685 ± 0.0505 | 0.9492 ± 0.0256 | 0.9077 ± 0.0066 | 0.0416 ± 0.0279 | 0.8616 ± 0.0557 | 0.7881 ± 0.0168 | 0.0734 ± 0.0481 |
| Regularized | HiFuse-Small | 5 | 19.0 ± 5.1 | 0.2324 ± 0.0162 | 0.2342 ± 0.0105 | 0.2611 ± 0.0147 | 0.0287 ± 0.0040 | 0.9261 ± 0.0094 | 0.9041 ± 0.0089 | 0.0220 ± 0.0030 | 0.7655 ± 0.0202 | 0.7295 ± 0.0248 | 0.0360 ± 0.0094 |
| Regularized | MedViT-Small | 5 | 15.8 ± 6.5 | 0.2260 ± 0.0164 | 0.2377 ± 0.0057 | 0.2517 ± 0.0072 | 0.0257 ± 0.0121 | 0.9324 ± 0.0107 | 0.9140 ± 0.0067 | 0.0184 ± 0.0086 | 0.7917 ± 0.0284 | 0.7588 ± 0.0397 | 0.0330 ± 0.0139 |
