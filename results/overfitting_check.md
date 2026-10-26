# Overfitting check at the selected checkpoint

All partitions scored identically (validation transform, no resampling, unweighted cross-entropy, threshold 0.5). Mean ± SD over seeds.

| Experiment | Model | Runs | Best epoch | Loss (train) | Loss (val) | Loss (test) | Loss gap (test − train) | ROC-AUC (train) | ROC-AUC (test) | ROC-AUC gap (train − test) | Balanced acc. (train) | Balanced acc. (test) | Bal. acc. gap (train − test) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Main | HiFuse-Small | 5 | 15.2 ± 6.8 | 0.2509 ± 0.0358 | 0.2688 ± 0.0186 | 0.3017 ± 0.0213 | 0.0508 ± 0.0192 | 0.9327 ± 0.0190 | 0.8980 ± 0.0098 | 0.0347 ± 0.0118 | 0.8231 ± 0.0410 | 0.7755 ± 0.0386 | 0.0475 ± 0.0183 |
| Main | MedViT-Small | 5 | 20.6 ± 11.2 | 0.2266 ± 0.0381 | 0.2794 ± 0.0153 | 0.2951 ± 0.0143 | 0.0685 ± 0.0505 | 0.9492 ± 0.0256 | 0.9077 ± 0.0066 | 0.0416 ± 0.0279 | 0.8616 ± 0.0557 | 0.7881 ± 0.0168 | 0.0734 ± 0.0481 |
