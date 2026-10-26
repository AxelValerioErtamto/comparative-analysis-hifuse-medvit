# Tuning on the validation set (test set not used)

Seed 0, both models, max 30 epochs, early stopping patience 7. Setting chosen by the mean validation MCC (threshold 0.5) of both models, compared at 2 decimals; ties broken by the smaller validation − train loss gap.

| regularisation | melanoma weight | mean val MCC | mean val bal. acc | mean val recall | mean val precision | mean val AUPRC | mean loss gap (val − train) |
|---|---|---|---|---|---|---|---|
| strong | 2 | 0.6380 | 0.8171 | 0.6853 | 0.6952 | 0.7202 | +0.0037 |
| strong | 3 | 0.6194 | 0.7956 | 0.6353 | 0.7114 | 0.7108 | +0.0041 |
| official | 2 | 0.6129 | 0.7814 | 0.6000 | 0.7336 | 0.7329 | +0.0061 |
| official | 4 | 0.6075 | 0.8019 | 0.6588 | 0.6693 | 0.7223 | +0.0102 |
| official | 3 | 0.5997 | 0.7813 | 0.6059 | 0.7053 | 0.7110 | +0.0079 |
| official | 6 | 0.5840 | 0.8069 | 0.6882 | 0.6096 | 0.6854 | +0.0069 |
| strong | 4 | 0.5604 | 0.7734 | 0.6029 | 0.6427 | 0.6559 | +0.0077 |
| strong | 6 | 0.4904 | 0.7566 | 0.6000 | 0.5377 | 0.6309 | +0.0007 |

**Chosen:** regularisation = strong {'drop_path': 0.2, 'dropout': 0.1, 'wd': 0.05}, melanoma weight = 2
