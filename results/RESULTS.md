# Results

All numbers are produced by the scripts in `scripts/`. CV rows: mean ± std over all folds of all repeats. LODO: pooled predictions.


## Fdataset - cv5

| Method         | AUC             | AUPR            |
|:---------------|:----------------|:----------------|
| MV-HGAT (ours) | 0.9392 ± 0.0072 | 0.4878 ± 0.0273 |
| MBiRW          | 0.8831 ± 0.0095 | 0.3105 ± 0.0187 |
| DRRS           | 0.8787 ± 0.0123 | 0.3893 ± 0.0174 |
| SCMFDD         | 0.8934 ± 0.0102 | 0.4946 ± 0.0199 |
| NIMCGCN        | 0.8384 ± 0.0089 | 0.0956 ± 0.0137 |
| LAGCN          | 0.8333 ± 0.0152 | 0.1328 ± 0.0181 |


## Cdataset - cv5

| Method         | AUC             | AUPR            |
|:---------------|:----------------|:----------------|
| MV-HGAT (ours) | 0.9602 ± 0.0052 | 0.5862 ± 0.0260 |
| MBiRW          | 0.9108 ± 0.0064 | 0.4260 ± 0.0172 |
| DRRS           | 0.9069 ± 0.0096 | 0.5021 ± 0.0208 |
| SCMFDD         | 0.9125 ± 0.0080 | 0.5879 ± 0.0207 |
| NIMCGCN        | 0.8746 ± 0.0086 | 0.1174 ± 0.0150 |
| LAGCN          | 0.8649 ± 0.0094 | 0.2256 ± 0.0259 |
