# Model metrics (test sets, real executed results)
- Created (UTC): 2026-11-05T16:42:56.521906+00:00

## AChE (5727 molecules)

| Split | Model | R2 | RMSE | MAE | Spearman | R2 excl. high-var |
|---|---|---|---|---|---|---|
| scaffold | mean_baseline | -0.001 | 1.251 | 1.050 | nan | -0.001 |
| scaffold | RandomForest | 0.475 | 0.906 | 0.727 | 0.696 | 0.479 |
| scaffold | XGBoost | 0.445 | 0.932 | 0.752 | 0.670 | 0.448 |
| random | mean_baseline | -0.002 | 1.308 | 1.073 | nan | -0.000 |
| random | RandomForest | 0.693 | 0.724 | 0.508 | 0.811 | 0.689 |
| random | XGBoost | 0.685 | 0.733 | 0.514 | 0.807 | 0.680 |

Applicability domain (scaffold test, XGBoost):

| Max Tanimoto to train | n | RMSE |
|---|---|---|
| 0.0-0.3 | 51 | 0.951 |
| 0.3-0.5 | 98 | 1.036 |
| 0.5-0.7 | 179 | 0.953 |
| 0.7-1.0 | 245 | 0.865 |

## BChE (3511 molecules)

| Split | Model | R2 | RMSE | MAE | Spearman | R2 excl. high-var |
|---|---|---|---|---|---|---|
| scaffold | mean_baseline | -0.001 | 1.237 | 0.987 | nan | -0.001 |
| scaffold | RandomForest | 0.561 | 0.819 | 0.616 | 0.705 | 0.561 |
| scaffold | XGBoost | 0.557 | 0.823 | 0.617 | 0.695 | 0.557 |
| random | mean_baseline | -0.008 | 1.121 | 0.918 | nan | -0.008 |
| random | RandomForest | 0.709 | 0.603 | 0.432 | 0.846 | 0.708 |
| random | XGBoost | 0.692 | 0.620 | 0.447 | 0.823 | 0.692 |

Applicability domain (scaffold test, XGBoost):

| Max Tanimoto to train | n | RMSE |
|---|---|---|
| 0.0-0.3 | 20 | 1.508 |
| 0.3-0.5 | 60 | 0.697 |
| 0.5-0.7 | 115 | 0.749 |
| 0.7-1.0 | 157 | 0.796 |

## BACE1 (8082 molecules)

| Split | Model | R2 | RMSE | MAE | Spearman | R2 excl. high-var |
|---|---|---|---|---|---|---|
| scaffold | mean_baseline | -0.082 | 1.230 | 1.013 | nan | -0.099 |
| scaffold | RandomForest | 0.544 | 0.798 | 0.629 | 0.738 | 0.554 |
| scaffold | XGBoost | 0.550 | 0.794 | 0.619 | 0.734 | 0.559 |
| random | mean_baseline | -0.002 | 1.211 | 1.020 | nan | -0.001 |
| random | RandomForest | 0.675 | 0.690 | 0.498 | 0.826 | 0.679 |
| random | XGBoost | 0.706 | 0.657 | 0.477 | 0.842 | 0.710 |

Applicability domain (scaffold test, XGBoost):

| Max Tanimoto to train | n | RMSE |
|---|---|---|
| 0.0-0.3 | 36 | 1.325 |
| 0.3-0.5 | 62 | 0.939 |
| 0.5-0.7 | 217 | 0.76 |
| 0.7-1.0 | 494 | 0.734 |
