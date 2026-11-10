# Phase 7 — Chemical Space and Applicability Domain Analysis

- **Created (UTC):** 2026-11-05T19:53:25.816142
- **Library:** 18214 phytochemical compounds, 0 with invalid SMILES
- **Fingerprint:** Morgan radius=2, 2048 bits
- **Bootstrap CI:** n=500 resamples, seed=42

> **Scientific labelling reminder**
> - `*_pred_*` columns are ML predictions (ensemble, RF, XGBoost).
> - `*_exp_pIC50` are experimental ChEMBL median pIC50 values.
> - `unc_tree_std` and `unc_disagree` are **uncertainty proxies**, not calibrated confidence intervals.
> - Tanimoto thresholds have been evaluated empirically here; none is universally validated.

---

## A. Chemical-Space Overview

### AChE
- Training set: 5727 molecules | Library (valid SMILES): 18214
- Library invalid SMILES: 0
- Median max Tanimoto to training: 0.298

**Similarity bins (Morgan radius=2, 2048 bits):**

| Bin | Count | % of library |
|-----|-------|--------------|
| 0.0–0.3 | 9167 | 50.33 |
| 0.3–0.5 | 7243 | 39.77 |
| 0.5–0.7 | 1362 | 7.48 |
| 0.7–1.0 | 442 | 2.43 |

**Molecular-property comparison (median [Q25–Q75]):**

| Property | Library | Training |
|----------|---------|----------|
| MolWt | 314.25 [207.2–444.7] | 400.43 [335.43–481.54] |
| LogP | 2.83 [1.52–4.25] | 4.38 [3.33–5.57] |
| TPSA | 57.53 [26.3–106.04] | 56.79 [41.57–81.01] |
| HBD | 1.0 [0.0–3.0] | 1.0 [0.0–2.0] |
| HBA | 4.0 [2.0–7.0] | 4.0 [3.0–6.0] |

### BChE
- Training set: 3511 molecules | Library (valid SMILES): 18214
- Library invalid SMILES: 0
- Median max Tanimoto to training: 0.247

**Similarity bins (Morgan radius=2, 2048 bits):**

| Bin | Count | % of library |
|-----|-------|--------------|
| 0.0–0.3 | 12583 | 69.08 |
| 0.3–0.5 | 4841 | 26.58 |
| 0.5–0.7 | 580 | 3.18 |
| 0.7–1.0 | 210 | 1.15 |

**Molecular-property comparison (median [Q25–Q75]):**

| Property | Library | Training |
|----------|---------|----------|
| MolWt | 314.25 [207.2–444.7] | 424.63 [356.45–507.63] |
| LogP | 2.83 [1.52–4.25] | 4.81 [3.73–6.09] |
| TPSA | 57.53 [26.3–106.04] | 58.2 [42.43–83.14] |
| HBD | 1.0 [0.0–3.0] | 1.0 [0.0–2.0] |
| HBA | 4.0 [2.0–7.0] | 4.0 [3.0–6.0] |

### BACE1
- Training set: 8082 molecules | Library (valid SMILES): 18214
- Library invalid SMILES: 0
- Median max Tanimoto to training: 0.250

**Similarity bins (Morgan radius=2, 2048 bits):**

| Bin | Count | % of library |
|-----|-------|--------------|
| 0.0–0.3 | 12726 | 69.87 |
| 0.3–0.5 | 4331 | 23.78 |
| 0.5–0.7 | 862 | 4.73 |
| 0.7–1.0 | 295 | 1.62 |

**Molecular-property comparison (median [Q25–Q75]):**

| Property | Library | Training |
|----------|---------|----------|
| MolWt | 314.25 [207.2–444.7] | 443.57 [397.38–502.61] |
| LogP | 2.83 [1.52–4.25] | 3.52 [2.79–4.32] |
| TPSA | 57.53 [26.3–106.04] | 91.85 [73.39–111.72] |
| HBD | 1.0 [0.0–3.0] | 2.0 [1.0–2.0] |
| HBA | 4.0 [2.0–7.0] | 6.0 [5.0–7.0] |

---

## B. Applicability-Domain Validation (Scaffold Test Sets)

Error is computed on the scaffold-held-out test set only. Numbers come from executed code.

### AChE  (n_test = 573)

Spearman ρ(max_tanimoto, |err|):  XGBoost ρ=-0.0842  p=0.044003  |  RF ρ=-0.0827  p=0.047786

| Bin | n | XGB RMSE | XGB MAE | XGB MedAE | RF RMSE | RF MAE |
|-----|---|----------|---------|-----------|---------|--------|
| 0.0–0.3 | 51 | 0.9511 | 0.797 | 0.7472 | 0.9146 | 0.7525 |
| 0.3–0.5 | 98 | 1.036 | 0.8463 | 0.8058 | 1.006 | 0.8223 |
| 0.5–0.7 | 179 | 0.9535 | 0.7601 | 0.637 | 0.8977 | 0.7207 |
| 0.7–1.0 | 245 | 0.8647 | 0.6997 | 0.637 | 0.8667 | 0.6876 |

> Negative rho means higher similarity → lower error. Inspect p-value before interpreting. BChE lowest-bin has very few samples (n≤20); statistics are noisy.

### BChE  (n_test = 352)

Spearman ρ(max_tanimoto, |err|):  XGBoost ρ=-0.0123  p=0.817638  |  RF ρ=-0.0024  p=0.96423

| Bin | n | XGB RMSE | XGB MAE | XGB MedAE | RF RMSE | RF MAE |
|-----|---|----------|---------|-----------|---------|--------|
| 0.0–0.3 | 20 | 1.508 | 1.1269 | 1.0929 | 1.4374 | 1.0247 |
| 0.3–0.5 | 60 | 0.6965 | 0.4916 | 0.355 | 0.7231 | 0.5204 |
| 0.5–0.7 | 115 | 0.7487 | 0.5956 | 0.543 | 0.7119 | 0.5731 |
| 0.7–1.0 | 157 | 0.7961 | 0.615 | 0.535 | 0.819 | 0.6312 |

> Negative rho means higher similarity → lower error. Inspect p-value before interpreting. BChE lowest-bin has very few samples (n≤20); statistics are noisy.

### BACE1  (n_test = 809)

Spearman ρ(max_tanimoto, |err|):  XGBoost ρ=-0.1498  p=1.9e-05  |  RF ρ=-0.148  p=2.4e-05

| Bin | n | XGB RMSE | XGB MAE | XGB MedAE | RF RMSE | RF MAE |
|-----|---|----------|---------|-----------|---------|--------|
| 0.0–0.3 | 36 | 1.3247 | 1.0931 | 1.0118 | 1.2118 | 0.9598 |
| 0.3–0.5 | 62 | 0.9391 | 0.7753 | 0.7511 | 1.0302 | 0.8666 |
| 0.5–0.7 | 217 | 0.7604 | 0.6151 | 0.5125 | 0.7739 | 0.6242 |
| 0.7–1.0 | 494 | 0.7341 | 0.5669 | 0.4653 | 0.7349 | 0.5766 |

> Negative rho means higher similarity → lower error. Inspect p-value before interpreting. BChE lowest-bin has very few samples (n≤20); statistics are noisy.

---

## C. Threshold Sensitivity

RMSE computed on scaffold test compounds above / below each cutoff (XGBoost). `Supported = True` means retained set has lower RMSE than excluded set.

### AChE

| Cutoff | Lib retained | % retained | Test n above | RMSE above | RMSE below | Supported? |
|--------|-------------|------------|--------------|------------|------------|------------|
| 0.3 | 9047 | 49.67 | 522 | 0.9296 | 0.9511 | True |
| 0.4 | 4158 | 22.83 | 473 | 0.9206 | 0.9817 | True |
| 0.5 | 1804 | 9.9 | 424 | 0.9032 | 1.0077 | True |
| 0.6 | 827 | 4.54 | 357 | 0.8927 | 0.9923 | True |
| 0.7 | 442 | 2.43 | 245 | 0.8647 | 0.9785 | True |

### BChE

| Cutoff | Lib retained | % retained | Test n above | RMSE above | RMSE below | Supported? |
|--------|-------------|------------|--------------|------------|------------|------------|
| 0.3 | 5631 | 30.92 | 332 | 0.7626 | 1.508 | True |
| 0.4 | 2105 | 11.56 | 304 | 0.7608 | 1.1418 | True |
| 0.5 | 790 | 4.34 | 272 | 0.7764 | 0.9656 | True |
| 0.6 | 372 | 2.04 | 237 | 0.7714 | 0.9209 | True |
| 0.7 | 210 | 1.15 | 157 | 0.7961 | 0.8445 | True |

### BACE1

| Cutoff | Lib retained | % retained | Test n above | RMSE above | RMSE below | Supported? |
|--------|-------------|------------|--------------|------------|------------|------------|
| 0.3 | 5488 | 30.13 | 773 | 0.7599 | 1.3247 | True |
| 0.4 | 2313 | 12.7 | 744 | 0.7475 | 1.2019 | True |
| 0.5 | 1157 | 6.35 | 711 | 0.7423 | 1.0966 | True |
| 0.6 | 591 | 3.24 | 648 | 0.7309 | 1.0073 | True |
| 0.7 | 295 | 1.62 | 494 | 0.7341 | 0.8789 | True |

---

## D. Uncertainty-Proxy Analysis

**All proxies are labelled as proxies — not calibrated predictive intervals.**
Per-tree RF predictions are not stored in test_predictions.csv; tree_std analysis limited to library screening output.
RF–XGBoost disagreement is computed as |pred_xgb − pred_rf| and evaluated against scaffold test errors.

### AChE
- Library `unc_tree_std` median: 1.0246 (Q25=0.8966, Q75=1.1619)
- Library `unc_disagree` median: 0.1891 (Q25=0.0879, Q75=0.332)
- Spearman ρ(disagree, |err_xgb|) on scaffold test: ρ=0.0819  p=0.050052  n=573
- RF tree std vs test error: NOT AVAILABLE — per-tree predictions not saved in test_predictions.csv

### BChE
- Library `unc_tree_std` median: 0.8549 (Q25=0.7594, Q75=0.9675)
- Library `unc_disagree` median: 0.1732 (Q25=0.0809, Q75=0.3011)
- Spearman ρ(disagree, |err_xgb|) on scaffold test: ρ=0.0563  p=0.292274  n=352
- RF tree std vs test error: NOT AVAILABLE — per-tree predictions not saved in test_predictions.csv

### BACE1
- Library `unc_tree_std` median: 0.9465 (Q25=0.8366, Q75=1.0607)
- Library `unc_disagree` median: 0.2214 (Q25=0.1051, Q75=0.3926)
- Spearman ρ(disagree, |err_xgb|) on scaffold test: ρ=0.0137  p=0.69695  n=809
- RF tree std vs test error: NOT AVAILABLE — per-tree predictions not saved in test_predictions.csv

---

## E. Candidate-Level Audit

| Target | Candidates | Out-of-domain | Structural flags | In multitarget top |
|--------|-----------|---------------|-----------------|-------------------|
| AChE | 200 | 0 | 145 | 90 |
| BChE | 200 | 0 | 25 | 4 |
| BACE1 | 200 | 0 | 189 | 88 |

Full per-compound audit in `reports/applicability_domain/candidate_reliability_audit.csv`.
Structural flags: glycoside-pattern match, Michael acceptor, aldehyde, epoxide, acyl halide, MW outlier.
These flags require manual structural review — compounds are not removed automatically.

---

## Figures

| File | Description |
|------|-------------|
| fig1_similarity_distributions.png | Tanimoto similarity histograms per target |
| fig2_property_distributions.png | Library vs. training molecular properties |
| fig3_error_vs_similarity.png | Held-out absolute error vs. max Tanimoto |
| fig4_coverage_vs_error.png | Coverage and RMSE at alternative cutoffs |
| fig5_uncertainty_vs_error.png | Uncertainty proxy vs. absolute error |

---

## Scientific Limitations

1. **Low coverage**: Most phytochemicals are far from the ChEMBL training chemical space (median Tanimoto < 0.4). Predictions for out-of-domain compounds should be treated as low-confidence rankings only.
2. **Model shrinkage**: Tree-based models underestimate extreme pIC50 values. Use predictions to rank, not as potency estimates.
3. **Tanimoto thresholds**: No threshold tested here is universally validated. The empirical support column in threshold_sensitivity.csv shows whether a given cutoff correlates with improved accuracy on this test set.
4. **Uncertainty proxies**: RF tree spread and RF–XGBoost disagreement are not calibrated uncertainty estimates. High disagreement is a warning, not a quantified interval.
5. **Glycosylated compounds**: Flagged for review. Similarity to training may be artificially inflated by the sugar moiety.
6. **Scaffold split artefacts**: Low-similarity test compounds are genuinely novel scaffolds; their error may be worse purely because tree models extrapolate poorly, not because they are inactive.

---

## Recommended Candidate-Review Strategy (Phase 8)

1. **Prioritise experimental evidence first**: compounds in `*_known_experimental.csv` have real ChEMBL IC50 data.
2. **Within novel candidates, prefer in-domain compounds** (max_tanimoto ≥ 0.5), especially where the threshold sensitivity analysis shows lower RMSE.
3. **Review all structural flags manually** before docking. Glycosylated flavonoids should be docked as the aglycone unless plant-bioavailability is explicitly in scope.
4. **Multi-target compounds** (n=91) are highest priority for selectivity profiling.
5. **For out-of-domain borderline compounds** that rank highly, consider literature-mining or public bioactivity databases before committing to synthesis or wet-lab assays.
6. **Do not use predicted pIC50 values as potency estimates** — present them as rankings with the domain tier and uncertainty proxy alongside.
