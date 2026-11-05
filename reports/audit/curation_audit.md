# Curation Audit
- Created (UTC): 2026-11-05T01:24:03.832522+00:00
- pIC50 kept: 3 to 12

## AChE

| Step | Rows | Unique molecules |
|---|---|---|
| raw rows read | 10079 | 8372 |
| relation '=' | 7901 | 6345 |
| units 'nM' | 7827 | 6289 |
| numeric value > 0 | 7826 | 6288 |
| pIC50 in [3, 12] | 7681 | 6180 |
| no data_validity_comment | 7275 | 5831 |
| potential_duplicate != 1 | 6456 | 5831 |
| SMILES standardised | 6455 | 5830 |
| merged by standardised SMILES (final molecules) | 5727 | 5727 |

- pChEMBL cross-check: 7275 compared, max abs diff 0.005496221825573855, 0 differ by >0.01
- Final median pIC50 min/median/max: 4.00 / 5.96 / 10.96
- High-variance molecules (spread > 1): 94
- Dropped-record log: data/interim/AChE_dropped_records.csv

## BChE

| Step | Rows | Unique molecules |
|---|---|---|
| raw rows read | 5534 | 4720 |
| relation '=' | 4513 | 3781 |
| units 'nM' | 4497 | 3770 |
| numeric value > 0 | 4497 | 3770 |
| pIC50 in [3, 12] | 4455 | 3739 |
| no data_validity_comment | 4272 | 3568 |
| potential_duplicate != 1 | 3876 | 3568 |
| SMILES standardised | 3874 | 3566 |
| merged by standardised SMILES (final molecules) | 3511 | 3511 |

- pChEMBL cross-check: 4272 compared, max abs diff 0.005496221825573855, 0 differ by >0.01
- Final median pIC50 min/median/max: 4.00 / 5.86 / 10.67
- High-variance molecules (spread > 1): 26
- Dropped-record log: data/interim/BChE_dropped_records.csv

## BACE1

| Step | Rows | Unique molecules |
|---|---|---|
| raw rows read | 13777 | 8938 |
| relation '=' | 12913 | 8274 |
| units 'nM' | 12911 | 8272 |
| numeric value > 0 | 12911 | 8272 |
| pIC50 in [3, 12] | 12879 | 8251 |
| no data_validity_comment | 12715 | 8112 |
| potential_duplicate != 1 | 10643 | 8112 |
| SMILES standardised | 10643 | 8112 |
| merged by standardised SMILES (final molecules) | 8082 | 8082 |

- pChEMBL cross-check: 12721 compared, max abs diff 0.005496221825573855, 0 differ by >0.01
- Final median pIC50 min/median/max: 3.03 / 6.84 / 10.96
- High-variance molecules (spread > 1): 582
- Dropped-record log: data/interim/BACE1_dropped_records.csv
