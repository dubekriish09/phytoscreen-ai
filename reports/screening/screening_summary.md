# Screening summary (ML predictions, not docking, not experiment)
- Created (UTC): 2026-11-05T16:57:53.708799+00:00
- Library size: 18214
- Docking-candidate rules: not in ChEMBL training data, max Tanimoto >= 0.3, MolWt 150-700, top 200 by ensemble prediction

## AChE
- Library compounds with ChEMBL IC50 evidence (experimental): 107
- Domain tiers (max Tanimoto to training): {'out_of_domain': 9167, 'borderline': 7243, 'in_domain': 1804}
- Predicted pIC50 (ensemble) min/median/max: 4.30 / 5.37 / 7.63
- Eligible novel compounds: 7901; written top 200

## BChE
- Library compounds with ChEMBL IC50 evidence (experimental): 58
- Domain tiers (max Tanimoto to training): {'out_of_domain': 12583, 'borderline': 4841, 'in_domain': 790}
- Predicted pIC50 (ensemble) min/median/max: 4.31 / 5.19 / 7.85
- Eligible novel compounds: 5161; written top 200

## BACE1
- Library compounds with ChEMBL IC50 evidence (experimental): 66
- Domain tiers (max Tanimoto to training): {'out_of_domain': 12726, 'borderline': 4331, 'in_domain': 1157}
- Predicted pIC50 (ensemble) min/median/max: 4.09 / 5.33 / 7.79
- Eligible novel compounds: 4902; written top 200

## Multi-target
- Compounds in the top-200 of at least 2 targets: 91

## Reading guide
- `*_pred_*` are ML predictions; `*_exp_pIC50` are experimental ChEMBL medians. Never mix them.
- Out-of-domain predictions are unreliable. Phytochemicals are often far from the ChEMBL chemical space.
- Next step is docking of selected candidates (Phase 6) and expert review.