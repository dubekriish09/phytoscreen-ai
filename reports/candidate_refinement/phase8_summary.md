# Phase 8 Candidate Refinement Summary

- Created (UTC): 2026-11-06T14:14:01.808102+00:00
- Rules: top 200 novel per target, max Tanimoto >= 0.3, MolWt 150-700
- Consistency check: Multi-target compounds: Phase 5 = 91, rebuilt here = 91, overlap = 91

## Selection groups per target (within the top-N lists)
A = in-domain, no alerts; B = borderline, no alerts; D = alert, needs structural review

selection_group   A    B    D
Target                       
AChE             40   12  148
BACE1             2   10  188
BChE             12  163   25

## Docking set

Target  role                
AChE    experimental_control     5
        novel_candidate         20
BACE1   experimental_control     5
        novel_candidate         12
BChE    experimental_control     5
        novel_candidate         25

## Diversity of selected novel compounds

```
{
  "AChE": {
    "n_novel_selected": 20,
    "n_unique_scaffolds": 15,
    "mean_pairwise_tanimoto": 0.20655622477090052
  },
  "BChE": {
    "n_novel_selected": 25,
    "n_unique_scaffolds": 21,
    "mean_pairwise_tanimoto": 0.11587504055512209
  },
  "BACE1": {
    "n_novel_selected": 12,
    "n_unique_scaffolds": 10,
    "mean_pairwise_tanimoto": 0.1880653644875318
  }
}
```

- Experimental positive controls available (library compounds with ChEMBL IC50): {'AChE': 107, 'BACE1': 66, 'BChE': 58}
- Glycosides flagged: 236; aglycones generated (to re-score): 236
- PAINS matches among top-N candidates (informational, not used for exclusion): 126

## Caveats
- Predictions are ML estimates; the models are moderate (scaffold-split R2 0.45-0.56).
- Controls are in the training data, so their predictions are in-sample and not a validation.
- No docking results are used here.