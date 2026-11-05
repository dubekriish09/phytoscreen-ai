# ChEMBL Acquisition Audit

- Created (UTC): 2026-11-04T18:53:59.495375+00:00
- ChEMBL release: ChEMBL_37
- Types requested: IC50
- Partial smoke test: False

## AChE (CHEMBL220)
- Verified: Acetylcholinesterase | Homo sapiens | UniProt ['P22303'] | passed=True
- **IC50**: downloaded 10079 / API total 10079 (match=True)
  - molecules: 8372, assays: 1031, documents: 685
  - with standard_value: 8793, pChEMBL: 7275, SMILES: 10040
  - validity‑comment flagged: 736, potential duplicates: 858
  - relations: {'=': 7901, 'MISSING': 1286, '>': 875, '<': 13, '~': 2, '>>': 2}
  - units: {'nM': 8723, 'MISSING': 1265, 'ug.mL-1': 83, "10'5pM": 3, "10'6pM": 2, '10^-4microM': 2, "10'3pM": 1}

## BChE (CHEMBL1914)
- Verified: Cholinesterase | Homo sapiens | UniProt ['P06276'] | passed=True
- **IC50**: downloaded 5534 / API total 5534 (match=True)
  - molecules: 4720, assays: 553, documents: 413
  - with standard_value: 5217, pChEMBL: 4272, SMILES: 5531
  - validity‑comment flagged: 329, potential duplicates: 423
  - relations: {'=': 4513, '>': 701, 'MISSING': 317, '<': 3}
  - units: {'nM': 5207, 'MISSING': 306, 'ug.mL-1': 21}

## BACE1 (CHEMBL4822)
- Verified: Beta-secretase 1 | Homo sapiens | UniProt ['P56817'] | passed=True
- **IC50**: downloaded 13777 / API total 13777 (match=True)
  - molecules: 8938, assays: 725, documents: 462
  - with standard_value: 13616, pChEMBL: 12721, SMILES: 13775
  - validity‑comment flagged: 305, potential duplicates: 2126
  - relations: {'=': 12913, '>': 603, 'MISSING': 161, '<': 100}
  - units: {'nM': 13614, 'MISSING': 161, 'ug.mL-1': 2}

## Notes
- Raw pages are unmodified API responses (gzipped JSON). Flat CSVs are unfiltered.
- No curation, deduplication or modelling has been performed (Phase 2+).
- Non‑human ChEMBL AChE/BChE/BACE entries appear only in discovery files, not downloaded.