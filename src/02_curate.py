#!/usr/bin/env python3
"""
PhytoScreen-AI | Phase 2: curation of raw ChEMBL IC50 data (per target).

Filters are applied SEQUENTIALLY; every audit number is len(df) measured after that step.
Raw files are never modified. Every dropped record is logged with a reason.

Steps:
  1 relation '='            2 units 'nM'            3 numeric value > 0
  4 pIC50 in [3, 12]        5 no data_validity_comment
  6 potential_duplicate != 1
  7 RDKit standardisation (cleanup, largest fragment, neutralise); failures dropped
  8 merge records with identical standardised SMILES (median pIC50)

Run from anywhere:  python src/02_curate.py
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "chembl"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
AUDIT = ROOT / "reports" / "audit"
TARGETS = ["AChE", "BChE", "BACE1"]
PIC50_MIN, PIC50_MAX = 3.0, 12.0
VARIANCE_THRESHOLD = 1.0  # log units

_lfc = rdMolStandardize.LargestFragmentChooser()
_unch = rdMolStandardize.Uncharger()


def standardize(smi):
    """Return (standardised isomeric SMILES, InChIKey first block) or (None, None)."""
    if not isinstance(smi, str) or not smi.strip():
        return None, None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None, None
        m = rdMolStandardize.Cleanup(m)
        m = _lfc.choose(m)
        m = _unch.uncharge(m)
        std = Chem.MolToSmiles(m)
        if not std:
            return None, None
        ik = Chem.MolToInchiKey(m)
        return std, (ik.split("-")[0] if ik else None)
    except Exception:
        return None, None


def curate(target: str):
    src = RAW / target / f"{target}_IC50_raw_flat.csv"
    df = pd.read_csv(src, low_memory=False)
    id_col = "activity_id" if "activity_id" in df.columns else None
    steps, dropped = [], []

    def log(name, d):
        steps.append({"step": name, "rows": int(len(d)),
                      "unique_molecules": int(d["molecule_chembl_id"].nunique())})

    def keep(d, mask, reason):
        mask = mask.fillna(False)
        gone = d[~mask]
        if len(gone):
            rec = pd.DataFrame({
                "activity_id": gone[id_col] if id_col else gone.index,
                "molecule_chembl_id": gone["molecule_chembl_id"],
                "reason": reason})
            dropped.append(rec)
        return d[mask]

    log("raw rows read", df)

    df = keep(df, df["standard_relation"] == "=", "relation_not_equals")
    log("relation '='", df)

    df = keep(df, df["standard_units"] == "nM", "units_not_nM")
    log("units 'nM'", df)

    df = df.copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = keep(df, df["standard_value"] > 0, "value_missing_or_nonpositive")
    log("numeric value > 0", df)

    df = df.copy()
    df["pIC50"] = 9.0 - np.log10(df["standard_value"])

    # Cross-check against ChEMBL's own pChEMBL (information only, no filtering)
    pch = pd.to_numeric(df.get("pchembl_value"), errors="coerce")
    both = df["pIC50"].notna() & pch.notna()
    diff = (df.loc[both, "pIC50"] - pch[both]).abs()
    pchembl_check = {"n_compared": int(both.sum()),
                     "max_abs_diff": float(diff.max()) if both.any() else None,
                     "n_diff_gt_0.01": int((diff > 0.01).sum())}

    df = keep(df, df["pIC50"].between(PIC50_MIN, PIC50_MAX), "pIC50_outside_3_12")
    log(f"pIC50 in [{PIC50_MIN:g}, {PIC50_MAX:g}]", df)

    df = keep(df, df["data_validity_comment"].isna(), "data_validity_comment_present")
    log("no data_validity_comment", df)

    dup = pd.to_numeric(df["potential_duplicate"], errors="coerce").fillna(0)
    df = keep(df, dup != 1, "potential_duplicate_flag")
    log("potential_duplicate != 1", df)

    df = df.copy()
    res = df["canonical_smiles"].map(standardize)
    df["smiles_std"] = res.map(lambda x: x[0])
    df["inchikey14"] = res.map(lambda x: x[1])
    df = keep(df, df["smiles_std"].notna(), "smiles_standardisation_failed")
    log("SMILES standardised", df)

    # Aggregate identical standardised structures
    g = df.groupby("smiles_std")
    out = g.agg(
        inchikey14=("inchikey14", "first"),
        molecule_chembl_id=("molecule_chembl_id", "first"),
        molecule_chembl_ids=("molecule_chembl_id", lambda s: ";".join(sorted(set(s)))),
        median_pIC50=("pIC50", "median"),
        min_pIC50=("pIC50", "min"),
        max_pIC50=("pIC50", "max"),
        n_records=("pIC50", "size"),
        n_assays=("assay_chembl_id", "nunique"),
    ).reset_index().rename(columns={"smiles_std": "canonical_smiles"})
    out["spread"] = out["max_pIC50"] - out["min_pIC50"]
    out["high_variance_flag"] = out["spread"] > VARIANCE_THRESHOLD
    steps.append({"step": "merged by standardised SMILES (final molecules)",
                  "rows": int(len(out)), "unique_molecules": int(len(out))})

    # Integrity: counts must never increase
    rows = [s["rows"] for s in steps]
    assert all(a >= b for a, b in zip(rows, rows[1:])), f"{target}: counts increased {rows}"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / f"{target}_IC50_curated.csv"
    out.to_csv(out_path, index=False)
    if dropped:
        pd.concat(dropped).to_csv(INTERIM / f"{target}_dropped_records.csv", index=False)

    stats = {"min": float(out.median_pIC50.min()), "median": float(out.median_pIC50.median()),
             "max": float(out.median_pIC50.max()),
             "n_high_variance": int(out.high_variance_flag.sum()),
             "n_with_multiple_records": int((out.n_records > 1).sum())}
    return {"source_csv": str(src.relative_to(ROOT)), "steps": steps,
            "pchembl_crosscheck": pchembl_check, "curated_csv": str(out_path.relative_to(ROOT)),
            "final_pIC50_stats": stats}


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    audit = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "pIC50_range": [PIC50_MIN, PIC50_MAX], "targets": {}}
    for t in TARGETS:
        print(f"Curating {t} ...")
        audit["targets"][t] = curate(t)
    (AUDIT / "curation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = ["# Curation Audit", f"- Created (UTC): {audit['created_utc']}",
             f"- pIC50 kept: {PIC50_MIN:g} to {PIC50_MAX:g}", ""]
    for t, a in audit["targets"].items():
        lines += [f"## {t}", "", "| Step | Rows | Unique molecules |", "|---|---|---|"]
        lines += [f"| {s['step']} | {s['rows']} | {s['unique_molecules']} |" for s in a["steps"]]
        c, s = a["pchembl_crosscheck"], a["final_pIC50_stats"]
        lines += ["", f"- pChEMBL cross-check: {c['n_compared']} compared, max abs diff {c['max_abs_diff']}, "
                      f"{c['n_diff_gt_0.01']} differ by >0.01",
                  f"- Final median pIC50 min/median/max: {s['min']:.2f} / {s['median']:.2f} / {s['max']:.2f}",
                  f"- High-variance molecules (spread > {VARIANCE_THRESHOLD:g}): {s['n_high_variance']}",
                  f"- Dropped-record log: data/interim/{t}_dropped_records.csv", ""]
    (AUDIT / "curation_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
