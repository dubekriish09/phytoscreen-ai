#!/usr/bin/env python3
"""
PhytoScreen-AI | Phase 3: standardise a phytochemical library (IMPPAT export) for screening.

Reads a CSV/TSV/XLSX/SDF from data/raw/imppat/, standardises structures exactly as in Phase 2,
merges identical structures, flags overlap with the ChEMBL training sets (= compounds that already
have experimental IC50 evidence) and writes an audit. Raw files are never modified.

Usage:
  python src/03_prepare_library.py --input data/raw/imppat/<file>.csv
  python src/03_prepare_library.py --input <file> --smiles-col SMILES --id-col "IMPPAT ID" --name-col Name
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
AUDIT = ROOT / "reports" / "audit"
TARGETS = ["AChE", "BChE", "BACE1"]
_lfc = rdMolStandardize.LargestFragmentChooser()
_unch = rdMolStandardize.Uncharger()


def standardize(smi):
    if not isinstance(smi, str) or not smi.strip():
        return None, None, None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None, None, None
        m = rdMolStandardize.Cleanup(m)
        m = _lfc.choose(m)
        m = _unch.uncharge(m)
        std = Chem.MolToSmiles(m)
        ik = Chem.MolToInchiKey(m)
        props = (Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
                 rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m))
        return std, (ik.split("-")[0] if ik else None), props
    except Exception:
        return None, None, None


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".sdf":
        rows = []
        for m in Chem.SDMolSupplier(str(path)):
            if m is None:
                rows.append({"smiles": None})
                continue
            d = m.GetPropsAsDict()
            d["smiles"] = Chem.MolToSmiles(m)
            rows.append(d)
        return pd.DataFrame(rows)
    raise SystemExit(f"Unsupported file type: {ext}")


def pick(cols, user, keys):
    if user:
        if user not in cols:
            raise SystemExit(f"Column '{user}' not found. Columns are: {list(cols)}")
        return user
    for k in keys:
        for c in cols:
            if k in str(c).lower():
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--smiles-col")
    ap.add_argument("--id-col")
    ap.add_argument("--name-col")
    a = ap.parse_args()

    path = Path(a.input)
    raw = read_table(path)
    print("Columns found:", list(raw.columns))
    smi_c = pick(raw.columns, a.smiles_col, ["smiles"])
    id_c = pick(raw.columns, a.id_col, ["imphy", "identifier", "phytochemical id", "id"])
    name_c = pick(raw.columns, a.name_col, ["name"])
    if smi_c is None:
        raise SystemExit("No SMILES column detected. Re-run with --smiles-col <name>.")
    print(f"Using SMILES={smi_c!r}, ID={id_c!r}, NAME={name_c!r}")

    df = pd.DataFrame({"source_id": raw[id_c] if id_c else raw.index.astype(str),
                       "name": raw[name_c] if name_c else None,
                       "smiles_raw": raw[smi_c]})
    steps = [{"step": "rows read", "rows": int(len(df))}]
    df = df[df["smiles_raw"].notna() & (df["smiles_raw"].astype(str).str.strip() != "")]
    steps.append({"step": "has SMILES", "rows": int(len(df))})

    res = df["smiles_raw"].map(standardize)
    df["smiles_std"] = res.map(lambda x: x[0])
    df["inchikey14"] = res.map(lambda x: x[1])
    df["_props"] = res.map(lambda x: x[2])
    df = df[df["smiles_std"].notna()].copy()
    steps.append({"step": "RDKit standardisation succeeded", "rows": int(len(df))})

    props = pd.DataFrame(df["_props"].tolist(), index=df.index,
                         columns=["MolWt", "LogP", "TPSA", "HBD", "HBA"])
    df = pd.concat([df.drop(columns="_props"), props], axis=1)

    lib = df.groupby("smiles_std").agg(
        inchikey14=("inchikey14", "first"),
        source_ids=("source_id", lambda s: ";".join(sorted(set(map(str, s))))),
        names=("name", lambda s: ";".join(sorted(set(map(str, s.dropna()))))[:300]),
        MolWt=("MolWt", "first"), LogP=("LogP", "first"), TPSA=("TPSA", "first"),
        HBD=("HBD", "first"), HBA=("HBA", "first"),
    ).reset_index().rename(columns={"smiles_std": "canonical_smiles"})
    steps.append({"step": "merged identical standardised structures", "rows": int(len(lib))})

    overlap = {}
    for t in TARGETS:
        f = PROCESSED / f"{t}_IC50_curated.csv"
        if f.exists():
            cur = pd.read_csv(f)
            # first InChIKey block = connectivity only (ignores stereo): flags likely known compounds
            lib[f"in_{t}_training"] = lib["inchikey14"].isin(set(cur["inchikey14"].dropna()))
            overlap[t] = int(lib[f"in_{t}_training"].sum())
        else:
            overlap[t] = "curated file not found"

    out = PROCESSED / "phytochemical_library.csv"
    lib.to_csv(out, index=False)
    audit = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "source_file": str(path), "columns_used": {"smiles": smi_c, "id": id_c, "name": name_c},
             "steps": steps, "overlap_with_chembl_training": overlap, "output": str(out.relative_to(ROOT))}
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "library_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md = ["# Phytochemical Library Audit", f"- Created (UTC): {audit['created_utc']}", f"- Source: {path}", "",
          "| Step | Rows |", "|---|---|"] + [f"| {s['step']} | {s['rows']} |" for s in steps]
    md += ["", "Overlap with ChEMBL training sets (compounds that may already have experimental IC50 data):"]
    md += [f"- {t}: {n}" for t, n in overlap.items()]
    (AUDIT / "library_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()