import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# RDKit imports – will raise if not installed
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception as e:
    raise ImportError("RDKit is required for SMILES standardisation. Install it before running this script.")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
RAW = ROOT / "data" / "raw" / "chembl"
PROCESSED = ROOT / "data" / "processed"
AUDIT_DIR = ROOT / "reports" / "audit"
PROCESSED.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def neutralize_molecule(mol):
    """Simple neutralisation: convert common charged groups to neutral forms.
    Returns the modified molecule or None if conversion fails.
    """
    # Define patterns for common charged groups
    patts = {
        "[+\-]": "[H]",
        "[NH3+]": "N",
        "[O-]": "O",
    }
    for pat, repl in patts.items():
        try:
            mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmarts(pat), Chem.MolFromSmiles(repl), replaceAll=True)[0]
        except Exception:
            continue
    return mol

def standardise_smiles(smiles):
    """Return a canonical SMILES after taking the largest fragment, neutralising, and sanitising.
    If any step fails, returns None.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Keep only the largest fragment (by atom count)
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return None
        mol = max(frags, key=lambda m: m.GetNumAtoms())
        # Neutralise common charges
        mol = neutralize_molecule(mol)
        Chem.SanitizeMol(mol)
        # Kekulize and generate canonical SMILES
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None

def curate_target(name: str, csv_path: Path):
    audit = {
        "target": name,
        "source_csv": str(csv_path.relative_to(ROOT)),
        "total_rows": 0,
        "after_relation_filter": 0,
        "after_units_filter": 0,
        "after_value_filter": 0,
        "after_validity_filter": 0,
        "after_duplicate_filter": 0,
        "smiles_success": 0,
        "final_rows": 0,
    }
    df = pd.read_csv(csv_path, low_memory=False)
    audit["total_rows"] = len(df)

    # Keep only rows where standard_relation == "="
    df = df[df["standard_relation"] == "="]
    audit["after_relation_filter"] = len(df)

    # Keep only rows where standard_units == "nM"
    df = df[df["standard_units"] == "nM"]
    audit["after_units_filter"] = len(df)

    # Keep only positive standard_value
    df = df[pd.to_numeric(df["standard_value"], errors="coerce") > 0]
    audit["after_value_filter"] = len(df)
    # Compute pIC50 (9 - log10(nM))
    df["pIC50_computed"] = 9 - np.log10(pd.to_numeric(df["standard_value"], errors="coerce"))
    # Filter to biologically relevant range (IC50 1 mM to 1 pM => pIC50 3–12)
    before_pIC50 = len(df)
    df = df[(df["pIC50_computed"] >= 3) & (df["pIC50_computed"] <= 12)]
    audit["after_pIC50_range_filter"] = len(df)
    audit["before_dedup"] = before_pIC50
    audit["after_value_filter"] = len(df)

    # Exclude rows with data_validity_comment (non‑empty) or potential_duplicate==1
    # data_validity_comment may be NaN; treat NaN as empty
    df = df[(df["data_validity_comment"].isna()) | (df["data_validity_comment"] == "")]
    audit["after_validity_filter"] = len(df)
    df = df[df["potential_duplicate"].fillna(0) != 1]
    audit["after_duplicate_filter"] = len(df)

    # Standardise SMILES – create a new column canonical_smiles_std
    std_smiles = []
    for smi in df["canonical_smiles"].fillna(""):
        std = standardise_smiles(smi)
        if std is None:
            std_smiles.append(None)
        else:
            std_smiles.append(std)
            audit["smiles_success"] += 1
    df = df.copy()
    df["canonical_smiles_std"] = std_smiles
    # Drop rows where SMILES standardisation failed
    df = df[~df["canonical_smiles_std"].isna()]

    # Compute pIC50 from standard_value (nM) -> 9 - log10(value)
    df["pIC50_computed"] = 9 - np.log10(pd.to_numeric(df["standard_value"], errors="coerce"))

    # Optional: compare to existing pchembl_value – we just keep both for reference
    # Aggregate per molecule
    agg = (
        df.groupby("molecule_chembl_id")
        .apply(
            lambda g: pd.Series(
                {
                    "canonical_smiles": g["canonical_smiles_std"].iloc[0],  # use first std SMILES
                    "median_pIC50": np.median(g["pIC50_computed"].values),
                    "n_records": len(g),
                    "pIC50_min": np.min(g["pIC50_computed"].values),
                    "pIC50_max": np.max(g["pIC50_computed"].values),
                }
            )
        )
        .reset_index()
    )
    agg["spread"] = agg["pIC50_max"] - agg["pIC50_min"]
    agg["high_variance_flag"] = agg["spread"] > 1.0
    # Keep only columns needed for downstream
    final = agg[[
        "molecule_chembl_id",
        "canonical_smiles",
        "median_pIC50",
        "n_records",
        "spread",
        "high_variance_flag",
    ]]
    audit["final_rows"] = len(final)
    # Deduplicate by standardized SMILES (merge identical SMILES)
    before_dedup = len(df)
    audit["before_dedup"] = before_dedup
    dedup = (
        df.groupby("canonical_smiles_std")
        .apply(
            lambda g: pd.Series(
                {
                    "canonical_smiles": g["canonical_smiles_std"].iloc[0],
                    "median_pIC50": np.median(g["pIC50_computed"].values),
                    "n_records": int(len(g)),
                    "pIC50_min": np.min(g["pIC50_computed"].values),
                    "pIC50_max": np.max(g["pIC50_computed"].values),
                }
            )
        )
        .reset_index(drop=True)
    )
    dedup["spread"] = dedup["pIC50_max"] - dedup["pIC50_min"]
    dedup["high_variance_flag"] = dedup["spread"] > 1.0
    audit["after_dedup"] = len(dedup)
    # Use dedup as final dataframe for output (no molecule_chembl_id column needed)
    final = dedup[["canonical_smiles", "median_pIC50", "n_records", "spread", "high_variance_flag"]]
    audit["final_rows"] = len(final)

    # Delete any previous curated outputs to ensure fresh regeneration
for f in PROCESSED.glob("*_IC50_curated.csv"):
    try:
        f.unlink()
    except Exception:
        pass

    out_path = PROCESSED / f"{name}_IC50_curated.csv"
    final.to_csv(out_path, index=False)
    audit["processed_csv"] = str(out_path.relative_to(ROOT))
    return audit

def main():
    overall_audit = {"created_utc": pd.Timestamp.utcnow().isoformat(), "targets": {}}
    for target_dir in RAW.iterdir():
        if not target_dir.is_dir():
            continue
        name = target_dir.name
        csv_path = target_dir / f"{name}_IC50_raw_flat.csv"
        if not csv_path.exists():
            logging.warning(f"CSV not found for {name}: {csv_path}")
            continue
        logging.info(f"Curating {name} …")
        audit = curate_target(name, csv_path)
        overall_audit["targets"][name] = audit

    # Write audit JSON
    json_path = AUDIT_DIR / "curation_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_audit, f, indent=2)

    # Human‑readable MD summary
    lines = ["# Curation Audit", f"- Created (UTC): {overall_audit['created_utc']}", ""]
    for name, a in overall_audit["targets"].items():
        lines.append(f"## {name}")
        lines.append(f"- Source CSV: {a['source_csv']}")
        lines.append(f"- Total rows read: {a['total_rows']}")
        lines.append(f"- After relation '=' filter: {a['after_relation_filter']}")
        lines.append(f"- After units 'nM' filter: {a['after_units_filter']}")
        lines.append(f"- After positive value filter: {a['after_value_filter']}")
        lines.append(f"- After validity‑comment filter: {a['after_validity_filter']}")
        lines.append(f"- After duplicate filter: {a['after_duplicate_filter']}")
        lines.append(f"- SMILES successfully standardised: {a['smiles_success']}")
        lines.append(f"- Final curated records (molecules): {a['final_rows']}")
        lines.append(f"- Processed CSV: {a['processed_csv']}")
        lines.append("")
    md_path = AUDIT_DIR / "curation_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Curation complete. Audit written to {md_path}")

if __name__ == "__main__":
    main()
