#!/usr/bin/env python3
"""
PhytoScreen-AI | Phase 5: score the phytochemical library with the Phase 4 *_full models.

For every library compound and each target (AChE, BChE, BACE1) it reports:
  * pred_rf, pred_xgb, pred_ens  (ensemble mean)       -> ML PREDICTION
  * unc_tree_std  (spread across RF trees) and unc_disagree (|RF - XGB|)   -> uncertainty proxies
  * max_tanimoto_train and domain tier (in_domain >=0.5, borderline 0.3-0.5, out_of_domain <0.3)
  * exp_pIC50 / exp_n_records when the compound (InChIKey first block) is in the ChEMBL training data
    -> EXPERIMENTAL EVIDENCE (kept separate from predictions)

Outputs:
  data/processed/screening_results.csv
  reports/screening/<T>_top_candidates.csv        (novel compounds, not in training, ranked by prediction)
  reports/screening/<T>_known_experimental.csv    (library compounds with ChEMBL IC50 evidence)
  reports/screening/multitarget_candidates.csv    (in the top-N of >=2 targets)
  reports/screening/screening_summary.md

Docking has NOT been done; these are ML predictions only.

Run:  python src/05_screen.py            (defaults: top 200 per target, min similarity 0.3)
"""
import argparse
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "screening"
TARGETS = ["AChE", "BChE", "BACE1"]
FP_BITS = 2048
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def featurize(smiles):  # IDENTICAL to 04_train.py
    X, fps = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = _gen.GetFingerprint(m)
        arr = np.zeros((FP_BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        desc = [Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
                rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                m.GetNumHeavyAtoms()]
        X.append(np.concatenate([arr.astype(np.float32), np.array(desc, dtype=np.float32)]))
        fps.append(fp)
    return np.vstack(X), fps


def tier(s):
    return "in_domain" if s >= 0.5 else ("borderline" if s >= 0.3 else "out_of_domain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--min-sim", type=float, default=0.3, help="min max-Tanimoto for a docking candidate")
    ap.add_argument("--mw-min", type=float, default=150)
    ap.add_argument("--mw-max", type=float, default=700)
    a = ap.parse_args()

    lib = pd.read_csv(PROCESSED / "phytochemical_library.csv")
    print(f"Library: {len(lib)} compounds. Featurising ...")
    X, lib_fps = featurize(lib["canonical_smiles"].tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    res = lib[["canonical_smiles", "inchikey14", "source_ids", "names", "MolWt", "LogP", "TPSA", "HBD", "HBA"]].copy()
    summary = ["# Screening summary (ML predictions, not docking, not experiment)",
               f"- Created (UTC): {dt.datetime.now(dt.timezone.utc).isoformat()}",
               f"- Library size: {len(lib)}",
               f"- Docking-candidate rules: not in ChEMBL training data, max Tanimoto >= {a.min_sim}, "
               f"MolWt {a.mw_min:g}-{a.mw_max:g}, top {a.top_n} by ensemble prediction", ""]
    ranked = {}

    for t in TARGETS:
        print(f"Scoring {t} ...")
        cur = pd.read_csv(PROCESSED / f"{t}_IC50_curated.csv")
        rf = joblib.load(MODELS / f"{t}_rf_full.joblib")
        xg = joblib.load(MODELS / f"{t}_xgb_full.joblib")
        p_rf, p_xg = rf.predict(X), xg.predict(X)
        tree_std = np.std(np.stack([e.predict(X) for e in rf.estimators_]), axis=0)
        train_fps = [_gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in cur["canonical_smiles"]]
        maxsim = np.array([max(DataStructs.BulkTanimotoSimilarity(f, train_fps)) for f in lib_fps])

        res[f"{t}_pred_rf"], res[f"{t}_pred_xgb"] = p_rf, p_xg
        res[f"{t}_pred_ens"] = (p_rf + p_xg) / 2
        res[f"{t}_unc_tree_std"], res[f"{t}_unc_disagree"] = tree_std, np.abs(p_rf - p_xg)
        res[f"{t}_max_tanimoto"] = maxsim
        res[f"{t}_domain"] = [tier(s) for s in maxsim]

        exp = cur.groupby("inchikey14").agg(exp_pIC50=("median_pIC50", "median"),
                                            exp_n_records=("n_records", "sum")).reset_index()
        res = res.merge(exp.rename(columns={"exp_pIC50": f"{t}_exp_pIC50",
                                            "exp_n_records": f"{t}_exp_n_records"}), on="inchikey14", how="left")
        known = res[f"{t}_exp_pIC50"].notna()

        base = ["source_ids", "names", "canonical_smiles", "MolWt", "LogP",
                f"{t}_pred_ens", f"{t}_pred_rf", f"{t}_pred_xgb", f"{t}_unc_tree_std",
                f"{t}_unc_disagree", f"{t}_max_tanimoto", f"{t}_domain"]
        res[known].sort_values(f"{t}_exp_pIC50", ascending=False)[
            base[:5] + [f"{t}_exp_pIC50", f"{t}_exp_n_records", f"{t}_pred_ens"]
        ].to_csv(OUT / f"{t}_known_experimental.csv", index=False)

        elig = (~known) & (res[f"{t}_max_tanimoto"] >= a.min_sim) & res["MolWt"].between(a.mw_min, a.mw_max)
        top = res[elig].sort_values(f"{t}_pred_ens", ascending=False).head(a.top_n)
        top[base].to_csv(OUT / f"{t}_top_candidates.csv", index=False)
        ranked[t] = set(top["canonical_smiles"])

        dom = res[f"{t}_domain"].value_counts().to_dict()
        summary += [f"## {t}",
                    f"- Library compounds with ChEMBL IC50 evidence (experimental): {int(known.sum())}",
                    f"- Domain tiers (max Tanimoto to training): {dom}",
                    f"- Predicted pIC50 (ensemble) min/median/max: {res[f'{t}_pred_ens'].min():.2f} / "
                    f"{res[f'{t}_pred_ens'].median():.2f} / {res[f'{t}_pred_ens'].max():.2f}",
                    f"- Eligible novel compounds: {int(elig.sum())}; written top {len(top)}",
                    ""]

    res.to_csv(PROCESSED / "screening_results.csv", index=False)
    counts = pd.Series([s for t in TARGETS for s in ranked[t]]).value_counts()
    multi = counts[counts >= 2].index
    mt = res[res["canonical_smiles"].isin(multi)].copy()
    mt["n_targets_in_top"] = mt["canonical_smiles"].map(counts)
    mt.sort_values("n_targets_in_top", ascending=False).to_csv(OUT / "multitarget_candidates.csv", index=False)
    summary += [f"## Multi-target", f"- Compounds in the top-{a.top_n} of at least 2 targets: {len(multi)}", "",
                "## Reading guide",
                "- `*_pred_*` are ML predictions; `*_exp_pIC50` are experimental ChEMBL medians. Never mix them.",
                "- Out-of-domain predictions are unreliable. Phytochemicals are often far from the ChEMBL chemical space.",
                "- Next step is docking of selected candidates (Phase 6) and expert review."]
    (OUT / "screening_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()