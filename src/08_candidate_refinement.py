"""
PhytoScreen-AI | Phase 8: candidate refinement and docking-set selection.
 
Inputs (all produced by earlier phases; nothing is modified):
  data/processed/screening_results.csv           (Phase 5, one row per compound, wide format)
  reports/screening/multitarget_candidates.csv   (Phase 5, optional: used only for a consistency check)
 
What it does, per target (AChE, BChE, BACE1):
  1. Rebuilds the Phase 5 candidate ranking: novel compounds (no ChEMBL IC50 evidence), max Tanimoto
     to training >= 0.3, MolWt 150-700, ranked by ensemble predicted pIC50, top N.
  2. Flags structural alerts (Michael acceptor, epoxide, aldehyde, O-glycoside) and PAINS (info only).
  3. Assigns selection groups:  A = in-domain (Tanimoto >= 0.5), no alerts
                                B = borderline (0.3-0.5), no alerts
                                D = has a reactive/glycoside alert -> manual structural review
  4. Builds the docking set: up to 25 novel compounds (A first, then B; scaffold cap) plus up to 5
     experimental positive controls spanning the potency range.
  5. Writes aglycones of glycosides to a SEPARATE file. They are new structures: predictions of the
     parent do NOT apply, so they must be re-scored with the models before use.
 
Evidence types are kept separate: experimental (ChEMBL) / ML-predicted / (later) docking-derived.
 
Run:  python src/08_candidate_refinement.py
"""
import argparse
import datetime as dt
import json
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
 
RDLogger.DisableLog("rdApp.*")
 
ROOT = Path(__file__).resolve().parents[1]
SCREEN_CSV = ROOT / "data" / "processed" / "screening_results.csv"
MULTI_CSV = ROOT / "reports" / "screening" / "multitarget_candidates.csv"
OUT = ROOT / "reports" / "candidate_refinement"
FIG = ROOT / "reports" / "figures" / "phase8"
TARGETS = ["AChE", "BChE", "BACE1"]
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
 
 
# ----------------------------------------------------------------------------- substructure patterns
def _smarts(s):
    p = Chem.MolFromSmarts(s)
    assert p is not None, f"bad SMARTS: {s}"
    return p
 
 
# Alpha,beta-unsaturated carbonyl with a NON-aromatic C=C (aromatic flavones/coumarins are not matched)
MICHAEL = _smarts("[CX3]=[CX3]-[CX3]=[OX1]")
EPOXIDE = _smarts("[CX4]1[OX2][CX4]1")
ALDEHYDE = _smarts("[CX3H1](=[OX1])[#6]")
# Pyranose ring (hexose / deoxyhexose, stereo-agnostic). Match order: 0 ring O, 1 anomeric C, 2 anomeric exo O
SUGAR = _smarts("[OX2;r6]1[CX4;r6]([OX2])[CX4;r6]([OX2H1])[CX4;r6]([OX2H1])[CX4;r6]([OX2H1])[CX4;r6]1")
 
_params = FilterCatalogParams()
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS = FilterCatalog(_params)
 
 
def find_glycosidic_bond(mol):
    """Return (anomeric_C_idx, exocyclic_O_idx) of the first O-glycosidic linkage, or None."""
    for m in mol.GetSubstructMatches(SUGAR):
        ano, exo = m[1], m[2]
        others = [n.GetIdx() for n in mol.GetAtomWithIdx(exo).GetNeighbors() if n.GetIdx() != ano]
        if others and others[0] not in set(m):  # O bonded to something outside the sugar = glycosidic
            return ano, exo
    return None
 
 
def flags_for(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"glycoside": False, "michael": False, "epoxide": False, "aldehyde": False,
                "pains": False, "pains_name": ""}
    pains = PAINS.GetFirstMatch(mol)
    return {"glycoside": find_glycosidic_bond(mol) is not None,
            "michael": mol.HasSubstructMatch(MICHAEL),
            "epoxide": mol.HasSubstructMatch(EPOXIDE),
            "aldehyde": mol.HasSubstructMatch(ALDEHYDE),
            "pains": pains is not None,
            "pains_name": pains.GetDescription() if pains is not None else ""}
 
 
def aglycone_of(smiles, max_sugars=5):
    """Strip O-linked pyranose sugars by cutting anomeric C - O bonds (the O stays on the aglycone as OH)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    removed = 0
    for _ in range(max_sugars):
        hit = find_glycosidic_bond(mol)
        if hit is None:
            break
        ano, exo = hit
        bond = mol.GetBondBetweenAtoms(ano, exo).GetIdx()
        frag = Chem.FragmentOnBonds(mol, [bond], addDummies=False)
        maps = []
        frags = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=maps)
        agl = next((f for f, mp in zip(frags, maps) if exo in mp), None)
        if agl is None:
            return None
        try:
            Chem.SanitizeMol(agl)
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(agl))
        except Exception:
            return None
        if mol is None:
            return None
        removed += 1
    if removed == 0 or mol.GetNumHeavyAtoms() < 6:
        return None
    return Chem.MolToSmiles(mol)
 
 
def scaffold_of(smiles):
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
    except Exception:
        sc = ""
    return sc if sc else "ACYCLIC"
 
 
def mean_pairwise_tanimoto(smiles_list):
    fps = [_gen.GetFingerprint(m) for m in (Chem.MolFromSmiles(s) for s in smiles_list) if m is not None]
    if len(fps) < 2:
        return None
    sims = [s for i in range(1, len(fps)) for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]
    return float(np.mean(sims))
 
 
def md_table(series, name):
    lines = [f"| {name} | Count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in series.items()]
    return "\n".join(lines)
 
 
# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--min-sim", type=float, default=0.3)
    ap.add_argument("--mw-min", type=float, default=150)
    ap.add_argument("--mw-max", type=float, default=700)
    ap.add_argument("--n-novel", type=int, default=25, help="novel compounds per target in the docking set")
    ap.add_argument("--n-controls", type=int, default=5, help="experimental controls per target")
    ap.add_argument("--max-per-scaffold", type=int, default=2)
    a = ap.parse_args()
 
    if not SCREEN_CSV.exists():
        raise SystemExit(f"Missing {SCREEN_CSV}. Run Phase 5 (05_screen.py) first.")
    scr = pd.read_csv(SCREEN_CSV)
    need = ["canonical_smiles", "inchikey14", "source_ids", "names", "MolWt"]
    for t in TARGETS:
        need += [f"{t}_pred_ens", f"{t}_unc_tree_std", f"{t}_unc_disagree", f"{t}_max_tanimoto",
                 f"{t}_domain", f"{t}_exp_pIC50"]
    missing = [c for c in need if c not in scr.columns]
    if missing:
        raise SystemExit(f"screening_results.csv is missing columns: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
 
    # ---- 1. long-format ranking per target (same rules as Phase 5) -------------------------------
    rank_frames, exp_frames = [], []
    for t in TARGETS:
        d = scr.rename(columns={f"{t}_pred_ens": "pred_ens", f"{t}_unc_tree_std": "unc_tree_std",
                                f"{t}_unc_disagree": "unc_disagree", f"{t}_max_tanimoto": "max_tanimoto",
                                f"{t}_domain": "domain_tier", f"{t}_exp_pIC50": "exp_pIC50",
                                f"{t}_exp_n_records": "exp_n_records"}).copy()
        if "exp_n_records" not in d.columns:
            d["exp_n_records"] = np.nan
        d = d[["canonical_smiles", "inchikey14", "source_ids", "names", "MolWt", "pred_ens", "unc_tree_std",
               "unc_disagree", "max_tanimoto", "domain_tier", "exp_pIC50", "exp_n_records"]]
        d.insert(0, "Target", t)
        known = d["exp_pIC50"].notna()
        exp_frames.append(d[known].copy())
        elig = (~known) & (d["max_tanimoto"] >= a.min_sim) & d["MolWt"].between(a.mw_min, a.mw_max)
        top = d[elig].sort_values("pred_ens", ascending=False).head(a.top_n).copy()
        top["rank"] = np.arange(1, len(top) + 1)
        rank_frames.append(top)
    ranked = pd.concat(rank_frames, ignore_index=True)
    experimental = pd.concat(exp_frames, ignore_index=True)
 
    ranked["n_targets_in_top"] = ranked.groupby("canonical_smiles")["Target"].transform("count")
 
    # consistency check against Phase 5
    check = "Phase 5 multitarget file not found (check skipped)."
    if MULTI_CSV.exists():
        p5 = set(pd.read_csv(MULTI_CSV)["canonical_smiles"])
        mine = set(ranked.loc[ranked["n_targets_in_top"] >= 2, "canonical_smiles"])
        check = (f"Multi-target compounds: Phase 5 = {len(p5)}, rebuilt here = {len(mine)}, "
                 f"overlap = {len(p5 & mine)}")
    print(check)
 
    # ---- 2. flags and groups (only for compounds that matter) -------------------------------------
    uniq = ranked["canonical_smiles"].drop_duplicates()
    flag_df = pd.DataFrame([flags_for(s) for s in uniq], index=uniq.values)
    ranked = ranked.join(flag_df, on="canonical_smiles")
    ranked["scaffold_id"] = ranked["canonical_smiles"].map(scaffold_of)
    ranked["any_alert"] = ranked[["glycoside", "michael", "epoxide", "aldehyde"]].any(axis=1)
 
    def group(r):
        if r["domain_tier"] == "out_of_domain":
            return "Excluded"
        if r["any_alert"]:
            return "D"
        return "A" if r["domain_tier"] == "in_domain" else "B"
 
    ranked["selection_group"] = ranked.apply(group, axis=1)
    ranked["evidence"] = "ML-predicted (" + ranked["domain_tier"] + ")"
 
    # ---- 3. docking set ----------------------------------------------------------------------------
    sets, ctrl_rows = [], []
    for t in TARGETS:
        r = ranked[(ranked["Target"] == t) & ranked["selection_group"].isin(["A", "B"])]
        r = pd.concat([r[r["selection_group"] == "A"].sort_values("rank"),
                       r[r["selection_group"] == "B"].sort_values("rank")])
        picked, per_scaf = [], {}
        for _, row in r.iterrows():
            if per_scaf.get(row["scaffold_id"], 0) >= a.max_per_scaffold:
                continue
            picked.append(row)
            per_scaf[row["scaffold_id"]] = per_scaf.get(row["scaffold_id"], 0) + 1
            if len(picked) >= a.n_novel:
                break
        nov = pd.DataFrame(picked)
        nov["role"] = "novel_candidate"
        sets.append(nov)
 
        e = experimental[experimental["Target"] == t].sort_values("exp_pIC50", ascending=False).reset_index(drop=True)
        if len(e):
            k = min(a.n_controls, len(e))
            idx = sorted(set(np.linspace(0, len(e) - 1, k).round().astype(int)))  # span potency range
            c = e.iloc[idx].copy()
            c["role"] = "experimental_control"
            c["evidence"] = "Experimental (ChEMBL IC50)"
            c["selection_group"] = "control"
            ctrl_rows.append(c)
    controls = pd.concat(ctrl_rows, ignore_index=True) if ctrl_rows else pd.DataFrame()
    docking = pd.concat(sets + ([controls] if len(controls) else []), ignore_index=True)
 
    # ---- 4. glycoside aglycones (separate file, no predictions) ------------------------------------
    gly = ranked[ranked["glycoside"]].drop_duplicates("canonical_smiles")
    agl_rows = []
    for _, r in gly.iterrows():
        ag = aglycone_of(r["canonical_smiles"])
        if ag:
            agl_rows.append({"parent_source_ids": r["source_ids"], "parent_names": r["names"],
                             "parent_smiles": r["canonical_smiles"], "aglycone_smiles": ag,
                             "note": "NEW structure: re-score with Phase 4/5 models before any use"})
    aglycones = pd.DataFrame(agl_rows)
 
    # ---- 5. diversity ------------------------------------------------------------------------------
    diversity = {}
    for t in TARGETS:
        s = docking[(docking["Target"] == t) & (docking["role"] == "novel_candidate")]
        diversity[t] = {"n_novel_selected": int(len(s)),
                        "n_unique_scaffolds": int(s["scaffold_id"].nunique()) if len(s) else 0,
                        "mean_pairwise_tanimoto": mean_pairwise_tanimoto(s["canonical_smiles"].tolist())}
 
    # multi-target shortlist: A/B in at least two targets
    ab = ranked[ranked["selection_group"].isin(["A", "B"])]
    cnt = ab.groupby("canonical_smiles")["Target"].nunique()
    multi = ab[ab["canonical_smiles"].isin(cnt[cnt >= 2].index)].sort_values(["canonical_smiles", "Target"])
 
    # ---- 6. write outputs --------------------------------------------------------------------------
    ranked.to_csv(OUT / "target_candidate_rankings.csv", index=False)
    docking.to_csv(OUT / "docking_candidate_set.csv", index=False)
    multi.to_csv(OUT / "multitarget_shortlist.csv", index=False)
    experimental.sort_values(["Target", "exp_pIC50"], ascending=[True, False]).to_csv(
        OUT / "experimental_controls.csv", index=False)
    ranked[ranked["selection_group"] == "D"].to_csv(OUT / "structural_review.csv", index=False)
    aglycones.to_csv(OUT / "aglycones_to_rescore.csv", index=False)
 
    group_counts = ranked.groupby(["Target", "selection_group"]).size().unstack(fill_value=0)
    summary = ["# Phase 8 Candidate Refinement Summary", "",
               f"- Created (UTC): {dt.datetime.now(dt.timezone.utc).isoformat()}",
               f"- Rules: top {a.top_n} novel per target, max Tanimoto >= {a.min_sim}, MolWt {a.mw_min:g}-{a.mw_max:g}",
               f"- Consistency check: {check}", "",
               "## Selection groups per target (within the top-N lists)",
               "A = in-domain, no alerts; B = borderline, no alerts; D = alert, needs structural review", "",
               group_counts.to_markdown() if hasattr(group_counts, "to_markdown") and _has_tabulate()
               else group_counts.to_string(), "",
               "## Docking set", "",
               md_table(docking.groupby(["Target", "role"]).size().rename(lambda x: x), "Target / role")
               if False else docking.groupby(["Target", "role"]).size().to_string(), "",
               "## Diversity of selected novel compounds", "", "```", json.dumps(diversity, indent=2), "```", "",
               f"- Experimental positive controls available (library compounds with ChEMBL IC50): "
               f"{experimental.groupby('Target').size().to_dict()}",
               f"- Glycosides flagged: {int(gly.shape[0])}; aglycones generated (to re-score): {len(aglycones)}",
               f"- PAINS matches among top-N candidates (informational, not used for exclusion): "
               f"{int(ranked.drop_duplicates('canonical_smiles')['pains'].sum())}", "",
               "## Caveats",
               "- Predictions are ML estimates; the models are moderate (scaffold-split R2 0.45-0.56).",
               "- Controls are in the training data, so their predictions are in-sample and not a validation.",
               "- No docking results are used here."]
    (OUT / "phase8_summary.md").write_text("\n".join(summary), encoding="utf-8")
    (OUT / "phase8_results.json").write_text(json.dumps(
        {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "parameters": vars(a),
         "consistency_check": check, "diversity": diversity,
         "group_counts": {t: group_counts.loc[t].to_dict() for t in group_counts.index},
         "docking_set_size": int(len(docking))}, indent=2), encoding="utf-8")
 
    # ---- 7. figures --------------------------------------------------------------------------------
    colors = {"A": "tab:green", "B": "tab:orange", "D": "tab:red", "Excluded": "tab:gray"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, t in zip(axes, TARGETS):
        for g, c in colors.items():
            s = ranked[(ranked["Target"] == t) & (ranked["selection_group"] == g)]
            ax.scatter(s["max_tanimoto"], s["pred_ens"], s=14, alpha=0.6, c=c, label=f"{g} (n={len(s)})")
        ax.set_title(t); ax.set_xlabel("Max Tanimoto to training set"); ax.legend(fontsize=8)
    axes[0].set_ylabel("Predicted pIC50 (ensemble)")
    fig.tight_layout(); fig.savefig(FIG / "domain_vs_prediction.png", dpi=150); plt.close(fig)
 
    fig, ax = plt.subplots(figsize=(6, 4))
    group_counts.plot.bar(ax=ax, rot=0, color=[colors.get(c, "gray") for c in group_counts.columns])
    ax.set_ylabel("Compounds"); ax.set_title("Selection groups per target")
    fig.tight_layout(); fig.savefig(FIG / "selection_groups.png", dpi=150); plt.close(fig)
 
    fig, ax = plt.subplots(figsize=(6, 4))
    ranked["scaffold_id"].value_counts().hist(bins=20, ax=ax)
    ax.set_xlabel("Compounds per scaffold"); ax.set_ylabel("Number of scaffolds"); ax.set_title("Scaffold size distribution")
    fig.tight_layout(); fig.savefig(FIG / "scaffold_histogram.png", dpi=150); plt.close(fig)
 
    fig, ax = plt.subplots(figsize=(5, 4))
    ranked.drop_duplicates("canonical_smiles")["n_targets_in_top"].value_counts().sort_index().plot.bar(ax=ax, rot=0)
    ax.set_xlabel("Number of targets (top-N lists)"); ax.set_ylabel("Compounds"); ax.set_title("Multi-target overlap")
    fig.tight_layout(); fig.savefig(FIG / "multitarget_overlap.png", dpi=150); plt.close(fig)
 
    print("\n".join(summary))
    print(f"\nOutputs: {OUT}\nFigures: {FIG}")
 
 
def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
 
 
if __name__ == "__main__":
    main()
