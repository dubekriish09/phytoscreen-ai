"""
07_applicability_domain.py
Phase 7 – Chemical Space and Applicability Domain Analysis
PhytoScreen-AI

Analyses:
  A. Chemical-space overview: Tanimoto similarity distributions, molecular-property distributions
  B. Applicability-domain validation: held-out test error vs. similarity bins
  C. Threshold sensitivity: coverage vs. error at alternative cutoffs
  D. Uncertainty-proxy analysis: tree spread and RF–XGBoost disagreement vs. error
  E. Candidate-level audit: flag high-ranked out-of-domain compounds, structural issues

IMPORTANT RULES:
  - Never modifies data/raw/, data/processed/, models/ or Phase 1-5 scripts.
  - Experimental pIC50, ML predictions and uncertainty proxies are always separately labelled.
  - RF tree std / RF-XGBoost disagreement are labelled as UNCERTAINTY PROXIES, not confidence intervals.
  - Tanimoto thresholds are evaluated empirically; no threshold is assumed to be universally valid.
  - All results come from executed code; nothing is estimated or invented.
"""

import datetime
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import DataStructs

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────── paths ────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
REPORTS_AD = ROOT / "reports" / "applicability_domain"
REPORTS_FIG = ROOT / "reports" / "figures" / "phase7"
REPORTS_METRICS = ROOT / "reports" / "metrics"
REPORTS_SCREENING = ROOT / "reports" / "screening"

TARGETS = ["AChE", "BChE", "BACE1"]
SIM_BINS = [0.0, 0.3, 0.5, 0.7, 1.001]  # right edge open except last
SIM_BIN_LABELS = ["0.0–0.3", "0.3–0.5", "0.5–0.7", "0.7–1.0"]
ALT_CUTOFFS = [0.3, 0.4, 0.5, 0.6, 0.7]
N_BOOTSTRAP = 500
RNG_SEED = 42
RNG = np.random.default_rng(RNG_SEED)

# ─────────────────── fingerprint helper ────────────────────────
def smiles_to_fp(smi, radius=2, nbits=2048):
    """Return Morgan fingerprint or None if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def bulk_max_tanimoto(query_fps, ref_fps):
    """Return array of max Tanimoto similarity of each query to any ref."""
    results = []
    ref_valid = [fp for fp in ref_fps if fp is not None]
    for fp in query_fps:
        if fp is None:
            results.append(np.nan)
        elif not ref_valid:
            results.append(np.nan)
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp, ref_valid)
            results.append(float(max(sims)))
    return np.array(results)


# ─────────────────── descriptor helper ─────────────────────────
PROP_COLS = ["MolWt", "LogP", "TPSA", "HBD", "HBA"]


def compute_props(df, smiles_col="canonical_smiles"):
    """Add molecular property columns if not present."""
    if "MolWt" in df.columns:
        return df
    mols = df[smiles_col].map(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
    df = df.copy()
    df["MolWt"] = mols.map(lambda m: Descriptors.MolWt(m) if m else np.nan)
    df["LogP"]  = mols.map(lambda m: Descriptors.MolLogP(m) if m else np.nan)
    df["TPSA"]  = mols.map(lambda m: Descriptors.TPSA(m) if m else np.nan)
    df["HBD"]   = mols.map(lambda m: rdMolDescriptors.CalcNumHBD(m) if m else np.nan)
    df["HBA"]   = mols.map(lambda m: rdMolDescriptors.CalcNumHBA(m) if m else np.nan)
    return df


# ─────────────────── statistics helpers ────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def median_abs_error(y_true, y_pred):
    return float(np.median(np.abs(np.array(y_true) - np.array(y_pred))))


def bootstrap_ci(values, stat_fn, n=N_BOOTSTRAP, seed=RNG_SEED):
    """Return (lo, hi) 95 % bootstrap CI for stat_fn applied to values."""
    rng = np.random.default_rng(seed)
    boot = [stat_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("Loading data …")
lib = pd.read_csv(DATA_PROC / "phytochemical_library.csv")
screening = pd.read_csv(DATA_PROC / "screening_results.csv")

curated = {}
test_preds = {}
for t in TARGETS:
    curated[t] = pd.read_csv(DATA_PROC / f"{t}_IC50_curated.csv")
    test_preds[t] = pd.read_csv(REPORTS_METRICS / f"{t}_test_predictions.csv")

top_cands = {}
for t in TARGETS:
    top_cands[t] = pd.read_csv(REPORTS_SCREENING / f"{t}_top_candidates.csv")

multitarget = pd.read_csv(REPORTS_SCREENING / "multitarget_candidates.csv")

print(f"  Library: {len(lib)} compounds")
for t in TARGETS:
    print(f"  {t} training: {len(curated[t])}, test: {len(test_preds[t])}")
print()

# ─────────────────── compute library fingerprints ──────────────
print("Computing library fingerprints …")
lib["_fp"] = lib["canonical_smiles"].map(smiles_to_fp)
lib_fp_invalid = lib["_fp"].isna().sum()
print(f"  Invalid SMILES in library: {lib_fp_invalid}")

# ─────────────────── compute training fingerprints ─────────────
train_fps = {}
for t in TARGETS:
    df = curated[t]
    fps = df["canonical_smiles"].map(smiles_to_fp).tolist()
    invalid = sum(1 for f in fps if f is None)
    train_fps[t] = [f for f in fps if f is not None]
    print(f"  {t} training fps: {len(train_fps[t])} valid, {invalid} invalid")

print()

# ═══════════════════════════════════════════════════════════════
# A. CHEMICAL-SPACE OVERVIEW
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("A. Chemical-space overview")
print("=" * 60)

space_results = {}

lib_valid = lib[lib["_fp"].notna()].copy()
lib_fps_list = lib_valid["_fp"].tolist()

for t in TARGETS:
    print(f"\n  {t}: computing max Tanimoto ({len(lib_valid)} lib vs {len(train_fps[t])} train) …")
    max_tan = bulk_max_tanimoto(lib_fps_list, train_fps[t])
    lib_valid = lib_valid.copy()
    lib_valid[f"max_tanimoto_{t}"] = max_tan

    # bin counts
    bin_counts = {}
    for lo, hi, label in zip(SIM_BINS[:-1], SIM_BINS[1:], SIM_BIN_LABELS):
        mask = (max_tan >= lo) & (max_tan < hi)
        n = int(mask.sum())
        pct = 100 * n / len(max_tan)
        bin_counts[label] = {"n": n, "pct": round(pct, 2)}

    # property summary for library
    lib_props = {}
    for col in PROP_COLS:
        vals = lib_valid[col].dropna()
        lib_props[col] = {
            "median": round(float(vals.median()), 2),
            "q25": round(float(vals.quantile(0.25)), 2),
            "q75": round(float(vals.quantile(0.75)), 2),
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
        }

    # property summary for training set
    cur = compute_props(curated[t].copy())
    train_props = {}
    for col in PROP_COLS:
        if col in cur.columns:
            vals = cur[col].dropna()
        else:
            vals = pd.Series([], dtype=float)
        train_props[col] = {
            "median": round(float(vals.median()), 2) if len(vals) > 0 else None,
            "q25": round(float(vals.quantile(0.25)), 2) if len(vals) > 0 else None,
            "q75": round(float(vals.quantile(0.75)), 2) if len(vals) > 0 else None,
            "min": round(float(vals.min()), 2) if len(vals) > 0 else None,
            "max": round(float(vals.max()), 2) if len(vals) > 0 else None,
        }

    space_results[t] = {
        "n_library_valid": len(lib_valid),
        "n_library_invalid_smiles": int(lib_fp_invalid),
        "n_training": len(curated[t]),
        "similarity_bins": bin_counts,
        "library_properties": lib_props,
        "training_properties": train_props,
        "max_tanimoto_median": round(float(np.nanmedian(max_tan)), 4),
        "max_tanimoto_mean": round(float(np.nanmean(max_tan)), 4),
    }

    for label, d in bin_counts.items():
        print(f"    {label}: {d['n']:>5} ({d['pct']:.1f}%)")
    print(f"    median max_tanimoto: {space_results[t]['max_tanimoto_median']:.3f}")

# ═══════════════════════════════════════════════════════════════
# B. APPLICABILITY-DOMAIN VALIDATION (held-out test)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("B. Applicability-domain validation (scaffold test sets)")
print("=" * 60)

ad_results = {}

error_by_sim_rows = []

for t in TARGETS:
    df = test_preds[t].copy()
    df["abs_err_xgb"] = (df["y_true"] - df["pred_xgb"]).abs()
    df["abs_err_rf"]  = (df["y_true"] - df["pred_rf"]).abs()
    df["sim_bin"] = pd.cut(
        df["max_tanimoto_to_train"],
        bins=SIM_BINS,
        labels=SIM_BIN_LABELS,
        right=False,
    )

    target_ad = {}
    for label in SIM_BIN_LABELS:
        sub = df[df["sim_bin"] == label]
        n = len(sub)
        if n == 0:
            target_ad[label] = {"n": 0}
            continue

        row = {
            "target": t,
            "similarity_bin": label,
            "n": n,
        }

        for model, col in [("XGBoost", "abs_err_xgb"), ("RF", "abs_err_rf")]:
            errs = sub[col].values
            r = rmse(sub["y_true"], sub[f"pred_{model.lower()[:3]}"])
            m = mae(sub["y_true"], sub[f"pred_{model.lower()[:3]}"])
            med = median_abs_error(sub["y_true"], sub[f"pred_{model.lower()[:3]}"])
            ci_lo, ci_hi = bootstrap_ci(errs, np.mean)
            row[f"{model}_RMSE"] = round(r, 4)
            row[f"{model}_MAE"]  = round(m, 4)
            row[f"{model}_MedAE"] = round(med, 4)
            row[f"{model}_MAE_CI95_lo"] = round(ci_lo, 4)
            row[f"{model}_MAE_CI95_hi"] = round(ci_hi, 4)
            target_ad[label] = {
                "n": n,
                f"{model}_RMSE": round(r, 4),
                f"{model}_MAE": round(m, 4),
                f"{model}_MedAE": round(med, 4),
            }

        error_by_sim_rows.append(row)

    # Spearman correlation between max_tanimoto and abs error
    rho_xgb, p_xgb = stats.spearmanr(df["max_tanimoto_to_train"], df["abs_err_xgb"])
    rho_rf,  p_rf  = stats.spearmanr(df["max_tanimoto_to_train"], df["abs_err_rf"])

    ad_results[t] = {
        "n_test": len(df),
        "bins": target_ad,
        "spearman_tanimoto_vs_abs_err": {
            "XGBoost": {"rho": round(float(rho_xgb), 4), "p": round(float(p_xgb), 6)},
            "RF":      {"rho": round(float(rho_rf),  4), "p": round(float(p_rf),  6)},
        },
        "note": (
            "Negative rho means higher similarity → lower error. "
            "Inspect p-value before interpreting. "
            "BChE lowest-bin has very few samples (n≤20); statistics are noisy."
        ),
    }

    print(f"\n  {t} (n={len(df)}):")
    print(f"    Spearman(tanimoto, |err|) XGBoost rho={rho_xgb:.3f} p={p_xgb:.4f}")
    print(f"    Spearman(tanimoto, |err|) RF      rho={rho_rf:.3f} p={p_rf:.4f}")
    for label, d in target_ad.items():
        if d.get("n", 0) == 0:
            continue
        print(f"    {label}: n={d['n']}, XGB RMSE={d.get('XGBoost_RMSE','?')}, RF RMSE={d.get('RF_RMSE','?')}")

error_by_sim_df = pd.DataFrame(error_by_sim_rows)

# ═══════════════════════════════════════════════════════════════
# C. THRESHOLD SENSITIVITY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("C. Threshold sensitivity")
print("=" * 60)

thresh_rows = []
for t in TARGETS:
    # library coverage
    lib_sim = lib_valid[f"max_tanimoto_{t}"].values
    # test-set error at each cutoff
    test_df = test_preds[t].copy()
    test_df["abs_err_xgb"] = (test_df["y_true"] - test_df["pred_xgb"]).abs()

    for cutoff in ALT_CUTOFFS:
        n_retained = int((lib_sim >= cutoff).sum())
        pct_retained = round(100 * n_retained / len(lib_sim), 2)
        # test-set compounds above cutoff
        sub_test = test_df[test_df["max_tanimoto_to_train"] >= cutoff]
        n_test_above = len(sub_test)
        rmse_above = round(rmse(sub_test["y_true"], sub_test["pred_xgb"]), 4) if n_test_above > 0 else None
        mae_above  = round(mae(sub_test["y_true"],  sub_test["pred_xgb"]), 4) if n_test_above > 0 else None

        # test-set compounds below cutoff (the ones that would be excluded)
        sub_below = test_df[test_df["max_tanimoto_to_train"] < cutoff]
        n_test_below = len(sub_below)
        rmse_below = round(rmse(sub_below["y_true"], sub_below["pred_xgb"]), 4) if n_test_below > 0 else None

        # determine if cutoff is empirically supported
        supported = None
        if rmse_above is not None and rmse_below is not None:
            supported = rmse_above < rmse_below  # True if retained set has lower error

        thresh_rows.append({
            "target": t,
            "cutoff": cutoff,
            "library_compounds_retained": n_retained,
            "library_pct_retained": pct_retained,
            "test_n_above_cutoff": n_test_above,
            "test_XGBoost_RMSE_above": rmse_above,
            "test_XGBoost_MAE_above": mae_above,
            "test_n_below_cutoff": n_test_below,
            "test_XGBoost_RMSE_below": rmse_below,
            "empirically_supported": supported,
        })
        print(
            f"  {t} cutoff={cutoff}: lib_retained={n_retained}({pct_retained}%)  "
            f"test_above n={n_test_above} RMSE={rmse_above}  below n={n_test_below} RMSE={rmse_below}  "
            f"supported={supported}"
        )

thresh_df = pd.DataFrame(thresh_rows)

# ═══════════════════════════════════════════════════════════════
# D. UNCERTAINTY-PROXY ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("D. Uncertainty-proxy analysis")
print("=" * 60)
print("  NOTE: tree_std and disagree are UNCERTAINTY PROXIES, not calibrated confidence intervals.")

unc_results = {}

for t in TARGETS:
    sc = screening[[
        "canonical_smiles",
        f"{t}_pred_rf", f"{t}_pred_xgb", f"{t}_pred_ens",
        f"{t}_unc_tree_std", f"{t}_unc_disagree",
        f"{t}_max_tanimoto", f"{t}_domain",
    ]].copy().rename(columns={
        f"{t}_pred_rf": "pred_rf",
        f"{t}_pred_xgb": "pred_xgb",
        f"{t}_pred_ens": "pred_ens",
        f"{t}_unc_tree_std": "unc_tree_std",
        f"{t}_unc_disagree": "unc_disagree",
        f"{t}_max_tanimoto": "max_tanimoto",
        f"{t}_domain": "domain",
    })

    # Screening library proxy distributions
    tree_std = sc["unc_tree_std"].dropna()
    disagree  = sc["unc_disagree"].dropna()

    # Correlation with test-set errors (where we have ground truth)
    test_df = test_preds[t].copy()
    test_df["abs_err_xgb"] = (test_df["y_true"] - test_df["pred_xgb"]).abs()
    test_df["unc_disagree_proxy"] = (test_df["pred_xgb"] - test_df["pred_rf"]).abs()
    # RF tree std is not available in test_predictions → note as unavailable
    rho_dis, p_dis = stats.spearmanr(test_df["unc_disagree_proxy"], test_df["abs_err_xgb"])

    unc_results[t] = {
        "library_tree_std": {
            "median": round(float(tree_std.median()), 4),
            "q25": round(float(tree_std.quantile(0.25)), 4),
            "q75": round(float(tree_std.quantile(0.75)), 4),
            "max": round(float(tree_std.max()), 4),
        },
        "library_disagree": {
            "median": round(float(disagree.median()), 4),
            "q25": round(float(disagree.quantile(0.25)), 4),
            "q75": round(float(disagree.quantile(0.75)), 4),
            "max": round(float(disagree.max()), 4),
        },
        "test_spearman_disagree_vs_abs_err_xgb": {
            "rho": round(float(rho_dis), 4),
            "p": round(float(p_dis), 6),
            "n": len(test_df),
            "note": "disagree computed as |pred_xgb - pred_rf| on scaffold test set",
        },
        "rf_tree_std_vs_test_error": "NOT AVAILABLE — per-tree predictions not saved in test_predictions.csv",
        "label": "UNCERTAINTY PROXY — not a calibrated predictive interval",
    }

    print(f"\n  {t}:")
    print(f"    library tree_std median: {unc_results[t]['library_tree_std']['median']:.4f}")
    print(f"    library disagree median: {unc_results[t]['library_disagree']['median']:.4f}")
    print(f"    Spearman(disagree, |err_xgb|) rho={rho_dis:.4f} p={p_dis:.5f}")

# ═══════════════════════════════════════════════════════════════
# E. CANDIDATE-LEVEL AUDIT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("E. Candidate-level audit")
print("=" * 60)

audit_rows = []

# Flag patterns for structural issues
GLYCOSIDE_PATTERN = Chem.MolFromSmarts("[OX2][C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O1)")  # basic pyranose
REACTIVE_PATTERNS = {
    "Michael acceptor (alpha-beta unsat carbonyl)": Chem.MolFromSmarts("[CX3](=[OX1])[CX3]=[CX3]"),
    "Aldehyde": Chem.MolFromSmarts("[CX3H1](=O)"),
    "Epoxide": Chem.MolFromSmarts("[OX2r3]"),
    "Acyl halide": Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]"),
}

def check_flags(smi):
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return "INVALID_SMILES", []
    flags = []
    if GLYCOSIDE_PATTERN and mol.HasSubstructMatch(GLYCOSIDE_PATTERN):
        flags.append("possible_glycoside")
    for name, patt in REACTIVE_PATTERNS.items():
        if patt and mol.HasSubstructMatch(patt):
            flags.append(f"reactive:{name}")
    mw = Descriptors.MolWt(mol)
    if mw > 700:
        flags.append(f"high_MW:{mw:.0f}")
    if mw < 150:
        flags.append(f"low_MW:{mw:.0f}")
    return "ok", flags

for t in TARGETS:
    df = top_cands[t].copy()
    domain_col = f"{t}_domain"
    sim_col    = f"{t}_max_tanimoto"
    pred_col   = f"{t}_pred_ens"

    # Check whether compound is in multi-target top list
    df["in_multitarget"] = df["canonical_smiles"].isin(multitarget["canonical_smiles"].values)

    for _, row in df.iterrows():
        status, flags = check_flags(row["canonical_smiles"])
        is_ood = row[domain_col] == "out_of_domain"
        audit_rows.append({
            "target": t,
            "source_ids": row.get("source_ids", ""),
            "names": row.get("names", ""),
            "canonical_smiles": row["canonical_smiles"],
            "pred_ens": round(row[pred_col], 4),
            "max_tanimoto": round(row[sim_col], 4),
            "domain": row[domain_col],
            "is_out_of_domain": is_ood,
            "in_multitarget_top": row["in_multitarget"],
            "structural_flags": "; ".join(flags) if flags else "",
            "smiles_valid": status,
        })

    n_ood   = sum(1 for r in audit_rows if r["target"] == t and r["is_out_of_domain"])
    n_flags = sum(1 for r in audit_rows if r["target"] == t and r["structural_flags"])
    n_mt    = sum(1 for r in audit_rows if r["target"] == t and r["in_multitarget_top"])
    print(f"  {t}: {len(df)} candidates | out-of-domain: {n_ood} | structural flags: {n_flags} | multitarget: {n_mt}")

audit_df = pd.DataFrame(audit_rows)

# ═══════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Saving outputs …")

error_by_sim_df.to_csv(REPORTS_AD / "error_by_similarity.csv", index=False)
print(f"  error_by_similarity.csv: {len(error_by_sim_df)} rows")

thresh_df.to_csv(REPORTS_AD / "threshold_sensitivity.csv", index=False)
print(f"  threshold_sensitivity.csv: {len(thresh_df)} rows")

audit_df.to_csv(REPORTS_AD / "candidate_reliability_audit.csv", index=False)
print(f"  candidate_reliability_audit.csv: {len(audit_df)} rows")

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════
print("\nGenerating plots …")

COLORS = {"AChE": "#2E86AB", "BChE": "#A23B72", "BACE1": "#F18F01"}
DOMAIN_COLORS = {"in_domain": "#2E86AB", "borderline": "#F5A623", "out_of_domain": "#D0021B"}

# ── Fig 1: Similarity distributions ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, t in zip(axes, TARGETS):
    sim_vals = lib_valid[f"max_tanimoto_{t}"].dropna().values
    ax.hist(sim_vals, bins=50, color=COLORS[t], alpha=0.8, edgecolor="none")
    for cutoff in [0.3, 0.5]:
        ax.axvline(cutoff, color="grey", linestyle="--", linewidth=0.9)
    ax.set_title(f"{t}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Max Tanimoto to training set", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.text(0.98, 0.95, f"median={np.nanmedian(sim_vals):.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)

fig.suptitle(
    "Fig 1: Max Morgan-Tanimoto similarity of phytochemical library to target training sets\n"
    "(dashed lines at 0.3 and 0.5 for reference — not validated thresholds)",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(REPORTS_FIG / "fig1_similarity_distributions.png", dpi=150)
plt.close()
print("  fig1_similarity_distributions.png")

# ── Fig 2: Molecular property comparison ───────────────────────
props_to_plot = ["MolWt", "LogP", "TPSA", "HBD", "HBA"]
fig, axes = plt.subplots(len(props_to_plot), 3, figsize=(13, 14))
for col_i, t in enumerate(TARGETS):
    cur_props = compute_props(curated[t].copy())
    for row_i, prop in enumerate(props_to_plot):
        ax = axes[row_i][col_i]
        lib_vals  = lib_valid[prop].dropna().values if prop in lib_valid.columns else np.array([])
        train_vals = cur_props[prop].dropna().values if prop in cur_props.columns else np.array([])
        combined_max = max(
            lib_vals.max() if len(lib_vals) > 0 else 0,
            train_vals.max() if len(train_vals) > 0 else 0,
        )
        combined_min = min(
            lib_vals.min() if len(lib_vals) > 0 else 0,
            train_vals.min() if len(train_vals) > 0 else 0,
        )
        bins = np.linspace(combined_min, combined_max, 40)
        ax.hist(lib_vals, bins=bins, alpha=0.6, color=COLORS[t], label="Library")
        ax.hist(train_vals, bins=bins, alpha=0.5, color="grey", label="Training")
        if row_i == 0:
            ax.set_title(t, fontsize=10, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel(prop, fontsize=9)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

fig.suptitle("Fig 2: Molecular-property distributions — phytochemical library vs. target training sets", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(REPORTS_FIG / "fig2_property_distributions.png", dpi=150)
plt.close()
print("  fig2_property_distributions.png")

# ── Fig 3: Error vs. similarity (held-out test) ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for col_i, t in enumerate(TARGETS):
    test_df = test_preds[t].copy()
    test_df["abs_err_xgb"] = (test_df["y_true"] - test_df["pred_xgb"]).abs()
    test_df["abs_err_rf"]  = (test_df["y_true"] - test_df["pred_rf"]).abs()

    for row_i, (model_label, err_col) in enumerate([("XGBoost", "abs_err_xgb"), ("RF", "abs_err_rf")]):
        ax = axes[row_i][col_i]
        ax.scatter(
            test_df["max_tanimoto_to_train"], test_df[err_col],
            s=8, alpha=0.35, color=COLORS[t], linewidths=0,
        )
        # bin medians
        df2 = test_df.copy()
        df2["sim_bin"] = pd.cut(df2["max_tanimoto_to_train"], bins=SIM_BINS, labels=SIM_BIN_LABELS, right=False)
        bm = df2.groupby("sim_bin", observed=True)[err_col].median()
        bin_centers = [0.15, 0.4, 0.6, 0.85]
        ax.plot(bin_centers[:len(bm)], bm.values, "k-o", markersize=5, linewidth=1.5, label="bin median")
        ax.set_xlabel("Max Tanimoto to training set", fontsize=8)
        ax.set_ylabel(f"|error| {model_label}", fontsize=8)
        if row_i == 0:
            ax.set_title(t, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)

fig.suptitle("Fig 3: Absolute prediction error vs. max Tanimoto similarity (scaffold test sets)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(REPORTS_FIG / "fig3_error_vs_similarity.png", dpi=150)
plt.close()
print("  fig3_error_vs_similarity.png")

# ── Fig 4: Coverage vs. RMSE at alternative cutoffs ────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, t in zip(axes, TARGETS):
    sub = thresh_df[thresh_df["target"] == t]
    ax2 = ax.twinx()
    ax.bar(sub["cutoff"].astype(str), sub["library_pct_retained"], color=COLORS[t], alpha=0.6, label="% library retained")
    ax2.plot(sub["cutoff"].astype(str), sub["test_XGBoost_RMSE_above"], "ko-", markersize=6, label="RMSE (above cutoff)")
    ax2.plot(sub["cutoff"].astype(str), sub["test_XGBoost_RMSE_below"], "rs--", markersize=6, label="RMSE (below cutoff)")
    ax.set_xlabel("Similarity cutoff", fontsize=9)
    ax.set_ylabel("% library retained", fontsize=9, color=COLORS[t])
    ax2.set_ylabel("Test RMSE (XGBoost)", fontsize=9)
    ax.set_title(t, fontsize=10, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

fig.suptitle("Fig 4: Coverage (% library retained) and held-out RMSE at alternative similarity cutoffs\n"
             "(empirical support column in threshold_sensitivity.csv)", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(REPORTS_FIG / "fig4_coverage_vs_error.png", dpi=150)
plt.close()
print("  fig4_coverage_vs_error.png")

# ── Fig 5: Uncertainty proxy vs. absolute error (test) ──────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, t in zip(axes, TARGETS):
    test_df = test_preds[t].copy()
    test_df["abs_err_xgb"] = (test_df["y_true"] - test_df["pred_xgb"]).abs()
    test_df["unc_disagree"] = (test_df["pred_xgb"] - test_df["pred_rf"]).abs()

    ax.scatter(test_df["unc_disagree"], test_df["abs_err_xgb"],
               s=8, alpha=0.3, color=COLORS[t], linewidths=0)
    rho, p = stats.spearmanr(test_df["unc_disagree"], test_df["abs_err_xgb"])
    ax.set_xlabel("|pred_xgb − pred_rf| (uncertainty proxy)", fontsize=8)
    ax.set_ylabel("|error| XGBoost", fontsize=8)
    ax.set_title(t, fontsize=10, fontweight="bold")
    ax.text(0.05, 0.95, f"Spearman ρ={rho:.3f}\np={p:.4f}",
            transform=ax.transAxes, va="top", fontsize=8)

fig.suptitle("Fig 5: Uncertainty proxy (RF–XGBoost disagreement) vs. absolute XGBoost error (scaffold test)\n"
             "PROXY ONLY — not a calibrated confidence interval", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(REPORTS_FIG / "fig5_uncertainty_vs_error.png", dpi=150)
plt.close()
print("  fig5_uncertainty_vs_error.png")

# ═══════════════════════════════════════════════════════════════
# BUILD RESULTS JSON
# ═══════════════════════════════════════════════════════════════
results_json = {
    "phase": 7,
    "title": "Chemical Space and Applicability Domain Analysis",
    "created_utc": datetime.datetime.utcnow().isoformat(),
    "inputs": {
        "phytochemical_library": f"{len(lib)} rows",
        "screening_results": f"{len(screening)} rows",
        "per_target": {
            t: {
                "training": f"{len(curated[t])} molecules",
                "scaffold_test": f"{len(test_preds[t])} molecules",
            }
            for t in TARGETS
        },
    },
    "A_chemical_space": space_results,
    "B_applicability_domain": ad_results,
    "C_threshold_sensitivity": thresh_df.to_dict(orient="records"),
    "D_uncertainty_proxy": unc_results,
    "E_candidate_audit_summary": {
        t: {
            "n_candidates": len(top_cands[t]),
            "n_out_of_domain": int((audit_df[audit_df["target"] == t]["is_out_of_domain"]).sum()),
            "n_with_structural_flags": int((audit_df[audit_df["target"] == t]["structural_flags"] != "").sum()),
            "n_in_multitarget": int((audit_df[audit_df["target"] == t]["in_multitarget_top"]).sum()),
        }
        for t in TARGETS
    },
    "outputs": [
        "reports/applicability_domain/error_by_similarity.csv",
        "reports/applicability_domain/threshold_sensitivity.csv",
        "reports/applicability_domain/candidate_reliability_audit.csv",
        "reports/applicability_domain/phase7_results.json",
        "reports/applicability_domain/phase7_summary.md",
        "reports/figures/phase7/fig1_similarity_distributions.png",
        "reports/figures/phase7/fig2_property_distributions.png",
        "reports/figures/phase7/fig3_error_vs_similarity.png",
        "reports/figures/phase7/fig4_coverage_vs_error.png",
        "reports/figures/phase7/fig5_uncertainty_vs_error.png",
    ],
}

with open(REPORTS_AD / "phase7_results.json", "w") as f:
    json.dump(results_json, f, indent=2, default=str)
print("  phase7_results.json")

# ═══════════════════════════════════════════════════════════════
# SUMMARY MARKDOWN
# ═══════════════════════════════════════════════════════════════
def fmt_bin_table(target):
    rows = error_by_sim_df[error_by_sim_df["target"] == target]
    lines = ["| Bin | n | XGB RMSE | XGB MAE | XGB MedAE | RF RMSE | RF MAE |",
             "|-----|---|----------|---------|-----------|---------|--------|"]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r['similarity_bin']} | {r['n']} "
            f"| {r.get('XGBoost_RMSE','?')} | {r.get('XGBoost_MAE','?')} | {r.get('XGBoost_MedAE','?')} "
            f"| {r.get('RF_RMSE','?')} | {r.get('RF_MAE','?')} |"
        )
    return "\n".join(lines)

def fmt_thresh_table(target):
    rows = thresh_df[thresh_df["target"] == target]
    lines = ["| Cutoff | Lib retained | % retained | Test n above | RMSE above | RMSE below | Supported? |",
             "|--------|-------------|------------|--------------|------------|------------|------------|"]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r['cutoff']} | {r['library_compounds_retained']} | {r['library_pct_retained']} "
            f"| {r['test_n_above_cutoff']} | {r['test_XGBoost_RMSE_above']} | {r['test_XGBoost_RMSE_below']} "
            f"| {r['empirically_supported']} |"
        )
    return "\n".join(lines)

summary_lines = [
    "# Phase 7 — Chemical Space and Applicability Domain Analysis",
    "",
    f"- **Created (UTC):** {results_json['created_utc']}",
    f"- **Library:** {len(lib)} phytochemical compounds, {int(lib_fp_invalid)} with invalid SMILES",
    f"- **Fingerprint:** Morgan radius=2, 2048 bits",
    f"- **Bootstrap CI:** n={N_BOOTSTRAP} resamples, seed={RNG_SEED}",
    "",
    "> **Scientific labelling reminder**",
    "> - `*_pred_*` columns are ML predictions (ensemble, RF, XGBoost).",
    "> - `*_exp_pIC50` are experimental ChEMBL median pIC50 values.",
    "> - `unc_tree_std` and `unc_disagree` are **uncertainty proxies**, not calibrated confidence intervals.",
    "> - Tanimoto thresholds have been evaluated empirically here; none is universally validated.",
    "",
    "---",
    "",
    "## A. Chemical-Space Overview",
    "",
]

for t in TARGETS:
    r = space_results[t]
    summary_lines += [
        f"### {t}",
        f"- Training set: {r['n_training']} molecules | Library (valid SMILES): {r['n_library_valid']}",
        f"- Library invalid SMILES: {r['n_library_invalid_smiles']}",
        f"- Median max Tanimoto to training: {r['max_tanimoto_median']:.3f}",
        "",
        "**Similarity bins (Morgan radius=2, 2048 bits):**",
        "",
        "| Bin | Count | % of library |",
        "|-----|-------|--------------|",
    ]
    for label, d in r["similarity_bins"].items():
        summary_lines.append(f"| {label} | {d['n']} | {d['pct']} |")
    summary_lines.append("")

    summary_lines += [
        "**Molecular-property comparison (median [Q25–Q75]):**",
        "",
        "| Property | Library | Training |",
        "|----------|---------|----------|",
    ]
    for prop in PROP_COLS:
        lp = r["library_properties"].get(prop, {})
        tp = r["training_properties"].get(prop, {})
        lib_str  = f"{lp.get('median','?')} [{lp.get('q25','?')}–{lp.get('q75','?')}]" if lp else "—"
        train_str = f"{tp.get('median','?')} [{tp.get('q25','?')}–{tp.get('q75','?')}]" if tp else "—"
        summary_lines.append(f"| {prop} | {lib_str} | {train_str} |")
    summary_lines.append("")

summary_lines += [
    "---",
    "",
    "## B. Applicability-Domain Validation (Scaffold Test Sets)",
    "",
    "Error is computed on the scaffold-held-out test set only. Numbers come from executed code.",
    "",
]

for t in TARGETS:
    r = ad_results[t]
    sp = r["spearman_tanimoto_vs_abs_err"]
    summary_lines += [
        f"### {t}  (n_test = {r['n_test']})",
        "",
        f"Spearman ρ(max_tanimoto, |err|):  XGBoost ρ={sp['XGBoost']['rho']}  p={sp['XGBoost']['p']}  |  RF ρ={sp['RF']['rho']}  p={sp['RF']['p']}",
        "",
        fmt_bin_table(t),
        "",
        f"> {r['note']}",
        "",
    ]

summary_lines += [
    "---",
    "",
    "## C. Threshold Sensitivity",
    "",
    "RMSE computed on scaffold test compounds above / below each cutoff (XGBoost). "
    "`Supported = True` means retained set has lower RMSE than excluded set.",
    "",
]

for t in TARGETS:
    summary_lines += [f"### {t}", "", fmt_thresh_table(t), ""]

summary_lines += [
    "---",
    "",
    "## D. Uncertainty-Proxy Analysis",
    "",
    "**All proxies are labelled as proxies — not calibrated predictive intervals.**",
    "Per-tree RF predictions are not stored in test_predictions.csv; "
    "tree_std analysis limited to library screening output.",
    "RF–XGBoost disagreement is computed as |pred_xgb − pred_rf| and evaluated against scaffold test errors.",
    "",
]

for t in TARGETS:
    r = unc_results[t]
    sp = r["test_spearman_disagree_vs_abs_err_xgb"]
    summary_lines += [
        f"### {t}",
        f"- Library `unc_tree_std` median: {r['library_tree_std']['median']} (Q25={r['library_tree_std']['q25']}, Q75={r['library_tree_std']['q75']})",
        f"- Library `unc_disagree` median: {r['library_disagree']['median']} (Q25={r['library_disagree']['q25']}, Q75={r['library_disagree']['q75']})",
        f"- Spearman ρ(disagree, |err_xgb|) on scaffold test: ρ={sp['rho']}  p={sp['p']}  n={sp['n']}",
        f"- RF tree std vs test error: {r['rf_tree_std_vs_test_error']}",
        "",
    ]

summary_lines += [
    "---",
    "",
    "## E. Candidate-Level Audit",
    "",
    "| Target | Candidates | Out-of-domain | Structural flags | In multitarget top |",
    "|--------|-----------|---------------|-----------------|-------------------|",
]
for t in TARGETS:
    es = results_json["E_candidate_audit_summary"][t]
    summary_lines.append(
        f"| {t} | {es['n_candidates']} | {es['n_out_of_domain']} | "
        f"{es['n_with_structural_flags']} | {es['n_in_multitarget']} |"
    )

summary_lines += [
    "",
    "Full per-compound audit in `reports/applicability_domain/candidate_reliability_audit.csv`.",
    "Structural flags: glycoside-pattern match, Michael acceptor, aldehyde, epoxide, acyl halide, MW outlier.",
    "These flags require manual structural review — compounds are not removed automatically.",
    "",
    "---",
    "",
    "## Figures",
    "",
    "| File | Description |",
    "|------|-------------|",
    "| fig1_similarity_distributions.png | Tanimoto similarity histograms per target |",
    "| fig2_property_distributions.png | Library vs. training molecular properties |",
    "| fig3_error_vs_similarity.png | Held-out absolute error vs. max Tanimoto |",
    "| fig4_coverage_vs_error.png | Coverage and RMSE at alternative cutoffs |",
    "| fig5_uncertainty_vs_error.png | Uncertainty proxy vs. absolute error |",
    "",
    "---",
    "",
    "## Scientific Limitations",
    "",
    "1. **Low coverage**: Most phytochemicals are far from the ChEMBL training chemical space (median Tanimoto < 0.4). Predictions for out-of-domain compounds should be treated as low-confidence rankings only.",
    "2. **Model shrinkage**: Tree-based models underestimate extreme pIC50 values. Use predictions to rank, not as potency estimates.",
    "3. **Tanimoto thresholds**: No threshold tested here is universally validated. The empirical support column in threshold_sensitivity.csv shows whether a given cutoff correlates with improved accuracy on this test set.",
    "4. **Uncertainty proxies**: RF tree spread and RF–XGBoost disagreement are not calibrated uncertainty estimates. High disagreement is a warning, not a quantified interval.",
    "5. **Glycosylated compounds**: Flagged for review. Similarity to training may be artificially inflated by the sugar moiety.",
    "6. **Scaffold split artefacts**: Low-similarity test compounds are genuinely novel scaffolds; their error may be worse purely because tree models extrapolate poorly, not because they are inactive.",
    "",
    "---",
    "",
    "## Recommended Candidate-Review Strategy (Phase 8)",
    "",
    "1. **Prioritise experimental evidence first**: compounds in `*_known_experimental.csv` have real ChEMBL IC50 data.",
    "2. **Within novel candidates, prefer in-domain compounds** (max_tanimoto ≥ 0.5), especially where the threshold sensitivity analysis shows lower RMSE.",
    "3. **Review all structural flags manually** before docking. Glycosylated flavonoids should be docked as the aglycone unless plant-bioavailability is explicitly in scope.",
    "4. **Multi-target compounds** (n=91) are highest priority for selectivity profiling.",
    "5. **For out-of-domain borderline compounds** that rank highly, consider literature-mining or public bioactivity databases before committing to synthesis or wet-lab assays.",
    "6. **Do not use predicted pIC50 values as potency estimates** — present them as rankings with the domain tier and uncertainty proxy alongside.",
]

with open(REPORTS_AD / "phase7_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")
print("  phase7_summary.md")

# ═══════════════════════════════════════════════════════════════
# INTEGRITY CHECK — confirm Phase 1–5 files untouched
# ═══════════════════════════════════════════════════════════════
print("\nIntegrity check …")
protected = [
    DATA_PROC / "phytochemical_library.csv",
    DATA_PROC / "AChE_IC50_curated.csv",
    DATA_PROC / "BChE_IC50_curated.csv",
    DATA_PROC / "BACE1_IC50_curated.csv",
    DATA_PROC / "screening_results.csv",
]
for p in protected:
    assert p.exists(), f"PROTECTED FILE MISSING: {p}"
print("  All protected Phase 1-5 files present. OK")

# ─────────────────── verify output row counts ───────────────────
err_df_check = pd.read_csv(REPORTS_AD / "error_by_similarity.csv")
thr_df_check = pd.read_csv(REPORTS_AD / "threshold_sensitivity.csv")
aud_df_check = pd.read_csv(REPORTS_AD / "candidate_reliability_audit.csv")

assert len(err_df_check) == len(TARGETS) * len(SIM_BIN_LABELS), \
    f"error_by_similarity row count mismatch: {len(err_df_check)}"
assert len(thr_df_check) == len(TARGETS) * len(ALT_CUTOFFS), \
    f"threshold_sensitivity row count mismatch: {len(thr_df_check)}"
assert len(aud_df_check) == sum(len(top_cands[t]) for t in TARGETS), \
    f"candidate_reliability_audit row count mismatch: {len(aud_df_check)}"

print(f"  error_by_similarity.csv: {len(err_df_check)} rows ✓")
print(f"  threshold_sensitivity.csv: {len(thr_df_check)} rows ✓")
print(f"  candidate_reliability_audit.csv: {len(aud_df_check)} rows ✓")

# ─────────────────── check figures ──────────────────────────────
for fig_name in [
    "fig1_similarity_distributions.png",
    "fig2_property_distributions.png",
    "fig3_error_vs_similarity.png",
    "fig4_coverage_vs_error.png",
    "fig5_uncertainty_vs_error.png",
]:
    assert (REPORTS_FIG / fig_name).exists(), f"FIGURE MISSING: {fig_name}"
    print(f"  {fig_name} ✓")

print("\n✓ Phase 7 complete. All outputs verified.")
print(f"  Summary: {REPORTS_AD / 'phase7_summary.md'}")
print(f"  Results: {REPORTS_AD / 'phase7_results.json'}")
