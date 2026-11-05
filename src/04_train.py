#!/usr/bin/env python3
"""
PhytoScreen-AI | Phase 4: target-specific QSAR models (pIC50 regression).

Per target:
  * Bemis-Murcko SCAFFOLD split 80/10/10 (primary validation) and a RANDOM split (comparison only).
  * Features: Morgan fingerprint (r=2, 2048 bits) + 8 simple RDKit descriptors. No scaling, no
    feature selection -> nothing is fitted on test data (no leakage).
  * Models: mean-predictor baseline, RandomForest, XGBoost (early stopping on the validation set only).
  * Metrics on the held-out test set: R2, RMSE, MAE, Spearman; also on the non-high-variance subset.
  * Applicability domain: max Tanimoto similarity of each test compound to the training set,
    with error reported per similarity bin.
  * After evaluation, final models are refit on ALL curated data for later screening (saved as *_full).

Run:  python src/04_train.py            (all targets)
      python src/04_train.py --targets AChE
"""
import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
METRICS = ROOT / "reports" / "metrics"
SEED = 42
FP_BITS = 2048
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def featurize(smiles):
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


def scaffold_of(smi):
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi)
    except Exception:
        sc = ""
    return sc if sc else smi  # acyclic molecules each form their own group


def scaffold_split(smiles, frac=(0.8, 0.1, 0.1)):
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[scaffold_of(s)].append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(smiles)
    tr_cut, va_cut = frac[0] * n, (frac[0] + frac[1]) * n
    tr, va, te = [], [], []
    for g in ordered:
        if len(tr) + len(g) > tr_cut:
            if len(tr) + len(va) + len(g) > va_cut:
                te += g
            else:
                va += g
        else:
            tr += g
    return np.array(tr), np.array(va), np.array(te)


def random_split(n, sizes):
    idx = np.random.RandomState(SEED).permutation(n)
    a, b = sizes
    return idx[:a], idx[a:a + b], idx[a + b:]


def metrics(y, p):
    return {"n": int(len(y)), "R2": float(r2_score(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)),
            "Spearman": float(spearmanr(y, p)[0])}


def make_models():
    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED, max_features=0.3)
    xg = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=6, subsample=0.8,
                          colsample_bytree=0.5, min_child_weight=2, n_jobs=-1, random_state=SEED,
                          early_stopping_rounds=50)
    return rf, xg


def run_split(name, tr, va, te, X, y, hv):
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    base = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    rf, xg = make_models()
    rf.fit(Xtr, ytr)
    xg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    preds = {"mean_baseline": base.predict(Xte), "RandomForest": rf.predict(Xte),
             "XGBoost": xg.predict(Xte)}
    res = {"split": name, "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
           "xgb_best_iteration": int(xg.best_iteration), "models": {}}
    keep = ~hv[te]
    for m, p in preds.items():
        res["models"][m] = {"all_test": metrics(yte, p),
                            "excluding_high_variance": metrics(yte[keep], p[keep]) if keep.sum() > 2 else None}
    return res, preds, rf, xg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["AChE", "BChE", "BACE1"])
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "seed": SEED, "targets": {}}

    for t in args.targets:
        print(f"\n=== {t} ===")
        df = pd.read_csv(PROCESSED / f"{t}_IC50_curated.csv")
        smiles = df["canonical_smiles"].tolist()
        y = df["median_pIC50"].to_numpy(dtype=float)
        hv = df["high_variance_flag"].to_numpy(dtype=bool)
        X, fps = featurize(smiles)

        # ---- primary: scaffold split ----
        tr, va, te = scaffold_split(smiles)
        sc = lambda idx: {scaffold_of(smiles[i]) for i in idx}
        assert not (sc(tr) & sc(te)) and not (sc(tr) & sc(va)) and not (sc(va) & sc(te)), "scaffold leakage!"
        res_s, preds_s, rf, xg = run_split("scaffold", tr, va, te, X, y, hv)
        res_s["scaffold_overlap_train_test"] = 0

        # ---- applicability domain (scaffold test set) ----
        train_fps = [fps[i] for i in tr]
        maxsim = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in te])
        err = np.abs(y[te] - preds_s["XGBoost"])
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
        ad = []
        for lo, hi in bins:
            mk = (maxsim >= lo) & (maxsim < hi)
            ad.append({"max_tanimoto_bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(mk.sum()),
                       "XGB_RMSE": float(np.sqrt(np.mean(err[mk] ** 2))) if mk.sum() else None})
        res_s["applicability_domain_xgb"] = ad

        # ---- comparison: random split (expected to look better; do NOT report as primary) ----
        rtr, rva, rte = random_split(len(df), (len(tr), len(va)))
        res_r, _, _, _ = run_split("random", rtr, rva, rte, X, y, hv)

        # save split assignment and test predictions for reproducibility
        split_col = np.empty(len(df), dtype=object)
        split_col[tr], split_col[va], split_col[te] = "train", "valid", "test"
        out = df[["molecule_chembl_id", "canonical_smiles", "median_pIC50"]].copy()
        out["scaffold_split"] = split_col
        out.to_csv(PROCESSED / f"{t}_scaffold_split.csv", index=False)
        pd.DataFrame({"canonical_smiles": [smiles[i] for i in te], "y_true": y[te],
                      "pred_xgb": preds_s["XGBoost"], "pred_rf": preds_s["RandomForest"],
                      "max_tanimoto_to_train": maxsim}).to_csv(METRICS / f"{t}_test_predictions.csv", index=False)

        # ---- final models on ALL curated data (for screening) ----
        joblib.dump(rf, MODELS / f"{t}_rf_trainonly.joblib")
        joblib.dump(xg, MODELS / f"{t}_xgb_trainonly.joblib")
        rf_full, xg_full = make_models()
        rf_full.fit(X, y)
        xg_full.set_params(n_estimators=int(xg.best_iteration) + 1, early_stopping_rounds=None)
        xg_full.fit(X, y, verbose=False)
        joblib.dump(rf_full, MODELS / f"{t}_rf_full.joblib")
        joblib.dump(xg_full, MODELS / f"{t}_xgb_full.joblib")

        report["targets"][t] = {"n_molecules": int(len(df)), "scaffold": res_s, "random": res_r}
        for r in (res_s, res_r):
            for m, v in r["models"].items():
                a = v["all_test"]
                print(f"{r['split']:8s} {m:14s} R2={a['R2']:.3f} RMSE={a['RMSE']:.3f} "
                      f"MAE={a['MAE']:.3f} Spearman={a['Spearman']:.3f}")

    (METRICS / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Model metrics (test sets, real executed results)", f"- Created (UTC): {report['created_utc']}", ""]
    for t, d in report["targets"].items():
        lines += [f"## {t} ({d['n_molecules']} molecules)", "",
                  "| Split | Model | R2 | RMSE | MAE | Spearman | R2 excl. high-var |", "|---|---|---|---|---|---|---|"]
        for key in ("scaffold", "random"):
            r = d[key]
            for m, v in r["models"].items():
                a, b = v["all_test"], v["excluding_high_variance"]
                lines.append(f"| {r['split']} | {m} | {a['R2']:.3f} | {a['RMSE']:.3f} | {a['MAE']:.3f} | "
                             f"{a['Spearman']:.3f} | {b['R2']:.3f} |" if b else
                             f"| {r['split']} | {m} | {a['R2']:.3f} | {a['RMSE']:.3f} | {a['MAE']:.3f} | {a['Spearman']:.3f} | n/a |")
        lines += ["", "Applicability domain (scaffold test, XGBoost):", "",
                  "| Max Tanimoto to train | n | RMSE |", "|---|---|---|"]
        lines += [f"| {a['max_tanimoto_bin']} | {a['n']} | {a['XGB_RMSE'] if a['XGB_RMSE'] is None else round(a['XGB_RMSE'], 3)} |"
                  for a in d["scaffold"]["applicability_domain_xgb"]]
        lines.append("")
    (METRICS / "model_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote reports/metrics/model_metrics.md")


if __name__ == "__main__":
    main()