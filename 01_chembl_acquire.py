#!/usr/bin/env python3
"""
PhytoScreen-AI | Phase 1: ChEMBL target discovery, verification, raw download, audit.

What this script does (in order):
  1. Records the ChEMBL release version (provenance).
  2. DISCOVERY: searches ChEMBL by name for each target and saves every hit (audit trail).
  3. VERIFICATION: checks that the chosen target is human (tax_id 9606), a SINGLE PROTEIN,
     and carries the expected UniProt accession. Stops if anything mismatches.
  4. DOWNLOAD: pulls ALL activity records of the requested standard types, page by page,
     and saves each raw page as JSON (never edited).
  5. FLATTEN: writes one un‑filtered CSV per target/type (still raw content, just tabular).
  6. AUDIT: writes audit JSON + Markdown with counts, checksums and data‑quality flags.

It does NOT filter, clean, deduplicate, or model. Curation is Phase 2.
Ki/Kd are stored in separate files from IC50 (IC50 is the default and only primary type).
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE = "https://www.ebi.ac.uk"
API = BASE + "/chembl/api/data"

# Expected identities. We still VERIFY them against the live API before downloading.
TARGETS = {
    "AChE": {"chembl_id": "CHEMBL220", "uniprot": "P22303", "search": "acetylcholinesterase"},
    "BChE": {"chembl_id": "CHEMBL1914", "uniprot": "P06276", "search": "butyrylcholinesterase"},
    "BACE1": {"chembl_id": "CHEMBL4822", "uniprot": "P56817", "search": "beta-secretase 1"},
}
HUMAN_TAX_ID = 9606

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "chembl"
AUDIT = ROOT / "reports" / "audit"
LOGS = ROOT / "logs"


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"Accept": "application/json", "User-Agent": "PhytoScreen-AI-research/0.1"})
    return s


def get_json(session, url, params=None):
    r = session.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ---------- Step 2: discovery ----------
def discover(session, name, query):
    """Save all search hits so a human can see what ChEMBL offers (incl. other species)."""
    hits, url, params = [], f"{API}/target/search.json", {"q": query, "limit": 100}
    while url:
        data = get_json(session, url, params)
        hits.extend(data.get("targets", []))
        nxt = data.get("page_meta", {}).get("next")
        url, params = (BASE + nxt, None) if nxt else (None, None)
    write_json(RAW / name / "target_discovery_search.json", hits)
    summary = [
        {
            "target_chembl_id": t["target_chembl_id"],
            "pref_name": t["pref_name"],
            "organism": t["organism"],
            "target_type": t["target_type"],
        }
        for t in hits
    ]
    return summary

# ---------- Step 3: verification ----------
def verify_target(session, name, spec):
    t = get_json(session, f"{API}/target/{spec['chembl_id']}.json")
    write_json(RAW / name / "target_record.json", t)
    accessions = [c.get("accession") for c in t.get("target_components", [])]
    checks = {
        "target_type_is_single_protein": t.get("target_type") == "SINGLE PROTEIN",
        "organism_is_human": t.get("organism") == "Homo sapiens",
        "tax_id_is_9606": int(t.get("tax_id") or -1) == HUMAN_TAX_ID,
        "uniprot_matches_expected": accessions == [spec["uniprot"]],
    }
    result = {
        "name": name,
        "chembl_id": spec["chembl_id"],
        "pref_name": t.get("pref_name"),
        "organism": t.get("organism"),
        "tax_id": t.get("tax_id"),
        "target_type": t.get("target_type"),
        "uniprot_accessions": accessions,
        "expected_uniprot": spec["uniprot"],
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not result["passed"]:
        print(f"[FAIL] Target verification failed for {name}: {json.dumps(result, indent=2)}")
        sys.exit(1)
    print(f"[OK] {name}: {spec['chembl_id']} = {t.get('pref_name')} | {t.get('organism')} | {accessions}")
    return result

# ---------- Step 4/5: download ----------
def download_activities(session, name, chembl_id, std_type, max_pages=None):
    out_dir = RAW / name / std_type
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{API}/activity.json"
    params = {"target_chembl_id": chembl_id, "standard_type": std_type, "limit": 1000}
    records, files, page, total = [], [], 0, None
    pbar = tqdm(desc=f"{name} {std_type}", unit="page")
    while url:
        data = get_json(session, url, params)
        meta = data["page_meta"]
        total = meta.get("total_count", total)
        page += 1
        fp = out_dir / f"page_{page:04d}.json.gz"
        with gzip.open(fp, "wt", encoding="utf-8") as f:
            json.dump({"request_url": url, "params": params, "page_meta": meta, "activities": data["activities"]}, f)
        files.append(fp)
        records.extend(data["activities"])
        pbar.update(1)
        nxt = meta.get("next")
        url, params = (BASE + nxt, None) if nxt else (None, None)
        if max_pages and page >= max_pages:
            print(f"[WARN] --max-pages reached; {name} {std_type} is PARTIAL (smoke test).")
            break
    pbar.close()
    df = pd.json_normalize(records) if records else pd.DataFrame()
    csv_path = RAW / name / f"{name}_{std_type}_raw_flat.csv"
    df.to_csv(csv_path, index=False)
    return df, total, files, csv_path

# ---------- Step 6: audit ----------
def profile(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_records": 0}
    def vc(col):
        return df[col].fillna("MISSING").value_counts().to_dict() if col in df else {}
    def nn(col):
        return int(df[col].notna().sum()) if col in df else None
    return {
        "n_records": int(len(df)),
        "n_unique_molecules": int(df["molecule_chembl_id"].nunique()) if "molecule_chembl_id" in df else None,
        "n_unique_assays": int(df["assay_chembl_id"].nunique()) if "assay_chembl_id" in df else None,
        "n_unique_documents": int(df["document_chembl_id"].nunique()) if "document_chembl_id" in df else None,
        "n_with_standard_value": nn("standard_value"),
        "n_with_pchembl_value": nn("pchembl_value"),
        "n_with_canonical_smiles": nn("canonical_smiles"),
        "n_with_data_validity_comment": nn("data_validity_comment"),
        "n_flagged_potential_duplicate": int((df["potential_duplicate"] == 1).sum()) if "potential_duplicate" in df else None,
        "standard_relation_counts": vc("standard_relation"),
        "standard_units_counts": vc("standard_units"),
        "assay_type_counts": vc("assay_type"),
        "target_organism_counts": vc("target_organism"),
        "target_chembl_id_counts": vc("target_chembl_id"),
        "assay_variant_mutation_present": nn("assay_variant_mutation"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["IC50"], help="standard types, kept in SEPARATE files")
    ap.add_argument("--max-pages", type=int, default=None, help="smoke test limit")
    args = ap.parse_args()

    for d in (RAW, AUDIT, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    session = make_session()

    status = get_json(session, f"{API}/status.json")
    print(f"ChEMBL release: {status.get('chembl_db_version')} ({status.get('chembl_release_date')})")

    audit = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "chembl_status": status,
        "types_requested": args.types,
        "partial_smoke_test": bool(args.max_pages),
        "targets": {},
    }

    for name, spec in TARGETS.items():
        print(f"\n=== {name} ===")
        disc = discover(session, name, spec["search"])
        ver = verify_target(session, name, spec)
        entry = {"discovery_hits": disc, "verification": ver, "downloads": {}}
        for std_type in args.types:
            df, total, files, csv_path = download_activities(
                session, name, spec["chembl_id"], std_type, args.max_pages
            )
            entry["downloads"][std_type] = {
                "api_reported_total": total,
                "records_downloaded": int(len(df)),
                "count_matches_api": (total == len(df)) if not args.max_pages else None,
                "n_raw_page_files": len(files),
                "flat_csv": str(csv_path.relative_to(ROOT)),
                "flat_csv_sha256": sha256(csv_path),
                "raw_page_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
                "profile": profile(df),
            }
        audit["targets"][name] = entry

    write_json(AUDIT / "chembl_acquisition_audit.json", audit)

    # Human‑readable summary
    lines = ["# ChEMBL Acquisition Audit", "", f"- Created (UTC): {audit['created_utc']}", f"- ChEMBL release: {status.get('chembl_db_version')}", f"- Types requested: {', '.join(args.types)}", f"- Partial smoke test: {audit['partial_smoke_test']}", ""]
    for name, e in audit["targets"].items():
        v = e["verification"]
        lines += [
            f"## {name} ({v['chembl_id']})",
            f"- Verified: {v['pref_name']} | {v['organism']} | UniProt {v['uniprot_accessions']} | passed={v['passed']}",
        ]
        for t, d in e["downloads"].items():
            p = d["profile"]
            lines += [
                f"- **{t}**: downloaded {d['records_downloaded']} / API total {d['api_reported_total']} (match={d['count_matches_api']})",
                f"  - molecules: {p.get('n_unique_molecules')}, assays: {p.get('n_unique_assays')}, documents: {p.get('n_unique_documents')}",
                f"  - with standard_value: {p.get('n_with_standard_value')}, pChEMBL: {p.get('n_with_pchembl_value')}, SMILES: {p.get('n_with_canonical_smiles')}",
                f"  - validity‑comment flagged: {p.get('n_with_data_validity_comment')}, potential duplicates: {p.get('n_flagged_potential_duplicate')}",
                f"  - relations: {p.get('standard_relation_counts')}",
                f"  - units: {p.get('standard_units_counts')}",
            ]
        lines.append("")
    lines += [
        "## Notes",
        "- Raw pages are unmodified API responses (gzipped JSON). Flat CSVs are unfiltered.",
        "- No curation, deduplication or modelling has been performed (Phase 2+).",
        "- Non‑human ChEMBL AChE/BChE/BACE entries appear only in discovery files, not downloaded.",
    ]
    (AUDIT / "chembl_acquisition_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Audit: {AUDIT / 'chembl_acquisition_audit.md'}")


if __name__ == "__main__":
    main()
