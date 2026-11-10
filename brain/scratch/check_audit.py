import pandas as pd

aud = pd.read_csv("reports/applicability_domain/candidate_reliability_audit.csv")

# NaN in structural_flags = no flags raised (empty string was written as NaN by CSV round-trip)
aud["structural_flags"] = aud["structural_flags"].fillna("")

print("smiles_valid values:", aud["smiles_valid"].value_counts().to_dict())
print()

for t in ["AChE", "BChE", "BACE1"]:
    sub = aud[aud["target"] == t].copy()
    sub["structural_flags"] = sub["structural_flags"].fillna("")
    has_flags = (sub["structural_flags"] != "").sum()
    print(f"{t}:")
    print(f"  total: {len(sub)}")
    print(f"  out_of_domain: {sub['is_out_of_domain'].sum()}")
    print(f"  with structural flags: {has_flags}")
    print(f"  in multitarget: {sub['in_multitarget_top'].sum()}")
    print(f"  domain breakdown: {sub['domain'].value_counts().to_dict()}")
    print(f"  flag types sample:")
    flags = sub[sub["structural_flags"] != ""]["structural_flags"].value_counts().head(5)
    for flag, cnt in flags.items():
        print(f"    {flag}: {cnt}")
    print()
