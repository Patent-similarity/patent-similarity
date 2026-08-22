import pandas as pd

files_to_check = [
    "g_patent.tsv",
    "g_patent_abstract.tsv",
    "g_cpc_current.tsv",
    "g_claims_2024.tsv"
]

for f in files_to_check:
    print(f"\n{'='*50}")
    print(f"FILE: {f}")
    print('='*50)
    df = pd.read_csv(f"datasets/{f}", sep="\t", nrows=5)
    print("Columns:", df.columns.tolist())
    print(df.head(3))

# This runs once, after the loop finishes — not inside it
df_claims = pd.read_csv("datasets/g_claims_2024.tsv", sep="\t", nrows=1000)

print("\n" + "="*50)
print("INSPECTING 'dependent' COLUMN")
print("="*50)

print("dtype:", df_claims['dependent'].dtype)
print(df_claims['dependent'].value_counts())

print("\nMissing (NaN) count in 'dependent':", df_claims['dependent'].isna().sum())

print("\n--- Sample where dependent is NaN (likely independent claims) ---")
print(df_claims[df_claims['dependent'].isna()]['claim_text'].iloc[0][:300])

print("\n--- Sample where dependent = 'claim 1' (likely dependent claims) ---")
print(df_claims[df_claims['dependent'] == 'claim 1']['claim_text'].iloc[0][:300])