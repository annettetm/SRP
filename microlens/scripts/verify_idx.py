from pathlib import Path
import ast
import pandas as pd

BASE = Path("/home/mathew/SRP/Amazon_beauty")
PROCESSED = BASE / "processed"
REPORT = PROCESSED / "verification_report.txt"

item_df = pd.read_csv(PROCESSED / "item_df.csv")
interaction_df = pd.read_csv(PROCESSED / "interaction_df.csv")
seq_df = pd.read_csv(PROCESSED / "user_sequences.csv")

for df in [item_df, interaction_df, seq_df]:
    for col in df.columns:
        if col == "parent_asin":
            df[col] = df[col].astype(str).str.strip()

item_df["parent_asin"] = item_df["parent_asin"].astype(str).str.strip()
interaction_df["parent_asin"] = interaction_df["parent_asin"].astype(str).str.strip()
seq_df["user_id"] = seq_df["user_id"].astype(str).str.strip()

item_asins = set(item_df["parent_asin"])
inter_asins = set(interaction_df["parent_asin"])

seq_asins = set()
for s in seq_df["item_sequence"].astype(str):
    try:
        seq_asins.update(ast.literal_eval(s))
    except Exception:
        pass

lines = []
lines.append(f"item_df rows: {len(item_df)}")
lines.append(f"unique item ASINs: {item_df['parent_asin'].nunique()}")
lines.append(f"interaction rows: {len(interaction_df)}")
lines.append(f"unique interaction ASINs: {len(inter_asins)}")
lines.append(f"sequence ASINs: {len(seq_asins)}")
lines.append(f"duplicate ASINs in item_df: {item_df['parent_asin'].duplicated().sum()}")
lines.append(f"missing interaction ASINs in item_df: {len(inter_asins - item_asins)}")
lines.append(f"missing sequence ASINs in item_df: {len(seq_asins - item_asins)}")

print("\n".join(lines))
REPORT.write_text("\n".join(lines))