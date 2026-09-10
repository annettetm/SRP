import pandas as pd

interaction_df = pd.read_csv("/home/mathew/SRP/Amazon_beauty/processed/interaction_df.csv")
item_df = pd.read_csv("/home/mathew/SRP/Amazon_beauty/processed/item_df.csv")

print(interaction_df["parent_asin"].head().tolist())
print(item_df["parent_asin"].head().tolist())
print("intersection:", len(set(interaction_df["parent_asin"]) & set(item_df["parent_asin"])))