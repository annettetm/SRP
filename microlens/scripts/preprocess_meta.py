import pandas as pd

BASE = "/home/mathew/SRP/microlens"

meta_df = pd.read_csv(
    f"{BASE}/data/microlens100k/MicroLens-100k_title_en.csv",
    header=None,
    names=["item_id", "title"]
)

meta_df["item_id"] = meta_df["item_id"].astype(str).str.strip()
meta_df["title"] = meta_df["title"].astype(str).str.strip()

valid_items = meta_df["item_id"].unique()
meta_df = meta_df[meta_df["item_id"].isin(valid_items)]
meta_df = meta_df.drop_duplicates(subset="item_id", keep="first")

meta_df.to_csv(f"{BASE}/processed/preprocessed_meta.csv", index=False)

print("Done preprocessing meta!")
print(meta_df.shape)