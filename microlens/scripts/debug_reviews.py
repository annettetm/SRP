import pandas as pd

BASE = "/home/mathew/SRP/microlens"
df = pd.read_csv(f"{BASE}/data/microlens100k/MicroLens-100k_pairs.csv")
print("loaded:", df.shape)

reviews_df = df.rename(columns={"item": "item_id"})
reviews_df["item_id"] = reviews_df["item_id"].astype(str).str.strip()
reviews_df["user"] = reviews_df["user"].astype(str).str.strip()
print("after rename/cast:", reviews_df.shape)

print("dtypes:")
print(reviews_df.dtypes)
print("timestamp NaN count:", reviews_df["timestamp"].isna().sum())
print("timestamp sample:", reviews_df["timestamp"].head().tolist())

reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df.iloc[reviews_df["timestamp"].to_numpy().argsort()]
reviews_df = reviews_df.drop_duplicates(subset=["user", "item_id"], keep="last")
print("after dedup:", reviews_df.shape)

user_counts = reviews_df["user"].value_counts()
valid_users = user_counts[user_counts >= 5].index
print("valid users count:", len(valid_users))

reviews_df = reviews_df[reviews_df["user"].isin(valid_users)]
print("after user filter:", reviews_df.shape)