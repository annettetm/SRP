import pandas as pd
import numpy as np

BASE = "/home/mathew/SRP/microlens"

reviews_df = pd.read_csv(f"{BASE}/data/microlens100k/MicroLens-100k_pairs.csv")

reviews_df = reviews_df.rename(columns={"item": "item_id"})
reviews_df["item_id"] = reviews_df["item_id"].astype(str).str.strip()
reviews_df["user"] = reviews_df["user"].astype(str).str.strip()

reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df.iloc[reviews_df["timestamp"].to_numpy().argsort()]
reviews_df = reviews_df.drop_duplicates(subset=["user", "item_id"], keep="last")

user_counts = reviews_df["user"].value_counts()
valid_users = user_counts[user_counts >= 5].index
reviews_df = reviews_df[reviews_df["user"].isin(valid_users)]

reviews_df = reviews_df.reset_index(drop=True)
sort_idx = np.lexsort((reviews_df["timestamp"].to_numpy(), reviews_df["user"].to_numpy()))
reviews_df = reviews_df.iloc[sort_idx]

reviews_df.to_csv(f"{BASE}/processed/preprocessed_reviews.csv", index=False)

print("Done preprocessing reviews!")
print(reviews_df.shape)