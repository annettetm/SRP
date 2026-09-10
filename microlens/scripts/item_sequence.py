import os
import pandas as pd
import numpy as np

# --- toggle this ---
INCLUDE_IMAGE_EMBEDDINGS = True # 1st run: False. 2nd run (after image/text embedding): True
# --------------------

meta_path = "/home/mathew/SRP/microlens/processed/preprocessed_meta.csv"
review_path = "/home/mathew/SRP/microlens/processed/preprocessed_reviews.csv"
image_emb_path = "/home/mathew/SRP/microlens/embeddings/image_embeddings.npy"
image_ids_path = "/home/mathew/SRP/microlens/embeddings/image_ids.npy"
out_dir = "/home/mathew/SRP/microlens/processed"

item_out = os.path.join(out_dir, "item_df.csv")
interaction_out = os.path.join(out_dir, "interaction_df.csv")
seq_out = os.path.join(out_dir, "user_sequences.csv")
item2idx_out = os.path.join(out_dir, "item2idx.npy")
user2idx_out = os.path.join(out_dir, "user2idx.npy")
item_feature_out = os.path.join(out_dir, "item_features.csv")

os.makedirs(out_dir, exist_ok=True)

meta_df = pd.read_csv(meta_path)
review_df = pd.read_csv(review_path)

meta_df["item_id"] = meta_df["item_id"].astype(str).str.strip()
review_df["item_id"] = review_df["item_id"].astype(str).str.strip()
review_df["user"] = review_df["user"].astype(str).str.strip()

interaction_df = review_df[["user", "item_id", "timestamp"]].copy()
interaction_df = interaction_df.dropna(subset=["user", "item_id", "timestamp"])
interaction_df["timestamp"] = pd.to_numeric(interaction_df["timestamp"], errors="coerce")
interaction_df = interaction_df.dropna(subset=["timestamp"])
interaction_df["timestamp"] = interaction_df["timestamp"].astype(int)

interaction_df = interaction_df.reset_index(drop=True)
sort_idx = np.lexsort((interaction_df["timestamp"].to_numpy(), interaction_df["user"].to_numpy()))
interaction_df = interaction_df.iloc[sort_idx].reset_index(drop=True)

user_sequences = (
    interaction_df.groupby("user")["item_id"]
    .apply(list)
    .reset_index(name="item_sequence")
)

users = interaction_df["user"].unique().tolist()
review_items = interaction_df["item_id"].unique().tolist()
user2idx = {u: i for i, u in enumerate(users)}

meta_subset = meta_df[meta_df["item_id"].isin(review_items)].copy()
meta_subset = meta_subset.drop_duplicates(subset="item_id").reset_index(drop=True)
item2idx = {a: i for i, a in enumerate(meta_subset["item_id"].tolist())}

interaction_df["user_idx"] = interaction_df["user"].map(user2idx)
interaction_df["item_idx"] = interaction_df["item_id"].map(item2idx)

user_sequences["user_idx"] = user_sequences["user"].map(user2idx)
user_sequences["item_sequence_idx"] = user_sequences["item_sequence"].apply(
    lambda seq: [item2idx[a] for a in seq if a in item2idx]
)

item_df = meta_subset[["item_id", "title"]].copy()
item_df["title"] = item_df["title"].fillna("")
item_df["text"] = item_df["title"].str.strip()
item_df["item_idx"] = item_df["item_id"].map(item2idx)

if INCLUDE_IMAGE_EMBEDDINGS:
    image_embeddings = np.load(image_emb_path)
    image_ids = np.load(image_ids_path, allow_pickle=True)
    emb_map = {str(a): image_embeddings[i] for i, a in enumerate(image_ids)}
    item_df["has_image_embedding"] = item_df["item_id"].isin(emb_map.keys())
    item_df["image_embedding"] = item_df["item_id"].map(lambda a: emb_map.get(a))
else:
    item_df["has_image_embedding"] = False
    item_df["image_embedding"] = None

item_df.to_csv(item_out, index=False)
interaction_df.to_csv(interaction_out, index=False)
user_sequences.to_csv(seq_out, index=False)
np.save(item2idx_out, item2idx, allow_pickle=True)
np.save(user2idx_out, user2idx, allow_pickle=True)

item_feature_df = item_df[["item_id", "item_idx", "has_image_embedding"]].copy()
item_feature_df.to_csv(item_feature_out, index=False)

print("Saved:", item_out, interaction_out, seq_out, item2idx_out, user2idx_out, item_feature_out)
print("Items:", len(item_df))
print("Interactions:", len(interaction_df))
print("Users:", len(user_sequences))
print("Missing item_idx:", interaction_df["item_idx"].isna().sum())