from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = Path("/home/mathew/SRP/microlens")
PROCESSED = BASE / "processed"
EMB = BASE / "embeddings"
EMB.mkdir(parents=True, exist_ok=True)

item_df = pd.read_csv(PROCESSED / "item_df.csv")
item_df["item_id"] = item_df["item_id"].astype(str).str.strip()

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x in ["[]", "nan", "None"]:
        return ""
    return x

cols = [c for c in ["title", "text"] if c in item_df.columns]
if not cols:
    raise ValueError("No usable text columns found in item_df.csv")

item_df["text_for_embed"] = ""
for c in cols:
    item_df["text_for_embed"] = item_df["text_for_embed"] + " " + item_df[c].apply(clean_text)

item_df["text_for_embed"] = item_df["text_for_embed"].str.strip()

model = SentenceTransformer("/home/mathew/SRP/microlens/models/all-mpnet-base-v2", device="cpu")
texts = item_df["text_for_embed"].tolist()
text_embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

np.save(EMB / "text_embeddings.npy", text_embeddings)
np.save(EMB / "text_ids.npy", item_df["item_id"].to_numpy(dtype=object))

print("saved:", EMB / "text_embeddings.npy")
print("saved:", EMB / "text_ids.npy")
print("shape:", text_embeddings.shape)