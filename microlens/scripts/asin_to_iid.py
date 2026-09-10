import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/home/mathew/SRP/microlens")
EMB_DIR = BASE / "embeddings"
PROCESSED = BASE / "processed"
OUT_DIR = BASE / "proxyrca_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

item_df = pd.read_csv(PROCESSED / "item_df.csv")
item_df["item_id"] = item_df["item_id"].astype(str).str.strip()

img = np.load(EMB_DIR / "image_embeddings.npy")
txt = np.load(EMB_DIR / "text_embeddings.npy")
img_ids = np.load(EMB_DIR / "image_ids.npy", allow_pickle=True)
txt_ids = np.load(EMB_DIR / "text_ids.npy", allow_pickle=True)

img_map = {str(a).strip(): img[i] for i, a in enumerate(img_ids)}
txt_map = {str(a).strip(): txt[i] for i, a in enumerate(txt_ids)}

iid2ifeature = {}
for iid, item_id in enumerate(item_df["item_id"].tolist(), start=1):
    if item_id in img_map and item_id in txt_map:
        v = np.concatenate([img_map[item_id], txt_map[item_id]]).astype(np.float32)
    elif item_id in img_map:
        v = img_map[item_id].astype(np.float32)
    elif item_id in txt_map:
        v = txt_map[item_id].astype(np.float32)
    else:
        v = np.zeros(2048 + 768, dtype=np.float32)
    iid2ifeature[iid] = tuple(v.tolist())

with open(OUT_DIR / "iid2ifeature.pkl", "wb") as f:
    pickle.dump(iid2ifeature, f)

print("Saved:", OUT_DIR / "iid2ifeature.pkl")
print("Items:", len(iid2ifeature))