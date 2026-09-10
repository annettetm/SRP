import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Paths
# ----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
emb_dir = os.path.join(project_root, "embeddings")
proxy_dir = os.path.join(project_root, "proxyrca_data")
out_dir = os.path.join(project_root, "embedding_quality_out")
os.makedirs(out_dir, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
image_embeddings = np.load(os.path.join(emb_dir, "image_embeddings.npy"))
text_embeddings = np.load(os.path.join(emb_dir, "text_embeddings.npy"))
image_asins = np.load(os.path.join(emb_dir, "image_asins.npy"), allow_pickle=True)
text_asins = np.load(os.path.join(emb_dir, "text_asins.npy"), allow_pickle=True)

def decode_asins(arr):
    return np.array([
        a.decode() if isinstance(a, (bytes, np.bytes_)) else str(a)
        for a in arr
    ])

image_asins = decode_asins(image_asins)
text_asins = decode_asins(text_asins)

with open(os.path.join(proxy_dir, "iid2iindex.pkl"), "rb") as f:
    iid2index = pickle.load(f)

with open(os.path.join(proxy_dir, "iid2imagefeature.pkl"), "rb") as f:
    iid2imagefeature = pickle.load(f)

with open(os.path.join(proxy_dir, "iid2textfeature.pkl"), "rb") as f:
    iid2textfeature = pickle.load(f)

# ----------------------------
# Basic checks
# ----------------------------
print("image_embeddings:", image_embeddings.shape)
print("text_embeddings:", text_embeddings.shape)
print("image_asins:", len(image_asins))
print("text_asins:", len(text_asins))

assert image_embeddings.shape[0] == len(image_asins), "Mismatch: image embeddings vs ASINs"
assert text_embeddings.shape[0] == len(text_asins), "Mismatch: text embeddings vs ASINs"

# ----------------------------
# Helpers
# ----------------------------
def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def topk_neighbors(emb, idx, k=5):
    sims = cosine_similarity(emb[idx:idx+1], emb).ravel()
    order = np.argsort(-sims)
    order = [o for o in order if o != idx][:k]
    return order, sims[order]

# Normalize for same-modality similarity search
image_embeddings_n = l2_normalize(image_embeddings)
text_embeddings_n = l2_normalize(text_embeddings)

# ----------------------------
# Same-modality neighbor checks
# ----------------------------
sample_n = min(20, len(image_asins), len(text_asins))
sample_image_idx = np.linspace(0, len(image_asins) - 1, num=min(10, len(image_asins)), dtype=int)
sample_text_idx = np.linspace(0, len(text_asins) - 1, num=min(10, len(text_asins)), dtype=int)

rows = []

for idx in sample_image_idx:
    asin = image_asins[idx]
    nn_idx, nn_sim = topk_neighbors(image_embeddings_n, idx, k=5)
    for rank, (j, sim) in enumerate(zip(nn_idx, nn_sim), start=1):
        rows.append({
            "query_modality": "image",
            "query_asin": asin,
            "rank": rank,
            "neighbor_asin": image_asins[j],
            "similarity": float(sim)
        })

for idx in sample_text_idx:
    asin = text_asins[idx]
    nn_idx, nn_sim = topk_neighbors(text_embeddings_n, idx, k=5)
    for rank, (j, sim) in enumerate(zip(nn_idx, nn_sim), start=1):
        rows.append({
            "query_modality": "text",
            "query_asin": asin,
            "rank": rank,
            "neighbor_asin": text_asins[j],
            "similarity": float(sim)
        })

neighbor_df = pd.DataFrame(rows)
neighbor_df.to_csv(os.path.join(out_dir, "nearest_neighbors_sample.csv"), index=False)

# ----------------------------
# Cross-modal alignment only if dimensions match
# ----------------------------
common_asins = [a for a in image_asins if a in set(text_asins)]
cross_rows = []

if image_embeddings.shape[1] == text_embeddings.shape[1]:
    text_map = {a: i for i, a in enumerate(text_asins)}
    image_map = {a: i for i, a in enumerate(image_asins)}

    for asin in common_asins[:min(5000, len(common_asins))]:
        i = image_map[asin]
        j = text_map[asin]
        cos = float(cosine_similarity(
            image_embeddings_n[i:i+1],
            text_embeddings_n[j:j+1]
        )[0, 0])
        cross_rows.append({
            "asin": asin,
            "img_text_cos": cos,
            "img_norm": float(np.linalg.norm(image_embeddings[i])),
            "txt_norm": float(np.linalg.norm(text_embeddings[j]))
        })

    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(os.path.join(out_dir, "cross_modal_alignment.csv"), index=False)
    print("Cross-modal cosine computed.")
    print(cross_df["img_text_cos"].describe())
else:
    print(
        f"Skipping cross-modal cosine: image dim {image_embeddings.shape[1]} != text dim {text_embeddings.shape[1]}"
    )
    print("To compare them, project both to the same dimension first.")

# ----------------------------
# Summary
# ----------------------------
summary = pd.DataFrame([{
    "n_image": image_embeddings.shape[0],
    "n_text": text_embeddings.shape[0],
    "dim_image": image_embeddings.shape[1],
    "dim_text": text_embeddings.shape[1],
    "common_asins": len(common_asins),
    "mean_image_norm": float(np.mean(np.linalg.norm(image_embeddings, axis=1))),
    "mean_text_norm": float(np.mean(np.linalg.norm(text_embeddings, axis=1))),
}])

summary.to_csv(os.path.join(out_dir, "embedding_quality_summary.csv"), index=False)
print(summary.T)
print("Saved outputs in:", out_dir)