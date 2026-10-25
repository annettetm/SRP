#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EvalDataset
from models.carca_gated import CarcaGated
from solvers.carca import _load_feature_matrix


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_POPULARITY_BUCKETS = 3


def find_run_dir() -> str:
    """Return the most likely trained gated run directory."""
    candidates = [
        os.path.join("runs", "gated_threeway", "proxyrca"),
        os.path.join("runs", "gated_ID", "proxyrca"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config-lock.json")):
            return candidate

    # Fallback: search for a run with a config-lock and a model checkpoint.
    runs_root = os.path.join(os.getcwd(), "runs")
    if not os.path.isdir(runs_root):
        raise FileNotFoundError("No runs/ directory found in the current project root.")

    for root, _, files in os.walk(runs_root):
        if "config-lock.json" not in files:
            continue
        run_dir = root
        config_path = os.path.join(run_dir, "config-lock.json")
        with open(config_path, "r") as fp:
            config = json.load(fp)
        model_cfg = config.get("model", {})
        feature_mode = model_cfg.get("feature_mode")
        if feature_mode in {"three_way", "fusion", "image", "text"}:
            pth_path = os.path.join(run_dir, "pths", "model.pth")
            if os.path.isfile(pth_path):
                return run_dir

    raise FileNotFoundError(
        "Could not locate a trained gated run. Set RUN_DIR inside this script or create the expected run directory."
    )




def parse_args():
    parser = argparse.ArgumentParser(description="Analyze learned modality gates for gated Carca runs.")
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Optional run dirs to analyze, e.g. runs/gated_ID/proxyrca runs/gated_threeway/proxyrca",
    )
    return parser.parse_args()


ARGS = parse_args()
RUN_DIRS = ARGS.run_dirs if ARGS.run_dirs else [find_run_dir()]


def load_config(run_dir):
    with open(os.path.join(run_dir, "config-lock.json"), "r") as fp:
        return json.load(fp)


def build_model(config, num_items, icontext_dim, data_dir, run_dir):
    CM = config["model"]
    feature_mode = CM.get("feature_mode", "fusion")
    num_known_item = CM["num_known_item"]
    if isinstance(num_known_item, float):
        num_known_item = int(num_items * num_known_item)

    if feature_mode == "id":
        ifeatures, ifeature_dim = None, 0
        image_features, text_features = None, None
    elif feature_mode == "three_way":
        ifeatures, ifeature_dim = None, 0
        image_features = _load_feature_matrix(data_dir, "image")
        text_features = _load_feature_matrix(data_dir, "text")
    else:
        ifeatures = _load_feature_matrix(data_dir, feature_mode)
        ifeature_dim = ifeatures.shape[1]
        image_features, text_features = None, None

    model = CarcaGated(
        num_items=num_items,
        ifeatures=ifeatures,
        ifeature_dim=ifeature_dim,
        image_features=image_features,
        text_features=text_features,
        icontext_dim=icontext_dim,
        hidden_dim=CM["hidden_dim"],
        num_known_item=num_known_item,
        num_layers=CM["num_layers"],
        num_heads=CM["num_heads"],
        dropout_prob=0.0,
        random_seed=CM["random_seed"],
        feature_mode=feature_mode,
        id_emb_dim=CM.get("id_emb_dim", 64),
    )

    state_path = os.path.join(run_dir, "pths", "model.pth")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Model checkpoint not found at {state_path}")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, feature_mode


@torch.no_grad()
def three_way_gate_weights(model, tokens, icontexts):
    """Return per-modality softmax gate weights: [image, text, id]."""
    assert model.feature_mode == "three_way"

    image_vec = model.image_encoder(tokens, icontexts)
    text_vec = model.text_encoder(tokens, icontexts)

    id_tokens = tokens.clamp(min=0, max=model.num_items)
    id_vec = model.id_proj(model.item_id_emb(id_tokens))

    gate_input = torch.cat([image_vec, text_vec, id_vec], dim=-1)
    weights = torch.softmax(model.three_way_gate_proj(gate_input), dim=-1)
    return weights


@torch.no_grad()
def two_way_gate_weights(model, tokens, icontexts):
    """Return [content, id] gate weights for fused or single-modality content gates."""
    assert model.use_id and model.use_content and model.feature_mode != "three_way"

    content_vec = model.item_encoder(tokens, icontexts)
    id_tokens = tokens.clamp(min=0, max=model.num_items)
    id_vec = model.id_proj(model.item_id_emb(id_tokens))

    gate = torch.sigmoid(model.gate_proj(torch.cat([content_vec, id_vec], dim=-1)))
    content_weight = gate.mean(dim=-1, keepdim=True)
    id_weight = 1.0 - content_weight
    return torch.cat([content_weight, id_weight], dim=-1)


def compute_item_popularity(eval_ds):
    counts = defaultdict(int)
    for _, urows in eval_ds.uindex2urows_train.items():
        for iindex, _, _ in urows:
            counts[iindex] += 1
    return counts


def assign_popularity_buckets(counts, n_buckets=3):
    labels = ["cold", "mid", "popular"] if n_buckets == 3 else [f"bucket_{i}" for i in range(n_buckets)]
    values = np.array(sorted(counts.values()), dtype=np.float64)
    if len(values) == 0:
        return lambda iindex: "unseen"
    if len(values) == 1:
        edges = np.array([0.0, float(values[0]), float(values[0])])
    else:
        edges = np.quantile(values, np.linspace(0, 1, n_buckets + 1))

    def bucket_for(iindex):
        c = counts.get(iindex, 0)
        if c == 0:
            return "unseen"
        idx = np.searchsorted(edges[1:-1], c, side="right")
        idx = min(idx, n_buckets - 1)
        return labels[idx]

    return bucket_for


def analyze_run(run_dir):
    config = load_config(run_dir)
    data_dir = os.path.join(config["envs"]["DATA_ROOT"], config["dataset"])

    with open(os.path.join(data_dir, "iid2iindex.pkl"), "rb") as fp:
        num_items = len(pickle.load(fp))

    eval_ds = EvalDataset(
        name=config["dataset"],
        target="test",
        sequence_len=config["dataloader"]["sequence_len"],
    )
    # The evaluation set has variable-length negatives per user, so batching must
    # be single-sample to avoid torch.stack() errors on mismatched `extract_tokens`.
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)

    model, feature_mode = build_model(config, num_items, eval_ds.icontext_dim, data_dir, run_dir)

    if feature_mode == "three_way":
        gate_fn = three_way_gate_weights
        modality_names = ["image", "text", "id"]
    elif feature_mode not in {"id"}:
        gate_fn = two_way_gate_weights
        modality_names = ["content", "id"]
    else:
        raise ValueError(
            f"feature_mode='{feature_mode}' has no gate; this analysis is only for gated architectures."
        )

    print(f"Detected feature_mode='{feature_mode}' -> modalities: {modality_names}")
    print(f"Using trained run: {run_dir}")

    pop_counts = compute_item_popularity(eval_ds)
    bucket_for = assign_popularity_buckets(pop_counts, N_POPULARITY_BUCKETS)

    item_gate_sum = defaultdict(lambda: np.zeros(len(modality_names), dtype=np.float64))
    item_gate_count = defaultdict(int)

    print("Computing item gate weights...")
    for batch in loader:
        tokens = batch["extract_tokens"].to(DEVICE)
        icontexts = batch["extract_icontexts"].to(DEVICE)

        weights = gate_fn(model, tokens, icontexts).cpu().numpy()
        flat_tokens = tokens.cpu().numpy().reshape(-1)
        flat_weights = weights.reshape(-1, len(modality_names))

        for iindex, gate in zip(flat_tokens, flat_weights):
            iindex = int(iindex)
            if iindex <= 0:
                continue
            item_gate_sum[iindex] += gate
            item_gate_count[iindex] += 1

    item_mean_weights = {
        iindex: item_gate_sum[iindex] / item_gate_count[iindex]
        for iindex in item_gate_sum
    }
    print(f"Collected gate weights for {len(item_mean_weights)} unique items.")

    bucket_order = ["unseen", "cold", "mid", "popular"]
    bucket_weights = {b: [] for b in bucket_order}
    for iindex, w in item_mean_weights.items():
        bucket_weights[bucket_for(iindex)].append(w)

    by_bucket_modality = {b: {m: [] for m in modality_names} for b in bucket_order}
    for iindex, w in item_mean_weights.items():
        bucket = bucket_for(iindex)
        for i, modality in enumerate(modality_names):
            by_bucket_modality.setdefault(bucket, {m: [] for m in modality_names})
            by_bucket_modality[bucket][modality].append(float(w[i]))

    present_buckets = []
    for b in bucket_order:
        if any(by_bucket_modality[b][m] for m in modality_names):
            present_buckets.append(b)

    out_csv = os.path.join(run_dir, "gate_weights_by_item.csv")
    with open(out_csv, "w") as fp:
        fp.write("iindex,popularity_bucket," + ",".join(modality_names) + "\n")
        for iindex, w in sorted(item_mean_weights.items()):
            row = ",".join(f"{v:.6f}" for v in w)
            fp.write(f"{iindex},{bucket_for(iindex)},{row}\n")
    print(f"Saved per-item gate weights to {out_csv}")

    heatmap_data = np.array([
        [np.median(by_bucket_modality[b][m]) if by_bucket_modality[b][m] else np.nan for m in modality_names]
        for b in present_buckets
    ])

    fig, ax = plt.subplots(figsize=(max(5, len(modality_names) * 2.2), 3.2 + 0.8 * len(present_buckets)))
    threshold = 1.0 / len(modality_names)
    cmap = plt.get_cmap("coolwarm")
    im = ax.imshow(heatmap_data, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(modality_names)))
    ax.set_xticklabels(modality_names)
    ax.set_yticks(np.arange(len(present_buckets)))
    ax.set_yticklabels(present_buckets)

    for i in range(len(present_buckets)):
        for j in range(len(modality_names)):
            val = heatmap_data[i, j]
            if np.isnan(val):
                text = "NA"
                color = "black"
            else:
                text = f"{val:.2f}"
                color = "white" if val < 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    ax.axhline(-0.5, color="black", linewidth=0)
    ax.axvline(-0.5, color="black", linewidth=0)
    ax.set_title(f"Median gate weight by popularity bucket ({feature_mode})")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Popularity bucket")
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Median gate weight")
    fig.tight_layout()

    out_png = os.path.join(run_dir, "gate_weight_analysis.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved plot to {out_png}")


def main():
    for run_dir in RUN_DIRS:
        analyze_run(run_dir)


if __name__ == "__main__":
    main()
