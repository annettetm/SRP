import gc
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/mathew/SRP/ProxyRCAmicro_latefusion")

if PROJECT_ROOT != EXPECTED_ROOT:
    raise RuntimeError(
        f"Safety stop: expected {EXPECTED_ROOT}, got {PROJECT_ROOT}"
    )

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EvalDataset
from models import Carca
from solvers.carca import _load_feature_matrix


MODES = ("id", "image", "text")


def load_config(mode):
    path = (
        PROJECT_ROOT
        / "runs"
        / f"micro_{mode}"
        / "proxyrca"
        / "config-lock.json"
    )
    with path.open("r") as f:
        config = json.load(f)

    if config["model"]["feature_mode"] != mode:
        raise RuntimeError(
            f"Mode mismatch for {mode}: "
            f"{config['model']['feature_mode']}"
        )

    # Override copied absolute paths in memory only.
    config["envs"]["RUN_ROOT"] = str(PROJECT_ROOT / "runs")
    config["envs"]["DATA_ROOT"] = str(PROJECT_ROOT / "data")
    config["envs"]["RAW_ROOT"] = str(PROJECT_ROOT / "raw")
    config["run_dir"] = str(
        PROJECT_ROOT / "runs" / f"micro_{mode}" / "proxyrca"
    )
    return config


def build_model(mode, config, icontext_dim, device):
    model_config = config["model"]
    data_dir = PROJECT_ROOT / "data" / f"micro_{mode}"

    with (data_dir / "iid2iindex.pkl").open("rb") as f:
        import pickle
        iid2iindex = pickle.load(f)

    num_items = len(iid2iindex)

    if mode == "id":
        ifeatures = None
        ifeature_dim = 0
    else:
        ifeatures = _load_feature_matrix(str(data_dir), mode)
        ifeature_dim = ifeatures.shape[1]

    num_known_item = model_config["num_known_item"]
    if isinstance(num_known_item, float):
        num_known_item = int(num_items * num_known_item)

    model = Carca(
        num_items=num_items,
        ifeatures=ifeatures,
        ifeature_dim=ifeature_dim,
        icontext_dim=icontext_dim,
        hidden_dim=model_config["hidden_dim"],
        num_known_item=num_known_item,
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout_prob=model_config["dropout_prob"],
        random_seed=model_config["random_seed"],
        feature_mode=mode,
        id_emb_dim=model_config.get("id_emb_dim", 64),
    ).to(device)

    model_path = (
        PROJECT_ROOT
        / "runs"
        / f"micro_{mode}"
        / "proxyrca"
        / "pths"
        / "model.pth"
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Project: {PROJECT_ROOT}")
    print(f"Device: {device}")

    config_id = load_config("id")
    dataset = EvalDataset(
        name="micro_id",
        target="test",
        sequence_len=config_id["dataloader"]["sequence_len"],
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    batch = next(iter(loader))
    print(f"Shared batch users: {batch['uindex'].tolist()}")
    print(f"Candidates per user: {batch['extract_tokens'].shape[1]}")

    for mode in MODES:
        config = load_config(mode)
        model = build_model(
            mode,
            config,
            dataset.icontext_dim,
            device,
        )

        with torch.no_grad():
            logits = model(
                batch["profile_tokens"].to(device),
                batch["profile_icontexts"].to(device),
                batch["extract_tokens"].to(device),
                batch["extract_icontexts"].to(device),
            )

        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{mode}: non-finite logits detected")

        print(
            f"{mode}: shape={tuple(logits.shape)}, "
            f"min={logits.min().item():.6f}, "
            f"max={logits.max().item():.6f}"
        )

        del model, logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("SUCCESS: all three models loaded and scored the shared batch.")


if __name__ == "__main__":
    main()
