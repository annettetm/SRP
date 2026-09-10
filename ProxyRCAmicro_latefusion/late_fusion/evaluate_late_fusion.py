import gc
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/mathew/SRP/ProxyRCAmicro_latefusion")

if PROJECT_ROOT != EXPECTED_ROOT:
    raise RuntimeError(
        f"Safety stop: expected {EXPECTED_ROOT}, got {PROJECT_ROOT}"
    )

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import EvalDataset
from models import Carca
from solvers.carca import _load_feature_matrix
from tools.metrics import METRIC_NAMES, calc_batch_rec_metrics_per_k


MODES = ("id", "image", "text")
NORMALIZATIONS = ("raw", "zscore", "minmax", "rank")
KS_TEST = [1, 5, 10, 20, 50, 100]
BASELINE_TOLERANCE = 1e-4

SCORES_DIR = PROJECT_ROOT / "late_fusion" / "scores"
RESULTS_DIR = PROJECT_ROOT / "late_fusion" / "results"

REFERENCE_RESULTS = {
    "id": PROJECT_ROOT
    / "runs/micro_id/proxyrca/data/results_mean.json",
    "image": PROJECT_ROOT
    / "runs/micro_image/proxyrca/data/results_mean.json",
    "text": PROJECT_ROOT
    / "runs/micro_text/proxyrca/data/results_mean.json",
}


def load_config(mode):
    path = (
        PROJECT_ROOT
        / "runs"
        / f"micro_{mode}"
        / "proxyrca"
        / "config-lock.json"
    )

    with path.open("r") as file:
        config = json.load(file)

    if config["model"]["feature_mode"] != mode:
        raise RuntimeError(
            f"Mode mismatch for {mode}: "
            f"{config['model']['feature_mode']}"
        )

    # Replace old paths in memory only.
    config["envs"]["RUN_ROOT"] = str(PROJECT_ROOT / "runs")
    config["envs"]["DATA_ROOT"] = str(PROJECT_ROOT / "data")
    config["envs"]["RAW_ROOT"] = str(PROJECT_ROOT / "raw")
    config["run_dir"] = str(
        PROJECT_ROOT / "runs" / f"micro_{mode}" / "proxyrca"
    )
    return config


def build_dataset(target, config):
    dataloader_config = config["dataloader"]

    kwargs = {
        "name": "micro_id",
        "target": target,
        "sequence_len": dataloader_config["sequence_len"],
    }

    if target == "valid":
        kwargs["valid_num_negatives"] = dataloader_config[
            "valid_num_negatives"
        ]
        kwargs["random_seed"] = dataloader_config["random_seed"]

    return EvalDataset(**kwargs)


def build_loader(target, config):
    dataset = build_dataset(target, config)
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    return dataset, loader


def build_model(mode, config, icontext_dim, device):
    model_config = config["model"]
    data_dir = PROJECT_ROOT / "data" / f"micro_{mode}"

    with (data_dir / "iid2iindex.pkl").open("rb") as file:
        iid2iindex = pickle.load(file)

    num_items = len(iid2iindex)

    if mode == "id":
        item_features = None
        item_feature_dim = 0
    else:
        item_features = _load_feature_matrix(str(data_dir), mode)
        item_feature_dim = item_features.shape[1]

    num_known_item = model_config["num_known_item"]
    if isinstance(num_known_item, float):
        num_known_item = int(num_items * num_known_item)

    model = Carca(
        num_items=num_items,
        ifeatures=item_features,
        ifeature_dim=item_feature_dim,
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


def collect_scores(mode, target, device, reference_metadata=None):
    config = load_config(mode)
    dataset, loader = build_loader(target, config)
    model = build_model(mode, config, dataset.icontext_dim, device)

    score_parts = []
    user_parts = []
    candidate_parts = []
    label_parts = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["profile_tokens"].to(device),
                batch["profile_icontexts"].to(device),
                batch["extract_tokens"].to(device),
                batch["extract_icontexts"].to(device),
            )

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"{target}/{mode}: non-finite logits detected"
                )

            score_parts.append(logits.detach().cpu().numpy().astype(np.float32))
            user_parts.append(batch["uindex"].cpu().numpy())
            candidate_parts.append(batch["extract_tokens"].cpu().numpy())
            label_parts.append(batch["labels"].cpu().numpy())

    scores = np.concatenate(score_parts, axis=0)
    metadata = {
        "users": np.concatenate(user_parts, axis=0),
        "candidates": np.concatenate(candidate_parts, axis=0),
        "labels": np.concatenate(label_parts, axis=0),
    }

    if scores.shape != metadata["labels"].shape:
        raise RuntimeError(
            f"{target}/{mode}: score/label shape mismatch: "
            f"{scores.shape} vs {metadata['labels'].shape}"
        )

    if reference_metadata is not None:
        for key in ("users", "candidates", "labels"):
            if not np.array_equal(metadata[key], reference_metadata[key]):
                raise RuntimeError(
                    f"{target}/{mode}: {key} alignment mismatch"
                )

    del model, loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores, metadata


def calculate_metrics(scores, labels, ks):
    score_tensor = torch.from_numpy(scores)
    label_tensor = torch.from_numpy(labels).long()

    _, rankers = torch.topk(
        score_tensor,
        max(ks),
        dim=1,
    )

    values = calc_batch_rec_metrics_per_k(
        rankers,
        label_tensor,
        ks,
    )

    means = {}
    for key, metric_values in values.items():
        means[key] = float(np.mean(metric_values))

    return means


def normalize_scores(scores, method):
    scores = scores.astype(np.float32, copy=False)

    if method == "raw":
        return scores

    if method == "zscore":
        means = scores.mean(axis=1, keepdims=True)
        standard_deviations = scores.std(axis=1, keepdims=True)
        standard_deviations = np.maximum(
            standard_deviations,
            np.float32(1e-8),
        )
        return (scores - means) / standard_deviations

    if method == "minmax":
        minimums = scores.min(axis=1, keepdims=True)
        maximums = scores.max(axis=1, keepdims=True)
        ranges = np.maximum(
            maximums - minimums,
            np.float32(1e-8),
        )
        return (scores - minimums) / ranges

    if method == "rank":
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rank_values = np.broadcast_to(
            np.arange(scores.shape[1]),
            scores.shape,
        )
        np.put_along_axis(ranks, order, rank_values, axis=1)

        denominator = max(scores.shape[1] - 1, 1)
        return 1.0 - ranks.astype(np.float32) / denominator

    raise ValueError(f"Unknown normalization: {method}")


def generate_weight_grid(denominator):
    for id_units in range(denominator + 1):
        for image_units in range(denominator - id_units + 1):
            text_units = denominator - id_units - image_units
            yield (
                id_units / denominator,
                image_units / denominator,
                text_units / denominator,
            )


def fuse_scores(normalized_scores, weights):
    return (
        weights[0] * normalized_scores["id"]
        + weights[1] * normalized_scores["image"]
        + weights[2] * normalized_scores["text"]
    ).astype(np.float32)


def selection_key(metrics, weights):
    return (
        metrics["NDCG@10"],
        metrics["Recall@10"],
        weights[1],
        weights[2],
        weights[0],
    )


def search_weights(valid_scores, valid_labels):
    best = None

    # Coarse search: increments of 0.05.
    for normalization in NORMALIZATIONS:
        normalized = {
            mode: normalize_scores(valid_scores[mode], normalization)
            for mode in MODES
        }

        for weights in generate_weight_grid(20):
            fused = fuse_scores(normalized, weights)
            metrics = calculate_metrics(fused, valid_labels, [10])

            candidate = {
                "normalization": normalization,
                "weights": weights,
                "metrics": metrics,
            }

            if best is None or selection_key(
                metrics,
                weights,
            ) > selection_key(
                best["metrics"],
                best["weights"],
            ):
                best = candidate

    # Fine search around the coarse optimum: increments of 0.01.
    normalization = best["normalization"]
    coarse_weights = best["weights"]
    normalized = {
        mode: normalize_scores(valid_scores[mode], normalization)
        for mode in MODES
    }

    for weights in generate_weight_grid(100):
        if any(
            abs(weights[index] - coarse_weights[index]) > 0.051
            for index in range(3)
        ):
            continue

        fused = fuse_scores(normalized, weights)
        metrics = calculate_metrics(fused, valid_labels, [10])

        if selection_key(metrics, weights) > selection_key(
            best["metrics"],
            best["weights"],
        ):
            best = {
                "normalization": normalization,
                "weights": weights,
                "metrics": metrics,
            }

    return best


def load_reference_metrics(mode):
    with REFERENCE_RESULTS[mode].open("r") as file:
        return json.load(file)


def validate_baseline(mode, calculated):
    reference = load_reference_metrics(mode)

    differences = {}
    for key in reference:
        differences[key] = abs(calculated[key] - reference[key])

    largest_difference = max(differences.values())
    if largest_difference > BASELINE_TOLERANCE:
        raise RuntimeError(
            f"{mode}: baseline reproduction failed; "
            f"largest difference={largest_difference:.8f}"
        )

    return reference, largest_difference


def save_score_file(run_id, target, mode, scores, metadata):
    path = SCORES_DIR / f"{target}_{mode}_{run_id}.npz"

    np.savez_compressed(
        path,
        scores=scores,
        users=metadata["users"],
        candidates=metadata["candidates"],
        labels=metadata["labels"],
    )

    return str(path)


def main():
    if not SCORES_DIR.is_dir() or not RESULTS_DIR.is_dir():
        raise RuntimeError(
            "Safety stop: late_fusion/scores and results must already exist"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("Safety stop: CUDA GPU is required")

    run_id = os.environ.get("SLURM_JOB_ID", "manual")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Device: {device}")
    print(f"Run ID: {run_id}")

    valid_scores = {}
    valid_metadata = None
    score_files = []

    # Validation scoring and alignment checks.
    for mode in MODES:
        scores, metadata = collect_scores(
            mode,
            "valid",
            device,
            valid_metadata,
        )

        if valid_metadata is None:
            valid_metadata = metadata

        valid_scores[mode] = scores
        score_files.append(
            save_score_file(
                run_id,
                "valid",
                mode,
                scores,
                metadata,
            )
        )

        metrics = calculate_metrics(
            scores,
            metadata["labels"],
            [10],
        )
        print(
            f"valid/{mode}: "
            f"Recall@10={metrics['Recall@10']:.6f}, "
            f"NDCG@10={metrics['NDCG@10']:.6f}"
        )

    # Select normalization and weights using validation only.
    best = search_weights(
        valid_scores,
        valid_metadata["labels"],
    )
    locked_normalization = best["normalization"]
    locked_weights = tuple(best["weights"])

    print(
        "LOCKED FROM VALIDATION: "
        f"normalization={locked_normalization}, "
        f"weights(id,image,text)={locked_weights}, "
        f"NDCG@10={best['metrics']['NDCG@10']:.6f}"
    )

    del valid_scores
    gc.collect()

    # Test scoring after weights are locked.
    test_scores = {}
    test_metadata = None
    individual_test_metrics = {}
    baseline_checks = {}

    for mode in MODES:
        scores, metadata = collect_scores(
            mode,
            "test",
            device,
            test_metadata,
        )

        if test_metadata is None:
            test_metadata = metadata

        test_scores[mode] = scores
        score_files.append(
            save_score_file(
                run_id,
                "test",
                mode,
                scores,
                metadata,
            )
        )

        metrics = calculate_metrics(
            scores,
            metadata["labels"],
            KS_TEST,
        )
        reference, largest_difference = validate_baseline(
            mode,
            metrics,
        )

        individual_test_metrics[mode] = metrics
        baseline_checks[mode] = {
            "reference": reference,
            "largest_absolute_difference": largest_difference,
            "passed": True,
        }

        print(
            f"test/{mode}: baseline reproduced; "
            f"NDCG@10={metrics['NDCG@10']:.6f}, "
            f"max_diff={largest_difference:.8f}"
        )

    # Apply the locked validation choice once to test.
    normalized_test_scores = {
        mode: normalize_scores(
            test_scores[mode],
            locked_normalization,
        )
        for mode in MODES
    }

    fused_test_scores = fuse_scores(
        normalized_test_scores,
        locked_weights,
    )
    fused_test_metrics = calculate_metrics(
        fused_test_scores,
        test_metadata["labels"],
        KS_TEST,
    )

    result = {
        "run_id": run_id,
        "project_root": str(PROJECT_ROOT),
        "selection_rule": "maximize validation NDCG@10",
        "normalization": locked_normalization,
        "weights": {
            "id": locked_weights[0],
            "image": locked_weights[1],
            "text": locked_weights[2],
        },
        "validation_metrics": best["metrics"],
        "individual_test_metrics": individual_test_metrics,
        "late_fusion_test_metrics": fused_test_metrics,
        "baseline_checks": baseline_checks,
        "score_files": score_files,
    }

    result_path = (
        RESULTS_DIR / f"late_fusion_results_{run_id}.json"
    )
    temporary_path = result_path.with_suffix(".json.tmp")

    with temporary_path.open("x") as file:
        json.dump(result, file, indent=2)

    temporary_path.replace(result_path)

    print(
        "LATE FUSION TEST: "
        f"Recall@10={fused_test_metrics['Recall@10']:.6f}, "
        f"NDCG@10={fused_test_metrics['NDCG@10']:.6f}"
    )
    print(f"Result file: {result_path}")
    print("SUCCESS: late-fusion evaluation completed.")


if __name__ == "__main__":
    main()