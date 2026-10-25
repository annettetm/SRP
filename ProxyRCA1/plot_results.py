import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "graphs"

METHODS = {
    "ID only": ROOT / "runs/beauty1_id/proxyrca/data/results_mean.json",
    "Image only": ROOT / "runs/beauty1_image/proxyrca/data/results_mean.json",
    "Text only": ROOT / "runs/beauty1_text/proxyrca/data/results_mean.json",
    "Fusion": ROOT / "runs/beauty1_fusion/proxyrca/data/results_mean.json",
    "2-way gated fusion": ROOT / "runs/gated_ID/proxyrca/data/results_mean.json",
    "3-way gated fusion": ROOT / "runs/gated_threeway/proxyrca/data/results_mean.json",
}

COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8a5a44"]


def load_results():
    results = {}
    for method, path in METHODS.items():
        if not path.is_file():
            print(f"Skipping missing result: {path}")
            continue
        with path.open() as file:
            results[method] = json.load(file)
    if not results:
        raise FileNotFoundError("No results_mean.json files were found.")
    return results


def metric_values(results, metric):
    ks = sorted(
        int(key.split("@")[1])
        for key in next(iter(results.values()))
        if key.startswith(f"{metric}@")
    )
    values = {
        method: [data[f"{metric}@{k}"] for k in ks]
        for method, data in results.items()
    }
    return ks, values


def plot_metric_curves(results, metric):
    ks, values = metric_values(results, metric)
    figure, axis = plt.subplots(figsize=(9, 5.5))

    for color, (method, method_values) in zip(COLORS, values.items()):
        axis.plot(ks, method_values, marker="o", linewidth=2.2, label=method, color=color)

    axis.set_title(f"{metric} across recommendation cutoffs")
    axis.set_xlabel("Top-K recommendations")
    axis.set_ylabel(metric)
    axis.set_xticks(ks)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{metric.lower()}_curves.png", dpi=300)
    plt.close(figure)


def plot_at_ten(results):
    metric_names = ["Recall", "NDCG"]
    methods = list(results)
    x = np.arange(len(methods))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 5.5))

    for offset, metric, color in [(-width / 2, "Recall", "#264653"),
                                  (width / 2, "NDCG", "#e76f51")]:
        values = [results[method][f"{metric}@10"] for method in methods]
        axis.bar(x + offset, values, width, label=metric, color=color)

    axis.set_title("Method comparison at Top-10")
    axis.set_ylabel("Score")
    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=20, ha="right")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "method_comparison_at_10.png", dpi=300)
    plt.close(figure)


def plot_heatmap(results):
    methods = list(results)
    labels = ["Recall@10", "NDCG@10"]
    matrix = np.array([
        [results[method][label] for label in labels]
        for method in methods
    ])
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto")

    axis.set_title("Top-10 metric heatmap")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(methods)), methods)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3f}",
                      ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis, label="Score")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "top10_heatmap.png", dpi=300)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = load_results()
    plot_metric_curves(results, "Recall")
    plot_metric_curves(results, "NDCG")
    plot_at_ten(results)
    plot_heatmap(results)
    print(f"Saved graphs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
