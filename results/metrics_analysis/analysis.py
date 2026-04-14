import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# python -m results.metrics_analysis.analysis --metrics_path  ./results/metrics_ultrakill/summary.json


def get_metrics(
    metrics_path: Path,
) -> dict:
    metrics_path = Path(metrics_path)

    with open(metrics_path) as f:
        summary = json.load(f)
        summary = summary["conditions"]

    metrics = {
        "condition": [],
        "fad": [],
        "mean_clap_similarity": [],
        "std_clap_similarity": [],
        "mean_repetition_score": [],
        "mean_loop_ratio": [],
        "diversity_score": []
    }
    for sample_metrics in summary.values():
        for metric, value in sample_metrics.items():
            metrics[metric].append(value)
    return metrics


def analyze_metrics(
    metrics_path: Path,
) -> dict:
    metrics = get_metrics(metrics_path)

    df = pd.DataFrame(metrics)
    corr_cols = [
        "fad", 
        "mean_clap_similarity",
        "mean_repetition_score",
        "diversity_score"
    ]
    # heat map of correlations
    plt.figure(figsize=(6, 4))
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Matrix of correlations of metrics and generation features")
    plt.tight_layout()
    plt.show()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", type=str, required=True)
    args = parser.parse_args()

    analyze_metrics(
        metrics_path=Path(args.metrics_path),
    )


if __name__ == "__main__":
    main()
