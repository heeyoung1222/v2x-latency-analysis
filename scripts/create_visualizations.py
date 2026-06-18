from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    output_dir = RESULTS / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_summary = read_csv("model_comparison_summary.csv")
    tx_summary = read_csv("tx_complexity_summary.csv")

    create_model_comparison(model_summary, output_dir / "model_comparison_bar.png")
    create_threshold_sweep(output_dir / "threshold_sweep_curve.png")
    create_prediction_scatter(output_dir / "prediction_scatter_thr50.png")
    create_tx_complexity(tx_summary, output_dir / "tx_complexity_dashboard.png")

    print(f"Wrote presentation figures to {output_dir}")


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def create_model_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    comparison = frame[
        frame["experiment"].eq("SEE-V2X 4_tx full, 50ms risk")
        & frame["model"].isin(["Temporal GNN", "GRU baseline", "MLP baseline"])
    ].copy()
    comparison = comparison.set_index("model").loc[
        ["MLP baseline", "GRU baseline", "Temporal GNN"]
    ].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#7b8da0", "#2f6f9f", "#0f8b6f"]
    metrics = [
        ("mae_ms", "MAE (ms)", False),
        ("r2", "R2", True),
        ("best_threshold_f1", "Best F1", True),
    ]
    for axis, (column, title, higher_is_better) in zip(axes, metrics):
        axis.bar(comparison["model"], comparison[column], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(comparison[column]):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        if not higher_is_better:
            axis.set_ylabel("lower is better")
    fig.suptitle("Model Comparison on SEE-V2X 4_tx, 50 ms Risk")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_threshold_sweep(output_path: Path) -> None:
    runs = {
        "50 ms": RESULTS / "see_v2x_4tx_full_thr50_real" / "risk_threshold_sweep.csv",
        "80 ms": RESULTS / "see_v2x_4tx_full_thr80_real" / "risk_threshold_sweep.csv",
        "100 ms": RESULTS / "see_v2x_4tx_full_thr100_real" / "risk_threshold_sweep.csv",
    }
    fig, axis = plt.subplots(figsize=(8, 4.6))
    for label, path in runs.items():
        if not path.exists():
            continue
        sweep = pd.read_csv(path)
        axis.plot(sweep["threshold"], sweep["f1"], marker="o", label=label)
    axis.set_title("Risk Threshold Sweep")
    axis.set_xlabel("Classification threshold")
    axis.set_ylabel("F1-score")
    axis.grid(alpha=0.25)
    axis.legend(title="Risk definition")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_prediction_scatter(output_path: Path) -> None:
    path = RESULTS / "see_v2x_4tx_full_thr50_real" / "deep_learning_test_predictions.csv"
    predictions = pd.read_csv(path)
    if len(predictions) > 2500:
        predictions = predictions.sample(2500, random_state=1222)

    fig, axis = plt.subplots(figsize=(6, 5.4))
    scatter = axis.scatter(
        predictions["actual_latency_ms"],
        predictions["predicted_latency_ms"],
        c=predictions["predicted_high_latency_probability"],
        cmap="viridis",
        s=18,
        alpha=0.78,
    )
    lower = min(
        predictions["actual_latency_ms"].min(),
        predictions["predicted_latency_ms"].min(),
    )
    upper = max(
        predictions["actual_latency_ms"].max(),
        predictions["predicted_latency_ms"].max(),
    )
    axis.plot([lower, upper], [lower, upper], color="#c51b29", linewidth=2)
    axis.set_title("Temporal GNN Prediction Scatter")
    axis.set_xlabel("Actual latency (ms)")
    axis.set_ylabel("Predicted latency (ms)")
    fig.colorbar(scatter, ax=axis, label="High-latency probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_tx_complexity(frame: pd.DataFrame, output_path: Path) -> None:
    plot_data = frame.sort_values("links")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    specs = [
        ("mae_ms", "Latency MAE", "ms", "#1f77b4"),
        ("r2", "Regression R2", "", "#2ca02c"),
        ("best_threshold_f1", "Best Risk F1", "", "#d62728"),
    ]
    for axis, (column, title, ylabel, color) in zip(axes, specs):
        axis.plot(plot_data["tx_setting"], plot_data[column], marker="o", color=color)
        axis.set_title(title)
        axis.set_xlabel("Tx setting")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    fig.suptitle("SEE-V2X Tx Complexity Comparison (50 ms Risk)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Visualization failed: {exc}", file=sys.stderr)
        raise
