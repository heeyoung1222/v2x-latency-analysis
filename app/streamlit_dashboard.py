from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


st.set_page_config(
    page_title="V2X Latency Research Dashboard",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


@st.cache_data(show_spinner=False)
def load_predictions(run_dir: str, max_rows: int = 4000) -> pd.DataFrame:
    frame = read_csv(f"results/{run_dir}/deep_learning_test_predictions.csv")
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=1222)
    return frame


def format_metric(value: float, suffix: str = "", precision: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def run_label_to_dir(label: str) -> str:
    return {
        "Temporal GNN / 50 ms": "see_v2x_4tx_full_thr50_real",
        "Temporal GNN / 80 ms": "see_v2x_4tx_full_thr80_real",
        "Temporal GNN / 100 ms": "see_v2x_4tx_full_thr100_real",
        "GRU baseline / 50 ms": "see_v2x_4tx_full_thr50_gru",
        "MLP baseline / 50 ms": "see_v2x_4tx_full_thr50_mlp",
    }[label]


def main() -> None:
    model_summary = read_csv("results/model_comparison_summary.csv")
    tx_summary = read_csv("results/tx_complexity_summary.csv")

    st.title("V2X Latency Prediction Dashboard")
    st.caption("SEE-V2X C-V2X receiver traces · Temporal GNN · latency risk detection")

    run_label = st.sidebar.selectbox(
        "Run",
        [
            "Temporal GNN / 50 ms",
            "Temporal GNN / 80 ms",
            "Temporal GNN / 100 ms",
            "GRU baseline / 50 ms",
            "MLP baseline / 50 ms",
        ],
    )
    run_dir = run_label_to_dir(run_label)
    metrics = read_csv(f"results/{run_dir}/deep_learning_metrics.csv").iloc[0]
    predictions = load_predictions(run_dir)
    sweep = read_csv(f"results/{run_dir}/risk_threshold_sweep.csv")

    show_top_metrics(metrics)

    tab_overview, tab_models, tab_thresholds, tab_tx = st.tabs(
        ["Overview", "Models", "Thresholds", "Tx Complexity"]
    )

    with tab_overview:
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(prediction_scatter(predictions), use_container_width=True)
        with right:
            st.plotly_chart(error_histogram(predictions), use_container_width=True)

    with tab_models:
        filtered = model_summary[
            model_summary["experiment"].eq("SEE-V2X 4_tx full, 50ms risk")
            & model_summary["model"].isin(
                ["Temporal GNN", "GRU baseline", "MLP baseline"]
            )
        ].copy()
        col_a, col_b = st.columns([1.05, 1])
        with col_a:
            st.plotly_chart(model_metric_chart(filtered), use_container_width=True)
        with col_b:
            st.dataframe(
                filtered[
                    [
                        "model",
                        "mae_ms",
                        "rmse_ms",
                        "r2",
                        "best_threshold_f1",
                        "p95_absolute_error_ms",
                    ]
                ].round(3),
                use_container_width=True,
                hide_index=True,
            )

    with tab_thresholds:
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            st.plotly_chart(threshold_curve(sweep), use_container_width=True)
        with col_b:
            best = sweep.sort_values("f1", ascending=False).iloc[0]
            threshold_table = pd.DataFrame(
                [
                    ["Best threshold", format_metric(best["threshold"], precision=2)],
                    ["Precision", format_metric(best["precision"])],
                    ["Recall", format_metric(best["recall"])],
                    ["F1", format_metric(best["f1"])],
                    ["Accuracy", format_metric(best["accuracy"])],
                ],
                columns=["Metric", "Value"],
            )
            st.dataframe(threshold_table, use_container_width=True, hide_index=True)

    with tab_tx:
        col_a, col_b = st.columns([1.1, 1])
        with col_a:
            st.plotly_chart(tx_complexity_chart(tx_summary), use_container_width=True)
        with col_b:
            st.dataframe(
                tx_summary[
                    [
                        "tx_setting",
                        "rows",
                        "scenarios",
                        "links",
                        "mae_ms",
                        "rmse_ms",
                        "r2",
                        "best_threshold_f1",
                    ]
                ].round(3),
                use_container_width=True,
                hide_index=True,
            )


def show_top_metrics(metrics: pd.Series) -> None:
    cols = st.columns(6)
    cols[0].metric("MAE", format_metric(metrics["mae_ms"], " ms", 2))
    cols[1].metric("RMSE", format_metric(metrics["rmse_ms"], " ms", 2))
    cols[2].metric("R2", format_metric(metrics["r2"]))
    cols[3].metric("Risk F1", format_metric(metrics["risk_f1"]))
    cols[4].metric("Best F1", format_metric(metrics["best_threshold_f1"]))
    cols[5].metric("p95 Error", format_metric(metrics["p95_absolute_error_ms"], " ms", 2))


def prediction_scatter(frame: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        frame,
        x="actual_latency_ms",
        y="predicted_latency_ms",
        color="predicted_high_latency_probability",
        color_continuous_scale="Viridis",
        labels={
            "actual_latency_ms": "Actual latency (ms)",
            "predicted_latency_ms": "Predicted latency (ms)",
            "predicted_high_latency_probability": "Risk probability",
        },
        title="Predicted vs Actual Latency",
        opacity=0.78,
    )
    lower = min(frame["actual_latency_ms"].min(), frame["predicted_latency_ms"].min())
    upper = max(frame["actual_latency_ms"].max(), frame["predicted_latency_ms"].max())
    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            line={"color": "#c51b29", "width": 2},
            name="Ideal",
        )
    )
    fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 55, "b": 10})
    return fig


def error_histogram(frame: pd.DataFrame) -> go.Figure:
    errors = frame["predicted_latency_ms"] - frame["actual_latency_ms"]
    fig = px.histogram(
        x=errors,
        nbins=45,
        labels={"x": "Prediction error (ms)", "y": "Count"},
        title="Prediction Error Distribution",
    )
    fig.add_vline(x=0, line_color="#c51b29", line_width=2)
    fig.update_traces(marker_color="#2f6f9f")
    fig.update_layout(height=520, margin={"l": 10, "r": 10, "t": 55, "b": 10})
    return fig


def model_metric_chart(frame: pd.DataFrame) -> go.Figure:
    melted = frame.melt(
        id_vars="model",
        value_vars=["mae_ms", "rmse_ms", "best_threshold_f1"],
        var_name="metric",
        value_name="value",
    )
    labels = {
        "mae_ms": "MAE",
        "rmse_ms": "RMSE",
        "best_threshold_f1": "Best F1",
    }
    melted["metric"] = melted["metric"].map(labels)
    fig = px.bar(
        melted,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="Model Comparison",
        labels={"model": "Model", "value": "Value", "metric": "Metric"},
        color_discrete_sequence=["#2f6f9f", "#7b8da0", "#0f8b6f"],
    )
    fig.update_layout(height=480, margin={"l": 10, "r": 10, "t": 55, "b": 10})
    return fig


def threshold_curve(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, color in [
        ("precision", "#2f6f9f"),
        ("recall", "#0f8b6f"),
        ("f1", "#c51b29"),
        ("accuracy", "#7b8da0"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=frame["threshold"],
                y=frame[column],
                mode="lines+markers",
                name=column.title(),
                line={"color": color},
            )
        )
    fig.update_layout(
        title="Classification Threshold Sweep",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=500,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    return fig


def tx_complexity_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, label, color in [
        ("mae_ms", "MAE", "#2f6f9f"),
        ("r2", "R2", "#0f8b6f"),
        ("best_threshold_f1", "Best F1", "#c51b29"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=frame["tx_setting"],
                y=frame[column],
                mode="lines+markers",
                name=label,
                line={"color": color},
            )
        )
    fig.update_layout(
        title="Tx Complexity Comparison",
        xaxis_title="Tx setting",
        yaxis_title="Metric value",
        height=500,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    return fig


if __name__ == "__main__":
    main()
