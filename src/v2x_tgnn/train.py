from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .data import TemporalGraphDataset
from .models import create_model


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    risk_loss_weight: float = 0.25
    use_auto_pos_weight: bool = True
    max_pos_weight: float = 20.0
    patience: int = 18
    seed: int = 1222
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    train_dataset: TemporalGraphDataset,
    val_dataset: TemporalGraphDataset,
    config: TrainConfig,
    model_type: str = "tgnn",
) -> tuple[nn.Module, pd.DataFrame]:
    set_seed(config.seed)
    model = create_model(model_type=model_type, input_dim=train_dataset.x.shape[-1]).to(
        config.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    regression_loss = nn.SmoothL1Loss()
    pos_weight = None
    if config.use_auto_pos_weight:
        pos_weight_value = _estimate_pos_weight(train_dataset, config.max_pos_weight)
        pos_weight = torch.tensor(pos_weight_value, device=config.device)
    risk_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            config,
            regression_loss,
            risk_loss,
            optimizer,
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            config,
            regression_loss,
            risk_loss,
            optimizer=None,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_model(
    model: nn.Module,
    dataset: TemporalGraphDataset,
    config: TrainConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    model.eval()

    predictions = []
    targets = []
    risk_probs = []
    risk_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(config.device)
            latency_pred, risk_logit = model(x)
            latency_pred = latency_pred.cpu().numpy() * dataset.target_std + dataset.target_mean
            mask = batch.get("mask", torch.ones_like(batch["risk"], dtype=torch.bool))
            mask_np = mask.numpy().astype(bool).reshape(-1)
            predictions.append(latency_pred.reshape(-1)[mask_np])
            targets.append(batch["latency_raw"].numpy().reshape(-1)[mask_np])
            risk_probs.append(torch.sigmoid(risk_logit).cpu().numpy().reshape(-1)[mask_np])
            risk_targets.append(batch["risk"].numpy().reshape(-1)[mask_np])

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    p_risk = np.concatenate(risk_probs)
    y_risk = np.concatenate(risk_targets).astype(int)
    pred_risk = (p_risk >= 0.5).astype(int)
    threshold_metrics = _threshold_sweep(y_risk, p_risk)
    best_threshold = threshold_metrics.sort_values("f1", ascending=False).iloc[0]

    metrics = {
        "mae_ms": float(mean_absolute_error(y_true, y_pred)),
        "rmse_ms": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "risk_accuracy": float(accuracy_score(y_risk, pred_risk)),
        "risk_f1": float(f1_score(y_risk, pred_risk, zero_division=0)),
        "risk_positive_rate": float(y_risk.mean()),
        "best_risk_threshold": float(best_threshold["threshold"]),
        "best_threshold_accuracy": float(best_threshold["accuracy"]),
        "best_threshold_precision": float(best_threshold["precision"]),
        "best_threshold_recall": float(best_threshold["recall"]),
        "best_threshold_f1": float(best_threshold["f1"]),
        "mean_predicted_latency_ms": float(y_pred.mean()),
        "mean_actual_latency_ms": float(y_true.mean()),
        "p95_absolute_error_ms": float(np.percentile(np.abs(y_true - y_pred), 95)),
    }
    predictions_frame = pd.DataFrame(
        {
            "actual_latency_ms": y_true,
            "predicted_latency_ms": y_pred,
            "actual_high_latency": y_risk,
            "predicted_high_latency_probability": p_risk,
        }
    )
    return metrics, predictions_frame


def save_training_artifacts(
    output_dir: Path,
    history: pd.DataFrame,
    metrics: dict[str, float],
    predictions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "deep_learning_training_history.csv", index=False)
    predictions.to_csv(output_dir / "deep_learning_test_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "deep_learning_metrics.csv", index=False)
    threshold_metrics = _threshold_sweep(
        predictions["actual_high_latency"].to_numpy(dtype=int),
        predictions["predicted_high_latency_probability"].to_numpy(dtype=float),
    )
    threshold_metrics.to_csv(output_dir / "risk_threshold_sweep.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Temporal GNN Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "deep_learning_training_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5.8, 5.8))
    plt.scatter(
        predictions["actual_latency_ms"],
        predictions["predicted_latency_ms"],
        c=predictions["predicted_high_latency_probability"],
        cmap="viridis",
        s=28,
        alpha=0.78,
    )
    lower = min(predictions["actual_latency_ms"].min(), predictions["predicted_latency_ms"].min())
    upper = max(predictions["actual_latency_ms"].max(), predictions["predicted_latency_ms"].max())
    plt.plot([lower, upper], [lower, upper], color="#cb181d", linewidth=2)
    plt.xlabel("Actual latency (ms)")
    plt.ylabel("Predicted latency (ms)")
    plt.title("Temporal GNN Latency Prediction")
    plt.colorbar(label="High-latency probability")
    plt.tight_layout()
    plt.savefig(output_dir / "deep_learning_prediction_scatter.png", dpi=150)
    plt.close()


def _run_epoch(
    model: TemporalGraphLatencyModel,
    loader: DataLoader,
    config: TrainConfig,
    regression_loss: nn.Module,
    risk_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        x = batch["x"].to(config.device)
        latency = batch["latency"].to(config.device)
        risk = batch["risk"].to(config.device)
        mask = batch.get("mask", torch.ones_like(risk, dtype=torch.bool)).to(config.device)

        if optimizer is not None:
            optimizer.zero_grad()

        latency_pred, risk_logit = model(x)
        if not bool(mask.any()):
            continue
        loss = regression_loss(latency_pred[mask], latency[mask]) + (
            config.risk_loss_weight * risk_loss(risk_logit[mask], risk[mask])
        )

        if optimizer is not None:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def _estimate_pos_weight(dataset: TemporalGraphDataset, max_pos_weight: float) -> float:
    risk = dataset.risk
    mask = getattr(dataset, "mask", torch.ones_like(risk, dtype=torch.bool))
    observed = risk[mask].float()
    positives = float(observed.sum())
    negatives = float(len(observed) - positives)
    if positives <= 0:
        return 1.0
    return float(min(negatives / positives, max_pos_weight))


def _threshold_sweep(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        predicted = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "accuracy": float(accuracy_score(y_true, predicted)),
                "precision": float(precision_score(y_true, predicted, zero_division=0)),
                "recall": float(recall_score(y_true, predicted, zero_division=0)),
                "f1": float(f1_score(y_true, predicted, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)
