from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2x_tgnn.data import (
    RealTemporalGraphDataset,
    TemporalGraphDataset,
    load_temporal_v2x_csv,
    load_v2x_csv,
    prepare_splits,
    prepare_temporal_splits,
)
from v2x_tgnn.train import TrainConfig, evaluate_model, save_training_artifacts, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Temporal GNN prototype for V2X latency prediction.",
    )
    parser.add_argument("--data", default="data/simulated_v2x_dataset.csv")
    parser.add_argument(
        "--dataset-type",
        choices=["simulated", "temporal"],
        default="simulated",
        help="Use 'temporal' for canonical real V2X traces such as SEE-V2X.",
    )
    parser.add_argument(
        "--model-type",
        choices=["tgnn", "gru", "mlp"],
        default="tgnn",
        help="Model to train: tgnn, gru baseline, or mlp baseline.",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--time-bin-ms", type=int, default=100)
    parser.add_argument("--min-valid-targets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1222)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_type == "temporal":
        frame = load_temporal_v2x_csv(str(ROOT / args.data), time_bin_ms=args.time_bin_ms)
        splits = prepare_temporal_splits(frame, seed=args.seed)
        dataset_cls = RealTemporalGraphDataset
        dataset_kwargs = {
            "seq_len": args.seq_len,
            "num_nodes": args.num_nodes,
            "stride": args.stride,
            "min_valid_targets": args.min_valid_targets,
        }
    else:
        frame = load_v2x_csv(str(ROOT / args.data))
        splits = prepare_splits(frame, seed=args.seed)
        dataset_cls = TemporalGraphDataset
        dataset_kwargs = {
            "seq_len": args.seq_len,
            "num_nodes": args.num_nodes,
            "stride": args.stride,
        }

    train_dataset = dataset_cls(
        splits.train,
        splits.feature_columns,
        splits.target_mean,
        splits.target_std,
        **dataset_kwargs,
    )
    val_dataset = dataset_cls(
        splits.val,
        splits.feature_columns,
        splits.target_mean,
        splits.target_std,
        **dataset_kwargs,
    )
    test_dataset = dataset_cls(
        splits.test,
        splits.feature_columns,
        splits.target_mean,
        splits.target_std,
        **dataset_kwargs,
    )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model, history = train_model(
        train_dataset,
        val_dataset,
        config,
        model_type=args.model_type,
    )
    metrics, predictions = evaluate_model(model, test_dataset, config)
    save_training_artifacts(ROOT / args.output_dir, history, metrics, predictions)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": splits.feature_columns,
            "target_mean": splits.target_mean,
            "target_std": splits.target_std,
            "seq_len": args.seq_len,
            "num_nodes": args.num_nodes,
            "stride": args.stride,
            "dataset_type": args.dataset_type,
            "model_type": args.model_type,
            "time_bin_ms": args.time_bin_ms,
        },
        ROOT / args.output_dir / f"{args.model_type}_model.pt",
    )

    print(f"{args.model_type} training complete")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
