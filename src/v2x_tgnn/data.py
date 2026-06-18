from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


NUMERIC_COLUMNS = [
    "speed_kmh",
    "signal_strength_dbm",
    "network_stability_index",
]

CATEGORICAL_COLUMNS = [
    "vehicle_density",
    "scheduling_algorithm",
]

TARGET_COLUMN = "latency_ms"
RISK_COLUMN = "high_latency"


@dataclass(frozen=True)
class PreparedSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    target_mean: float
    target_std: float
    temporal: bool = False


class TemporalGraphDataset(Dataset):
    """Converts tabular V2X records into small temporal graph episodes.

    The current public dataset has no vehicle IDs or timestamps, so each sample is
    a reproducible pseudo-episode made from consecutive rows after shuffling.
    When real V2X traces are available, this class can be replaced by a loader
    that groups rows by scenario, timestamp, and transmitter/receiver IDs.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        target_mean: float,
        target_std: float,
        seq_len: int = 6,
        num_nodes: int = 8,
        stride: int | None = None,
    ) -> None:
        if len(frame) < seq_len * num_nodes:
            raise ValueError(
                f"Need at least {seq_len * num_nodes} rows, got {len(frame)}."
            )
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.feature_columns = feature_columns
        self.target_mean = target_mean
        self.target_std = target_std
        self.stride = stride or num_nodes

        x = frame[feature_columns].to_numpy(dtype=np.float32)
        y = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
        risk = frame["risk_label"].to_numpy(dtype=np.float32)

        block = seq_len * num_nodes
        starts = range(0, len(frame) - block + 1, self.stride)
        x = np.stack(
            [
                x[start : start + block].reshape(seq_len, num_nodes, len(feature_columns))
                for start in starts
            ]
        ).copy()
        y = np.stack(
            [y[start : start + block].reshape(seq_len, num_nodes) for start in starts]
        ).copy()
        risk = np.stack(
            [risk[start : start + block].reshape(seq_len, num_nodes) for start in starts]
        ).copy()

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(((y[:, -1, :] - target_mean) / target_std).astype(np.float32))
        self.risk = torch.from_numpy(risk[:, -1, :].astype(np.float32))
        self.y_raw = torch.from_numpy(y[:, -1, :].astype(np.float32))
        self.mask = torch.ones_like(self.y, dtype=torch.bool)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "latency": self.y[index],
            "risk": self.risk[index],
            "latency_raw": self.y_raw[index],
            "mask": self.mask[index],
        }


class RealTemporalGraphDataset(Dataset):
    """Builds graph episodes from real V2X traces with timestamps and link IDs.

    Each node represents a communication link, for example `tx_1->rx_2`.
    Rows are grouped into fixed-width time bins and then converted into sliding
    windows. Missing observations are masked out of the loss and metrics.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        target_mean: float,
        target_std: float,
        seq_len: int = 8,
        num_nodes: int = 8,
        stride: int = 1,
        min_valid_targets: int = 1,
    ) -> None:
        required = {"scenario_id", "time_bin", "node_id", TARGET_COLUMN, "risk_label"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Missing temporal columns: {missing}")

        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.feature_columns = feature_columns
        self.target_mean = target_mean
        self.target_std = target_std
        self.stride = stride

        episodes: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        risks: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        for _, scenario in frame.groupby("scenario_id", sort=False):
            node_counts = scenario["node_id"].value_counts()
            selected_nodes = node_counts.head(num_nodes).index.tolist()
            if len(selected_nodes) < num_nodes:
                continue

            scenario = scenario[scenario["node_id"].isin(selected_nodes)].copy()
            node_index = {node_id: index for index, node_id in enumerate(selected_nodes)}
            time_bins = sorted(scenario["time_bin"].unique())
            if len(time_bins) < seq_len:
                continue

            feature_cube = np.zeros(
                (len(time_bins), num_nodes, len(feature_columns)),
                dtype=np.float32,
            )
            target_cube = np.full((len(time_bins), num_nodes), target_mean, dtype=np.float32)
            risk_cube = np.zeros((len(time_bins), num_nodes), dtype=np.float32)
            mask_cube = np.zeros((len(time_bins), num_nodes), dtype=bool)

            time_index = {time_bin: index for index, time_bin in enumerate(time_bins)}
            time_positions = scenario["time_bin"].map(time_index).to_numpy(dtype=np.int64)
            node_positions = scenario["node_id"].map(node_index).to_numpy(dtype=np.int64)
            feature_values = scenario[feature_columns].to_numpy(dtype=np.float32)

            feature_cube[time_positions, node_positions, :] = feature_values
            target_cube[time_positions, node_positions] = scenario[TARGET_COLUMN].to_numpy(
                dtype=np.float32
            )
            risk_cube[time_positions, node_positions] = scenario["risk_label"].to_numpy(
                dtype=np.float32
            )
            mask_cube[time_positions, node_positions] = True

            for start in range(0, len(time_bins) - seq_len + 1, stride):
                end = start + seq_len
                final_mask = mask_cube[end - 1]
                if int(final_mask.sum()) < min_valid_targets:
                    continue
                episodes.append(feature_cube[start:end])
                targets.append(target_cube[end - 1])
                risks.append(risk_cube[end - 1])
                masks.append(final_mask)

        if not episodes:
            raise ValueError(
                "No temporal graph episodes were created. Try smaller --num-nodes, "
                "larger --time-bin-ms, or lower --min-valid-targets."
            )

        x = np.stack(episodes).copy()
        y_raw = np.stack(targets).copy()
        risk = np.stack(risks).copy()
        mask = np.stack(masks).copy()

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(((y_raw - target_mean) / target_std).astype(np.float32))
        self.risk = torch.from_numpy(risk.astype(np.float32))
        self.y_raw = torch.from_numpy(y_raw.astype(np.float32))
        self.mask = torch.from_numpy(mask)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "latency": self.y[index],
            "risk": self.risk[index],
            "latency_raw": self.y_raw[index],
            "mask": self.mask[index],
        }


def load_v2x_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN, RISK_COLUMN])
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame.copy()
    frame["risk_label"] = (frame[RISK_COLUMN].astype(str).str.lower() == "high").astype(int)
    return frame


def prepare_splits(
    frame: pd.DataFrame,
    seed: int = 1222,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> PreparedSplits:
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["risk_label"],
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val["risk_label"],
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_features = pd.get_dummies(
        train[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS],
        columns=CATEGORICAL_COLUMNS,
        dtype=float,
    )
    feature_columns = train_features.columns.tolist()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    def transform(split: pd.DataFrame) -> pd.DataFrame:
        features = pd.get_dummies(
            split[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS],
            columns=CATEGORICAL_COLUMNS,
            dtype=float,
        )
        features = features.reindex(columns=feature_columns, fill_value=0.0)
        scaled = pd.DataFrame(scaler.transform(features), columns=feature_columns)
        scaled[TARGET_COLUMN] = split[TARGET_COLUMN].to_numpy(dtype=float)
        scaled["risk_label"] = split["risk_label"].to_numpy(dtype=int)
        return scaled

    target_mean = float(train[TARGET_COLUMN].mean())
    target_std = float(train[TARGET_COLUMN].std(ddof=0))
    if target_std == 0:
        target_std = 1.0

    train_ready = pd.DataFrame(train_scaled, columns=feature_columns)
    train_ready[TARGET_COLUMN] = train[TARGET_COLUMN].to_numpy(dtype=float)
    train_ready["risk_label"] = train["risk_label"].to_numpy(dtype=int)

    return PreparedSplits(
        train=train_ready,
        val=transform(val),
        test=transform(test),
        feature_columns=feature_columns,
        target_mean=target_mean,
        target_std=target_std,
    )


def load_temporal_v2x_csv(path: str, time_bin_ms: int = 100) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp_us", "node_id", TARGET_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required temporal columns: {missing}")

    frame = frame.copy()
    if "scenario_id" not in frame.columns:
        frame["scenario_id"] = "scenario_0"
    if "risk_label" not in frame.columns:
        if RISK_COLUMN in frame.columns:
            frame["risk_label"] = (
                frame[RISK_COLUMN].astype(str).str.lower() == "high"
            ).astype(int)
        else:
            threshold = float(frame[TARGET_COLUMN].quantile(0.8))
            frame["risk_label"] = (frame[TARGET_COLUMN] >= threshold).astype(int)

    frame["timestamp_us"] = pd.to_numeric(frame["timestamp_us"], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = frame.dropna(subset=["timestamp_us", TARGET_COLUMN, "node_id"])
    frame["time_bin"] = (
        frame["timestamp_us"] // max(time_bin_ms * 1000, 1)
    ).astype("int64")

    return frame.sort_values(["scenario_id", "time_bin", "node_id"]).reset_index(drop=True)


def prepare_temporal_splits(
    frame: pd.DataFrame,
    seed: int = 1222,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> PreparedSplits:
    excluded = {
        "scenario_id",
        "timestamp_us",
        "time_bin",
        "node_id",
        "transmitter_id",
        "receiver_id",
        TARGET_COLUMN,
        RISK_COLUMN,
        "risk_label",
    }
    candidate_columns = [column for column in frame.columns if column not in excluded]
    numeric_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [
        column
        for column in candidate_columns
        if column not in numeric_columns
    ]
    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns found in temporal dataset.")

    frame = frame.sort_values(["scenario_id", "time_bin", "node_id"]).reset_index(drop=True)
    train_parts = []
    val_parts = []
    test_parts = []

    for _, scenario in frame.groupby("scenario_id", sort=False):
        times = np.array(sorted(scenario["time_bin"].unique()))
        if len(times) < 3:
            train_parts.append(scenario)
            continue
        train_end = int(len(times) * (1 - val_size - test_size))
        val_end = int(len(times) * (1 - test_size))
        train_times = set(times[: max(train_end, 1)])
        val_times = set(times[max(train_end, 1) : max(val_end, train_end + 1)])
        test_times = set(times[max(val_end, train_end + 1) :])
        train_parts.append(scenario[scenario["time_bin"].isin(train_times)])
        if val_times:
            val_parts.append(scenario[scenario["time_bin"].isin(val_times)])
        if test_times:
            test_parts.append(scenario[scenario["time_bin"].isin(test_times)])

    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts or train_parts).reset_index(drop=True)
    test = pd.concat(test_parts or val_parts or train_parts).reset_index(drop=True)

    train_features = pd.get_dummies(
        train[numeric_columns + categorical_columns],
        columns=categorical_columns,
        dtype=float,
    )
    feature_columns = train_features.columns.tolist()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    def transform(split: pd.DataFrame) -> pd.DataFrame:
        features = pd.get_dummies(
            split[numeric_columns + categorical_columns],
            columns=categorical_columns,
            dtype=float,
        )
        features = features.reindex(columns=feature_columns, fill_value=0.0)
        scaled = pd.DataFrame(scaler.transform(features), columns=feature_columns)
        for column in ["scenario_id", "time_bin", "node_id", TARGET_COLUMN, "risk_label"]:
            scaled[column] = split[column].to_numpy()
        return scaled

    target_mean = float(train[TARGET_COLUMN].mean())
    target_std = float(train[TARGET_COLUMN].std(ddof=0))
    if target_std == 0:
        target_std = 1.0

    train_ready = pd.DataFrame(train_scaled, columns=feature_columns)
    for column in ["scenario_id", "time_bin", "node_id", TARGET_COLUMN, "risk_label"]:
        train_ready[column] = train[column].to_numpy()

    return PreparedSplits(
        train=train_ready,
        val=transform(val),
        test=transform(test),
        feature_columns=feature_columns,
        target_mean=target_mean,
        target_std=target_std,
        temporal=True,
    )
