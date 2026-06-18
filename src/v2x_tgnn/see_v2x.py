from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


CANONICAL_COLUMNS = [
    "scenario_id",
    "timestamp_us",
    "receiver_id",
    "transmitter_id",
    "node_id",
    "application",
    "latency_ms",
    "high_latency",
    "risk_label",
    "packet_size_bytes",
    "tx_priority",
    "channel_busy_percentage",
    "per_ue_loss_pct",
    "ipg_ms",
    "avg_throughput_10ms_bps",
    "avg_throughput_100ms_bps",
    "avg_packet_loss_10ms",
    "avg_packet_loss_100ms",
]


def convert_see_v2x_directory(
    input_dir: Path,
    output_csv: Path,
    high_latency_ms: float = 100.0,
    max_files: int | None = None,
    max_rows_per_file: int | None = None,
) -> pd.DataFrame:
    """Convert SEE-V2X receiver traces into the project temporal schema."""

    input_dir = input_dir.resolve()
    rx_files = sorted(input_dir.rglob("rx_*.csv"))
    if max_files is not None:
        rx_files = rx_files[:max_files]
    if not rx_files:
        raise FileNotFoundError(f"No rx_*.csv files found under {input_dir}")

    frames = []
    for csv_path in rx_files:
        frame = pd.read_csv(csv_path)
        if max_rows_per_file is not None:
            frame = frame.head(max_rows_per_file)
        frame = _normalize_columns(frame)

        filename_info = _parse_rx_filename(csv_path.name)
        converted = pd.DataFrame(index=frame.index)
        converted["scenario_id"] = _scenario_id(input_dir, csv_path)
        converted["timestamp_us"] = _first_available(
            frame,
            ["rx_timestamp_us", "rx_timestamp", "timestamp_us", "timestamp"],
        )
        converted["receiver_id"] = filename_info["receiver_id"]
        converted["transmitter_id"] = _first_available(
            frame,
            ["tx_equipment_id", "transmitter_id", "tx_id"],
            default=filename_info["transmitter_id"],
        ).astype(str)
        converted["node_id"] = (
            converted["transmitter_id"].astype(str)
            + "->"
            + converted["receiver_id"].astype(str)
        )
        converted["application"] = filename_info["application"]
        converted["latency_ms"] = _first_available(
            frame,
            ["latency_ms", "latency"],
        )
        converted["high_latency"] = converted["latency_ms"] >= high_latency_ms
        converted["risk_label"] = converted["high_latency"].astype(int)
        converted["packet_size_bytes"] = _first_available(
            frame,
            ["packet_size_bytes", "packet_size"],
            default=0,
        )
        converted["tx_priority"] = _first_available(
            frame,
            ["tx_priority", "priority"],
            default=0,
        )
        converted["channel_busy_percentage"] = _first_available(
            frame,
            ["channel_busy_percentage"],
            default=0,
        )
        converted["per_ue_loss_pct"] = _first_available(
            frame,
            ["per_ue_loss_pct", "per_ue_loss"],
            default=0,
        )
        converted["ipg_ms"] = _first_available(frame, ["ipg_ms", "ipg"], default=0)
        converted["avg_throughput_10ms_bps"] = _first_available(
            frame,
            ["avg_throughput_10ms_bps", "avg_throughput_10ms"],
            default=0,
        )
        converted["avg_throughput_100ms_bps"] = _first_available(
            frame,
            ["avg_throughput_100ms_bps", "avg_throughput_100ms"],
            default=0,
        )
        converted["avg_packet_loss_10ms"] = _first_available(
            frame,
            ["avg_packet_loss_10ms"],
            default=0,
        )
        converted["avg_packet_loss_100ms"] = _first_available(
            frame,
            ["avg_packet_loss_100ms"],
            default=0,
        )

        converted = converted.dropna(subset=["timestamp_us", "latency_ms"])
        frames.append(converted[CANONICAL_COLUMNS])

    output = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "timestamp_us",
        "latency_ms",
        "packet_size_bytes",
        "tx_priority",
        "channel_busy_percentage",
        "per_ue_loss_pct",
        "ipg_ms",
        "avg_throughput_10ms_bps",
        "avg_throughput_100ms_bps",
        "avg_packet_loss_10ms",
        "avg_packet_loss_100ms",
    ]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0)
    output = output.sort_values(["scenario_id", "timestamp_us", "node_id"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        renamed[column] = normalized
    return frame.rename(columns=renamed)


def _parse_rx_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    match = re.match(r"rx_([^_]+)_tx_([^_]+)(?:_(.+))?", stem)
    if not match:
        return {
            "receiver_id": "unknown_rx",
            "transmitter_id": "unknown_tx",
            "application": "unknown",
        }
    application = match.group(3) or "unknown"
    return {
        "receiver_id": match.group(1),
        "transmitter_id": match.group(2),
        "application": application,
    }


def _scenario_id(root: Path, csv_path: Path) -> str:
    parent = csv_path.parent.relative_to(root)
    return str(parent).replace("\\", "/") or "root"


def _first_available(
    frame: pd.DataFrame,
    candidates: list[str],
    default: object | None = None,
) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    if default is None:
        raise ValueError(f"Missing required columns. Tried: {candidates}")
    return pd.Series([default] * len(frame), index=frame.index)
