from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2x_tgnn.see_v2x import convert_see_v2x_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SEE-V2X rx_*.csv traces into a temporal training CSV.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Path to the decompressed SEE-V2X dataset or scenario folder.",
    )
    parser.add_argument(
        "--output",
        default="data/see_v2x_temporal.csv",
        help="Output CSV path in the canonical temporal schema.",
    )
    parser.add_argument("--high-latency-ms", type=float, default=100.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = convert_see_v2x_directory(
        input_dir=Path(args.input_dir),
        output_csv=ROOT / args.output,
        high_latency_ms=args.high_latency_ms,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
    )
    print(f"Wrote {len(output):,} rows to {ROOT / args.output}")
    print(f"Scenarios: {output['scenario_id'].nunique():,}")
    print(f"Communication links: {output['node_id'].nunique():,}")


if __name__ == "__main__":
    main()
