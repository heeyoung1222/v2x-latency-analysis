# Real Temporal V2X Dataset Workflow

## Recommended Dataset

The recommended real-world dataset for this project is **SEE-V2X: C-V2X Direct Communication Dataset**.

Official resources:

- Project page: https://cisl.ucr.edu/SEE-V2X/
- Code and dataset description: https://github.com/UCR-CISL/SEE-V2X/

SEE-V2X is appropriate for a graduation-level version of this project because it contains real C-V2X radio traces with:

- application-layer transmitter logs
- receiver-side latency and packet loss traces
- packet size, priority, throughput, jitter, and channel busy information
- MAC-layer traces and sidelink control information
- GNSS coordinates for selected emulating-application scenarios

## Dataset Preparation

Download and decompress the SEE-V2X dataset or one SEE-V2X scenario folder. The project converter searches for receiver trace files named like:

```text
rx_[receiver]_tx_[transmitter]_[application].csv
```

Then convert the raw traces into the canonical temporal schema:

```bash
python scripts/prepare_see_v2x_dataset.py ^
  --input-dir "C:/path/to/SEE-V2X Dataset/Emulating Applications/BSM Only" ^
  --output data/see_v2x_temporal.csv ^
  --high-latency-ms 100
```

The generated CSV contains:

- `scenario_id`
- `timestamp_us`
- `receiver_id`
- `transmitter_id`
- `node_id`
- `application`
- `latency_ms`
- `high_latency`
- `risk_label`
- packet size, priority, channel busy, packet loss, jitter, and throughput features

## Training With Real Temporal Traces

Train the Temporal GNN on the converted real trace:

```bash
python scripts/train_temporal_gnn.py ^
  --dataset-type temporal ^
  --data data/see_v2x_temporal.csv ^
  --output-dir results/see_v2x_real ^
  --seq-len 8 ^
  --num-nodes 8 ^
  --stride 1 ^
  --time-bin-ms 100 ^
  --min-valid-targets 4
```

Recommended experiment settings:

- Use `--time-bin-ms 100` for BSM-like traffic.
- Use `--time-bin-ms 20` for perception-like traffic.
- Start with `--num-nodes 4` or `--num-nodes 8`, then increase when more links are available.
- Use `--min-valid-targets` to avoid training on mostly missing graph snapshots.

## Research Upgrade

The real-data version should be evaluated as:

1. **Latency prediction**
   - MAE
   - RMSE
   - R2
   - p95 absolute error

2. **High-latency risk detection**
   - accuracy
   - F1-score
   - threshold sensitivity, such as 50 ms, 80 ms, and 100 ms

3. **Scenario generalization**
   - train on indoor traces and test on outdoor traces
   - train on BSM-only traces and test on BSM-plus-perception traces
   - compare low-load and high-load channel conditions

## Completed Real-Data Experiment

The current repository has completed a real SEE-V2X experiment using the `indoor_allconfigs` archive:

- Source archive: `data/raw/see_v2x/parameters_sweeping/indoor_allconfigs.tar.gz`
- Extracted subset: 900 `rx_*.csv` receiver traces from `4_tx`
- Converted dataset: `data/see_v2x_4tx_temporal.csv`
- Converted rows: 1,530,040
- Scenarios: 100
- Communication links: 9

The same Temporal GNN was trained under three high-latency definitions:

| Risk Threshold | MAE (ms) | RMSE (ms) | R2 | Positive Rate | Best F1 |
|---|---:|---:|---:|---:|---:|
| 50 ms | 9.51 | 16.42 | 0.318 | 0.093 | 0.452 |
| 80 ms | 9.60 | 16.55 | 0.308 | 0.025 | 0.308 |
| 100 ms | 9.54 | 16.18 | 0.338 | 0.009 | 0.293 |

The threshold sensitivity shows that risk detection becomes harder as the high-latency definition becomes rarer. For graduation-project reporting, the 50 ms threshold is the most balanced primary experiment, while 80 ms and 100 ms can be presented as stricter safety-critical sensitivity analyses.

## Baseline Comparison

The 50 ms risk task was also trained with non-graph baselines:

| Model | MAE (ms) | RMSE (ms) | R2 | Best F1 |
|---|---:|---:|---:|---:|
| Temporal GNN | 9.51 | 16.42 | 0.318 | 0.452 |
| GRU baseline | 9.98 | 16.85 | 0.282 | 0.442 |
| MLP baseline | 10.72 | 18.44 | 0.140 | 0.366 |

This comparison supports the use of temporal and graph-aware modeling. The GRU baseline is competitive because it captures temporal patterns, but the Temporal GNN gives the best overall regression and high-risk detection results on this experiment.

## Tx-Complexity Comparison

The project also compares different communication-load settings from `indoor_allconfigs`:

| Tx Setting | Links | Rows | Risk Positive Rate | MAE (ms) | RMSE (ms) | R2 | Best F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1_tx | 1 | 181,683 | 0.123 | 7.89 | 14.36 | 0.313 | 0.533 |
| 2_tx | 2 | 348,108 | 0.119 | 7.01 | 11.54 | 0.263 | 0.377 |
| 4_tx | 9 | 1,530,040 | 0.113 | 9.51 | 16.42 | 0.318 | 0.452 |

Generated files:

- `results/tx_dataset_summary.csv`
- `results/tx_complexity_summary.csv`
- `results/tx_complexity_comparison.png`

Interpretation:

- `2_tx` has the lowest absolute latency error in this run.
- `4_tx` has stronger high-risk recall and a higher best-threshold F1 than `2_tx`.
- `1_tx` has only one communication link, so its Temporal GNN is effectively a temporal model with a degenerate graph.
- The setting-level comparison should be reported as empirical behavior across SEE-V2X configurations, not as a perfectly controlled causal study of link count alone.

## Current Limitation

The repository does not include the full SEE-V2X dataset because it is distributed separately by the dataset authors. The included `examples/see_v2x_sample/` folder is a tiny SEE-V2X-style fixture used only to verify that the converter and real temporal training path work.
