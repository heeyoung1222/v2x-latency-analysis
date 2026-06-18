# Temporal Graph Neural Network Research Plan

## Proposed Topic

**Temporal Graph Neural Network-Based V2X Communication Latency Prediction and High-Risk Delay Detection**

This direction upgrades the original R-based exploratory latency analysis into a Python deep learning project. The main research question is:

> Can a temporal graph neural network predict V2X communication latency and detect high-latency risk from driving and network-condition features?

## Why This Topic Fits V2X

V2X communication is naturally graph-structured. Vehicles and roadside units can be represented as nodes, while communication links, distance, signal quality, and scheduling interactions can be represented as edges. Because traffic and wireless conditions change over time, the model should also learn temporal patterns rather than only static correlations.

## Current Prototype

The repository currently uses a simulated tabular V2X dataset with these variables:

- `speed_kmh`
- `signal_strength_dbm`
- `network_stability_index`
- `vehicle_density`
- `scheduling_algorithm`
- `latency_ms`
- `high_latency`

Since the current dataset does not include timestamps, vehicle IDs, transmitter IDs, receiver IDs, or RSU IDs, the first Python version builds reproducible pseudo temporal-graph episodes from the existing rows. This is enough to start model development, but the research should later be strengthened with real V2X traces.

The repository now includes a real-data path for SEE-V2X-style temporal traces:

1. Convert `rx_*.csv` receiver logs with `scripts/prepare_see_v2x_dataset.py`.
2. Train with `scripts/train_temporal_gnn.py --dataset-type temporal`.
3. Represent each communication link, such as `UE2->UE1`, as a graph node.
4. Use timestamp bins to create true temporal graph windows.
5. Mask missing link observations so the loss only uses observed latency labels.

## Model Design

The implemented prototype uses:

1. **Graph Attention Network layer**
   - Learns interactions among vehicles/nodes within each time step.
   - Builds k-nearest-neighbor graph masks from node features.

2. **GRU temporal encoder**
   - Learns how network and traffic states evolve over consecutive time steps.

3. **Multi-task prediction heads**
   - Regression head predicts `latency_ms`.
   - Classification head predicts high-latency risk.

## Recommended Next Research Milestones

1. Download SEE-V2X and run the converter on BSM-only, perception-only, and BSM-plus-perception scenarios.
2. Compare pseudo-episode training with true temporal trace training.
3. Add vehicle-to-vehicle and vehicle-to-infrastructure edge features.
4. Compare MLP, LSTM, GRU, Transformer, GCN-GRU, and GAT-GRU baselines.
5. Evaluate tail latency using p90, p95, and p99 error metrics.
6. Add explainability with attention visualization or SHAP-style feature attribution.
7. Build a Streamlit dashboard for scenario-level latency-risk simulation.

## Candidate Real Datasets

- SEE-V2X: real C-V2X traces with latency, throughput, packet loss, and scheduling-related information.
- Safety Pilot Model Deployment BSM: connected vehicle messages with GPS and CAN-derived values.
- V2X-Sim: synthetic collaborative perception dataset for autonomous driving research.
