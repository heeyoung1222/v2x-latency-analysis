from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLPBaseline(nn.Module):
    """Static baseline that predicts from the latest node features only."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.latency_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, num_nodes, _ = x.shape
        latest = x[:, -1, :, :].reshape(batch_size * num_nodes, -1)
        encoded = self.encoder(latest)
        latency = self.latency_head(encoded).reshape(batch_size, num_nodes)
        risk_logit = self.risk_head(encoded).reshape(batch_size, num_nodes)
        return latency, risk_logit


class GRUBaseline(nn.Module):
    """Temporal baseline that models each link independently without graph attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        head_input_dim = hidden_dim + input_dim
        self.latency_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_nodes, input_dim = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, input_dim)
        _, hidden = self.gru(sequence)
        hidden = self.dropout(hidden.squeeze(0))
        latest = x[:, -1, :, :].reshape(batch_size * num_nodes, -1)
        encoded = torch.cat([hidden, latest], dim=-1)
        latency = self.latency_head(encoded).reshape(batch_size, num_nodes)
        risk_logit = self.risk_head(encoded).reshape(batch_size, num_nodes)
        return latency, risk_logit


class GraphAttentionLayer(nn.Module):
    """Dense graph attention layer with a k-nearest-neighbor mask."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_src = nn.Linear(output_dim, 1, bias=False)
        self.attn_dst = nn.Linear(output_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        scores = self.attn_src(h) + self.attn_dst(h).transpose(1, 2)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        scores = scores.masked_fill(~adjacency, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.bmm(weights, h)


class TemporalGraphLatencyModel(nn.Module):
    """GAT + GRU model for node-level V2X latency and risk prediction."""

    def __init__(
        self,
        input_dim: int,
        graph_hidden_dim: int = 32,
        temporal_hidden_dim: int = 64,
        dropout: float = 0.15,
        knn_k: int = 4,
    ) -> None:
        super().__init__()
        self.knn_k = knn_k
        self.gat = GraphAttentionLayer(input_dim, graph_hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(graph_hidden_dim)
        self.gru = nn.GRU(
            input_size=graph_hidden_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        head_input_dim = temporal_hidden_dim + input_dim
        self.latency_head = nn.Sequential(
            nn.Linear(head_input_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim, 1),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(head_input_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_nodes, _ = x.shape
        encoded_steps = []

        for step in range(seq_len):
            node_features = x[:, step, :, :]
            adjacency = self._knn_adjacency(node_features)
            encoded = self.gat(node_features, adjacency)
            encoded = self.norm(F.relu(encoded))
            encoded_steps.append(encoded)

        encoded_sequence = torch.stack(encoded_steps, dim=1)
        encoded_sequence = encoded_sequence.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes,
            seq_len,
            -1,
        )
        _, hidden = self.gru(encoded_sequence)
        hidden = self.dropout(hidden.squeeze(0))
        last_step_features = x[:, -1, :, :].reshape(batch_size * num_nodes, -1)
        hidden = torch.cat([hidden, last_step_features], dim=-1)

        latency = self.latency_head(hidden).reshape(batch_size, num_nodes)
        risk_logit = self.risk_head(hidden).reshape(batch_size, num_nodes)
        return latency, risk_logit

    def _knn_adjacency(self, node_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_features.shape
        k = min(self.knn_k, num_nodes)
        distances = torch.cdist(node_features, node_features)
        nearest = distances.topk(k=k, largest=False).indices
        adjacency = torch.zeros(
            batch_size,
            num_nodes,
            num_nodes,
            dtype=torch.bool,
            device=node_features.device,
        )
        adjacency.scatter_(2, nearest, True)
        adjacency = adjacency | adjacency.transpose(1, 2)
        eye = torch.eye(num_nodes, dtype=torch.bool, device=node_features.device).unsqueeze(0)
        return adjacency | eye


def create_model(model_type: str, input_dim: int) -> nn.Module:
    if model_type == "mlp":
        return MLPBaseline(input_dim=input_dim)
    if model_type == "gru":
        return GRUBaseline(input_dim=input_dim)
    if model_type == "tgnn":
        return TemporalGraphLatencyModel(input_dim=input_dim)
    raise ValueError(f"Unsupported model_type: {model_type}")
