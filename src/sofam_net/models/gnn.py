"""
GNN-based Network Flow Analysis.

A simplified Graph Neural Network implemented in pure NumPy for analyzing
network flow graphs. Nodes represent hosts, edges represent connections.

This is a lightweight implementation suitable for federated learning
without requiring PyTorch Geometric or DGL.
"""

from __future__ import annotations

import numpy as np


class FlowGNN:
    """
    Graph Neural Network for network flow analysis.

    Implements a 2-layer message-passing GNN:
    1. Message: h_j → W_msg * h_j (neighbor features transformed)
    2. Aggregate: sum of neighbor messages
    3. Update: ReLU(W_update * [h_i || aggregated_messages])

    Graph structure:
    - Nodes: network hosts (features = traffic statistics)
    - Edges: connections between hosts (adjacency matrix)
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 2,
        seed: int | None = None,
    ):
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        rng = np.random.default_rng(seed)
        self.parameters: list[np.ndarray] = []

        # Build layer parameters
        in_dim = node_feature_dim
        for layer in range(num_layers):
            out_dim = hidden_dim if layer < num_layers - 1 else output_dim

            # Message weight
            scale = np.sqrt(2.0 / (in_dim + out_dim))
            W_msg = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
            b_msg = np.zeros(out_dim, dtype=np.float32)

            # Update weight (concatenation of self + aggregated)
            scale_up = np.sqrt(2.0 / (in_dim + out_dim + out_dim))
            W_update = rng.normal(0, scale_up, (in_dim + out_dim, out_dim)).astype(np.float32)
            b_update = np.zeros(out_dim, dtype=np.float32)

            self.parameters.extend([W_msg, b_msg, W_update, b_update])
            in_dim = out_dim

    def get_parameters(self) -> list[np.ndarray]:
        return self.parameters

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.parameters = [np.copy(p) for p in parameters]

    def forward(
        self, node_features: np.ndarray, adjacency: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through the GNN.

        Args:
            node_features: (num_nodes, feature_dim) node feature matrix
            adjacency: (num_nodes, num_nodes) adjacency matrix (0/1 or weighted)

        Returns:
            (num_nodes, output_dim) node embeddings/predictions
        """
        # Normalize adjacency (add self-loops, degree normalization)
        adj = adjacency + np.eye(len(adjacency))
        degree = np.sum(adj, axis=1, keepdims=True)
        adj_norm = adj / (degree + 1e-10)

        h = node_features.astype(np.float64)

        for layer in range(self.num_layers):
            idx = layer * 4
            W_msg = self.parameters[idx].astype(np.float64)
            b_msg = self.parameters[idx + 1].astype(np.float64)
            W_update = self.parameters[idx + 2].astype(np.float64)
            b_update = self.parameters[idx + 3].astype(np.float64)

            # Message passing: transform neighbor features
            messages = h @ W_msg + b_msg  # (nodes, out_dim)

            # Aggregate: normalized sum of neighbor messages
            aggregated = adj_norm @ messages  # (nodes, out_dim)

            # Update: concatenate self features with aggregated messages
            combined = np.concatenate([h, aggregated], axis=1)

            # Handle dimension mismatch by padding/truncating
            expected_in = W_update.shape[0]
            if combined.shape[1] < expected_in:
                pad = np.zeros((combined.shape[0], expected_in - combined.shape[1]))
                combined = np.concatenate([combined, pad], axis=1)
            elif combined.shape[1] > expected_in:
                combined = combined[:, :expected_in]

            h = combined @ W_update + b_update

            # Apply ReLU (except last layer)
            if layer < self.num_layers - 1:
                h = np.maximum(h, 0)

        return h.astype(np.float32)

    def predict_graph(
        self, node_features: np.ndarray, adjacency: np.ndarray
    ) -> np.ndarray:
        """Predict class labels for each node."""
        logits = self.forward(node_features, adjacency)
        return np.argmax(logits, axis=1)

    def graph_embedding(
        self, node_features: np.ndarray, adjacency: np.ndarray
    ) -> np.ndarray:
        """Compute a graph-level embedding by mean-pooling node embeddings."""
        node_emb = self.forward(node_features, adjacency)
        return np.mean(node_emb, axis=0)

    def evaluate(
        self,
        node_features: np.ndarray,
        adjacency: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate node classification performance."""
        preds = self.predict_graph(node_features, adjacency)
        labels_int = labels.astype(int)
        accuracy = float(np.mean(preds == labels_int))
        return {"accuracy": accuracy, "num_nodes": len(labels)}
