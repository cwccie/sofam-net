"""Pre-built IDS models — GNN flow analysis, MLP classifier, autoencoder anomaly detector."""

from sofam_net.models.autoencoder import AnomalyAutoencoder
from sofam_net.models.gnn import FlowGNN
from sofam_net.models.mlp import MLPClassifier

__all__ = ["MLPClassifier", "AnomalyAutoencoder", "FlowGNN"]
