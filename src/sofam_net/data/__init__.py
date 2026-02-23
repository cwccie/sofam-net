"""Data loaders — IDS dataset loading, non-IID splitting, synthetic data generation."""

from sofam_net.data.loader import IDSDataLoader
from sofam_net.data.splitter import DirichletSplitter
from sofam_net.data.synthetic import SyntheticIDSGenerator

__all__ = ["IDSDataLoader", "SyntheticIDSGenerator", "DirichletSplitter"]
