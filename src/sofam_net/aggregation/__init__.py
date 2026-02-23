"""Secure Aggregation — masked updates, threshold reconstruction, Byzantine robustness."""

from sofam_net.aggregation.byzantine import KrumAggregator, TrimmedMeanAggregator
from sofam_net.aggregation.secure import SecureAggregator

__all__ = ["SecureAggregator", "KrumAggregator", "TrimmedMeanAggregator"]
