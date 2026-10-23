"""Stateless schema adapters included in serialized prediction pipelines."""
import numpy as np


def canonicalize_missing(frame):
    """Normalize None from Parquet object columns before learned imputation."""
    return frame.replace({None: np.nan})
