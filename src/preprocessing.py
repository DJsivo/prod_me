"""Адаптеры схемы без обучаемых параметров для сохраняемого конвейера."""
import numpy as np


def canonicalize_missing(frame):
    """Заменяем None из Parquet на NaN перед обучаемым заполнением."""
    return frame.replace({None: np.nan})
