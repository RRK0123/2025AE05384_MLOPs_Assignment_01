"""Read and clean data without fitting any statistics."""

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CATEGORIES, FEATURES, NUMERIC, RAW_COLUMNS, TARGET


def read_raw(path: Path) -> pd.DataFrame:
    """Read the headerless original, keeping '?' as missing values."""
    frame = pd.read_csv(path, header=None, na_values=["?"])
    if frame.shape != (303, 14):
        raise ValueError(f"Expected Cleveland shape (303, 14); received {frame.shape}.")
    frame.columns = RAW_COLUMNS
    return frame.apply(pd.to_numeric, errors="raise")


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate source codes, retain missing features, and binarize diagnosis."""
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError("Unexpected source schema or column ordering.")
    if raw["num"].isna().any() or not raw["num"].isin(range(5)).all():
        raise ValueError("Diagnosis must be present and coded 0 through 4.")
    for column, levels in CATEGORIES.items():
        observed = raw[column].dropna()
        if not observed.isin(levels).all():
            raise ValueError(f"Unexpected category in {column}: {observed.unique()}.")
    values = raw[NUMERIC].to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError(
            "Continuous features must be non-negative, finite when observed, or missing."
        )
    if (raw[["age", "trestbps", "chol", "thalach"]] == 0).any().any():
        raise ValueError("Unexpected zero in a positive-valued measurement.")
    cleaned = raw[FEATURES].copy()
    for column in CATEGORIES:
        cleaned[column] = cleaned[column].astype("Int64")
    cleaned[TARGET] = (raw["num"] > 0).astype("int64")
    return cleaned
