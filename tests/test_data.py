"""Verify the cleaning contract before building model and API stages."""

import numpy as np
import pandas as pd
import pytest

from heart_disease.data import clean_data
from heart_disease.schema import FEATURES, RAW_COLUMNS, TARGET


@pytest.fixture
def raw():
    row = [54, 1, 4, 130, 245, 0, 0, 150, 0, 1.0, 1, 0, 3, 0]
    frame = pd.DataFrame([row] * 5, columns=RAW_COLUMNS, dtype=float)
    frame["num"] = pd.Series([0, 1, 2, 3, 4], dtype=float)
    return frame


def test_binary_target_replaces_original_diagnosis_without_mutating_source(raw):
    original = raw.copy(deep=True)
    cleaned = clean_data(raw)
    assert list(cleaned.columns) == FEATURES + [TARGET]
    assert cleaned[TARGET].tolist() == [0, 1, 1, 1, 1]
    assert "num" not in cleaned
    pd.testing.assert_frame_equal(raw, original)


def test_missing_predictors_are_retained_for_training_fitted_imputation(raw):
    raw.loc[0, "ca"] = np.nan
    raw.loc[1, "thal"] = np.nan
    raw.loc[2, "chol"] = np.nan
    cleaned = clean_data(raw)
    assert len(cleaned) == len(raw)
    assert cleaned[["ca", "thal", "chol"]].isna().sum().tolist() == [1, 1, 1]


@pytest.mark.parametrize("diagnosis", [np.nan, -1, 5, 1.5])
def test_missing_or_unsupported_diagnosis_is_rejected(raw, diagnosis):
    raw.loc[0, "num"] = diagnosis
    with pytest.raises(ValueError, match="Diagnosis"):
        clean_data(raw)


def test_unsupported_category_is_rejected(raw):
    raw.loc[0, "thal"] = 0
    with pytest.raises(ValueError, match="Unexpected category in thal"):
        clean_data(raw)


@pytest.mark.parametrize("value", [-20, np.inf])
def test_invalid_measurement_is_rejected(raw, value):
    raw.loc[0, "chol"] = value
    with pytest.raises(ValueError, match="Continuous features"):
        clean_data(raw)
