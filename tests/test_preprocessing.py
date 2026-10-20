"""Check missing-value handling and protection against test-set leakage."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from heart_disease.preprocessing import build_preprocessor
from heart_disease.schema import CATEGORIES, FEATURES, NUMERIC


def test_training_statistics_are_reused_for_test_inputs():
    train = pd.DataFrame({c: [float(next(iter(levels)))] * 3 for c, levels in CATEGORIES.items()})
    for column in NUMERIC:
        train[column] = [10.0, 20.0, 30.0]
    train = train[FEATURES]
    test = train.iloc[:2].copy()
    test.loc[test.index[0], NUMERIC] = np.nan
    test.loc[test.index[0], list(CATEGORIES)] = np.nan
    test.loc[test.index[1], NUMERIC] = 10000.0
    transformer = build_preprocessor()
    transformer.fit(train)
    numeric = transformer.named_transformers_["numeric"]
    before = numeric.named_steps["scaler"].mean_.copy()
    transformed = transformer.transform(test)
    assert np.isfinite(transformed).all()
    assert transformed.shape == (2, 28)
    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, [20.0] * 5)
    np.testing.assert_array_equal(numeric.named_steps["scaler"].mean_, before)
    np.testing.assert_allclose(transformed[0, :5], 0.0)
    assert list(transformer.feature_names_in_) == FEATURES


def test_preprocessor_can_be_cloned_without_reusing_fitted_state():
    transformer = build_preprocessor()
    assert clone(transformer) is not transformer
    assert not hasattr(clone(transformer), "transformers_")
