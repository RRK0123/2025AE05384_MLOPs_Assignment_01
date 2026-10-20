"""Reusable preprocessing for model training and inference."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORIES, NUMERIC


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Create an unfitted transformer; fit inside each future training fold."""
    numerical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if scale_numeric else "passthrough"),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    categories=[list(CATEGORIES[c]) for c in CATEGORIES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numerical, NUMERIC),
            ("categorical", categorical, list(CATEGORIES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
