import os

import pytest
from typing import List
from ml_example.enities.feature_params import FeatureParams


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "df_fake.csv")


@pytest.fixture()
def target_col():
    return "target"


@pytest.fixture()
def size_dataset():
    return 200


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
    ]


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
    )
    return params