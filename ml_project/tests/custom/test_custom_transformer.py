import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml_example.custom.CustomTransformer import CustomTransformer

FAKE_DATASET = np.array([12, 3, 5, 78, 39.6])


@pytest.fixture(scope='session')
def fake_dataset_to_transform() -> np.array:
    return FAKE_DATASET


def test_custom_transformer(fake_dataset_to_transform: np.array):
    custom_pipeline = Pipeline(
        [
            ("norm", CustomTransformer()),
        ]
    )
    transformed: pd.DataFrame = custom_pipeline.fit_transform(fake_dataset_to_transform)
    assert transformed.shape[0] == 5
    assert transformed.max() == 1
    assert transformed.min() >= 0
