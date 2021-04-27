# -*- coding: utf-8 -*-
from typing import Tuple
import pandas as pd
import numpy as np
from faker import Faker
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from ml_example.enities import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def generate_fake_dataset(size_dataset: int) -> pd.DataFrame:
    fake_data = Faker()
    age = np.random.randint(29, 77, size=size_dataset, dtype='int')
    sex = np.random.randint(0, 2, size=size_dataset, dtype='int')
    cp = np.random.randint(0, 4, size=size_dataset, dtype='int')
    trestbps = np.random.randint(94, 201, size=size_dataset, dtype='int')
    chol = np.random.randint(126, 565, size=size_dataset, dtype='int')
    fbs = fake_data.random_elements(
        elements=OrderedDict([
            (0, 0.75),
            (1, 0.25),
        ]), unique=False, length=size_dataset
    )
    restecg = fake_data.random_elements(
        elements=OrderedDict([
            (0, 0.48),
            (1, 0.48),
            (2, 0.04),
        ]), unique=False, length=size_dataset
    )
    thalach = np.random.randint(71, 203, size=size_dataset, dtype='int')
    exang = fake_data.random_elements(
        elements=OrderedDict([
            (0, 0.75),
            (1, 0.25),
        ]), unique=False, length=size_dataset
    )
    oldpeak = []
    for _ in range(size_dataset):
        oldpeak.append(round(np.random.uniform(0., 6.2), 1))
    slope = fake_data.random_elements(
        elements=OrderedDict([
            (0, 0.1),
            (1, 0.45),
            (2, 0.45),
        ]), unique=False, length=size_dataset
    )
    ca = fake_data.random_elements(
        elements=OrderedDict([
            (0, 0.5),
            (1, 0.3),
            (2, 0.15),
            (3, 0.05),
        ]), unique=False, length=size_dataset
    )
    thal = fake_data.random_elements(
        elements=OrderedDict([
            (0, 0.1),
            (1, 0.2),
            (2, 0.4),
            (3, 0.3),
        ]), unique=False, length=size_dataset
    )
    target = np.random.randint(0, 2, size=size_dataset, dtype='int')
    data = pd.DataFrame(list(zip(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal, target)),
                        columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])
    return data


def split_train_val_data(
        data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
