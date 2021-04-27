import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    plot_confusion_matrix,
    plot_roc_curve,
)

from ml_example.enities.train_params import TrainingParams

SklearnModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
        features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnModel:
    if train_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == 'LogisticRegression':
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(model: SklearnModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "auc": roc_auc_score(target, predicts),
        "accuracy": accuracy_score(target, predicts),
        "f1": f1_score(target, predicts)
    }


def report_model(model: SklearnModel, features: pd.DataFrame, target: np.ndarray, train_params: TrainingParams) -> str:
    report_file_path = f'reports/{train_params.model_type}_metrics.png'
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_confusion_matrix(model, features, target, ax=axes[0])
    plot_roc_curve(model, features, target, ax=axes[1])
    plt.title(train_params.model_type)
    plt.savefig(report_file_path)
    return report_file_path


def serialize_model(model: SklearnModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
