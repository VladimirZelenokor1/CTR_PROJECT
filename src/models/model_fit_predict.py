import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
from typing import Dict, List, Union, Tuple
import joblib

from src.entities.train_params import TrainingParams

Classifier = Union[CatBoostClassifier]

def train_model(
        features: pd.DataFrame, target: pd.Series, params: TrainingParams
    ) -> Classifier:
    """Train a CatBoostClassifier model"""
    model = CatBoostClassifier(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        depth=params.depth,
        bagging_temperature=params.bagging_temperature,
        random_state=params.random_state,
        verbose=True,
    )
    model.fit(features, target)
    return model

def predict_model(
        model: Classifier, features: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions using the trained model"""
    predictions = model.predict_proba(features)
    preds = np.argmax(predictions, axis=1)
    return predictions, preds

def evaluate_model(
        predictions: np.ndarray, predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    return {
        "f1_score": f1_score(target, predicts, average="weighted"),
        "log_loss": log_loss(target, predictions),
        "precision": precision_score(target, predicts),
        "recall": recall_score(target, predicts),
        "roc_auc_score": roc_auc_score(target, predictions[:, 1]),
    }

def serialize_model(model, output: str) -> str:
    """Serialize model from configs"""
    with open(output, "wb") as file:
        joblib.dump(model, file)
    return output