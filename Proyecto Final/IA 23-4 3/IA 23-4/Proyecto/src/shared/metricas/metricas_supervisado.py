from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def metricas_clasificacion(y_true: Any, y_pred: Any) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def metricas_regresion(y_true: Any, y_pred: Any) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = float(np.sum(np.abs(np.asarray(y_true))))
    er = float(np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denom) if denom else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "er": er,
        "r2": float(r2_score(y_true, y_pred)),
    }
