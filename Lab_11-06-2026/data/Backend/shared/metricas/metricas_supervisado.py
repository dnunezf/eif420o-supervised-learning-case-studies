from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def metricas_clasificacion(y_true: Any, y_pred: Any, y_proba: Any | None = None) -> dict[str, Any]:
    metricas = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    metricas["roc_auc"] = None
    if y_proba is not None:
        try:
            proba = np.asarray(y_proba)
            if proba.ndim == 2 and proba.shape[1] == 2:
                metricas["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
            elif proba.ndim == 2 and proba.shape[1] > 2:
                metricas["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except Exception:
            metricas["roc_auc"] = None
    return metricas


def metricas_regresion(y_true: Any, y_pred: Any) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = float(np.sum(np.abs(np.asarray(y_true))))
    er = float(np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denom) if denom else 0.0
    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        mape = float("nan")
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "er": er,
        "r2": float(r2_score(y_true, y_pred)),
    }
