from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def metricas_cluster(X: Any, labels: Any) -> dict[str, float]:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}

    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }

