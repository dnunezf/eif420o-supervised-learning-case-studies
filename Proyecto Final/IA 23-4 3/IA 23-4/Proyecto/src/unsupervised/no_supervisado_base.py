from __future__ import annotations

import pandas as pd

from src.eda.eda import EDAEngine
from src.shared.metricas.metricas_cluster import metricas_cluster


class NoSupervisadoBase(EDAEngine):
    """Shared logic for clustering and dimensionality reduction."""

    def __init__(self, df, scale: bool = True) -> None:
        super().__init__(df=df, target=None)
        self.scale = scale
        self._benchmark: list[dict[str, float | int | str]] = []

    def get_matrix(self, force: bool = False):
        key = "no_supervisado::matrix"
        if force:
            self._cache.pop(key, None)
        if key in self._cache:
            return self._cache[key]

        X = self.get_feature_matrix(use_scaled=self.scale, target=None, scaled_key="no_sup")
        self._cache[key] = X
        return X

    def fit_predict(self, model, nombre_modelo: str):
        X = self.get_matrix()
        labels = model.fit_predict(X)
        resumen = metricas_cluster(X, labels)
        self._benchmark.append(
            {
                "algoritmo": nombre_modelo,
                "n_clusters": len(set(labels)) - (1 if -1 in set(labels) else 0),
                "silhouette": resumen["silhouette"],
                "davies_bouldin": resumen["davies_bouldin"],
                "calinski_harabasz": resumen["calinski_harabasz"],
            }
        )
        return {"model": model, "labels": labels, "metricas": resumen}

    def benchmark_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._benchmark)

