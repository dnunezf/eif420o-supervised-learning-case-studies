from __future__ import annotations

import numpy as np
import pandas as pd

from Backend.unsupervised.clustering.Cluster import ClusterModelos
from Backend.unsupervised.pca.reduccion import ReduccionDimensional
from Backend.supervised.classification.Clasificacion import ClasificacionModelos
from Backend.supervised.regression.Regresion import RegresionModelos


class MotorExperimentos:
    """Single orchestrator to benchmark models with one coherent architecture."""

    def __init__(self, df: pd.DataFrame, target: str | None = None):
        self.df = df
        self.target = target

    def benchmark_clasificacion(self, target: str | None = None):
        objective = target or self.target
        if objective is None:
            raise ValueError("Target is required for classification benchmark.")
        if objective not in self.df.columns:
            raise KeyError(f"Target column '{objective}' does not exist in dataframe.")

        engine = ClasificacionModelos(self.df, target=objective, scale=True, stratify=True)
        results = engine.comparar_basico()
        return {name: payload["metricas"] for name, payload in results.items()}

    def benchmark_regresion(self, target: str | None = None):
        objective = target or self.target
        if objective is None:
            raise ValueError("Target is required for regression benchmark.")
        if objective not in self.df.columns:
            raise KeyError(f"Target column '{objective}' does not exist in dataframe.")

        df_local = self.df.copy()
        numeric_target = pd.to_numeric(df_local[objective], errors="coerce")
        if numeric_target.notna().sum() == 0:
            raise ValueError(f"El target '{objective}' no es numérico y no sirve para regresión.")
        df_local[objective] = numeric_target

        engine = RegresionModelos(df_local, target=objective, scale=True)
        results = engine.comparar_basico()
        return {name: payload["metricas"] for name, payload in results.items()}

    def benchmark_clustering(self):
        engine = ClusterModelos(self.df, scale=True)
        engine.kmeans()
        engine.mini_batch_kmeans()
        numpy_major = int(str(np.__version__).split(".", maxsplit=1)[0])
        if numpy_major < 2:
            try:
                engine.kmedoids()
            except Exception:
                pass
        engine.hac()
        engine.dbscan()
        return engine.benchmark_frame()

    def reduccion(self):
        engine = ReduccionDimensional(self.df, scale=True)
        return {
            "pca": engine.pca(),
            "tsne": engine.tsne(),
            "umap": engine.umap(),
        }
