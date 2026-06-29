from __future__ import annotations

import pandas as pd

from src.shared.pipelines.experimentos import MotorExperimentos


class AppController:
    def __init__(self, df: pd.DataFrame, target: str | None = None):
        self._motor = MotorExperimentos(df, target=target)

    def run_classification(self, target: str):
        return self._motor.benchmark_clasificacion(target=target)

    def run_regression(self, target: str):
        return self._motor.benchmark_regresion(target=target)

    def run_clustering(self):
        return self._motor.benchmark_clustering()

