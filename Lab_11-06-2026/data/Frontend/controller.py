from __future__ import annotations

import pandas as pd

from Backend.agents.coordinador import AgenteCoordinadorMAS
from Backend.shared.pipelines.experimentos import MotorExperimentos


class AppController:
    def __init__(self, df: pd.DataFrame, target: str | None = None):
        self._motor = MotorExperimentos(df, target=target)
        self._df = df
        self._target = target

    def run_classification(self, target: str):
        return self._motor.benchmark_clasificacion(target=target)

    def run_regression(self, target: str):
        return self._motor.benchmark_regresion(target=target)

    def run_clustering(self):
        return self._motor.benchmark_clustering()

    def run_mas(
        self,
        target: str | None = None,
        columnas_eliminar: list[str] | None = None,
        ejecutar_eda: bool = True,
        ejecutar_clustering: bool = True,
        ejecutar_clasificacion: bool = True,
        ejecutar_regresion: bool = True,
        modo_ejecucion: str = "completo",
    ):
        coordinador = AgenteCoordinadorMAS()
        return coordinador.ejecutar(
            self._df,
            target=target or self._target,
            columnas_eliminar=columnas_eliminar,
            ejecutar_eda=ejecutar_eda,
            ejecutar_clustering=ejecutar_clustering,
            ejecutar_clasificacion=ejecutar_clasificacion,
            ejecutar_regresion=ejecutar_regresion,
            modo_ejecucion=modo_ejecucion,
        )
