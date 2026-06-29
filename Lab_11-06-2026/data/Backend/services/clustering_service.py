from __future__ import annotations

from typing import Any

import pandas as pd

from Backend.shared.pipelines.experimentos import MotorExperimentos


class ClusteringService:
    def ejecutar(self, df: pd.DataFrame) -> dict[str, Any]:
        motor = MotorExperimentos(df)
        benchmark = motor.benchmark_clustering()
        tabla = benchmark.to_dict(orient="records")
        mejor = None
        if not benchmark.empty and "silhouette" in benchmark.columns:
            ordenada = benchmark.sort_values(
                by=["silhouette", "davies_bouldin"],
                ascending=[False, True],
                na_position="last",
            )
            mejor = ordenada.iloc[0].to_dict()
        recomendaciones = self.recomendaciones(mejor)
        return {"benchmark": tabla, "mejor": mejor, "recomendaciones": recomendaciones}

    @staticmethod
    def recomendaciones(mejor: dict[str, Any] | None) -> list[str]:
        recomendaciones = []
        if mejor:
            recomendaciones.append(
                f"Para agrupamiento, revisar primero {mejor['algoritmo']} con {int(mejor['n_clusters'])} grupos."
            )
        recomendaciones.append("Validar la interpretación de clusters con variables originales y conocimiento del dominio.")
        return recomendaciones
