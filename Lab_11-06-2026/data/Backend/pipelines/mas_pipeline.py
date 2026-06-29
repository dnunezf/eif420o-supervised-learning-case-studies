from __future__ import annotations

import pandas as pd

from Backend.agents.coordinador import AgenteCoordinadorMAS
from Backend.schemas.contexto import ContextoMAS


class MASPipeline:
    def __init__(self, coordinador: AgenteCoordinadorMAS | None = None) -> None:
        self.coordinador = coordinador or AgenteCoordinadorMAS()

    def run(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        columnas_eliminar: list[str] | None = None,
        ejecutar_eda: bool = True,
        ejecutar_clustering: bool = True,
        ejecutar_clasificacion: bool = True,
        ejecutar_regresion: bool = True,
        modo_ejecucion: str = "completo",
    ) -> ContextoMAS:
        return self.coordinador.ejecutar(
            df,
            target=target,
            columnas_eliminar=columnas_eliminar,
            ejecutar_eda=ejecutar_eda,
            ejecutar_clustering=ejecutar_clustering,
            ejecutar_clasificacion=ejecutar_clasificacion,
            ejecutar_regresion=ejecutar_regresion,
            modo_ejecucion=modo_ejecucion,
        )
