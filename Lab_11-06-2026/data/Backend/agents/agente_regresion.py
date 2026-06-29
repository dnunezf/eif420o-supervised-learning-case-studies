from __future__ import annotations

from typing import Any

from Backend.agents.base import BaseAgent
from Backend.schemas.contexto import ContextoMAS
from Backend.schemas.mensajes import MensajeAgente
from Backend.services.regression_service import RegressionService


class AgenteRegresion(BaseAgent):
    nombre = "Agente Regresión"

    def __init__(self, service: RegressionService | None = None) -> None:
        self.service = service or RegressionService()

    def ejecutar(self, contexto: ContextoMAS) -> dict[str, Any]:
        if not contexto.target:
            raise ValueError("El Agente Regresión requiere variable objetivo.")

        resultado = self.service.ejecutar(contexto.df_modelo, contexto.target, contexto.modo_ejecucion)
        recomendaciones = resultado["recomendaciones"]
        contexto.resultados["regresion"] = resultado
        contexto.agregar_recomendaciones(recomendaciones)
        contexto.registrar(
            MensajeAgente(
                agente=self.nombre,
                tipo="REPORTE_REGRESION",
                estado="completado",
                contenido={"mejor": resultado["mejor"], "recomendaciones": recomendaciones},
            )
        )
        return resultado
