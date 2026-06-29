from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ContextoMAS:
    df_original: pd.DataFrame
    df_modelo: pd.DataFrame
    target: str | None = None
    columnas_eliminadas: list[str] = field(default_factory=list)
    tipo_problema: str = "no_inferido"
    modo_ejecucion: str = "completo"
    plan_ejecucion: list[str] = field(default_factory=list)
    resultados: dict[str, Any] = field(default_factory=dict)
    recomendaciones: list[str] = field(default_factory=list)
    alertas: list[str] = field(default_factory=list)
    mensajes: list[dict[str, Any]] = field(default_factory=list)

    def registrar(self, mensaje: Any) -> None:
        if hasattr(mensaje, "to_dict"):
            self.mensajes.append(mensaje.to_dict())
        else:
            self.mensajes.append(dict(mensaje))

    def agregar_recomendaciones(self, recomendaciones: list[str]) -> None:
        for recomendacion in recomendaciones:
            if recomendacion not in self.recomendaciones:
                self.recomendaciones.append(recomendacion)

    def agregar_alertas(self, alertas: list[str]) -> None:
        for alerta in alertas:
            if alerta not in self.alertas:
                self.alertas.append(alerta)
