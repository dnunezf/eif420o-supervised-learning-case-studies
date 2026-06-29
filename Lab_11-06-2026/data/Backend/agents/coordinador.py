from __future__ import annotations

from typing import Any

import pandas as pd

from Backend.agents.agente_clasificacion import AgenteClasificacion
from Backend.agents.agente_clustering import AgenteClustering
from Backend.agents.agente_eda import AgenteEDA
from Backend.agents.agente_regresion import AgenteRegresion
from Backend.reports.executive_report import ExecutiveReportService
from Backend.schemas.contexto import ContextoMAS
from Backend.schemas.mensajes import MensajeAgente
from Backend.services.target_service import TargetService


class AgenteCoordinadorMAS:
    nombre = "Agente Coordinador MAS"

    def __init__(self) -> None:
        self.agente_eda = AgenteEDA()
        self.agente_clustering = AgenteClustering()
        self.agente_clasificacion = AgenteClasificacion()
        self.agente_regresion = AgenteRegresion()
        self.target_service = TargetService()
        self.report_service = ExecutiveReportService()

    def crear_contexto(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        columnas_eliminar: list[str] | None = None,
        modo_ejecucion: str = "completo",
    ) -> ContextoMAS:
        columnas_eliminar = columnas_eliminar or []
        df_modelo = df.drop(columns=columnas_eliminar, errors="ignore")
        contexto = ContextoMAS(
            df_original=df.copy(),
            df_modelo=df_modelo.copy(),
            target=target if target in df_modelo.columns else None,
            columnas_eliminadas=columnas_eliminar,
            modo_ejecucion=modo_ejecucion,
        )
        contexto.tipo_problema = self.inferir_tipo_problema(contexto)
        contexto.plan_ejecucion = self.generar_plan(contexto)
        contexto.registrar(
            MensajeAgente(
                agente=self.nombre,
                tipo="CONTEXTO_INICIAL",
                estado="completado",
                contenido={
                    "shape_original": df.shape,
                    "shape_modelo": df_modelo.shape,
                    "target": contexto.target,
                    "tipo_problema": contexto.tipo_problema,
                    "modo_ejecucion": contexto.modo_ejecucion,
                    "plan_ejecucion": contexto.plan_ejecucion,
                    "columnas_eliminadas": columnas_eliminar,
                },
            )
        )
        return contexto

    def sugerir_targets(self, df: pd.DataFrame) -> list[str]:
        return self.target_service.sugerir_targets(df)

    def inferir_tipo_problema(self, contexto: ContextoMAS) -> str:
        return self.target_service.inferir_tipo_problema(contexto.df_modelo, contexto.target)

    def generar_plan(self, contexto: ContextoMAS) -> list[str]:
        plan = [
            "Validar estructura del dataset y columnas eliminadas.",
            "Ejecutar EDA para detectar nulos, duplicados, outliers, correlaciones y riesgos.",
        ]
        if contexto.tipo_problema == "clustering":
            plan.append("No hay target seleccionado: priorizar análisis no supervisado y segmentación.")
        elif contexto.tipo_problema == "clasificacion":
            plan.append("Target categórico detectado: ejecutar benchmark de clasificación y priorizar F1/ROC-AUC.")
        elif contexto.tipo_problema == "regresion":
            plan.append("Target continuo detectado: ejecutar benchmark de regresión y priorizar RMSE/R².")
        plan.append("Comparar modelos, elegir candidatos ganadores y generar conclusiones ejecutivas.")
        if contexto.modo_ejecucion == "rapido":
            plan.append("Modo rápido activo: usar un subconjunto de modelos para reducir tiempo de ejecución.")
        return plan

    def ejecutar(
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
        contexto = self.crear_contexto(
            df,
            target=target,
            columnas_eliminar=columnas_eliminar,
            modo_ejecucion=modo_ejecucion,
        )
        if ejecutar_eda:
            self._ejecutar_seguro(contexto, "eda", self.agente_eda.ejecutar)
        if ejecutar_clustering:
            self._ejecutar_seguro(contexto, "clustering", self.agente_clustering.ejecutar)
        if ejecutar_clasificacion and contexto.target:
            self._ejecutar_seguro(contexto, "clasificacion", self.agente_clasificacion.ejecutar)
        if ejecutar_regresion and contexto.target:
            self._ejecutar_seguro(contexto, "regresion", self.agente_regresion.ejecutar)

        contexto.resultados["ejecutivo"] = self.generar_reporte_ejecutivo(contexto)
        contexto.registrar(
            MensajeAgente(
                agente=self.nombre,
                tipo="REPORTE_FINAL",
                estado="completado",
                contenido=contexto.resultados["ejecutivo"],
            )
        )
        return contexto

    def _ejecutar_seguro(self, contexto: ContextoMAS, clave: str, funcion) -> None:
        try:
            funcion(contexto)
        except Exception as exc:
            contexto.resultados[clave] = {"error": str(exc)}
            contexto.agregar_alertas([f"{clave}: {exc}"])
            contexto.registrar(
                MensajeAgente(
                    agente=self.nombre,
                    tipo=f"ERROR_{clave.upper()}",
                    estado="error",
                    contenido={"error": str(exc)},
                )
            )

    def generar_reporte_ejecutivo(self, contexto: ContextoMAS) -> dict[str, Any]:
        return self.report_service.generar(contexto)
