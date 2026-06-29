from __future__ import annotations

from typing import Any

from Backend.schemas.contexto import ContextoMAS


class ExecutiveReportService:
    def generar(self, contexto: ContextoMAS) -> dict[str, Any]:
        mejores = {
            "clustering": contexto.resultados.get("clustering", {}).get("mejor"),
            "clasificacion": contexto.resultados.get("clasificacion", {}).get("mejor"),
            "regresion": contexto.resultados.get("regresion", {}).get("mejor"),
        }
        conclusiones = []
        if mejores["clustering"]:
            conclusiones.append(f"Clustering sugerido: {mejores['clustering'].get('algoritmo')}.")
        if mejores["clasificacion"]:
            conclusiones.append(f"Clasificación sugerida: {mejores['clasificacion'].get('modelo')}.")
        if mejores["regresion"]:
            conclusiones.append(f"Regresión sugerida: {mejores['regresion'].get('modelo')}.")
        if contexto.alertas:
            conclusiones.append("Existen alertas de calidad o modelado que deben revisarse antes de producción.")
        return {
            "tipo_problema_inferido": contexto.tipo_problema,
            "mejores_modelos": mejores,
            "alertas": contexto.alertas,
            "recomendaciones": contexto.recomendaciones,
            "conclusiones": conclusiones,
            "plan_ejecucion": contexto.plan_ejecucion,
            "mensajes_generados": len(contexto.mensajes),
        }
