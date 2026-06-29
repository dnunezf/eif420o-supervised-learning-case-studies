from __future__ import annotations

import json
from typing import Any


def safe_json(value: Any) -> str:
    def default(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        return str(obj)

    return json.dumps(value, ensure_ascii=False, indent=2, default=default)


def context_export_payload(contexto) -> dict[str, Any]:
    return {
        "tipo_problema": contexto.tipo_problema,
        "modo_ejecucion": contexto.modo_ejecucion,
        "target": contexto.target,
        "columnas_eliminadas": contexto.columnas_eliminadas,
        "plan_ejecucion": contexto.plan_ejecucion,
        "alertas": contexto.alertas,
        "recomendaciones": contexto.recomendaciones,
        "resultados": contexto.resultados,
        "mensajes": contexto.mensajes,
    }


def markdown_report(contexto) -> str:
    ejecutivo = contexto.resultados.get("ejecutivo", {})
    lines = [
        "# Reporte Ejecutivo LAB04",
        "",
        f"- Tipo de problema inferido: {contexto.tipo_problema}",
        f"- Target: {contexto.target or 'No seleccionado'}",
        f"- Modo de ejecución: {contexto.modo_ejecucion}",
        "",
        "## Plan de ejecución",
    ]
    lines += [f"- {item}" for item in contexto.plan_ejecucion]
    lines += ["", "## Conclusiones"]
    lines += [f"- {item}" for item in ejecutivo.get("conclusiones", [])]
    lines += ["", "## Alertas"]
    lines += [f"- {item}" for item in contexto.alertas] or ["- No se generaron alertas críticas."]
    lines += ["", "## Recomendaciones"]
    lines += [f"- {item}" for item in contexto.recomendaciones]
    return "\n".join(lines)
