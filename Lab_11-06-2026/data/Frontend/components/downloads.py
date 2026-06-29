from __future__ import annotations

from Backend.reports.exporters import context_export_payload, markdown_report, safe_json


def render_report_downloads(st, contexto) -> None:
    payload = context_export_payload(contexto)
    col_json, col_md = st.columns(2)
    col_json.download_button(
        "Descargar reporte JSON",
        data=safe_json(payload),
        file_name="reporte_lab04_mas.json",
        mime="application/json",
    )
    col_md.download_button(
        "Descargar reporte Markdown",
        data=markdown_report(contexto),
        file_name="reporte_lab04_mas.md",
        mime="text/markdown",
    )
