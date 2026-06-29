from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from Frontend.components.alerts import render_alerts
from Frontend.components.downloads import render_report_downloads
from Frontend.state import init_state, require_context

st.set_page_config(page_title="Dashboard Ejecutivo", layout="wide")
init_state(st)
contexto = require_context(st)
if contexto is None:
    st.stop()

st.title("Página 7 - Dashboard Ejecutivo")
ejecutivo = contexto.resultados.get("ejecutivo", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Problema inferido", ejecutivo.get("tipo_problema_inferido", "N/D"))
c2.metric("Modo", contexto.modo_ejecucion)
c3.metric("Alertas", len(ejecutivo.get("alertas", [])))
c4.metric("Mensajes MAS", ejecutivo.get("mensajes_generados", 0))

st.subheader("Plan del coordinador")
for paso in ejecutivo.get("plan_ejecucion", []):
    st.write(f"- {paso}")

st.subheader("Mejores modelos")
mejores = ejecutivo.get("mejores_modelos", {})
cols = st.columns(3)
for col, clave, etiqueta in zip(cols, ["clustering", "clasificacion", "regresion"], ["Clustering", "Clasificación", "Regresión"]):
    mejor = mejores.get(clave)
    if mejor:
        nombre = mejor.get("algoritmo") or mejor.get("modelo")
        col.metric(etiqueta, nombre)
    else:
        col.metric(etiqueta, "No ejecutado")

st.subheader("Conclusiones automáticas")
for conclusion in ejecutivo.get("conclusiones", []):
    st.write(f"- {conclusion}")

st.subheader("Alertas")
alertas = ejecutivo.get("alertas", [])
render_alerts(st, alertas)

st.subheader("Recomendaciones")
for recomendacion in ejecutivo.get("recomendaciones", []):
    st.write(f"- {recomendacion}")

st.subheader("Comunicación entre agentes")
st.dataframe(pd.DataFrame(contexto.mensajes), width="stretch", hide_index=True)

st.subheader("Descargas")
render_report_downloads(st, contexto)
