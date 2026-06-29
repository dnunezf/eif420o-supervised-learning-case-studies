from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from Frontend.state import init_state, records_to_df, require_context
from Frontend.visualizaciones import plot_metric_bar

st.set_page_config(page_title="Regresión", layout="wide")
init_state(st)
contexto = require_context(st)
if contexto is None:
    st.stop()

st.title("Página 6 - Regresión")
resultado = contexto.resultados.get("regresion", {})
if "error" in resultado:
    st.error(resultado["error"])
    st.stop()

tabla = records_to_df(resultado.get("benchmark"))
if tabla.empty:
    st.info("No se ejecutó regresión. Seleccionar un target numérico continuo y activar el agente.")
    st.stop()

st.dataframe(tabla, width="stretch", hide_index=True)
fig = plot_metric_bar(tabla, "modelo", "rmse", "Ranking de regresión por RMSE", ascending=True)
if fig:
    st.pyplot(fig)

mejor = resultado.get("mejor")
if mejor:
    st.success(f"Mejor regresor inicial: {mejor.get('modelo')} con RMSE {mejor.get('rmse'):.4f}")

st.markdown(
    """
    **Criterio inteligente**:
    - RMSE bajo indica menor error promedio penalizando errores grandes.
    - MAE ayuda a interpretar error típico.
    - R² alto indica mayor capacidad explicativa.
    - MAPE se revisa como error relativo porcentual cuando los valores reales lo permiten.
    """
)
