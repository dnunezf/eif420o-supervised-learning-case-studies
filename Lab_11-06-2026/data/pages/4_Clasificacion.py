from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from Frontend.state import init_state, records_to_df, require_context
from Frontend.visualizaciones import plot_confusion_matrix, plot_metric_bar

st.set_page_config(page_title="Clasificación", layout="wide")
init_state(st)
contexto = require_context(st)
if contexto is None:
    st.stop()

st.title("Página 5 - Clasificación")
resultado = contexto.resultados.get("clasificacion", {})
if "error" in resultado:
    st.error(resultado["error"])
    st.stop()

tabla = records_to_df(resultado.get("benchmark"))
if tabla.empty:
    st.info("No se ejecutó clasificación. Seleccionar un target categórico y activar el agente.")
    st.stop()

if "confusion_matrix" in tabla.columns:
    tabla["confusion_matrix"] = tabla["confusion_matrix"].astype(str)
st.dataframe(tabla, width="stretch", hide_index=True)

metric = "roc_auc" if "roc_auc" in tabla.columns and tabla["roc_auc"].notna().any() else "f1_macro"
fig = plot_metric_bar(tabla, "modelo", metric, f"Ranking de clasificación por {metric}")
if fig:
    st.pyplot(fig)

mejor = resultado.get("mejor")
if mejor:
    st.success(f"Mejor clasificador inicial: {mejor.get('modelo')}")
    fig_cm = plot_confusion_matrix(mejor.get("confusion_matrix"))
    if fig_cm:
        st.pyplot(fig_cm)

st.markdown(
    """
    **Criterio inteligente**:
    - Si hay ROC-AUC disponible, se usa para priorizar clasificación binaria/multiclase.
    - Si no hay probabilidades confiables, se prioriza F1 macro.
    - En datasets desbalanceados, F1/Recall importan más que Accuracy.
    """
)
