from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from Frontend.state import init_state, records_to_df, require_context
from Frontend.visualizaciones import plot_cluster_metrics

st.set_page_config(page_title="Clustering", layout="wide")
init_state(st)
contexto = require_context(st)
if contexto is None:
    st.stop()

st.title("Página 4 - Clustering")
resultado = contexto.resultados.get("clustering", {})
if "error" in resultado:
    st.error(resultado["error"])
    st.stop()

tabla = records_to_df(resultado.get("benchmark"))
if tabla.empty:
    st.info("No se ejecutó clustering.")
    st.stop()

st.dataframe(tabla, width="stretch", hide_index=True)
fig = plot_cluster_metrics(tabla)
if fig:
    st.pyplot(fig)
mejor = resultado.get("mejor")
if mejor:
    st.success(f"Mejor agrupamiento inicial: {mejor.get('algoritmo')} con {int(mejor.get('n_clusters', 0))} grupos.")

st.markdown(
    """
    **Interpretación**:
    - Silhouette alto indica mejor separación.
    - Davies-Bouldin bajo indica menos solapamiento.
    - Calinski-Harabasz alto indica mejor compactación/separación global.
    """
)
