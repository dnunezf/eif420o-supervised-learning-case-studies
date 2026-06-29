from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from Frontend.state import init_state, require_context
from Frontend.visualizaciones import plot_boxplots, plot_correlation, plot_missing, plot_numeric_histograms

st.set_page_config(page_title="EDA", layout="wide")
init_state(st)
contexto = require_context(st)
if contexto is None:
    st.stop()

st.title("Página 3 - EDA")
eda = contexto.resultados.get("eda", {})
if "error" in eda:
    st.error(eda["error"])
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", eda.get("shape", [0, 0])[0])
c2.metric("Columnas", eda.get("shape", [0, 0])[1])
c3.metric("Duplicados", eda.get("duplicados", 0))
c4.metric("Nulos totales", int(sum(eda.get("nulos", {}).values())))

tab1, tab2, tab3, tab4 = st.tabs(["Calidad", "Correlación", "Distribuciones", "Outliers"])
with tab1:
    st.subheader("Tipos de datos")
    st.dataframe(pd.DataFrame(eda.get("tipos", {}).items(), columns=["columna", "tipo"]), width="stretch")
    nulos = pd.DataFrame({"nulos": eda.get("nulos", {}), "porcentaje": eda.get("porcentaje_nulos", {})})
    st.subheader("Valores nulos")
    st.dataframe(nulos, width="stretch")
    if not nulos.empty:
        st.pyplot(plot_missing(nulos))
with tab2:
    pearson = pd.DataFrame(eda.get("pearson", {}))
    spearman = pd.DataFrame(eda.get("spearman", {}))
    if not pearson.empty:
        st.pyplot(plot_correlation(pearson))
        st.dataframe(pearson, width="stretch")
    else:
        st.info("No hay suficientes variables numéricas para correlación.")
    if not spearman.empty:
        with st.expander("Spearman"):
            st.dataframe(spearman, width="stretch")
with tab3:
    numericas = eda.get("numericas", [])
    if numericas:
        st.pyplot(plot_numeric_histograms(contexto.df_modelo, numericas))
        st.pyplot(plot_boxplots(contexto.df_modelo, numericas))
    else:
        st.info("No hay variables numéricas.")
with tab4:
    st.dataframe(
        pd.DataFrame({"IQR": eda.get("outliers_iqr", {}), "Z-Score": eda.get("outliers_zscore", {})}),
        width="stretch",
    )
