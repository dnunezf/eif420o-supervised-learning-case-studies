from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from Backend.config.settings import EXECUTION_MODES
from Frontend.state import (
    init_state,
    load_sample,
    read_uploaded_csv,
    run_mas,
    sample_dataset_paths,
    set_dataset,
    suggest_targets,
)

st.set_page_config(page_title="Carga de Datos", layout="wide")
init_state(st)

st.title("Página 2 - Carga de Datos")

fuente = st.radio("Fuente de datos", ["Subir CSV", "Usar dataset de prueba"], horizontal=True)
if fuente == "Subir CSV":
    uploaded = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if uploaded:
        df, sep = read_uploaded_csv(uploaded)
        set_dataset(st, df, sep)
else:
    samples = sample_dataset_paths()
    seleccionado = st.selectbox("Dataset de prueba", list(samples.keys()))
    if st.button("Cargar dataset de prueba", type="secondary"):
        df, sep = load_sample(samples[seleccionado])
        set_dataset(st, df, sep)

df = st.session_state.get("df")
if df is None:
    st.info("Cargar o seleccionar un dataset para continuar.")
    st.stop()

st.caption(
    f"Separador: `{st.session_state.get('sep')}` | Filas: `{df.shape[0]}` | Columnas: `{df.shape[1]}`"
)
st.dataframe(df.head(30), width="stretch")

st.subheader("Preprocesamiento y selección")
st.session_state["columnas_eliminar"] = st.multiselect(
    "Columnas a eliminar antes del modelado",
    options=df.columns.tolist(),
    default=st.session_state.get("columnas_eliminar", []),
    help="Remover IDs, fechas, columnas constantes o variables que causen fuga de datos.",
)

df_modelo = df.drop(columns=st.session_state["columnas_eliminar"], errors="ignore")
candidatos = suggest_targets(df_modelo)
if candidatos:
    st.info("Targets sugeridos por el coordinador: " + ", ".join(candidatos))

target_options = [None] + df_modelo.columns.tolist()
default_target = st.session_state.get("target")
default_index = target_options.index(default_target) if default_target in target_options else 0
st.session_state["target"] = st.selectbox(
    "Variable objetivo para tareas supervisadas",
    target_options,
    index=default_index,
    format_func=lambda value: "Sin target: priorizar clustering" if value is None else value,
)

st.session_state["modo_ejecucion"] = st.segmented_control(
    "Modo de ejecución",
    options=EXECUTION_MODES,
    default=st.session_state.get("modo_ejecucion", "rapido"),
    help="Rápido usa menos modelos pesados. Completo ejecuta el benchmark más amplio.",
)

st.subheader("Agente Coordinador MAS")
col1, col2, col3, col4 = st.columns(4)
ejecutar = {
    "eda": col1.checkbox("EDA", value=True),
    "clustering": col2.checkbox("Clustering", value=True),
    "clasificacion": col3.checkbox("Clasificación", value=bool(st.session_state["target"])),
    "regresion": col4.checkbox("Regresión", value=bool(st.session_state["target"])),
}

if st.button("Ejecutar sistema multiagente", type="primary"):
    with st.spinner("Agentes trabajando..."):
        contexto = run_mas(st, ejecutar)
    st.success(f"Sistema ejecutado. Problema inferido: {contexto.tipo_problema}")
    st.write("Plan del coordinador:")
    for paso in contexto.plan_ejecucion:
        st.write(f"- {paso}")
    st.subheader("Continuar revisión")
    nav1, nav2, nav3, nav4, nav5 = st.columns(5)
    nav1.page_link("pages/2_EDA.py", label="Abrir EDA")
    nav2.page_link("pages/3_Clustering.py", label="Abrir Clustering")
    nav3.page_link("pages/4_Clasificacion.py", label="Abrir Clasificación")
    nav4.page_link("pages/5_Regresion.py", label="Abrir Regresión")
    nav5.page_link("pages/6_Dashboard_Ejecutivo.py", label="Abrir Ejecutivo")
