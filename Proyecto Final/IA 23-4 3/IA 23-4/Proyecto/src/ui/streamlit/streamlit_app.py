from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable even when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.ui.streamlit.controller import AppController
except ModuleNotFoundError:
    # Fallback when executed from inside the streamlit folder context.
    from controller import AppController


def _read_uploaded_csv(uploaded_file) -> tuple[pd.DataFrame, str]:
    raw = uploaded_file.getvalue()
    buffer = io.BytesIO(raw)

    # First try delimiter inference; it works for ';' and ',' in most CSVs.
    try:
        df = pd.read_csv(buffer, sep=None, engine="python")
        sep_used = "auto"
    except Exception:
        # Fallback path: semicolon files are common in local class datasets.
        buffer.seek(0)
        df = pd.read_csv(buffer, sep=";")
        sep_used = ";"

    # If still parsed as a single column with ';' in header, force semicolon.
    if df.shape[1] == 1 and ";" in df.columns[0]:
        buffer = io.BytesIO(raw)
        df = pd.read_csv(buffer, sep=";")
        sep_used = ";"

    return df, sep_used


def _tabla_clasificacion(resultados: dict) -> pd.DataFrame:
    filas = []
    for modelo, metricas in resultados.items():
        filas.append(
            {
                "Modelo": modelo,
                "Exactitud": metricas.get("accuracy"),
                "Precisión macro": metricas.get("precision_macro"),
                "Recall macro": metricas.get("recall_macro"),
                "F1 macro": metricas.get("f1_macro"),
                "Matriz de confusión": str(metricas.get("confusion_matrix")),
            }
        )
    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        tabla = tabla.sort_values(by="F1 macro", ascending=False, na_position="last")
    return tabla


def _tabla_regresion(resultados: dict) -> pd.DataFrame:
    filas = []
    for modelo, metricas in resultados.items():
        filas.append(
            {
                "Modelo": modelo,
                "RMSE": metricas.get("rmse"),
                "MAE": metricas.get("mae"),
                "Error relativo": metricas.get("er"),
                "R²": metricas.get("r2"),
            }
        )
    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        tabla = tabla.sort_values(by="RMSE", ascending=True, na_position="last")
    return tabla


def _tabla_agrupamiento(resultados: pd.DataFrame) -> pd.DataFrame:
    tabla = resultados.rename(
        columns={
            "algoritmo": "Algoritmo",
            "n_clusters": "Cantidad de grupos",
            "silhouette": "Silhouette",
            "davies_bouldin": "Davies-Bouldin",
            "calinski_harabasz": "Calinski-Harabasz",
        }
    ).copy()
    if "Silhouette" in tabla.columns:
        tabla = tabla.sort_values(by="Silhouette", ascending=False, na_position="last")
    return tabla


def run() -> None:
    try:
        import streamlit as st
    except Exception:
        print("Streamlit is not installed. Install it with: pip install streamlit")
        return

    st.set_page_config(page_title="Proyecto IA Modular", layout="wide")
    st.title("Proyecto IA")
    st.write("Arquitectura preparada: EDA -> Supervisado/No supervisado -> Modelos.")

    # Traducción visual del componente nativo de carga de archivos de Streamlit.
    st.markdown(
        """
        <style>
        [data-testid="stFileUploaderDropzone"] {
            align-items: center;
            gap: 1rem;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] {
            min-width: 0;
            flex: 1 1 auto;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] div {
            font-size: 0;
            line-height: 1.4;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] div::after {
            content: "Arrastra y suelta el archivo aquí";
            font-size: 1rem;
            display: block;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            font-size: 0;
            line-height: 1.4;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] small::after {
            content: "Límite 200MB por archivo • CSV";
            font-size: 0.875rem;
            display: block;
        }
        [data-testid="stFileUploaderDropzone"] button {
            flex: 0 0 auto;
            white-space: nowrap;
        }
        [data-testid="stFileUploaderDropzone"] button div {
            font-size: 0;
        }
        [data-testid="stFileUploaderDropzone"] button div::after {
            content: "Buscar archivos";
            font-size: 0.875rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if not uploaded:
        st.info("Cargar un archivo para iniciar.")
        return

    df, sep_used = _read_uploaded_csv(uploaded)
    st.caption(f"Separador detectado: `{sep_used}` | Filas: `{df.shape[0]}` | Columnas: `{df.shape[1]}`")

    st.subheader("Preprocesamiento rápido")
    st.info(
        "Para eliminar columnas: abrir el selector, marcar columnas (por ejemplo IDs o fechas) "
        "y luego ejecutar los modelos. El dataset se recalcula automáticamente."
    )
    columns_to_drop = st.multiselect(
        "Columnas a eliminar antes de entrenar",
        options=df.columns.tolist(),
        default=[],
        placeholder="Seleccionar columnas",
        help="Útil para remover IDs, fechas o columnas que no se quieran usar en los modelos.",
    )
    if columns_to_drop:
        df_model = df.drop(columns=columns_to_drop, errors="ignore")
    else:
        df_model = df

    st.caption(
        f"Dataset para modelado: `{df_model.shape[0]}` filas | `{df_model.shape[1]}` columnas "
        f"(eliminadas: {len(columns_to_drop)})"
    )
    st.write("Vista previa", df_model.head())

    if df_model.shape[1] == 0:
        st.error("Se eliminaron todas las columnas. Conservar al menos una para continuar.")
        return

    target_options = [None] + df_model.columns.tolist()
    target = st.selectbox(
        "Variable objetivo (solo supervisado)",
        target_options,
        index=0,
        format_func=lambda value: "Seleccionar variable objetivo" if value is None else value,
    )
    controller = AppController(df=df_model, target=target or None)

    with st.expander("Modelos incluidos en cada botón"):
        st.markdown(
            "- Clasificación: KNN, Árbol de Decisión, Bosque Aleatorio, Gradient Boosting, AdaBoost\n"
            "- Regresión: Lineal, LassoCV, RidgeCV, SVR, Árbol, Bosque, Boosting\n"
            "- Agrupamiento: KMeans, MiniBatchKMeans, HAC (y KMedoids si está disponible)"
        )

    info1, info2, info3 = st.columns(3)
    with info1.popover("Info clasificación"):
        st.markdown(
            "**Métricas de clasificación**\n\n"
            "- `Exactitud`: porcentaje total de aciertos.\n"
            "- `Precisión macro`: precisión promedio entre clases.\n"
            "- `Recall macro`: cobertura promedio entre clases.\n"
            "- `F1 macro`: balance entre precisión y recall.\n"
            "- `Matriz de confusión`: muestra aciertos y errores por clase.\n\n"
            "**Modelos**\n\n"
            "- `KNN`: clasifica por vecinos más cercanos.\n"
            "- `Árbol de Decisión`: reglas tipo árbol.\n"
            "- `Bosque Aleatorio`: conjunto de muchos árboles.\n"
            "- `Gradient Boosting`: árboles secuenciales corrigiendo errores.\n"
            "- `AdaBoost`: refuerza ejemplos difíciles."
        )
    with info2.popover("Info regresión"):
        st.markdown(
            "**Métricas de regresión**\n\n"
            "- `RMSE`: error cuadrático medio raíz (más bajo es mejor).\n"
            "- `MAE`: error absoluto medio (más bajo es mejor).\n"
            "- `Error relativo`: error proporcional respecto al valor real (más bajo es mejor).\n"
            "- `R²`: capacidad explicativa del modelo (más alto es mejor).\n\n"
            "**Modelos**\n\n"
            "- `Lineal`: relación lineal entre variables.\n"
            "- `LassoCV` / `RidgeCV`: regresión regularizada con búsqueda automática.\n"
            "- `SVR`: regresión con máquinas de soporte vectorial.\n"
            "- `Árbol`: regresión por particiones.\n"
            "- `Bosque`: promedio de múltiples árboles.\n"
            "- `Boosting`: mejora secuencial de árboles."
        )
    with info3.popover("Info agrupamiento"):
        st.markdown(
            "**Métricas de agrupamiento**\n\n"
            "- `Silhouette`: qué tan separados están los grupos (`-1` a `1`). Más alto es mejor.\n"
            "- `Davies-Bouldin`: qué tanto se solapan los grupos. Más bajo es mejor.\n"
            "- `Calinski-Harabasz`: relación entre separación y compactación. Más alto es mejor.\n"
            "- `Cantidad de grupos`: número de clusters detectados.\n\n"
            "**Modelos**\n\n"
            "- `KMeans`: agrupa por centroides.\n"
            "- `MiniBatchKMeans`: versión rápida de KMeans para más datos.\n"
            "- `HAC`: clustering jerárquico aglomerativo.\n"
            "- `KMedoids`: similar a KMeans usando puntos reales como centros (si está disponible)."
        )

    col1, col2, col3 = st.columns(3)
    run_cls = col1.button("Probar modelos de clasificación", disabled=not bool(target))
    run_reg = col2.button("Probar modelos de regresión", disabled=not bool(target))
    run_clu = col3.button("Probar modelos de agrupamiento")

    if not target:
        st.info("Seleccionar una variable objetivo para habilitar clasificación y regresión.")

    if run_cls and target:
        try:
            with st.spinner("Procesando clasificación..."):
                cls_result = controller.run_classification(target=target)
            st.success("Clasificación finalizada.")
            st.dataframe(_tabla_clasificacion(cls_result), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Error en clasificación: {exc}")
    if run_reg and target:
        try:
            with st.spinner("Procesando regresión..."):
                reg_result = controller.run_regression(target=target)
            st.success("Regresión finalizada.")
            st.dataframe(_tabla_regresion(reg_result), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Error en regresión: {exc}")
    if run_clu:
        try:
            with st.spinner("Procesando agrupamiento..."):
                clu_result = controller.run_clustering()
            st.success("Agrupamiento finalizado.")
            st.markdown(
                "**Cómo leer estos resultados de agrupamiento**\n\n"
                "- `Algoritmo`: modelo de agrupamiento evaluado.\n"
                "- `Cantidad de grupos`: número de clusters detectados.\n"
                "- `Silhouette` (más alto es mejor): separación entre grupos.\n"
                "- `Davies-Bouldin` (más bajo es mejor): solapamiento entre grupos.\n"
                "- `Calinski-Harabasz` (más alto es mejor): compactación y separación global."
            )
            st.dataframe(_tabla_agrupamiento(clu_result), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Error en agrupamiento: {exc}")


if __name__ == "__main__":
    run()
