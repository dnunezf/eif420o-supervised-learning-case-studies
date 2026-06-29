from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backend.agents.coordinador import AgenteCoordinadorMAS
from Backend.pipelines.mas_pipeline import MASPipeline


def init_state(st) -> None:
    defaults = {
        "df": None,
        "sep": None,
        "target": None,
        "columnas_eliminar": [],
        "contexto_mas": None,
        "modo_ejecucion": "rapido",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def read_uploaded_csv(uploaded_file) -> tuple[pd.DataFrame, str]:
    raw = uploaded_file.getvalue()
    buffer = io.BytesIO(raw)
    try:
        df = pd.read_csv(buffer, sep=None, engine="python")
        sep_used = "auto"
    except Exception:
        buffer.seek(0)
        df = pd.read_csv(buffer, sep=";")
        sep_used = ";"
    if df.shape[1] == 1 and ";" in df.columns[0]:
        buffer = io.BytesIO(raw)
        df = pd.read_csv(buffer, sep=";")
        sep_used = ";"
    return df, sep_used


def sample_dataset_paths() -> dict[str, Path]:
    base = PROJECT_ROOT / "Datasets"
    return {
        "Clasificación - clientes": base / "clasificacion" / "01_clasificacion_clientes.csv",
        "Regresión - viviendas": base / "regresion" / "02_regresion_viviendas.csv",
        "Agrupamiento - clientes": base / "agrupamiento" / "03_agrupamiento_clientes.csv",
        "Mixto semicolon": base / "mixto" / "04_mixto_todo_semicolon.csv",
    }


def load_sample(path: Path) -> tuple[pd.DataFrame, str]:
    sep = ";" if "semicolon" in path.name else ","
    return pd.read_csv(path, sep=sep), sep


def set_dataset(st, df: pd.DataFrame, sep: str) -> None:
    st.session_state["df"] = df
    st.session_state["sep"] = sep
    st.session_state["target"] = None
    st.session_state["columnas_eliminar"] = []
    st.session_state["contexto_mas"] = None


def suggest_targets(df: pd.DataFrame) -> list[str]:
    return AgenteCoordinadorMAS().sugerir_targets(df)


def run_mas(st, ejecutar: dict[str, bool]) -> Any:
    df = st.session_state.get("df")
    if df is None:
        raise ValueError("No hay dataset cargado.")
    contexto = MASPipeline().run(
        df,
        target=st.session_state.get("target"),
        columnas_eliminar=st.session_state.get("columnas_eliminar", []),
        ejecutar_eda=ejecutar.get("eda", True),
        ejecutar_clustering=ejecutar.get("clustering", True),
        ejecutar_clasificacion=ejecutar.get("clasificacion", True),
        ejecutar_regresion=ejecutar.get("regresion", True),
        modo_ejecucion=st.session_state.get("modo_ejecucion", "rapido"),
    )
    st.session_state["contexto_mas"] = contexto
    return contexto


def require_context(st):
    contexto = st.session_state.get("contexto_mas")
    if contexto is None:
        st.info("Ejecutar el sistema multiagente desde la página Carga de Datos.")
        return None
    return contexto


def records_to_df(records) -> pd.DataFrame:
    return pd.DataFrame(records or [])

