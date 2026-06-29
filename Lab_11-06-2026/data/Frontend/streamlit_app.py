from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Frontend.state import init_state
from Backend.config.settings import APP_TITLE


def run() -> None:
    try:
        import streamlit as st
    except Exception:
        print("Streamlit is not installed. Install it with: pip install streamlit")
        return

    st.set_page_config(page_title="LAB04 Agentic IA-ML", layout="wide")
    init_state(st)

    st.title(APP_TITLE)
    st.caption("EIF420O - Inteligencia Artificial")
    st.markdown(
        """
        Plataforma Streamlit con arquitectura MAS para automatizar EDA, clustering,
        clasificación, regresión, benchmarking y conclusiones ejecutivas.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Agentes", "5")
    c2.metric("Familias ML", "3")
    c3.metric("Dashboard", "7 páginas")

    st.subheader("Arquitectura MAS")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            Usuario -> Coordinador;
            Coordinador -> EDA;
            Coordinador -> Clustering;
            Coordinador -> Clasificacion;
            Coordinador -> Regresion;
            EDA -> Dashboard;
            Clustering -> Dashboard;
            Clasificacion -> Dashboard;
            Regresion -> Dashboard;
        }
        """
    )

    st.subheader("Flujo recomendado")
    st.markdown(
        """
        1. Ir a **Carga de Datos**.
        2. Cargar CSV o usar un dataset de prueba.
        3. Elegir columnas a eliminar, variable objetivo y modo de ejecución.
        4. Ejecutar el sistema multiagente.
        5. Revisar EDA, Clustering, Clasificación, Regresión y Dashboard Ejecutivo.
        """
    )

    st.subheader("Mejoras Agentic IA incluidas")
    st.markdown(
        """
        - Memoria compartida mediante `ContextoMAS`.
        - Mensajes estandarizados entre agentes.
        - Plan explícito del coordinador.
        - Sugerencia automática de variable objetivo.
        - Modo rápido y modo completo.
        - Alertas de calidad de datos y recomendaciones automáticas.
        - Reporte ejecutivo descargable en JSON y Markdown.
        """
    )

    st.subheader("Diagramas UML para documentación")
    st.code(
        """classDiagram
    class AgenteCoordinadorMAS
    class ContextoMAS
    class MensajeAgente
    class AgenteEDA
    class AgenteClustering
    class AgenteClasificacion
    class AgenteRegresion
    AgenteCoordinadorMAS --> ContextoMAS
    AgenteCoordinadorMAS --> AgenteEDA
    AgenteCoordinadorMAS --> AgenteClustering
    AgenteCoordinadorMAS --> AgenteClasificacion
    AgenteCoordinadorMAS --> AgenteRegresion
    ContextoMAS --> MensajeAgente""",
        language="mermaid",
    )


if __name__ == "__main__":
    run()
