from __future__ import annotations


def render_alerts(st, alertas: list[str]) -> None:
    if alertas:
        for alerta in alertas:
            st.warning(alerta)
    else:
        st.success("No se generaron alertas críticas.")
