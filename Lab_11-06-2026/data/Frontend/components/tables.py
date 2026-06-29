from __future__ import annotations


def render_dataframe(st, df, **kwargs) -> None:
    st.dataframe(df, width="stretch", **kwargs)
