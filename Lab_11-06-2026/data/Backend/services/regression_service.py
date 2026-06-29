from __future__ import annotations

from typing import Any

import pandas as pd

from Backend.supervised.regression.Regresion import RegresionModelos


class RegressionService:
    def ejecutar(self, df: pd.DataFrame, target: str, modo: str = "completo") -> dict[str, Any]:
        df_local = df.copy()
        df_local[target] = pd.to_numeric(df_local[target], errors="coerce")
        engine = RegresionModelos(df_local, target=target, scale=True)
        if modo == "rapido":
            payloads = {
                "lineal": engine.lineal(),
                "ridge": engine.ridge(),
                "arbol": engine.decision_tree(),
                "bosque": engine.random_forest(n_estimators=80, max_depth=8),
            }
        else:
            payloads = engine.comparar_basico()
        resultados = {name: payload["metricas"] for name, payload in payloads.items()}
        tabla = self.tabla(resultados)
        mejor = tabla.iloc[0].to_dict() if not tabla.empty else None
        recomendaciones = self.recomendaciones(mejor)
        return {"benchmark": tabla.to_dict(orient="records"), "mejor": mejor, "recomendaciones": recomendaciones}

    @staticmethod
    def tabla(resultados: dict[str, dict[str, Any]]) -> pd.DataFrame:
        filas = [{"modelo": modelo, **metricas} for modelo, metricas in resultados.items()]
        tabla = pd.DataFrame(filas)
        if tabla.empty:
            return tabla
        return tabla.sort_values(by="rmse", ascending=True, na_position="last")

    @staticmethod
    def recomendaciones(mejor: dict[str, Any] | None) -> list[str]:
        recomendaciones = []
        if mejor:
            recomendaciones.append(f"Para regresión, el mejor modelo inicial es {mejor['modelo']} por menor RMSE.")
        recomendaciones.append("Comparar RMSE con MAE para detectar si pocos errores grandes están dominando el desempeño.")
        return recomendaciones
