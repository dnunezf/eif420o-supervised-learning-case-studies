from __future__ import annotations

from typing import Any

import pandas as pd

from Backend.supervised.classification.Clasificacion import ClasificacionModelos


class ClassificationService:
    def ejecutar(self, df: pd.DataFrame, target: str, modo: str = "completo") -> dict[str, Any]:
        engine = ClasificacionModelos(df, target=target, scale=True, stratify=True)
        if modo == "rapido":
            payloads = {
                "decision_tree": engine.decision_tree(),
                "random_forest": engine.random_forest(n_estimators=80, max_depth=6),
                "logistic_regression": engine.logistic_regression(),
                "naive_bayes": engine.naive_bayes(),
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
        criterio = "roc_auc" if "roc_auc" in tabla.columns and tabla["roc_auc"].notna().any() else "f1_macro"
        return tabla.sort_values(by=criterio, ascending=False, na_position="last")

    @staticmethod
    def recomendaciones(mejor: dict[str, Any] | None) -> list[str]:
        recomendaciones = []
        if mejor:
            criterio = "ROC-AUC" if pd.notna(mejor.get("roc_auc")) else "F1 macro"
            recomendaciones.append(f"Para clasificación, el mejor modelo inicial es {mejor['modelo']} según {criterio}.")
        recomendaciones.append("Si hay desbalance, priorizar F1 macro, recall macro y ROC-AUC sobre accuracy.")
        return recomendaciones
