from __future__ import annotations

import pandas as pd


class TargetService:
    palabras_clave = [
        "target", "label", "class", "clase", "objetivo", "churn", "fuga",
        "riesgo", "precio", "valor", "salario", "score", "resultado",
    ]

    def sugerir_targets(self, df: pd.DataFrame, limite: int = 6) -> list[str]:
        candidatos = []
        for col in df.columns:
            nombre = col.lower()
            score = 0
            if any(p in nombre for p in self.palabras_clave):
                score += 4
            nunique = df[col].nunique(dropna=True)
            if 1 < nunique <= max(20, int(len(df) * 0.1)):
                score += 2
            if pd.api.types.is_numeric_dtype(df[col]) and nunique > max(20, int(len(df) * 0.1)):
                score += 1
            if score:
                candidatos.append((score, col))
        return [col for _, col in sorted(candidatos, key=lambda item: (-item[0], item[1]))[:limite]]

    def inferir_tipo_problema(self, df: pd.DataFrame, target: str | None) -> str:
        if not target or target not in df.columns:
            return "clustering"
        y = df[target]
        if pd.api.types.is_numeric_dtype(y):
            valores_unicos = y.nunique(dropna=True)
            if valores_unicos <= max(10, int(len(y) * 0.05)):
                return "clasificacion"
            return "regresion"
        return "clasificacion"
