from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class EDAService:
    def analizar(self, df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
        numericas = df.select_dtypes(include=np.number).columns.tolist()
        categoricas = df.select_dtypes(exclude=np.number).columns.tolist()
        nulos = df.isna().sum()
        porcentaje_nulos = (nulos / max(len(df), 1) * 100).round(2)
        duplicados = int(df.duplicated().sum())

        resultado = {
            "shape": df.shape,
            "tipos": df.dtypes.astype(str).to_dict(),
            "nulos": nulos.to_dict(),
            "porcentaje_nulos": porcentaje_nulos.to_dict(),
            "duplicados": duplicados,
            "numericas": numericas,
            "categoricas": categoricas,
            "estadisticas": df.describe(include="all").fillna("").astype(str).to_dict(),
            "pearson": df[numericas].corr(method="pearson").round(4).to_dict() if len(numericas) > 1 else {},
            "spearman": df[numericas].corr(method="spearman").round(4).to_dict() if len(numericas) > 1 else {},
            "outliers_iqr": self._outliers_iqr(df, numericas),
            "outliers_zscore": self._outliers_zscore(df, numericas),
        }
        alertas, recomendaciones = self.diagnosticar(df, target, resultado)
        return {"resultado": resultado, "alertas": alertas, "recomendaciones": recomendaciones}

    def diagnosticar(self, df: pd.DataFrame, target: str | None, resultado: dict[str, Any]) -> tuple[list[str], list[str]]:
        alertas: list[str] = []
        recomendaciones: list[str] = []
        if df.empty:
            alertas.append("El dataset está vacío.")
        if resultado["duplicados"]:
            recomendaciones.append(f"Eliminar o revisar {resultado['duplicados']} filas duplicadas antes de entrenar.")
        if sum(resultado["nulos"].values()) > 0:
            recomendaciones.append("Aplicar imputación diferenciada: mediana para numéricas y moda para categóricas.")
        if any(v > 40 for v in resultado["porcentaje_nulos"].values()):
            alertas.append("Hay columnas con más de 40% de valores nulos.")
        if resultado["numericas"]:
            recomendaciones.append("Escalar variables numéricas para KNN, SVM, K-Means, DBSCAN y SVR.")
        if resultado["categoricas"]:
            recomendaciones.append("Codificar variables categóricas con One-Hot Encoding antes del modelado.")
        if target and target in df.columns:
            serie = df[target]
            if serie.nunique(dropna=True) <= 20:
                min_ratio = serie.value_counts(normalize=True, dropna=True).min()
                if pd.notna(min_ratio) and min_ratio < 0.2:
                    alertas.append("La variable objetivo parece desbalanceada; priorizar F1 macro o ROC-AUC.")
        return alertas, recomendaciones

    @staticmethod
    def _outliers_iqr(df: pd.DataFrame, columnas: list[str]) -> dict[str, int]:
        salida = {}
        for col in columnas:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                salida[col] = 0
                continue
            li = q1 - 1.5 * iqr
            ls = q3 + 1.5 * iqr
            salida[col] = int(((df[col] < li) | (df[col] > ls)).sum())
        return salida

    @staticmethod
    def _outliers_zscore(df: pd.DataFrame, columnas: list[str]) -> dict[str, int]:
        salida = {}
        for col in columnas:
            serie = pd.to_numeric(df[col], errors="coerce")
            std = serie.std()
            if pd.isna(std) or std == 0:
                salida[col] = 0
                continue
            z = ((serie - serie.mean()) / std).abs()
            salida[col] = int((z > 3).sum())
        return salida
