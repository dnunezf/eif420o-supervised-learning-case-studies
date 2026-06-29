from __future__ import annotations

import ast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_missing(nulos: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    data = nulos.reset_index().rename(columns={"index": "columna"})
    sns.barplot(data=data, x="porcentaje", y="columna", ax=ax, color="#4c78a8")
    ax.set_xlabel("% nulos")
    ax.set_ylabel("")
    ax.set_title("Valores nulos por columna")
    fig.tight_layout()
    return fig


def plot_correlation(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="vlag", center=0, ax=ax)
    ax.set_title("Heatmap de correlación")
    fig.tight_layout()
    return fig


def plot_numeric_histograms(df: pd.DataFrame, columns: list[str]):
    cols = columns[:6]
    if not cols:
        return None
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, max(3, len(cols) * 2.2)))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(pd.to_numeric(df[col], errors="coerce").dropna(), kde=True, ax=ax, color="#59a14f")
        ax.set_title(col)
        ax.set_xlabel("")
    fig.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame, columns: list[str]):
    cols = columns[:8]
    if not cols:
        return None
    melted = df[cols].apply(pd.to_numeric, errors="coerce").melt(var_name="variable", value_name="valor")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=melted, x="variable", y="valor", ax=ax, color="#f28e2b")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Boxplots de variables numéricas")
    fig.tight_layout()
    return fig


def plot_metric_bar(tabla: pd.DataFrame, x: str, y: str, title: str, ascending: bool = False):
    if tabla.empty or x not in tabla.columns or y not in tabla.columns:
        return None
    data = tabla.sort_values(y, ascending=ascending).head(12)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=data, x=y, y=x, ax=ax, color="#4c78a8")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(value):
    if value is None:
        return None
    matrix = value
    if isinstance(value, str):
        try:
            matrix = ast.literal_eval(value)
        except Exception:
            return None
    data = pd.DataFrame(matrix)
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(data, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    return fig


def plot_cluster_metrics(tabla: pd.DataFrame):
    if tabla.empty:
        return None
    metrics = [c for c in ["silhouette", "davies_bouldin", "calinski_harabasz"] if c in tabla.columns]
    if not metrics:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=tabla, x=metric, y="algoritmo", ax=ax, color="#76b7b2")
        ax.set_title(metric)
        ax.set_ylabel("")
    fig.tight_layout()
    return fig
