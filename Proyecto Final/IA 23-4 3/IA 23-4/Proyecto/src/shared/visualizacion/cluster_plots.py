from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_cluster_scatter(componentes, labels, titulo: str = "Clusters"):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.scatter(componentes[:, 0], componentes[:, 1], c=labels, cmap="tab10", s=25, alpha=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(alpha=0.3)
    return fig, ax


def plot_silhouette_curve(k_values, scores, titulo: str = "Silhouette por K"):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.plot(k_values, scores, marker="o")
    ax.set_title(titulo)
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.grid(alpha=0.3)
    return fig, ax


def plot_radar_centroids(centros, labels):
    centros = np.asarray(centros)
    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140, subplot_kw={"polar": True})
    for idx, centro in enumerate(centros):
        values = centro.tolist() + [centro[0]]
        ax.plot(angles, values, linewidth=1.5, label=f"Cluster {idx}")
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Perfil de Centroides")
    ax.legend(loc="upper right")
    return fig, ax

