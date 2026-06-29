from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_plano_principal(componentes, labels=None, titulo: str = "Plano Principal"):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    if labels is None:
        ax.scatter(componentes[:, 0], componentes[:, 1], s=24, alpha=0.8)
    else:
        ax.scatter(componentes[:, 0], componentes[:, 1], c=labels, cmap="tab10", s=24, alpha=0.8)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.grid(alpha=0.3)
    return fig, ax


def plot_circulo_correlacion(loadings, nombres_variables, titulo: str = "Circulo de Correlacion"):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    circle = plt.Circle((0, 0), 1, color="steelblue", fill=False)
    ax.add_patch(circle)

    for i in range(loadings.shape[0]):
        x, y = loadings[i, 0], loadings[i, 1]
        ax.arrow(0, 0, x * 0.95, y * 0.95, color="steelblue", alpha=0.6, head_width=0.03)
        ax.text(x * 1.05, y * 1.05, nombres_variables[i], fontsize=9, ha="center", va="center")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", "box")
    ax.set_title(titulo)
    return fig, ax

