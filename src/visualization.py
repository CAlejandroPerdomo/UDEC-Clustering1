"""Utilidades de visualización para el análisis de clustering.

Contiene figuras y ayudas para representar clusters, curvas de evaluación y PCA, con estilos
unificados y legibles.
"""

import matplotlib.pyplot as plt
import numpy as np

# Paleta de colores por defecto para los clusters.
PALETA = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728", 3: "#9467bd", -1: "#111111"}


def _aplicar_estilo():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11.5,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def dispersion(x: np.ndarray, y: np.ndarray, labels: np.ndarray, *,
               xlabel: str = "", ylabel: str = "", title: str = "",
               archivo: str = None, dpi: int = 200):
    """Gráfico de dispersión 2D coloreado por etiqueta de cluster.

    Args:
        x, y: Coordenadas de los puntos.
        labels: Etiquetas de cluster (``-1`` se dibuja como ruido con una ``x``).
        xlabel, ylabel, title: Etiquetas de ejes y título.
        archivo: Ruta opcional para guardar la figura.
        dpi: Resolución en caso de guardar.
    """
    _aplicar_estilo()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for c in sorted(set(labels)):
        m = labels == c
        color = PALETA.get(c, "#888888")
        nombre = "Ruido (-1)" if c == -1 else f"Grupo {c + 1}"
        ax.scatter(x[m], y[m], s=16, alpha=0.55, edgecolor="none",
                   color=color, label=nombre,
                   marker="x" if c == -1 else "o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(markerscale=1.2, framealpha=0.9, title="Clúster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if archivo:
        plt.savefig(archivo, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, ax


def histograma_por_cluster(serie: np.ndarray, labels: np.ndarray, *,
                           xlabel: str = "", title: str = "", bins: int = 40,
                           archivo: str = None, dpi: int = 200):
    """Histograma de una variable apilado por cluster."""
    _aplicar_estilo()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c in sorted(set(labels)):
        if c == -1:
            continue
        ax.hist(serie[labels == c], bins=bins, alpha=0.55,
                color=PALETA.get(c, "#888888"), label=f"Grupo {c + 1}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if archivo:
        plt.savefig(archivo, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig, ax


def pca_proyeccion(X: np.ndarray, labels: np.ndarray, n_componentes: int = 2, *,
                   title: str = "", archivo: str = None, dpi: int = 200):
    """Ajusta PCA y grafica la proyección coloreada por cluster.

    Args:
        X: Matriz de características.
        labels: Etiquetas de cluster.
        n_componentes: Número de componentes (2 para 2D).
        title: Título de la figura.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_componentes).fit(X)
    X_pca = pca.transform(X)

    _aplicar_estilo()
    if n_componentes == 2:
        fig, ax = dispersion(X_pca[:, 0], X_pca[:, 1], labels,
                             xlabel="Componente principal 1",
                             ylabel="Componente principal 2",
                             title=title, archivo=archivo, dpi=dpi)
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        for c in sorted(set(labels)):
            m = labels == c
            ax.scatter(X_pca[m, 0], X_pca[m, 1], X_pca[m, 2], s=12, alpha=0.5,
                       color=PALETA.get(c, "#888888"),
                       label=("Ruido (-1)" if c == -1 else f"Grupo {c + 1}"))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(title)
        ax.legend()
        if archivo:
            plt.savefig(archivo, dpi=dpi, bbox_inches="tight")
        plt.show()
    return fig, ax, pca
