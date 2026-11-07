"""Aplicación de los tres algoritmos de clustering sobre el dataset IRCA.

Proporciona funciones para ajustar K-Means, DBSCAN y clustering jerárquico (Ward) a partir de
matrices ya estandarizadas, reproduciendo los parámetros documentados en la Etapa 3.
"""

import numpy as np
import pandas as pd

from .config import KMEANS, DBSCAN, JERARQUICO


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def ajustar_kmeans(X: np.ndarray, **kwargs):
    """Ajusta K-Means con los parámetros por defecto del proyecto.

    Args:
        X: Matriz de características.
        **kwargs: Sobrescriben los parámetros por defecto (``config.KMEANS``).

    Returns:
        Tupla (modelo, etiquetas).
    """
    from sklearn.cluster import KMeans

    params = dict(KMEANS)
    params.update(kwargs)
    modelo = KMeans(**params)
    etiquetas = modelo.fit_predict(X)
    return modelo, etiquetas


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
def ajustar_dbscan(X: np.ndarray, eps: float, min_samples: int = DBSCAN["min_samples"]):
    """Ajusta DBSCAN.

    Args:
        X: Matriz de características.
        eps: Radio de vecindad.
        min_samples: Número mínimo de puntos en la vecindad.

    Returns:
        Tupla (modelo, etiquetas).
    """
    from sklearn.cluster import DBSCAN as DBSCAN_sk

    modelo = DBSCAN_sk(eps=eps, min_samples=min_samples)
    etiquetas = modelo.fit_predict(X)
    return modelo, etiquetas


# ---------------------------------------------------------------------------
# Clustering jerárquico
# ---------------------------------------------------------------------------
def ajustar_jerarquico(X: np.ndarray, method: str = JERARQUICO["method"],
                       metric: str = JERARQUICO["metric"]):
    """Construye la jerarquía (linkage) con el método especificado.

    Args:
        X: Matriz de características.
        method: Método de enlace (``ward``, ``complete``, ``average``, ``single``).
        metric: Métrica de distancia.

    Returns:
        Matriz de linkage ``Z``.
    """
    from scipy.cluster.hierarchy import linkage

    return linkage(X, method=method, metric=metric)


def cortar_jerarquico(Z, k: int):
    """Extrae la partición de ``k`` clusters a partir del árbol de linkage.

    Args:
        Z: Matriz de linkage.
        k: Número de clusters.

    Returns:
        Array de etiquetas (1-based).
    """
    from scipy.cluster.hierarchy import fcluster

    return fcluster(Z, k, criterion="maxclust")


# ---------------------------------------------------------------------------
# Reporte de tamaños de clusters
# ---------------------------------------------------------------------------
def resumen_clusters(etiquetas: np.ndarray) -> pd.DataFrame:
    """Devuelve un resumen del número y tamaño (y % de ruido) de los clusters.

    Args:
        etiquetas: Array de etiquetas (con ``-1`` para ruido en DBSCAN).

    Returns:
        DataFrame con conteo y porcentaje por etiqueta.
    """
    ser = pd.Series(etiquetas)
    conteo = ser.value_counts().sort_index()
    res = pd.DataFrame({"conteo": conteo, "porcentaje": conteo / len(ser) * 100})
    return res
