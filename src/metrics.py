"""Métricas internas de evaluación de los modelos de clustering.

Incluye Silhouette, Davies-Bouldin y Calinski-Harabasz, con manejo correcto del ruido de
DBSCAN (etiquetas ``-1``) y de los casos en los que la métrica no es aplicable.
"""

import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def _sin_ruido(X: np.ndarray, labels: np.ndarray):
    """Elimina las observaciones clasificadas como ruido (``-1``).

    Returns:
        Tupla (X_filtrado, labels_filtrado).
    """
    mask = labels != -1
    return X[mask], labels[mask]


def n_clusters_validos(labels: np.ndarray) -> int:
    """Número de clusters excluyendo el ruido (``-1``)."""
    return len(set(labels)) - (1 if -1 in labels else 0)


def silhouette(X: np.ndarray, labels: np.ndarray, excluir_ruido: bool = False):
    """Coeficiente de silueta.

    Devuelve ``np.nan`` si hay menos de dos clusters válidos.
    """
    X_e, labels_e = (X, labels) if not excluir_ruido else _sin_ruido(X, labels)
    if n_clusters_validos(labels_e) < 2:
        return np.nan
    return silhouette_score(X_e, labels_e)


def davies_bouldin(X: np.ndarray, labels: np.ndarray, excluir_ruido: bool = False):
    """Índice de Davies-Bouldin (menor es mejor)."""
    X_e, labels_e = (X, labels) if not excluir_ruido else _sin_ruido(X, labels)
    if n_clusters_validos(labels_e) < 2:
        return np.nan
    return davies_bouldin_score(X_e, labels_e)


def calinski_harabasz(X: np.ndarray, labels: np.ndarray, excluir_ruido: bool = False):
    """Índice de Calinski-Harabasz (mayor es mejor)."""
    X_e, labels_e = (X, labels) if not excluir_ruido else _sin_ruido(X, labels)
    if n_clusters_validos(labels_e) < 2:
        return np.nan
    return calinski_harabasz_score(X_e, labels_e)


def evaluar(X: np.ndarray, labels: np.ndarray, excluir_ruido: bool = False) -> dict:
    """Calcula las tres métricas internas en un solo diccionario.

    Args:
        X: Matriz de características.
        labels: Etiquetas de clustering.
        excluir_ruido: Si es True, se descartan las observaciones con etiqueta ``-1``.

    Returns:
        Diccionario con las claves ``silhouette``, ``davies_bouldin`` y
        ``calinski_harabasz``.
    """
    return {
        "silhouette": silhouette(X, labels, excluir_ruido),
        "davies_bouldin": davies_bouldin(X, labels, excluir_ruido),
        "calinski_harabasz": calinski_harabasz(X, labels, excluir_ruido),
    }
