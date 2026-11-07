"""Segmentación de la calidad del agua para consumo humano en Colombia (IRCA).

Paquete de código fuente del proyecto de clustering sobre el Índice de Riesgo de la Calidad
del Agua (IRCA), publicado por el Instituto Nacional de Salud (INS) en el Portal de Datos
Abiertos de Colombia.

Módulos:
    config      : constantes y configuración global (URL de datos, semillas).
    data        : descarga, carga y preparación del dataset.
    clustering  : aplicación de K-Means, DBSCAN y clustering jerárquico (Ward).
    metrics     : métricas internas de evaluación (silueta, Davies-Bouldin, Calinski-Harabasz).
    visualization: figuras y ayudas de visualización.

Uso típico:
    from src.data import cargar_dataset, preparar_matrices
    df, metadatos, X_A_scaled, X_B_scaled = preparar_matrices()
"""

__version__ = "1.0.0"
__all__ = ["config", "data", "clustering", "metrics", "visualization"]
