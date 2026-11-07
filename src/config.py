"""Constantes y configuración global del proyecto.

Este módulo centraliza las rutas, la URL oficial de descarga y las semillas aleatorias
utilizadas a lo largo del análisis, garantizando la trazabilidad y la reproducibilidad.
"""

import os

# ---------------------------------------------------------------------------
# Fuente de datos oficial
# ---------------------------------------------------------------------------
# URL de descarga en formato CSV desde la API pública del portal de datos abiertos.
URL_DATASET = (
    "https://www.datos.gov.co/api/views/nxt2-39c3/rows.csv?accessType=DOWNLOAD"
)

# Identificador del dataset en el portal (Socrata).
DATASET_ID = "nxt2-39c3"

# Ruta local donde se guarda el archivo crudo descargado.
ARCHIVO_CSV = os.path.join("data_raw_irca.csv")

# ---------------------------------------------------------------------------
# Semilla aleatoria para reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Escenarios de características
# ---------------------------------------------------------------------------
# Escenario A: solo el IRCA general.
ESCENARIO_A_CAMPOS = ["IRCA"]
# Escenario B: IRCA + Año.
ESCENARIO_B_CAMPOS = ["IRCA", "Año"]

# ---------------------------------------------------------------------------
# Parámetros de los algoritmos de clustering
# ---------------------------------------------------------------------------
KMEANS = {"n_clusters": 3, "n_init": 10, "max_iter": 300, "random_state": SEED}

DBSCAN = {
    "eps_A": 0.02,      # Escenario A (1D)
    "eps_B": 0.20,      # Escenario B (2D)
    "min_samples": 5,
}

JERARQUICO = {
    "method": "ward",   # Método de enlace de Ward
    "metric": "euclidean",
    "k_principal": 2,
    "k_alternativo": 3,
}

# ---------------------------------------------------------------------------
# Nombre de los archivos de resultados persistidos
# ---------------------------------------------------------------------------
PKL_PREPARADOS = "data_etapa2_preparados.pkl"
PKL_KMEANS = "data_etapa3_kmeans.pkl"
PKL_DBSCAN = "data_etapa3_dbscan.pkl"
PKL_JERARQUICO = "data_etapa3_jerarquico.pkl"
