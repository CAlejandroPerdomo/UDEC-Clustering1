"""Carga y preparación del dataset IRCA.

Encapsula la descarga del archivo CSV desde la fuente oficial, su carga en un DataFrame y la
preparación de las matrices de características (Etapas 1 y 2 del proyecto).
"""

import os

import numpy as np
import pandas as pd

from .config import (
    ARCHIVO_CSV,
    URL_DATASET,
    ESCENARIO_A_CAMPOS,
    ESCENARIO_B_CAMPOS,
)


def descargar_dataset(archivo: str = ARCHIVO_CSV, url: str = URL_DATASET,
                      overwrite: bool = False) -> str:
    """Descarga el CSV oficial en el archivo local (si no existe o se pide sobrescribir).

    Args:
        archivo: Ruta de destino del archivo CSV.
        url: URL de descarga.
        overwrite: Si es True, vuelve a descargar aunque el archivo exista.

    Returns:
        Ruta del archivo local.
    """
    if os.path.exists(archivo) and not overwrite:
        return archivo

    import urllib.request

    req = urllib.request.Request(
        url,
        headers={"Accept": "text/csv", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        contenido = resp.read()
    with open(archivo, "wb") as f:
        f.write(contenido)
    return archivo


def cargar_dataset(archivo: str = ARCHIVO_CSV) -> pd.DataFrame:
    """Carga el dataset crudo desde el CSV en un DataFrame.

    Args:
        archivo: Ruta del archivo CSV.

    Returns:
        DataFrame con los datos crudos.
    """
    return pd.read_csv(archivo, encoding="utf-8")


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza de la Etapa 2 al dataset.

    - Convierte IRCAurbano / IRCArural a numérico (tratando el marcador ``"ND"`` como NaN).
    - Unifica la inconsistencia del nombre del archipiélago de San Andrés.
    - Convierte el código de departamento a cadena.

    Args:
        df: DataFrame crudo.

    Returns:
        DataFrame limpio (no modifica el original).
    """
    df_clean = df.copy()

    def _a_numerico(serie: pd.Series) -> pd.Series:
        return pd.to_numeric(serie.replace("ND", np.nan), errors="coerce")

    df_clean["IRCAurbano"] = _a_numerico(df_clean["IRCAurbano"])
    df_clean["IRCArural"] = _a_numerico(df_clean["IRCArural"])
    df_clean["Departamento"] = df_clean["Departamento"].replace(
        {"Archipiélago de San Andrés, Providencia Y Santa Catalina":
         "Archipiélago de San Andrés, Providencia y Santa Catalina"}
    )
    df_clean["DepartamentoCodigo"] = df_clean["DepartamentoCodigo"].astype(str)
    return df_clean


def separar_niveles(df_clean: pd.DataFrame):
    """Separa los registros municipales de los agregados departamentales ``#TODOS``.

    Args:
        df_clean: DataFrame limpio.

    Returns:
        Tupla (df_municipal, df_departamental).
    """
    mask = df_clean["MunicipioCodigo"] == "#TODOS"
    df_municipal = df_clean[~mask].reset_index(drop=True).copy()
    df_departamental = df_clean[mask].reset_index(drop=True).copy()
    return df_municipal, df_departamental


def construir_matrices(df_municipal: pd.DataFrame):
    """Construye las matrices de características de los escenarios A y B (escaladas).

    Args:
        df_municipal: DataFrame con los registros a nivel municipal.

    Returns:
        Diccionario con ``X_A_scaled``, ``X_B_scaled``, ``metadatos``, ``scaler_A``,
        ``scaler_B`` y los nombres de variables.
    """
    from sklearn.preprocessing import StandardScaler

    X_A = df_municipal[ESCENARIO_A_CAMPOS].reset_index(drop=True)
    X_B = df_municipal[ESCENARIO_B_CAMPOS].reset_index(drop=True)

    scaler_A = StandardScaler()
    scaler_B = StandardScaler()

    X_A_scaled = pd.DataFrame(
        scaler_A.fit_transform(X_A), columns=ESCENARIO_A_CAMPOS
    ).astype(float)
    X_B_scaled = pd.DataFrame(
        scaler_B.fit_transform(X_B), columns=ESCENARIO_B_CAMPOS
    ).astype(float)

    # Metadatos para la interpretación de resultados.
    metadatos = df_municipal[[
        "Departamento", "DepartamentoCodigo", "Municipio", "MunicipioCodigo",
        "Año", "IRCAurbano", "IRCArural",
        "Nivel de riesgo", "Nivel de riesgo urbano", "Nivel de riesgo rural",
    ]].reset_index(drop=True)

    return {
        "X_A": X_A,
        "X_B": X_B,
        "X_A_scaled": X_A_scaled,
        "X_B_scaled": X_B_scaled,
        "feature_names_A": ESCENARIO_A_CAMPOS,
        "feature_names_B": ESCENARIO_B_CAMPOS,
        "metadatos": metadatos,
        "scaler_A": scaler_A,
        "scaler_B": scaler_B,
    }


def preparar_matrices():
    """Descarga, carga, limpia y prepara las matrices en una sola llamada.

    Conveniencia para ejecutar el flujo completo (Etapas 1 y 2).

    Returns:
        Tupla (df, df_clean, df_municipal, df_departamental, X_A_scaled, X_B_scaled, metadatos).
    """
    archivo = descargar_dataset()
    df = cargar_dataset(archivo)
    df_clean = limpiar_datos(df)
    df_municipal, df_departamental = separar_niveles(df_clean)
    matrices = construir_matrices(df_municipal)
    return (
        df,
        df_clean,
        df_municipal,
        df_departamental,
        matrices["X_A_scaled"],
        matrices["X_B_scaled"],
        matrices["metadatos"],
    )
