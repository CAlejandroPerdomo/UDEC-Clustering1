# Código fuente del proyecto IRCA

Este paquete contiene el código Python reutilizable del proyecto de clustering sobre el IRCA,
organizado por módulos. El flujo completo de análisis interactivo se encuentra en el notebook
`notebook_clustering_etapa1.ipynb`; aquí se extrae la lógica en funciones reutilizables.

## Módulos

| Módulo | Descripción |
|---|---|
| `config.py` | Constantes globales: URL de datos, semilla, parámetros de los algoritmos y rutas. |
| `data.py` | Descarga, carga, limpieza y preparación de las matrices (Etapas 1 y 2). |
| `clustering.py` | Ajuste de K-Means, DBSCAN y clustering jerárquico (Ward) (Etapa 3). |
| `metrics.py` | Métricas internas: Silhouette, Davies-Bouldin y Calinski-Harabasz. |
| `visualization.py` | Figuras de dispersión, histogramas y PCA con estilo unificado. |

## Uso

```python
from src.data import preparar_matrices
df, df_clean, df_mun, df_dep, X_A, X_B, meta = preparar_matrices()

from src.clustering import ajustar_kmeans
modelo, etiquetas = ajustar_kmeans(X_A)
```

## Nota de reproducibilidad

Las funciones de `clustering` y `metrics` reproducen exactamente los resultados persistidos en
los archivos `.pkl` de la raíz del proyecto (verificado para K-Means, DBSCAN y jerárquico).
