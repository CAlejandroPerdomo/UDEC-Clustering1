# Segmentación de la Calidad del Agua para Consumo Humano en Colombia (IRCA)

Proyecto académico de **aprendizaje no supervisado (clustering)** aplicado al **Índice de Riesgo
de la Calidad del Agua para Consumo Humano (IRCA)** de Colombia.

Se agrupan 19.610 registros municipales (2007–2025) con tres algoritmos — **K-Means**, **DBSCAN**
y **clustering jerárquico (Ward)** — para descubrir patrones de riesgo, diferencias territoriales
y tendencias temporales que sirvan de apoyo a la toma de decisiones en salud pública.

---

## Datos

- **Fuente oficial:** [Portal Nacional de Datos Abiertos de Colombia](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Calidad-del-Agua-para-Consumo-Humano-en-Colombia/nxt2-39c3)
- **Entidad responsable:** Instituto Nacional de Salud (INS) — sistema **SIVICAP**.
- **Variable principal:** IRCA (0–100), magnitud del riesgo por no cumplimiento de características físicas, químicas y microbiológicas del agua.
- **Cobertura:** 32 departamentos + Distrito Capital de Bogotá · 1.016 municipios · 2007–2025.
- **Registros:** 19.610 (a nivel municipal, excluidos 602 agregados departamentales `#TODOS`).

> La escala oficial de interpretación del IRCA (Resolución 2115 de 2007): 0–5 sin riesgo,
> 5.1–14 riesgo bajo, 14.1–35 riesgo medio, 35.1–80 riesgo alto, 80.1–100 inviable sanitariamente.

---

## Estructura del proyecto

```
ma2/
├── notebook_clustering_etapa1.ipynb   # Notebook principal (Etapas 1 a 5)
├── informe_clustering.pdf             # Informe final en PDF
├── informe_clustering.tex             # Fuente LaTeX del informe
├── figs/                              # Figuras extraídas del notebook
├── data_raw_irca.csv                  # Datos crudos descargados de la fuente oficial
├── data_irca_etapa1.pkl               # Estado guardado tras la Etapa 1
├── data_etapa2_preparados.pkl         # Matrices y metadatos preparados (Etapa 2)
├── data_etapa3_kmeans.pkl             # Resultados de K-Means (Etapa 3.1)
├── data_etapa3_dbscan.pkl             # Resultados de DBSCAN (Etapa 3.2)
├── data_etapa3_jerarquico.pkl         # Resultados del jerárquico (Etapa 3.3)
└── README.md                          # Este archivo
```

---

## Etapas del proyecto

| Etapa | Contenido |
|---|---|
| **1** | Configuración del entorno, carga, exploración y comprensión del dataset. |
| **2** | Limpieza, selección y preparación de variables (dos escenarios de características). |
| **3.1** | **K-Means** (K=3) sobre los escenarios A (IRCA) y B (IRCA + Año). |
| **3.2** | **DBSCAN** (calibración de `eps`, análisis de ruido). |
| **3.3** | **Clustering jerárquico (Ward)**, dendrograma truncado y elección de k. |
| **4** | Evaluación con métricas, comparación de algoritmos, PCA y análisis territorial/temporal. |
| **5** | Informe final en PDF (LaTeX). |

---

## Escenarios de características

- **Escenario A:** solo `IRCA` estandarizado (perfil de riesgo puro).
- **Escenario B:** `IRCA` + `Año` estandarizados (riesgo + dimensión temporal).

Ambas matrices se construyeron con `StandardScaler` sobre las variables numéricas.

---

## Resultados principales

### Métricas de calidad (Silhouette / Davies-Bouldin / Calinski-Harabasz)

| Método y escenario | Silueta | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|
| K-Means (A) | 0.645 | 0.520 | 70 574.8 |
| K-Means (B) | 0.469 | 0.770 | 22 825.5 |
| DBSCAN (A) | 0.637 | 0.225 | 234.5 |
| DBSCAN (B) | 0.455 | 0.373 | 34.9 |
| Jerárquico (A, k=2) | **0.673** | 0.501 | 52 114.5 |
| Jerárquico (B, k=2) | 0.411 | 0.952 | 13 259.1 |

### Hallazgos clave

- El **clustering jerárquico de Ward con k=2** (escenario A) produce la partición más compacta
  (Silueta = 0.673).
- **K-Means (K=3)** separa los municipios en tres niveles de riesgo: bajo (IRCA ≈ 5.8), medio
  (≈ 31.2) y alto (≈ 63.2).
- **DBSCAN** no produjo agrupaciones representativas (un conglomerado dominante + micro-clusters
  extremos), pero **detectó observaciones extremas aisladas** como ruido.
- El **IRCA rural** (≈ 37.3) supera al **urbano** (≈ 14.9); no obstante, el 51.6 % de los datos
  rurales está ausente, lo que limita la comparación.
- El riesgo tiende a ser **mayor en años anteriores** (asociación descriptiva, no causal).
- Los departamentos con mayor presencia en los grupos de alto riesgo: **Nariño, Bolívar, Huila,
  Boyacá y Caldas**.

---

## Cómo reproducir el proyecto

1. Requisitos: Python 3.12+ y las librerías `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`,
   `scikit-learn`, `nbformat`, `nbclient`.
2. Abrir y **ejecutar las celdas del notebook** `notebook_clustering_etapa1.ipynb` en orden.
   - La Etapa 1 descarga el dataset desde la fuente oficial (guarda `data_raw_irca.csv`).
3. Los resultados intermedios se persisten en los archivos `.pkl`, que permiten retomar cualquier
   etapa posterior sin repetir los cálculos.
4. El informe se recompila con:
   ```bash
   pdflatex informe_clustering.tex
   ```

---

## Notas metodológicas

- Los **clusters no equivalen** a las categorías normativas del IRCA: son particiones estimadas
  por los algoritmos a partir de la distribución de los datos.
- Las métricas internas **no son unánimes** para elegir el número de grupos; la decisión se apoyó
  en la silueta y en la parsimonia.
- Los resultados son **descriptivos** y no permiten inferir relaciones causales.

---

## Autor

**Cristian Alejandro Perdomo Rivas** — Universidad de Cundinamarca
