# LAB04 - Sistema Multiagente Agentic IA para Machine Learning

Proyecto Streamlit para el Laboratorio 04 de EIF420O Inteligencia Artificial.

## Objetivo

Automatizar el ciclo de Machine Learning con un Sistema Multiagente capaz de:

- Analizar datasets con EDA.
- Ejecutar clustering no supervisado.
- Ejecutar clasificación supervisada.
- Ejecutar regresión supervisada.
- Comparar modelos mediante benchmarking.
- Generar alertas, recomendaciones y conclusiones ejecutivas.

## Arquitectura MAS

```text
Usuario
  -> Agente Coordinador MAS
      -> Agente EDA
      -> Agente Clustering
      -> Agente Clasificación
      -> Agente Regresión
  -> Dashboard Streamlit
```

## Estructura

```text
LAB4_AgenticIA_ML/
├── main.py
├── Datasets/
├── docs/
├── src/
│   ├── agents/
│   ├── schemas/
│   ├── services/
│   ├── pipelines/
│   ├── reports/
│   ├── config/
│   ├── eda/
│   ├── supervised/
│   ├── unsupervised/
│   ├── shared/
│   └── ui/streamlit/
└── README.md
```

## Cobertura del enunciado

- Agente Coordinador: carga contexto, infiere tipo de problema, activa agentes, consolida resultados y genera reporte final.
- Agente EDA: nulos, duplicados, tipos, estadísticas, Pearson, Spearman, outliers IQR y Z-Score.
- Agente Clustering: K-Means, MiniBatchKMeans, HAC, DBSCAN y KMedoids si la versión de NumPy lo permite.
- Agente Clasificación: KNN, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Naive Bayes, Logistic Regression, SVM y XGBoost si está instalado.
- Agente Regresión: Linear Regression, Lasso, LassoCV, Ridge, RidgeCV, SVR, Decision Tree, Random Forest, Gradient Boosting y XGBoost si está instalado.
- Dashboard: Inicio, Carga de Datos, EDA, Clustering, Clasificación, Regresión y Ejecutivo.
- Diagramas UML/MAS: `docs/arquitectura_mas.md`.
- Descargas: reporte JSON y Markdown desde el Dashboard Ejecutivo.
- Control de rendimiento: modo rápido y modo completo.

## Modularización aplicada

- `schemas/`: estructuras compartidas (`ContextoMAS`, `MensajeAgente`).
- `agents/`: agentes responsables de decisión, coordinación, comunicación y registro.
- `services/`: lógica ejecutable reutilizable para EDA, targets, clustering, clasificación y regresión.
- `pipelines/`: punto de entrada para ejecutar el MAS completo.
- `reports/`: generación de reporte ejecutivo y exportadores JSON/Markdown.
- `config/`: constantes globales de aplicación.
- `ui/streamlit/components/`: componentes visuales reutilizables.

## Instalación

```bash
cd /Users/billitas/Documents/proyectos/LAB4/LAB4_AgenticIA_ML
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install streamlit pandas numpy scikit-learn matplotlib seaborn scikit-learn-extra umap-learn xgboost
```

## Ejecución

```bash
cd /Users/billitas/Documents/proyectos/LAB4/LAB4_AgenticIA_ML
.venv/bin/streamlit run src/ui/streamlit/streamlit_app.py
```

## Validación

```bash
.venv/bin/python -m compileall src main.py
```

Prueba rápida del coordinador:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
from src.pipelines.mas_pipeline import MASPipeline

df = pd.read_csv("Datasets/clasificacion/01_clasificacion_clientes.csv")
ctx = MASPipeline().run(
    df,
    target="churn",
    columnas_eliminar=["id_cliente", "fecha_alta"],
    ejecutar_regresion=False,
)
print(ctx.tipo_problema)
print(ctx.resultados["clasificacion"]["mejor"]["modelo"])
print(ctx.resultados["ejecutivo"]["conclusiones"])
PY
```
