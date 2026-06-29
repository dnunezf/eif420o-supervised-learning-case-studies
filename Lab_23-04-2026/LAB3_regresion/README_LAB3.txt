LAB 3 - Regresión Supervisada

Archivos principales:
1. ModuloRegresionLab03.py
   Paquete pythónico con clases para EDA, preprocesamiento, modelos, validación cruzada,
   benchmark, selección del mejor modelo, gráficos y exportación de resultados.

2. InterfazRegresionLab03.ipynb
   Notebook de uso paso a paso. Por defecto usa diabetes_regression_lab03.csv.
   Para usar un CSV del profesor, cambiar:
      DATASET_PATH = 'archivo.csv'
      COLUMNA_OBJETIVO = 'nombre_columna_target'

3. main_lab3_regresion.tex
   Informe LaTeX con estructura académica: introducción, teoría, dataset,
   metodología, resultados, análisis, conclusiones y bibliografía.

4. diabetes_regression_lab03.csv
   Dataset reproducible basado en Diabetes de scikit-learn.

5. outputs_lab03/
   Carpeta con figuras y resultados ya generados para el informe.

Nota importante:
- Si el dataset Potabilidad tiene una variable objetivo binaria 0/1, corresponde a clasificación,
  no a regresión. Para regresión se requiere una variable objetivo numérica continua.
- El notebook valida esto con detectar_problema_regresion().
