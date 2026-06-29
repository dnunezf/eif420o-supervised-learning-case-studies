Dataset para LAB 3 - Regresión

Archivo: diabetes_regression_lab03.csv
Filas: 442
Columnas predictoras: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
Variable respuesta: target
Tipo de problema: regresión supervisada
Fuente técnica: dataset Diabetes incluido en scikit-learn, exportado a CSV para reproducibilidad del laboratorio.

Uso en Python:
import pandas as pd
df = pd.read_csv('data/diabetes_regression_lab03.csv')
