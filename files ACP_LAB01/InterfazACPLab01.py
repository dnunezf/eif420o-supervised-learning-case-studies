"""
InterfazACPLab01.py
===================
Script de interfaz / ejecucion del Analisis de Componentes Principales.
Laboratorio 01, EIF420O Inteligencia Artificial, I Ciclo 2026 - UNA

Este script demuestra el uso completo del modulo ModuloACPLab01.py
sobre el dataset BankChurners.csv.

Ejecutar:
    python InterfazACPLab01.py

O copiar las celdas a un Jupyter Notebook.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from ModuloACPLab01 import ACPAnalisis, ACPExperimento

# 1. CARGA DEL DATASET
print("=" * 60)
print("1. CARGA DEL DATASET")
print("=" * 60)

path = "/home/dnunezf/Documents/AI_LABS/Lab_19-03-2026/files ACP_LAB01/BankChurners.csv"   
df_raw = pd.read_csv(path, sep=",", decimal=".", index_col=0)

print(f"Dimensiones: {df_raw.shape}")
print(f"Columnas: {df_raw.columns.tolist()}")
print(f"\nTipos de datos:\n{df_raw.dtypes}")
print(f"\nValores nulos: {df_raw.isnull().sum().sum()}")

# 2. SELECCION DE VARIABLES NUMERICAS
print("\n" + "=" * 60)
print("2. SELECCION DE VARIABLES NUMERICAS")
print("=" * 60)

df_numerico = df_raw.select_dtypes(include=["number"])
print(f"Variables numericas ({df_numerico.shape[1]}): {df_numerico.columns.tolist()}")
print(f"\nEstadisticas descriptivas:")
print(df_numerico.describe().round(2))

# 3. EJECUTAR EXPERIMENTO ACP (6 configuraciones)
print("\n" + "=" * 60)
print("3. EXPERIMENTO ACP")
print("=" * 60)

# Crear el experimento pasando datos numericos y el df completo (para dummies)
# Si desea guardar figuras, pase carpeta="ruta/figuras"
exp = ACPExperimento(df_numerico, df_completo=df_raw, carpeta=None)

# Ejecutar: ajusta los 6 modelos y selecciona el mejor
exp.ejecutar()

# 4. BENCHMARK COMPARATIVO
print("\n" + "=" * 60)
print("4. BENCHMARK COMPARATIVO")
print("=" * 60)

tabla = exp.benchmark()
print(tabla.to_string(index=False))

# Grafico comparativo de varianza acumulada
exp.plot_comparacion_varianza()

# Circulos de correlacion comparativos
exp.plot_comparacion_circulos()

# 5. ANALISIS DEL MEJOR MODELO
print("\n" + "=" * 60)
print("5. ANALISIS DEL MEJOR MODELO")
print("=" * 60)

mejor = exp.mejor_modelo
print(f"\nModelo seleccionado: {exp.mejor_nombre}")
print(mejor)

# Resumen de eigenvalues
print("\n--- Resumen ---")
print(mejor.resumen().round(4))

# Correlaciones variable-componente
print("\n--- Correlaciones ---")
print(mejor.correlacion_var.round(4))

# Cos2 de variables
print("\n--- cos2 de variables ---")
print(mejor.cos2_var.round(4))

# Interpretacion textual
print("\n--- Interpretacion de componentes ---")
mejor.interpretacion(top_n=5)

# 6. GRAFICOS DEL MEJOR MODELO
print("\n" + "=" * 60)
print("6. GRAFICOS DEL MEJOR MODELO")
print("=" * 60)

# Scree plot
mejor.plot_varianza()

# Plano principal CP0-CP1
mejor.plot_plano_principal(ejes=[0, 1])

# Circulo de correlacion CP0-CP1
mejor.plot_circulo(ejes=[0, 1])

# Biplot CP0-CP1
mejor.plot_sobreposicion(ejes=[0, 1])

# Planos adicionales si hay 3+ componentes
if mejor.n_comp >= 3:
    mejor.plot_plano_principal(ejes=[0, 2])
    mejor.plot_circulo(ejes=[0, 2])
    mejor.plot_plano_principal(ejes=[1, 2])
    mejor.plot_circulo(ejes=[1, 2])

# Mapa de calor cos2
mejor.plot_cos2_variables()

# Contribuciones por componente
for cp in range(min(mejor.n_comp, 4)):
    mejor.plot_contribuciones(comp=cp)

# 7. USO INDIVIDUAL DE ACPAnalisis (ejemplo directo)
print("\n" + "=" * 60)
print("7. USO INDIVIDUAL DE ACPAnalisis")
print("=" * 60)

from sklearn.preprocessing import RobustScaler

# Escalar manualmente
datos_escalados = pd.DataFrame(
    RobustScaler().fit_transform(df_numerico),
    columns=df_numerico.columns,
    index=df_numerico.index
)

# Crear instancia directa
mi_acp = ACPAnalisis(datos_escalados, n_componentes=4, nombre='Mi ACP Robusto')

# Consultar
print(mi_acp)
print(f"\nKaiser sugiere: {mi_acp.kaiser()} componentes")
print(f"\nResumen:\n{mi_acp.resumen().round(4)}")

# Acceso a atributos
print(f"\nEigenvalues: {mi_acp.eigenvalues.round(4)}")
print(f"Varianza explicada: {mi_acp.var_explicada.round(2)}")
print(f"Varianza acumulada: {mi_acp.var_acumulada.round(2)}")

# Graficos individuales
mi_acp.plot_circulo(ejes=[0, 1])
mi_acp.plot_cos2_variables()

print("\n" + "=" * 60)
print("EJECUCION COMPLETADA")
print("=" * 60)
