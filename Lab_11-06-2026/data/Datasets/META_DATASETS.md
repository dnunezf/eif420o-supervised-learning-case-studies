# META de datasets de prueba

Esta carpeta contiene datasets sinteticos para validar todas las funciones de la app.

## Estructura por tipo de caso

- `clasificacion/01_clasificacion_clientes.csv`
  - Objetivo recomendado: `churn`
  - Tipo: clasificacion
  - Incluye: columnas numericas, categoricas, ID, fecha y valores nulos

- `regresion/02_regresion_viviendas.csv`
  - Objetivo recomendado: `precio`
  - Tipo: regresion
  - Incluye: columnas numericas, categoricas, ID, fecha y algunos nulos en target

- `agrupamiento/03_agrupamiento_clientes.csv`
  - Objetivo recomendado: no aplica
  - Tipo: agrupamiento
  - Incluye: patrones de 3 grupos + ruido y nulos

- `mixto/04_mixto_todo_semicolon.csv`
  - Objetivo clasificacion: `riesgo_fuga`
  - Objetivo regresion: `valor_vida`
  - Tipo: mixto
  - Formato: separador `;`
  - Incluye: ID, fecha, categoricas, numericas y nulos

## Pruebas sugeridas en UI

1. Cargar cada archivo y revisar deteccion de separador.
2. Eliminar columnas de ruido (`id_*`, `fecha_*`) desde "Columnas a eliminar antes de entrenar".
3. Ejecutar:
   - "Probar modelos de clasificación"
   - "Probar modelos de regresión"
   - "Probar modelos de agrupamiento"
