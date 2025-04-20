# Prediccion-de-la-calidad-del-vino
  Este proyecto tiene como objetivo predecir la calidad del vino utilizando modelos de aprendizaje automático, específicamente un Árbol de Decisión y un Random Forest. El conjunto de datos utilizado es el winequality-red.csv, que contiene información sobre diferentes características químicas del vino y su calidad.
El script ejercicio.py realiza lo siguiente:
1. Carga el conjunto de datos desde un archivo CSV.
2. Separación de Características (atributos del vino) y Variable Objetivo (calidad del vino).
3. Divide el conjunto de datos en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%).
4. Aplica técnicas de escalado para normalizar las características.
5. Entrena un modelo de Árbol de Decisión con los datos de entrenamiento.
6. Realiza predicciones con el modelo y se evalúa su rendimiento utilizando el error cuadrático medio (MSE).
7. Entrenamiento del Modelo de Random Forest: Se entrena un modelo de Random Forest con los mismos datos.
8. Realiza predicciones con el modelo Random Forest y se evalúa su rendimiento.
9. Calcula la importancia de cada característica en el modelo Random Forest.
10. Vsualiza la importancia de las características mediante un gráfico de barras.
