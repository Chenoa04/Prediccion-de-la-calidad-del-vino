import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Cargar el conjunto de datos
datos = pd.read_csv('winequality-red.csv')

# 2. Separar características y variable objetivo
X = datos.drop('quality', axis=1)
y = datos['quality']

# 3. Dividir en conjuntos de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Estandarización de los datos
escalador = StandardScaler()
X_entrenamiento = escalador.fit_transform(X_entrenamiento)
X_prueba = escalador.transform(X_prueba)

# 5. Entrenar el modelo de Árbol de Decisión
modelo_arbol = DecisionTreeRegressor(random_state=42)
modelo_arbol.fit(X_entrenamiento, y_entrenamiento)

# 6. Predecir y evaluar el rendimiento del Árbol de Decisión
y_pred_arbol = modelo_arbol.predict(X_prueba)
mse_arbol = mean_squared_error(y_prueba, y_pred_arbol)
print(f'MSE del Árbol de Decisión: {mse_arbol}')

# 7. Entrenar el modelo de Random Forest
modelo_rf = RandomForestRegressor(random_state=42)
modelo_rf.fit(X_entrenamiento, y_entrenamiento)

# 8. Predecir y evaluar el rendimiento del Random Forest
y_pred_rf = modelo_rf.predict(X_prueba)
mse_rf = mean_squared_error(y_prueba, y_pred_rf)
print(f'MSE del Random Forest: {mse_rf}')

# 9. Importancia de características
importancias = modelo_rf.feature_importances_
nombres_caracteristicas = X.columns

# Crear un DataFrame para visualizar la importancia
df_importancia = pd.DataFrame({'Característica': nombres_caracteristicas, 'Importancia': importancias})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

# 10. Visualizar la importancia de características
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Característica', data=df_importancia)
plt.title('Importancia de Características en Random Forest')
plt.show()
