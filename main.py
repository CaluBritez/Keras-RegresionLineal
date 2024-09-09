import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.api.models import Sequential
from keras.api.layers import Dense, Input
from keras.api.optimizers import SGD   


# FUNCIONES
def normalizar_datos(data):
    x = data['Altura'].values
    y = data['Peso'].values

    # Normalizamos en X
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_normalized = (x - x_mean) / x_std

    # Normalizamos en Y
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = (y - y_mean) / y_std

    return x, y, x_mean, y_mean, x_normalized, y_normalized, x_std, y_std

def generation_model():

    # Crear el modelo
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(Dense(1, activation='linear'))

    # Definir el optimizador
    optimizer = SGD(learning_rate=0.01)  # Ajustar la tasa de aprendizaje
    model.compile(optimizer=optimizer, loss='mse')

    return model
    
def train_model(model, x_normalized, y_normalized):
    # Entrenar el modelo
    history = model.fit(x_normalized, y_normalized, epochs=10000, batch_size=len(x_normalized), verbose=1)

    # Verificar la estructura del modelo
    model.summary()

    # Imprimir los parámetros del modelo
    weights, bias = model.layers[0].get_weights()
    print(f'Peso (w): {weights[0][0]}, Sesgo (b): {bias[0]}')

    return history



# EJECUTAMOS

# Leer los datos
data = pd.read_csv('altura_peso.csv')

# Normalizar los datos
x, y, x_mean, y_mean, x_normalized, y_normalized, x_std, y_std = normalizar_datos(data)

# Crear el modelo
modelo = generation_model()

# Entrenar el modelo
historial = train_model(modelo, x_normalized, y_normalized)

# GRAFICAMOS

# Graficar el error cuadrático medio vs. el número de épocas
plt.figure()
plt.plot(historial.history['loss'])
plt.xlabel('Épocas')
plt.ylabel('Error Cuadrático Medio')
plt.title('ECM vs. Número de Épocas')
plt.show()

# Superponer la recta de regresión sobre los datos originales
plt.figure()
plt.scatter(x, y, label='Datos Originales')
plt.plot(x, modelo.predict((x - x_mean) / x_std) * y_std + y_mean, color='red', label='Recta de Regresión')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.legend()
plt.show()

# Predicción dada una altura

altura_especifica = 180
altura_normalizada = (altura_especifica - x_mean) / x_std
peso_normalizado = modelo.predict(np.array([altura_normalizada]))
peso_predicho = peso_normalizado * y_std + y_mean
print(f'Predicción del peso para una altura de {altura_especifica} cm: {peso_predicho[0][0]} kg')