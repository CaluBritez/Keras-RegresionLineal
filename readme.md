# Objetivo

Desarrollar un script en Python que implemente un modelo de Regresión Lineal utilizando Keras, basado en un conjunto de datos que incluye altura y peso de personas. El modelo debe permitir encontrar la relación entre la altura y el peso, realizar predicciones y visualizar los resultados.


## Requisitos

- Python 3.x
- pandas
- numpy
- matplotlib
- keras (con TensorFlow)

## Pasos
1. Clonar este repositorio

```bash
git clone https://github.com/CaluBritez/Keras-RegresionLineal.git
```

2. Moverse al directorio

```bash
cd Keras-RegresionLineal
```

3. Instalar dependencias 
```bash
pip install pandas numpy matplotlib tensorflow keras
```

4. Ejecutar el script
```bash
python main.py
```

### Explicacion

Desarrollamos un modelo de regresión lineal simple para analizar la relación entre la altura y el peso de un conjunto de datos. El modelo fue implementado utilizando Keras, con un enfoque en la normalización de los datos antes del entrenamiento para mejorar la precisión y estabilidad del modelo.

### Resultados Obtenidos:

Tras entrenar el modelo con 10000 épocas, en primera instancia graficamos el error cuadrático medio vs. el número de épocas.
Luego graficamos la regresion lineal resultante del presente proyecto.
