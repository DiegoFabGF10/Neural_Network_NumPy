# Red Neuronal para Clasificación de Datos

Este proyecto implementa una **red neuronal multicapa** en **Python** para clasificar datos generados artificialmente con una distribución gaussiana.

## Características
- **Generación de datos** con `make_gaussian_quantiles()`.
- **Implementación de una red neuronal** con 3 capas ocultas.
- **Funciones de activación:** Sigmoide y ReLU.
- **Backpropagation y gradiente descendente** para entrenamiento.
- **Visualización de datos** con `matplotlib`.

## Requisitos

Asegúrate de tener **Python 3.7 o superior** y las siguientes librerías:

```bash
pip install numpy matplotlib scikit-learn

## **Instalacion**
## 1.- Clonar el repositorio:
```bash
git clone https://github.com/DiegoFabGF10/Neural_Network_NumPy.git
cd Neural_Network_NumPy

## 2.- (Opcional) Crear un entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

## 3.- Instalar las dependencias
```bash
pip install -r requirements.txt

## **Ejecutar el script principal**
```bash
python main.py

Esto generará un conjunto de datos, entrenará la red neuronal y mostrará la clasificación en un gráfico.
