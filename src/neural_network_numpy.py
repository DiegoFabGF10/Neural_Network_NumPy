# Importar librerías necesarias
import numpy as np  # Para cálculos numéricos
import matplotlib.pyplot as plt  # Para graficar resultados
from sklearn.datasets import make_gaussian_quantiles  # Para generar datos de clasificación

# Crear un dataset de clasificación con dos clases
def create_dataset(N=1000):
    """
    Genera un conjunto de datos de clasificación con distribución gaussiana.
    :param N: Número de muestras a generar.
    :return: X (características), Y (etiquetas)
    """

    gaussian_quantiles = make_gaussian_quantiles(
        mean=None,     # Media de la distribución (None usa la predeterminada)
        cov=0.1,       # Matriz de covarianza (dispersión de los puntos)
        n_samples=N,   # Número de muestras
        n_features=2,  # Número de características (dimensión)
        n_classes=2,   # Número de clases (binario en este caso)
        shuffle=True,  # Barajar los datos
        random_state=None  # Estado aleatorio (None significa que varía en cada ejecución)
    )
    
    X, Y = gaussian_quantiles  # Desempaquetar datos
    Y = Y[:, np.newaxis]  # Convertir a una matriz columna para compatibilidad con cálculos
    return X, Y

# Funciones de activación
def sigmoid(x, derivate=False):
    """
    Función de activación Sigmoide.
    :param x: Entrada.
    :param derivate: Si es True, devuelve la derivada de la sigmoide.
    :return: Valor de la sigmoide o su derivada.
    """
    if derivate:
        return np.exp(-x) / (np.exp(-x) + 1)**2  # Derivada de la sigmoide
    else:
        return 1 / (1 + np.exp(-x))  # Función sigmoide

def relu(x, derivate=False):
    """
    Función de activación ReLU (Rectified Linear Unit).
    :param x: Entrada.
    :param derivate: Si es True, devuelve la derivada de ReLU.
    :return: Valor de la función ReLU o su derivada.
    """
    if derivate:
        x[x <= 0] = 0  # Derivada de ReLU es 0 para x <= 0
        x[x > 0] = 1   # Derivada de ReLU es 1 para x > 0
        return x
    else:
        return np.maximum(0, x)  # ReLU: max(0, x)

# Función de pérdida (Error Cuadrático Medio - MSE)
def mse(y, y_hat, derivate=False):
    """
    Calcula el error cuadrático medio entre la predicción y la etiqueta real.
    :param y: Valores reales.
    :param y_hat: Valores predichos.
    :param derivate: Si es True, devuelve la derivada del MSE.
    :return: Valor del error o su derivada.
    """
    if derivate:
        return (y_hat - y)  # Derivada del MSE
    else:
        return np.mean((y_hat - y)**2)  # Fórmula del MSE

# Inicialización de pesos y sesgos de la red neuronal
def initialize_parameters_deep(layers_dims):
    """
    Inicializa los pesos y sesgos para una red neuronal profunda.
    :param layers_dims: Lista con el número de neuronas por capa.
    :return: Diccionario con los parámetros inicializados.
    """
    parameters = {}
    L = len(layers_dims)  # Número total de capas
    
    for l in range(0, L-1):
        parameters['W' + str(l+1)] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1  # Pesos aleatorios
        parameters['b' + str(l+1)] = (np.random.rand(1, layers_dims[l+1]) * 2) - 1  # Sesgos aleatorios
    
    return parameters

# Entrenamiento de la red neuronal
def train(x_data, y_data, learning_rate, params, training=True):
    """
    Realiza una pasada de forward y backward en la red neuronal.
    :param x_data: Datos de entrada.
    :param y_data: Etiquetas reales.
    :param learning_rate: Tasa de aprendizaje.
    :param params: Diccionario con los parámetros de la red.
    :param training: Si es True, realiza la retropropagación.
    :return: Salida de la red neuronal.
    """

    params['A0'] = x_data  # Asignar entrada a la primera capa

    # Propagación hacia adelante (Forward Propagation)
    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])

    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])

    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])  # Salida final

    output = params['A3']

    if training:
        # Retropropagación (Backpropagation)
        params['dZ3'] = mse(y_data, output, True) * sigmoid(params['A3'], True)
        params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])

        params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
        params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])

        params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
        params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])

        # Actualización de pesos y sesgos con descenso de gradiente
        params['W3'] -= params['dW3'] * learning_rate
        params['W2'] -= params['dW2'] * learning_rate
        params['W1'] -= params['dW1'] * learning_rate

        params['b3'] -= np.mean(params['dW3'], axis=0, keepdims=True) * learning_rate
        params['b2'] -= np.mean(params['dW2'], axis=0, keepdims=True) * learning_rate
        params['b1'] -= np.mean(params['dW1'], axis=0, keepdims=True) * learning_rate

    return output

# Función principal para entrenar el modelo
def train_model():
    """
    Entrena la red neuronal con los datos generados.
    """
    
    # Crear dataset
    X, Y = create_dataset()

    # Definir la arquitectura de la red neuronal (número de neuronas por capa)
    layers_dims = [2, 6, 10, 1]

    # Inicializar los parámetros de la red
    params = initialize_parameters_deep(layers_dims)

    # Lista para almacenar el error en cada iteración
    error = []

    # Ciclo de entrenamiento de 50,000 iteraciones
    for _ in range(50000):
        output = train(X, Y, 0.001, params)  # Ejecutar una iteración de entrenamiento
        
        # Cada 50 iteraciones, imprimir el error
        if _ % 50 == 0:
            print(mse(Y, output))
            error.append(mse(Y, output))

    # Graficar los datos de entrada con sus respectivas etiquetas
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()
