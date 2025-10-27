## ============================================================
## Implementación del método de Descenso de Gradiente
##    y Descenso de Gradiente con Momento
## ============================================================

import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activations=None, dropout_rates=None):
        # --- Inicialización idéntica al código original ---
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        if activations is None:
            self.activations = ['tanh'] * (self.num_layers - 1)
        else:
            self.activations = activations

        # Activaciones y sus derivadas
        self.activation_funcs = {
            'tanh': lambda x: np.tanh(x),
            'relu': lambda x: np.maximum(0, x),
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x))
        }
        self.activation_primes = {
            'tanh': lambda x: 1.0 - np.tanh(x)**2,
            'relu': lambda x: (x > 0).astype(float),
            'sigmoid': lambda x: self.activation_funcs['sigmoid'](x) * (1 - self.activation_funcs['sigmoid'](x))
        }

        # Inicialización de pesos (Glorot o He)
        def glorot(n_in, n_out):
            limit = np.sqrt(6 / (n_in + n_out))
            return np.random.uniform(-limit, limit, (n_out, n_in))

        def he(n_in, n_out):
            return np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)

        self.weights = []
        for i in range(len(layer_sizes) - 1):
            if self.activations[i] == 'relu':
                self.weights.append(he(layer_sizes[i], layer_sizes[i + 1]))
            else:
                self.weights.append(glorot(layer_sizes[i], layer_sizes[i + 1]))

        # Sesgos y velocidades (para momentum)
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        self.velocity_b = [np.zeros(b.shape) for b in self.biases]
        self.velocity_w = [np.zeros(w.shape) for w in self.weights]

        self.lambd = 0.0
        self.training = True

    # ============================================================
    # Feedforward: calcula la salida de la red
    # ============================================================
    def feedforward(self, a, return_intermediates=False):
        activations = [a]
        zs = []

        for b, w, act in zip(self.biases, self.weights, self.activations):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            a = self.activation_funcs[act](z)
            activations.append(a)

        if return_intermediates:
            return activations, zs
        else:
            return activations[-1]

    # ============================================================
    # Backpropagation: calcula gradientes de pesos y sesgos
    # ============================================================
    def backpropagation(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass
        activations, zs = self.feedforward(x, return_intermediates=True)

        # Error en la capa de salida
        delta = self.cost_derivative(activations[-1], y)
        delta *= self.activation_primes[self.activations[-1]](zs[-1])

        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].T)

        # Propagar hacia atrás
        for l in range(2, len(self.biases) + 1):
            z = zs[-l]
            sp = self.activation_primes[self.activations[-l]](z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l - 1].T)

        return grad_b, grad_w

    # ============================================================
    # Método NUEVO: Descenso de Gradiente con o sin Momento
    # ============================================================
    def gradient_descent(self, mini_batch, eta, mu=0.0):
        """
        Actualiza los pesos y sesgos de la red usando descenso de gradiente
        (mu=0 → descenso clásico, mu>0 → descenso con momento)

        Parámetros:
        -----------
        mini_batch : lista [(x, y), ...]
            Datos del batch
        eta : float
            Tasa de aprendizaje
        mu : float
            Coeficiente de momento (por defecto 0)
        """
        # Inicializar acumuladores de gradientes
        grad_b_total = [np.zeros(b.shape) for b in self.biases]
        grad_w_total = [np.zeros(w.shape) for w in self.weights]

        # Acumular gradientes de cada muestra del batch
        for x, y in mini_batch:
            grad_b, grad_w = self.backpropagation(x, y)
            grad_b_total = [gb + dgb for gb, dgb in zip(grad_b_total, grad_b)]
            grad_w_total = [gw + dgw for gw, dgw in zip(grad_w_total, grad_w)]

        # Actualización de pesos y sesgos
        for i in range(len(self.weights)):
            # Cálculo de velocidades (para momentum)
            self.velocity_w[i] = mu * self.velocity_w[i] - (eta / len(mini_batch)) * grad_w_total[i]
            self.velocity_b[i] = mu * self.velocity_b[i] - (eta / len(mini_batch)) * grad_b_total[i]

            # Actualizar parámetros
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    # ============================================================
    # Función de costo: derivada del error cuadrático medio
    # ============================================================
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    # ============================================================
    # Método de entrenamiento (ya lo tenías en tu notebook)
    # ============================================================
    def train(self, training_data, epochs, mini_batch_size, learning_rate, mu=0.0,
              validation_data=None, verbose=True):
        n = len(training_data)
        loss_history = []

        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.gradient_descent(mini_batch, learning_rate, mu)

            # Calcular pérdida promedio
            current_loss = np.mean([np.mean((self.feedforward(x) - y) ** 2) for x, y in training_data])
            loss_history.append(current_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {current_loss:.6f}")

        return loss_history
