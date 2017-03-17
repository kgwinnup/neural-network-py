
import numpy as np

class NeuralNet:

    def __init__(self, Y, X, layers=[1], iterations=10000, normalize=False, bias=0, activation='sigmoid'):
        """
        initialize this neural net object with training set, dependent variables and other optional parameters
        @param Y dependent variables
        @param X training set
        @param layers list of layers, which each integer describing how many nodes in that layer (don't include input layer)
        @param iterations, how many loops to 'learn'
        @param bias introduce a bias unit to the neuron, default is 0
        @param activation set the activation function, currently supports 'sigmoid' or 'htan'
        """
        self._X = X if not normalize else self.normalize(X)
        self._Y = Y
        self._layers = layers
        self._iterations = iterations
        self._synapses = None
        self._error = None
        self._normalize = normalize
        self._bias = bias

        activations = {
            'sigmoid': (self.sigmoid, self.sigmoid_derivitive),
            'htan': (self.htan, self.htan_derivitive)
        }

        if activation in activations:
            self._activation = activations[activation][0]
            self._activation_prime = activations[activation][1]
        else:
            self._activation = self.sigmoid
            self._activation_prime = self.sigmoid_derivitive

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def sigmoid_derivitive(self, t):
        return t * (1-t)

    def htan(self, t):
        return np.tanh(t)

    def htan_derivitive(self, t):
        return 1.0 - np.tanh(t)**2

    def forward(self, X, synapses):
        layers = []
        for i in range(len(synapses)):
            if i == 0:
                layers.append(self._activation(X.dot(synapses[i]) + self._bias))
            else:
                layers.append(self._activation(layers[i-1].dot(synapses[i]) + self._bias))
        return layers

    def predict(self, X):
        """given a new feature set, determine its learned dependent value"""
        Z = self.normalize(X) if self._normalize else X
        return {'yhat': self.forward(Z, self._synapses)[-1],
                'mse': np.mean(map(lambda x: x**2, self._error)),
                'normalize_X': True,
                'bias': self._bias,
                'layers': self._synapses}

    def normalize(self, M):
        return (M - M.mean()) / (M.max() - M.min())

    def learn(self):
        synapses = []
        synapses.append(2 * np.random.rand(self._X.shape[1], self._layers[0]) - 1)
        for i in range(1,len(self._layers)):
            synapses.append(2 * np.random.rand(self._layers[i-1], self._layers[i]) - 1)

        layers = []
        reverse = len(synapses) - 1
        for i in range(self._iterations):

            layers = self.forward(self._X, synapses)

            deltas = []
            for i in range(len(synapses)):
                if i == 0:
                    self._error = (self._Y - layers[reverse - i])
                    deltas.append( self._error * self._activation_prime(layers[reverse - i]) )
                else:
                    deltas.append( deltas[i-1].dot(synapses[reverse - i + 1].T) * self._activation_prime(layers[reverse - i]) )

            for i in range(len(synapses)):
                if i == len(synapses) - 1:
                    synapses[reverse - i] = synapses[reverse - i] + self._X.T.dot(deltas[i])
                else:
                    synapses[reverse - i] = synapses[reverse - i] + layers[reverse - i - 1].T.dot(deltas[i])

        self._synapses = synapses
