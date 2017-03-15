
import numpy as np

class NeuralNet:

    def __init__(self, Y, X, hidden=[1], iterations=10000, normalize=False):
        """
        initialize this neural net object with training set, dependent variables and other optional parameters
        @param Y dependent variables
        @param X training set
        @param hidden list of layers, which each integer describing how many nodes in that layer
        @param iterations, how many loops to 'learn'
        """
        self._X = X if not normalize else self.normalize(X)
        self._Y = Y
        self._hidden = hidden
        self._iterations = iterations
        self._synapses = None
        self._error = None
        self._normalize = normalize

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def sigmoid_derivitive(self, t):
        return t * (1-t)

    def forward(self, X, synapses):
        layers = []
        for i in range(0, len(synapses)):
            if i == 0:
                layers.append(self.sigmoid(X.dot(synapses[i])))
            else:
                layers.append(self.sigmoid(layers[i-1].dot(synapses[i])))
        return layers

    def predict(self, X):
        """given a new feature set, determine its learned dependent value"""
        Z = self.normalize(X) if self._normalize else X
        return {'yhat': self.forward(Z, self._synapses)[-1],
                'mse': np.mean(map(lambda x: x**2, self._error)),
                'normalize_X': True,
                'layers': self._synapses}

    def normalize(self, M):
        return (M - M.mean()) / (M.max() - M.min())

    def learn(self):
        synapses = []
        synapses.append(2 * np.random.rand(self._X.shape[1], self._hidden[0]) - 1)
        for i in range(1,len(self._hidden)):
            synapses.append(2 * np.random.rand(self._hidden[i-1], self._hidden[i]) - 1)
        synapses.append(2 * np.random.rand(self._hidden[-1], 1) - 1)

        layers = []
        reverse = len(synapses) - 1
        for i in range(0, self._iterations):

            layers = self.forward(self._X, synapses)

            deltas = []
            for i in range(0, len(synapses)):
                if i == 0:
                    self._error = (self._Y - layers[reverse - i])
                    deltas.append( self._error * self.sigmoid_derivitive(layers[reverse - i]) )
                else:
                    deltas.append( deltas[i-1].dot(synapses[reverse - i + 1].T) * self.sigmoid_derivitive(layers[reverse - i]) )

            for i in range(0, len(synapses)):
                if i == len(synapses) - 1:
                    synapses[reverse - i] = synapses[reverse - i] + self._X.T.dot(deltas[i])
                else:
                    synapses[reverse - i] = synapses[reverse - i] + layers[reverse - i - 1].T.dot(deltas[i])

        self._synapses = synapses
