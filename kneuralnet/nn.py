
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
        self.X = X if not normalize else self.normalize(X)
        self.Y = Y
        self.hidden = hidden
        self.iterations = iterations
        self.synapses = None
        self.error = None
        self.normalized = normalize

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
        Z = self.normalize(X) if self.normalized else X
        return {'yhat': self.forward(Z, self.synapses)[-1],
                'mse': np.mean(map(lambda x: x**2, self.error)),
                'normalized_X': True,
                'layers': self.synapses}

    def normalize(self, M):
        return (M - M.mean()) / (M.max() - M.min())

    def learn(self):
        synapses = []
        synapses.append(2 * np.random.rand(self.X.shape[1], self.hidden[0]) - 1)
        for i in range(1,len(self.hidden)):
            synapses.append(2 * np.random.rand(self.hidden[i-1], self.hidden[i]) - 1)
        synapses.append(2 * np.random.rand(self.hidden[-1], 1) - 1)

        layers = []
        reverse = len(synapses) - 1
        for i in range(0, self.iterations):

            layers = self.forward(self.X, synapses)

            deltas = []
            for i in range(0, len(synapses)):
                if i == 0:
                    self.error = (self.Y - layers[reverse - i])
                    deltas.append( self.error * self.sigmoid_derivitive(layers[reverse - i]) )
                else:
                    deltas.append( deltas[i-1].dot(synapses[reverse - i + 1].T) * self.sigmoid_derivitive(layers[reverse - i]) )

            for i in range(0, len(synapses)):
                if i == len(synapses) - 1:
                    synapses[reverse - i] = synapses[reverse - i] + self.X.T.dot(deltas[i])
                else:
                    synapses[reverse - i] = synapses[reverse - i] + layers[reverse - i - 1].T.dot(deltas[i])

        self.synapses = synapses
