

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from kneuralnet.nn import NeuralNet

categories = {'Iris-setosa': 0,
              'Iris-virginica': 1,
              'Iris-versicolor': 1}

if __name__ == '__main__':
    # get and clean the data
    df = pd.read_csv("iris.csv")
    df['category'] = df['Species'].map(lambda x: categories[x])
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].as_matrix()
    Y = df[['category']].as_matrix()

    # run the NN module and output results
    nn = NeuralNet(Y, X, normalize=True, hidden=[2], iterations=100000)
    nn.learn()
    out = nn.predict(nn.normalize(X))
    print map(lambda x: round(x), out['yhat'])
    print out['mse']
