
from kneuralnet.nn import NeuralNet
import pytest
import numpy as np

data = np.array([[0,0,0],
                 [0,1,1],
                 [1,0,1],
                 [1,1,0]])
X = np.array(data[:,0:-1])
Y = np.array([data[:,-1]]).T

def test_xor_11():
    hidden = [4,2,3]
    nn = NeuralNet(Y, X, hidden=hidden)
    nn.learn()
    res = nn.predict(np.array([1,1]))
    assert(round(res['yhat'][0]) == 0.0)

def test_xor_00():
    hidden = [4]
    nn = NeuralNet(Y, X, hidden=hidden)
    nn.learn()
    res = nn.predict(np.array([0,0]))
    assert(round(res['yhat'][0]) == 0.0)

def test_xor_10():
    hidden = [3]
    nn = NeuralNet(Y, X, hidden=hidden)
    nn.learn()
    res = nn.predict(np.array([1,0]))
    assert(round(res['yhat'][0]) == 1.0)

def test_xor_01():
    hidden = [2,2]
    nn = NeuralNet(Y, X, hidden=hidden)
    nn.learn()
    res = nn.predict(np.array([0,1]))
    assert(round(res['yhat'][0]) == 1.0)
