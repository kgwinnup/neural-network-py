
from kneuralnet.nn import NeuralNet
import pytest
import numpy as np

data = np.array([[0,0,0],
                 [0,1,1],
                 [1,0,1],
                 [1,1,0]])
X = np.array(data[:,0:-1])
Y = np.array([data[:,-1]]).T

nand = np.array([[0,0,1],
                 [0,1,1],
                 [1,0,1],
                 [1,1,0]])
nX = np.array(nand[:,0:-1])
nY = np.array([nand[:,-1]]).T

def test_xor_11():
    nn = NeuralNet(Y, X, layers=[4,1])
    nn.learn()
    res = nn.predict(np.array([1,1]))
    assert(round(res['yhat'][0]) == 0.0)

def test_xor_00():
    nn = NeuralNet(Y, X, layers=[4,1])
    nn.learn()
    res = nn.predict(np.array([0,0]))
    assert(round(res['yhat'][0]) == 0.0)

def test_xor_10():
    nn = NeuralNet(Y, X, layers=[4,1])
    nn.learn()
    res = nn.predict(np.array([1,0]))
    assert(round(res['yhat'][0]) == 1.0)

def test_xor_01():
    nn = NeuralNet(Y, X, layers=[4,1])
    nn.learn()
    res = nn.predict(np.array([0,1]))
    assert(round(res['yhat'][0]) == 1.0)

def test_xor_01_b1():
    nn = NeuralNet(Y, X, layers=[4,1], bias=1)
    nn.learn()
    res = nn.predict(np.array([0,1]))
    assert(round(res['yhat'][0]) == 1.0)

def test_nand_01():
    nn = NeuralNet(nY, nX, layers=[3,1], bias=1)
    nn.learn()
    res = nn.predict(np.array([0,1]))
    assert(round(res['yhat'][0]) == 1.0)

def test_nand_11():
    nn = NeuralNet(nY, nX, layers=[3,1], bias=1)
    nn.learn()
    res = nn.predict(np.array([1,1]))
    assert(round(res['yhat'][0]) == 0.0)
