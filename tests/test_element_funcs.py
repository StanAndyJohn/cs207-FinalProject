import pytest
import sys
sys.path.append('..')
import numpy as np
from autoDiff.autoDiff import Variable
from autoDiff import element_func as fun



def test_exp():
    x1 = fun.exp(Variable(1))
    # x2 = Variable(1)
    # x3 = AutoDiff.exp(x1 + x2)
    x4 = fun.exp(1)
    assert x1.val == np.exp(1)
    assert x1.der == {'x':np.exp(1)}
    # assert x3.der == {'x1': np.exp(1+1), 'x2': np.exp(1+1)}
    assert x4 == np.exp(1)


def test_log():
    x1 = fun.log(Variable(1))
    # x2 = Variable(1, name='x2')
    # x3 = AutoDiff.log(x1 + x2)
    x4 = fun.log(1)
    x5 = fun.log2(Variable(1))
    x6 = fun.log2(1)
    x7 = fun.log10(Variable(1))
    x8 = fun.log10(1)
    assert x1.val == np.log(1)
    assert x1.der == {'x':1}
    assert x5.val == np.log2(1)
    assert x5.der == {'x':1}
    assert x7.val == np.log10(1)
    assert x7.der == {'x':1}
    # assert x3.val == np.log(1+1)
    # assert x3.der == {'x1': 1/2, 'x2': 1/2}
    assert x4 == np.log(1)
    assert x6 == np.log2(1)
    assert x8 == np.log10(1)

def test_trigonometric():
    x1 = fun.sin(Variable(np.pi/4))
    x2 = fun.cos(Variable(np.pi/4))
    x3 = fun.tan(Variable(np.pi/4))
    # x4 = AutoDiff.sin(x1) + AutoDiff.cos(x2) + AutoDiff.tan(x3)
    x5 = np.pi/4
    x6 = np.pi/4
    x7 = np.pi/4
    x8 = fun.sin(x5) + fun.cos(x6) + fun.tan(x7)
    assert x1.val == np.sin(np.pi/4)
    assert x2.val == np.cos(np.pi/4)
    assert x3.val == np.tan(np.pi/4)
    assert x1.der == {'x':np.cos(np.pi/4)}
    assert x2.der == {'x':-np.sin(np.pi/4)}
    assert abs(x3.der['x'] - 1/np.cos(np.pi/4)**2 < 1e-8)
    # assert x4.val == 2**0.5 + 1
    # assert x4.der == {'x1': np.cos(np.pi/4), 
    #     'x2': -np.sin(np.pi/4),
    #     'x3': 1/np.cos(np.pi/4)**2}
    assert x8 == 2**0.5 + 1

def test_hyberbolic():
    x1 = fun.sinh(Variable(1))
    x2 = fun.cosh(Variable(1))
    x3 = fun.tanh(Variable(1))
    # x4 = AutoDiff.sinh(x1) + AutoDiff.cosh(x2) + AutoDiff.tanh(x3)
    x5 = 1
    x6 = 1
    x7 = 1
    x8 = fun.sinh(x5) + fun.cosh(x6) + fun.tanh(x7)
    assert x1.val == np.sinh(1)
    assert x2.val == np.cosh(1)
    assert x3.val == np.tanh(1)
    assert x1.der == {'x':np.cosh(1)}
    assert x2.der == {'x':np.sinh(1)}
    assert x3.der == {'x':1-np.tanh(1)**2}
    # assert x4.val == np.sinh(1)+np.cosh(1)+np.tanh(1)
    # assert x4.der == {'x1': np.cosh(1), 
    #     'x2': np.sinh(1),
    #     'x3': 1-np.tanh(1)**2}
    assert x8 == np.sinh(1)+np.cosh(1)+np.tanh(1)

def test_inverse_trigonometric():
    x1 = fun.arcsin(Variable(0.1))
    x2 = fun.arccos(Variable(0.2))
    x3 = fun.arctan(Variable(0.3))
    # x4 = AutoDiff.arcsin(x1) + AutoDiff.arccos(x2) + AutoDiff.arctan(x3)
    x5 = 0.1
    x6 = 0.2
    x7 = 0.3
    x8 = fun.arcsin(x5) + fun.arccos(x6) + fun.arctan(x7)
    assert x1.val == np.arcsin(0.1)
    assert x2.val == np.arccos(0.2)
    assert x3.val == np.arctan(0.3)
    assert x1.der == {'x':1.005037815259212}
    assert x2.der == {'x':-1.0206207261596576}
    assert x3.der == {'x':0.9174311926605504}
    # assert x4.val == 1.7610626216439926
    # assert x4.der == {'x1': 1.005037815259212, 
    #     'x2':  -1.0206207261596576,
    #     'x3': 0.9174311926605504}
    assert x8 == 1.7610626216439926

def test_sqrt():
    x1 = Variable(1)
    x2 = fun.sqrt(x1)
    x3 = 1 
    assert x2.val == 1
    assert x2.der == {'x': 0.5}
    assert x3 == 1

def test_error():
    try:
        fun.arcsin(Variable(2))
        assert False
    except ValueError:
        assert True
    try:
        fun.arcsin(Variable(-2))
        assert False
    except ValueError:
        assert True
    try:
        fun.arccos(Variable(2))
        assert False
    except ValueError:
        assert True
    try:
        fun.arccos(Variable(-2))
        assert False
    except ValueError:
        assert True
    try:
        fun.arcsin(2)
        assert False
    except ValueError:
        assert True
    try:
        fun.arcsin(-2)
        assert False
    except ValueError:
        assert True
    try:
        fun.arccos(2)
        assert False
    except ValueError:
        assert True
    try:
        fun.arccos(-2)
        assert False
    except ValueError:
        assert True


test_exp()
test_log()
test_trigonometric()
test_hyberbolic()
test_inverse_trigonometric()
test_sqrt()
test_error()

