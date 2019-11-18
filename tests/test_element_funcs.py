import pytest
import numpy as np
from AutoDiff.AutoDiff import Variable
from AutoDiff import AutoDiff

class Test_Elementary_functions():

    def test_exp(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x3 = AutoDiff.exp(x1 + x2)
        x4 = AutoDiff.exp(1)
        assert x3.val == np.exp(1+1)
        assert x3.der == {'x1': np.exp(1+1), 'x2': np.exp(1+1)}
        assert x4 == np.exp(1)

    def test_log(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x3 = AutoDiff.log(x1 + x2)
        x4 = AutoDiff.log(1)
        assert x3.val == np.log(1+1)
        assert x3.der == {'x1': 1/2, 'x2': 1/2}
        assert x4 == np.log(1)

    def test_trigonometric(self):
        x1 = Variable(np.pi/4, name='x1')
        x2 = Variable(np.pi/4, name='x2')
        x3 = Variable(np.pi/4, name='x3')
        x4 = AutoDiff.sin(x1) + AutoDiff.cos(x2) + AutoDiff.tan(x3)
        x5 = np.pi/4
        x6 = np.pi/4
        x7 = np.pi/4
        x8 = AutoDiff.sin(x5) + AutoDiff.cos(x6) + AutoDiff.tan(x7)
        assert x4.val == 2**0.5 + 1
        assert x4.der == {'x1': np.cos(np.pi/4), 
            'x2': -np.sin(np.pi/4),
            'x3': 1/np.cos(np.pi/4)**2}
        assert x8 == 2**0.5 + 1

    def test_hyberbolic(self):
        x1 = Variable(1, name='x1')
        x2 = Variable(1, name='x2')
        x3 = Variable(1, name='x3')
        x4 = AutoDiff.sinh(x1) + AutoDiff.cosh(x2) + AutoDiff.tanh(x3)
        x5 = 1
        x6 = 1
        x7 = 1
        x8 = AutoDiff.sinh(x5) + AutoDiff.cosh(x6) + AutoDiff.tanh(x7)
        assert x4.val == np.sinh(1)+np.cosh(1)+np.tanh(1)
        assert x4.der == {'x1': np.cosh(1), 
            'x2': np.sinh(1),
            'x3': 1-np.tanh(1)**2}
        assert x8 == np.sinh(1)+np.cosh(1)+np.tanh(1)

    def test_inverse_trigonometric(self):
        x1 = Variable(0.1, name='x1')
        x2 = Variable(0.2, name='x2')
        x3 = Variable(0.3, name='x3')
        x4 = AutoDiff.arcsin(x1) + AutoDiff.arccos(x2) + AutoDiff.arctan(x3)
        x5 = 0.1
        x6 = 0.2
        x7 = 0.3
        x8 = AutoDiff.arcsin(x5) + AutoDiff.arccos(x6) + AutoDiff.arctan(x7)
        assert x4.val == 1.7610626216439926
        assert x4.der == {'x1': 1.005037815259212, 
            'x2':  -1.0206207261596576,
            'x3': 0.9174311926605504}
        assert x8 == 1.7610626216439926

    def test_sqrt(self):
        x1 = Variable(1, name='x1')
        x2 = AutoDiff.sqrt(x1)
        x3 = 1 
        assert x2.val == 1
        assert x2.der == {'x1': 0.5}
        assert x3 == 1
