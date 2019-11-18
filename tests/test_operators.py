import pytest
import numpy as np
import sys
sys.path.append('..')
import autoDiff.autoDiff as ad

def test_add():
    x1 = ad.Variable(1)
    x2 = x1 + 2
    x3 = x1 + x2
    assert x1.val == 1
    assert x1.der == {'x': 1.0}
    assert x2.val == 3
    assert x2.der == {'x': 1.0}
    assert x3.val == 4
    assert x3.der == {'x': 2.0}

def test_sub():
    x1 = ad.Variable(4)
    x2 = x1 - 4
    x3 = x2 - x1
    x4 = 4 - x1
    x5 = x1 - 4
    assert x2.val == 0
    assert x2.der == {'x': 1}
    assert x3.val == -4
    assert x3.der == {'x': 0}
    assert x4.val == 0
    assert x4.der == {'x': -1}
    assert x5.val == 0
    assert x5.der == {'x': 1}

def test_mul():
    x1 = ad.Variable(4)
    x2 = 3*x1
    x3 = x1*x2
    assert x2.val == 12
    assert x2.der == {'x': 3}
    assert x3.val == 48
    assert x3.der == {'x': 24}

def test_div():
    x1 = ad.Variable(1)
    x2 = x1/5
    x3 = x1/x2
    x4 = 3/x1
    assert x2.val == 1/5
    assert x2.der == {'x': 1/5}
    assert x3.val == 5
    assert x3.der == {'x': 0}
    assert x4.val == 3
    assert x4.der == {'x': -3}

    # def test_pow(self):
    #     x1 = Variable(3)
    #     x2 = x1**2
    #     x3 = x1**x2
    #     x4 = 3**x1
    #     assert x2.val == 9
    #     assert x2.der == {'x': 6}
    #     print(x3)
    #     assert x3.val == 3**9
    #     assert x3.der == {'x': (6*np.log(3)+3)*3**9}
    #     assert x4.val == 27
    #     assert x4.der == {'x': np.log(3)*3**3}


# a = Test_Operators()
# a.test_add()
# a.test_sub()
# a.test_mul()
# a.test_div()
# a.test_pow()

