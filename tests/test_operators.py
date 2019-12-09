import pytest
import numpy as np
import sys
sys.path.append('..')
import autoDiff.operator as ad


def test_dup_vars():
    with np.testing.assert_raises(Exception):
        a = ad.StartUp(2)
        x1 = a.create_variable(1, 'x1')
        x2 = a.create_variable(2, 'x1')
    
def test_over_vars():
    with np.testing.assert_raises(Exception):
        a = ad.StartUp(1)
        x1 = a.create_variable(1, 'x1')
        x2 = a.create_variable(2, 'x2')

def test_add_1():
    a = ad.StartUp(1)
    x1 = a.create_variable(1, 'x1')
    x2 = x1 + 2
    x3 = 3 + x1
    assert x1.val == [1]
    assert x1.der == [1]
    assert x2.val == [3]
    assert x2.der == [1]
    assert x3.val == [4]
    assert x3.der == [1]
    
def test_sub_1():
    a = ad.StartUp(1)
    x1 = a.create_variable(2, 'x1')
    x2 = x1 - 4
    assert x2.val == [-2]
    assert x2.der == [1]

def test_sub_2():
    a = ad.StartUp(1)
    x1 = a.create_variable(2, 'x1')
    x2 = 4 - x1
    assert x2.val == [2]
    assert x2.der == [-1]

def test_sub_3():
    a = ad.StartUp(2)
    x1 = a.create_variable(2, 'x1')
    x2 = a.create_variable(5, 'x2')
    x3 = x2 - x1
    assert x3.val == [3]
    assert x3.der[0] == -1
    assert x3.der[1] == 1

def test_mul():
    a = ad.StartUp(2)
    x1 = a.create_variable(3, 'x1')
    x2 = a.create_variable(4, 'x2')
    x3 = x1 * x2 * x1
    assert x3.val == [36]
    assert x3.der[0] == 24
    assert x3.der[1] == 9

def test_div_1():
    a = ad.StartUp(1)
    x1 = a.create_variable(1, 'x1')
    x2 = x1/5
    assert x2.val == [0.2]
    assert x2.der[0] == 0.2

def test_div_2():
    a = ad.StartUp(1)
    x1 = a.create_variable(1, 'x1')
    x2 = 5/x1
    assert x2.val == [5]
    assert x2.der[0] == -5

def test_div_3():
    a = ad.StartUp(3)
    x1 = a.create_variable(10, 'x1')
    x2 = a.create_variable(2, 'x2')
    x3 = a.create_variable(1, 'x3')
    x4 = x1/x2/x3
    assert x4.val == [5]
    assert x4.der[0] == 0.5
    assert x4.der[1] == -2.5
    assert x4.der[2] == -5

def test_pow_1():
    a = ad.StartUp(1)
    x1 = a.create_variable(2, 'x1')
    x2 = 5**x1
    assert x2.val == [25]
    assert np.round(x2.der, 2) == 40.24

def test_pow_2():
    a = ad.StartUp(2)
    x1 = a.create_variable(2, 'x1')
    x2 = a.create_variable(3, 'x2')
    x3 = x1**x2
    assert x3.val == [8]
    assert x3.der[0] == 12
    assert np.round(x3.der[1], 2) == 5.55

def test_neg():
    a = ad.StartUp(1)
    x1 = a.create_variable(1, 'x1')
    x2 = -x1
    assert x2.val == [-1]
    assert x2.der == [-1]

def test_equal():
    a = ad.StartUp(2)
    x1 = a.create_variable(4, 'x1')
    x2 = a.create_variable(4, 'x2')
    x3 = (x1 == x2)
    assert x3 == False

def test_equal_2():
    a = ad.StartUp(2)
    x1 = a.create_variable(4, 'x1')
    x3 = (x1 == 5)
    assert x3 == False

def test_not_equal():
    a = ad.StartUp(2)
    x1 = a.create_variable(4, 'x1')
    x2 = a.create_variable(4, 'x2')
    x3 = (x1 != x2)
    assert x3 == True

test_add_1()
test_sub_1()
test_sub_2()
test_sub_3()
test_mul()
test_div_1()
test_div_2()
test_div_3()
test_pow_1()
test_pow_2()
test_neg()
test_equal()
test_equal_2()
test_not_equal()
