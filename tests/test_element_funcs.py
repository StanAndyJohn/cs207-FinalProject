import pytest
import sys
sys.path.append('..')
import numpy as np
from autoDiff.operator import *
from autoDiff import element_func as fun



def test_exp():
    # Scalar
    x1 = fun.exp(Variable(1,[1]))
    x2 = fun.exp(1)
    assert x1.val == [np.exp(1)]
    assert x1.der == [np.exp(1)]
    assert x2 == np.exp(1)
    # Vector
    x = Variable(0, [1, 0, 0])
    y = Variable(1.0, [0, 1, 0])
    z = Variable(2.0, [0, 0, 1])
    f = x + y + z
    f1 = fun.exp(f)
    np.testing.assert_array_equal(np.round(f1.val,2), np.array([20.09]))
    np.testing.assert_array_equal(np.round(f1.der,2), np.array([20.09, 20.09, 20.09]))
    # 1 to n
    x = Variable(3.0, [1])
    f = Variable().f([2 * x, x + 1, x ** 2])
    f1 = fun.exp(f)
    np.testing.assert_array_equal(np.round(f1.val, 2), np.array([ 403.43,   54.60,  8103.08 ]))
    np.testing.assert_array_equal(np.round(f1.der, 2), np.array([[  806.86],
                                                                 [   54.60],
                                                                 [48618.50]]))
    # m to n
    x = Variable(3.0, [1, 0, 0])
    y = Variable(1.0, [0, 1, 0])
    z = Variable(2.0, [0, 0, 1])
    f = Variable().f([2 * x, y + 1, z ** 2])
    f1 = fun.exp(f)
    np.testing.assert_array_equal(np.round(f1.val,2), np.array([403.43,   7.39,  54.6 ]))
    np.testing.assert_array_equal(np.round(f1.der,2), np.array([[806.86,   0.  ,   0.  ],
                                                                [  0.  ,   7.39,   0.  ],
                                                                [  0.  ,   0.  , 218.39]]))

def test_log():
    # Scalar
    x1 = fun.log(Variable(1,[1]))
    x2 = fun.log(1)
    x3 = fun.log2(Variable(1,[1]))
    x4 = fun.log2(1)
    x5 = fun.log10(Variable(1,[1]))
    x6 = fun.log10(1)
    assert x1.val == [np.log(1)]
    assert x1.der == [1]
    assert x3.val == [np.log2(1)]
    assert x3.der == [1]
    assert x5.val == [np.log10(1)]
    assert x5.der == [1]
    assert x2 == np.log(1)
    assert x4 == np.log2(1)
    assert x6 == np.log10(1)
    # Vector
    x = Variable(3.0, [1, 0, 0])
    y = Variable(1.0, [0, 1, 0])
    z = Variable(2.0, [0, 0, 1])
    f = x + y + z
    f1 = fun.log10(f)
    np.testing.assert_array_equal(np.round(f1.val,2), np.array([0.78]))
    np.testing.assert_array_equal(np.round(f1.der,2), np.array([0.17, 0.17, 0.17]))
    # with np.testing.assert_raises(ValueError):
    #     f2 = x - y - z
    #     f3 = fun.log2(f2)
    # 1 to n
    x = Variable(3.0, [1])
    f = Variable().f([2 * x, x + 1, x ** 2])
    f1 = fun.log10(f)
    np.testing.assert_array_equal(np.round(f1.val, 2), np.array([ 0.78, 0.6 , 0.95]))
    np.testing.assert_array_equal(np.round(f1.der, 2), np.array([[0.33],
                                                                 [0.25],
                                                                 [0.67]]))

    # with np.testing.assert_raises(ValueError):
    #     f2 = Variable([2 * x, x - 3, x ** 2])
    #     f3 = fun.log2(f2)
    # m to n
    x = Variable(3.0, [1, 0, 0])
    y = Variable(1.0, [0, 1, 0])
    z = Variable(2.0, [0, 0, 1])
    f = Variable().f([2 * x, y + 1, z ** 2])
    f1 = fun.log10(f)
    np.testing.assert_array_equal(np.round(f1.val,2), np.array([0.78, 0.3 , 0.6 ]))
    np.testing.assert_array_equal(np.round(f1.der,2), np.array([[0.33, 0.  , 0.  ],
                                                                [0.  , 0.5, 0.  ],
                                                                [0.  , 0.  , 1]]))

    # with np.testing.assert_raises(ValueError):
    #     f2 = Variable([2 * x, y - 1, z ** 2])
    #     f3 = fun.log2(f2)

def test_trigonometric():
    # Scalar
    x1 = Variable(np.pi/4,[1])
    f1 = fun.sin(x1)
    assert f1.val == [np.sin(np.pi/4)]
    assert f1.der == [np.cos(np.pi/4)]
    x2 = Variable(np.pi/4)
    f2 = fun.cos(x2)
    assert f2.val == [np.cos(np.pi/4)]
    assert f2.der == [-np.sin(np.pi/4)]
    x3 = fun.tan(Variable(np.pi/4))
    assert x3.val == [np.tan(np.pi/4)]
    assert abs(x3.der[0] - 1/np.cos(np.pi/4)**2 < 1e-8)
    # Vector
    x = Variable(np.pi / 2, [1, 0])
    y = Variable(0, [0, 1])
    f = 3 * fun.sin(x) + 2 * fun.sin(y)
    np.testing.assert_array_equal(np.round(f.val,2), np.array([3.]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([0., 2.]))
    x = Variable(np.pi / 2, [1, 0])
    y = Variable(0, [0, 1])
    f = 3 * fun.cos(x) + 2 * fun.cos(y)
    np.testing.assert_array_equal(np.round(f.val,2), np.array([2.]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([-3., 0.]))
    x = Variable(np.pi / 4, [1, 0])
    y = Variable(np.pi / 3, [0, 1])
    f = 3 * fun.tan(x) + 2 * fun.tan(y) 
    np.testing.assert_array_equal(np.round(f.val,2), np.array([6.46]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([6., 8.]))
    # with np.testing.assert_raises(ValueError):
    #     z = Variable(3 * np.pi / 2)
    #     f1 = fun.tan(z) + fun.tan(x)
    # 1 to n
    x = Variable(np.pi / 2, [1])
    f = Variable().f([fun.sin(x), fun.sin(x) + 1, fun.sin(x) ** 2])
    np.testing.assert_array_equal(np.round(f.val, 2), np.array([ 1., 2., 1.]))
    np.testing.assert_array_equal(np.round(f.der, 2), np.array([[ 0.],
                                                                [ 0.],
                                                                [ 0.]]))
    x = Variable(np.pi / 2, [ 1.])
    f = Variable().f([fun.cos(x) + 1, fun.cos(x), fun.cos(x) ** 2])
    np.testing.assert_array_equal(np.round(f.val, 2), np.array([ 1., 0., 0.]))
    np.testing.assert_array_equal(np.round(f.der, 2), np.array([[-1.],
                                                                [-1.],
                                                                [ 0.]]))
    x = Variable(np.pi / 3, [1])
    f = Variable().f([fun.tan(x) + 1, fun.tan(x), fun.tan(x) ** 2])
    np.testing.assert_array_equal(np.round(f.val, 2), np.array([ 2.73, 1.73, 3. ]))
    np.testing.assert_array_equal(np.round(f.der, 2), np.array([[ 4.],
                                                                [ 4.],
                                                                [ 13.86]]))
    # m to n
    x = Variable(np.pi / 2, [1, 0, 0])
    y = Variable(0, [0, 1, 0])
    z = Variable(np.pi / 3, [0, 0, 1])
    f = Variable().f([fun.sin(x), fun.sin(y) + 1, fun.sin(z) ** 2])
    np.testing.assert_array_equal(np.round(f.val,2), np.array([1., 1., 0.75]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([[0.  , 0.  , 0.  ],
                                                               [0.  , 1.  , 0.  ],
                                                               [0.  , 0.  , 0.87]]))    
    x = Variable(np.pi / 2, [1, 0, 0])
    y = Variable(0, [0, 1, 0])
    z = Variable(np.pi / 6, [0, 0, 1])
    f = Variable().f([fun.cos(x) + 1, fun.cos(y), fun.cos(z) ** 2])
    np.testing.assert_array_equal(np.round(f.val,2), np.array([1., 1., 0.75]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([[-1.  , 0.  , 0.  ],
                                                               [0.  , 0.  , 0.  ],
                                                               [0.  , 0.  , -0.87]]))
    x = Variable(np.pi / 3, [1, 0, 0])
    y = Variable(0, [0, 1, 0])
    z = Variable(np.pi / 4, [0, 0, 1])
    f = Variable().f([fun.tan(x) + 1, fun.tan(y), fun.tan(z) ** 2])
    np.testing.assert_array_equal(np.round(f.val,2), np.array([2.73, 0., 1. ]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([[4., 0., 0.],
                                                               [0., 1., 0.],
                                                               [0., 0., 4.]]))  

    with np.testing.assert_raises(ValueError):
        test_arccos = fun.arccos(2)
    
    x = Variable(1, [1])
    test_arccos2 = fun.arccos(x)
    np.testing.assert_array_equal(np.round(test_arccos2.val,2), np.array([0]))
    np.testing.assert_equal(test_arccos2.der, np.nan)  

    x = Variable(-1, [1])
    test_arccos2 = fun.arccos(x)
    np.testing.assert_array_equal(np.round(test_arccos2.val,2), np.array([3.14]))
    np.testing.assert_equal(test_arccos2.der, np.nan)  

    with np.testing.assert_raises(ValueError):
        test_arcsin = fun.arcsin(2)
    
    x = Variable(1, [1])
    test_arcsin = fun.arcsin(x)
    
    np.testing.assert_array_equal(np.round(test_arcsin.val,2), np.array([1.57]))
    np.testing.assert_equal(test_arcsin.der, np.nan)  

    x = Variable(-1, [1])
    test_arcsin = fun.arcsin(x)
    np.testing.assert_array_equal(np.round(test_arcsin.val,2), np.array([-1.57]))
    np.testing.assert_equal(test_arcsin.der, np.nan)  

    test_tan = fun.tan(1)
    assert np.round(test_tan, 2) == 1.56

    test_cos = fun.cos(1)
    assert np.round(test_cos, 2) == 0.54

    test_sin = fun.sin(1)
    assert np.round(test_sin, 2) == 0.84

def test_hyberbolic():
    # Scalar
    x1 = fun.sinh(Variable(1,[1]))
    x2 = fun.cosh(Variable(1,[1]))
    x3 = fun.tanh(Variable(1,[1]))
    x5 = 1
    x6 = 1
    x7 = 1
    x8 = fun.sinh(x5) + fun.cosh(x6) + fun.tanh(x7)
    assert x1.val == [np.sinh(1)]
    assert x2.val == [np.cosh(1)]
    assert x3.val == [np.tanh(1)]
    assert x1.der == [np.cosh(1)]
    assert x2.der == [np.sinh(1)]
    assert np.abs(x3.der-[1-np.tanh(1)**2])<1e-5
    assert x8 == np.sinh(1)+np.cosh(1)+np.tanh(1)
    # Vector
    x = Variable(-1, [1, 0])
    y = Variable(0, [0, 1])
    f = 3 * fun.sinh(x) + 2 * fun.sinh(y)
    np.testing.assert_array_equal(np.round(f.val,2), np.array([-3.53]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([4.63, 2.]))
    x = Variable(-1, [1, 0])
    y = Variable(0, [0, 1])
    f = 3 * fun.cosh(x) + 2 * fun.cosh(y)
    np.testing.assert_array_equal(np.round(f.val,2), np.array([6.63]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([-3.53, 0.]))
    x = Variable(-1, [1, 0])
    y = Variable(0, [0, 1])
    f = 3 * fun.tanh(x) + 2 * fun.tanh(y)
    np.testing.assert_array_equal(np.round(f.val,2), np.array([-2.28]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([1.26, 2.]))
    # 1 to n
    x = Variable(-1, [1])
    f = Variable().f([3 * fun.sinh(x), 2 * fun.sinh(x)])
    np.testing.assert_array_equal(np.round(f.val, 2), np.array([-3.53, -2.35]))
    np.testing.assert_array_equal(np.round(f.der, 2), np.array([[4.63],
                                                                [3.09]]))
    x = Variable(-1, [1])
    f = Variable().f([3 * fun.cosh(x), 2 * fun.cosh(x)])
    np.testing.assert_array_equal(np.round(f.val, 2), np.array([4.63, 3.09]))
    np.testing.assert_array_equal(np.round(f.der, 2), np.array([[-3.53],
                                                                [-2.35]]))
    x = Variable(-1, [1])
    f = Variable().f([3 * fun.tanh(x), 2 * fun.tanh(x)])
    np.testing.assert_array_equal(np.round(f.val, 2), np.array([-2.28, -1.52]))
    np.testing.assert_array_equal(np.round(f.der, 2), np.array([[1.26],
                                                                [0.84]]))
    # m to n
    x = Variable(-1, [1, 0])
    y = Variable(0, [0, 1])
    f = Variable().f([3 * fun.sinh(x), 2 * fun.sinh(y)])
    np.testing.assert_array_equal(np.round(f.val,2), np.array([-3.53,  0.]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([[4.63, 0.  ],
                                                               [0.  , 2.  ]]))
    x = Variable(-1, [1, 0])
    y = Variable(0, [0, 1])
    f = Variable().f([3 * fun.cosh(x), 2 * fun.cosh(y)])
    np.testing.assert_array_equal(np.round(f.val,2), np.array([4.63, 2.]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([[-3.53,  0.  ],
                                                               [ 0.  ,  0.  ]]))
    x = Variable(-1, [1, 0])
    y = Variable(0, [0, 1])
    f = Variable().f([3 * fun.tanh(x), 2 * fun.tanh(y)])
    np.testing.assert_array_equal(np.round(f.val,2), np.array([-2.28,  0.  ]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([[1.26, 0.  ],
                                                               [0.  , 2.  ]]))




def test_inverse_trigonometric():
    # Scalar
    x1 = fun.arcsin(Variable(0.1,[1]))
    x2 = fun.arccos(Variable(0.2,[1]))
    x3 = fun.arctan(Variable(0.3,[1]))
    x5 = 0.1
    x6 = 0.2
    x7 = 0.3
    x8 = fun.arcsin(x5) + fun.arccos(x6) + fun.arctan(x7)
    assert x1.val == np.arcsin(0.1)
    assert x2.val == np.arccos(0.2)
    assert x3.val == np.arctan(0.3)
    assert x1.der == [1.005037815259212]
    assert x2.der == [-1.0206207261596576]
    assert x3.der == [0.9174311926605504]
    assert x8 == 1.7610626216439926
    with np.testing.assert_raises(ValueError):
        x = Variable(-1.01)
        fun.arcsin(x)
    with np.testing.assert_raises(ValueError):
        x = Variable(-1.01)
        fun.arccos(x)
    # Vector
    x = Variable(0, [1, 0])
    y = Variable(0, [0, 1])
    f = 3 * fun.arcsin(x) + 2 * fun.arcsin(y)
    np.testing.assert_array_equal(np.round(f.val,2), np.array([0]))
    np.testing.assert_array_equal(np.round(f.der,2), np.array([3., 2]))
    # test on the boundary -1 and 1
    # z1 = Variable(1, [1, 0])
    # z2 = Variable(-1, [0, 1])
    # f_z = 3 * fun.arcsin(z1) + 2 * fun.arcsin(z2)
    # np.testing.assert_array_equal(np.round(f_z.val,2), np.array([1.57]))
    # np.testing.assert_array_equal(np.round(f_z.der,2), np.array([np.nan, np.nan]))
    # # test out of range [-1, 1]
    # with np.testing.assert_raises(ValueError):
    #     x = Variable(-1.01, [1, 0])
    #     f = 3 * fun.arcsin(x) + 2 * fun.arcsin(y)
    # x = Variable(0, [1, 0])
    # y = Variable(0.5, [0, 1])
    # f = 3 * fun.arccos(x) + 2 * fun.arccos(y)
    # np.testing.assert_array_equal(np.round(f.val,2), np.array([6.81]))
    # np.testing.assert_array_equal(np.round(f.der,2), np.array([-3., -2.31]))
    # # test on the boundary -1 and 1
    # z1 = Variable(1, [1, 0])
    # z2 = Variable(-1, [0, 1])
    # f_z = 3 * fun.arccos(z1) + 2 * fun.arccos(z2)
    # np.testing.assert_array_equal(np.round(f_z.val,2), np.array([6.28]))
    # np.testing.assert_array_equal(np.round(f_z.der,2), np.array([np.nan, np.nan]))

    # # test out of range [-1, 1]
    # with np.testing.assert_raises(ValueError):
    #     x = Variable(1.01, [1, 0])
    #     f = 3 * fun.arccos(x) + 2 * fun.arccos(y)
    # x = Variable(0.5, [1, 0])
    # y = Variable(np.pi/2, [0, 1])
    # f = 3 * fun.arctan(x) + 2 * fun.arctan(y)
    # np.testing.assert_array_equal(np.round(f.val,2), np.array([3.4]))
    # np.testing.assert_array_equal(np.round(f.der,2), np.array([2.4, 0.58]))
    # # 1 to n
    # x = Variable(1, [1])
    # print(fun.arcsin(x))
    # f = Variable().f([fun.arcsin(x), fun.arcsin(x) + 1, fun.arcsin(x) ** 2])
    # np.testing.assert_array_equal(np.round(f.val, 2), np.array([1.57, 2.57, 2.47]))
    # np.testing.assert_array_equal(np.round(f.der, 2), np.array([[np.nan],
    #                                                             [np.nan],
    #                                                             [np.nan]])) 
    # fun.arcsin(x)
    # # test on the boundary -1 and 1
    # z1 = Variable(-1, [1])
    # f_z = Variable().f([fun.arcsin(z1) + fun.arcsin(z1), fun.arcsin(z1) ** 2 + 1])
    # np.testing.assert_array_equal(np.round(f_z.val, 2), np.array([-3.14, 3.47]))
    # np.testing.assert_array_equal(np.round(f_z.der, 2), np.array([[np.nan],
    #                                                               [np.nan]]))
    # x = Variable(1, [1])
    # f = Variable().f([fun.arccos(x), fun.arccos(x) + 1, fun.arccos(x) ** 2])  
    # np.testing.assert_array_equal(np.round(f.val, 2), np.array([ 0., 1., 0.]))
    # np.testing.assert_array_equal(np.round(f.der, 2), np.array([[np.nan],
    #                                                             [np.nan],
    #                                                             [np.nan]])) 

    # # test on the boundary -1 and 1
    # z1 = Variable(-1, [1])
    # f_z = Variable().f([fun.arccos(z1) + fun.arccos(z1), fun.arccos(z1) ** 2 + 1])
    # np.testing.assert_array_equal(np.round(f_z.val, 2), np.array([ 6.28, 10.87]))
    # np.testing.assert_array_equal(np.round(f_z.der, 2), np.array([[np.nan],
    #                                                              [np.nan]]))
    # x = Variable(0.5, [1])
    # f = Variable().f([3 * fun.arctan(x), 2 * fun.arctan(x)])
    # np.testing.assert_array_equal(np.round(f.val, 2), np.array([1.39, 0.93]))
    # np.testing.assert_array_equal(np.round(f.der, 2), np.array([[2.4],
    #                                                             [1.6]]))
    # # m to n
    # x = Variable(1, [1, 0, 0])
    # y = Variable(0, [0, 1, 0])
    # z = Variable(np.pi / 4, [0, 0, 1])
    # f = Variable().f([fun.arcsin(x), fun.arcsin(y) + 1, fun.arcsin(z) ** 2])
    # np.testing.assert_array_equal(np.round(f.val,2), np.array([1.57, 1., 0.82]))
    # np.testing.assert_array_equal(np.round(f.der,2), np.array([[np.nan, np.nan  , np.nan],
    #                                                            [0.  , 1.  , 0.  ],
    #                                                            [0.  , 0.  , 2.92]]))    
    
    # # test on the boundary -1 and 1
    # z1 = Variable(1, [1, 0])
    # z2 = Variable(-1, [0, 1])
    # f_z = Variable().f([fun.arcsin(z1) + fun.arcsin(z2), fun.arcsin(z2) ** 2 + 1])
    # np.testing.assert_array_equal(np.round(f_z.val,2), np.array([0.  , 3.47]))
    # np.testing.assert_array_equal(np.round(f_z.der,2), np.array([[np.nan, np.nan],
    #                                                              [np.nan, np.nan]]))
    # x = Variable(1, [1, 0, 0])
    # y = Variable(0, [0, 1, 0])
    # z = Variable(np.pi / 4, [0, 0, 1]) 
    # f = Variable().f([fun.arccos(x), fun.arccos(y) + 1, fun.arccos(z) ** 2])   
    # np.testing.assert_array_equal(np.round(f.val,2), np.array([0.  , 2.57, 0.45]))
    # np.testing.assert_array_equal(np.round(f.der,2), np.array([[np.nan, np.nan  , np.nan],
    #                                                            [ 0.  , -1.  ,  0.  ],
    #                                                            [ 0.  ,  0.  , -2.16]])) 

    # # test on the boundary -1 and 1
    # z1 = Variable(1, [1, 0])
    # z2 = Variable(-1, [0, 1])
    # f_z = Variable().f([fun.arccos(z1) + fun.arccos(z2), fun.arccos(z2) ** 2 + 1])
    # np.testing.assert_array_equal(np.round(f_z.val,2), np.array([3.14, 10.87]))
    # np.testing.assert_array_equal(np.round(f_z.der,2), np.array([[np.nan, np.nan],
    #                                                              [np.nan, np.nan]]))
    # x = Variable(0.5, [1, 0])
    # y = Variable(np.pi/2, [0, 1])
    # f = Variable().f([3 * fun.arctan(x), 2 * fun.arctan(y)])
    # np.testing.assert_array_equal(np.round(f.val,2), np.array([1.39, 2.01]))
    # np.testing.assert_array_equal(np.round(f.der,2), np.array([[2.4 , 0.  ],
    #                                                            [0.  , 0.58]]))

def test_sqrt():
    a = StartUp(1)
    x1 = a.create_variable(4, 'x1')
    x2 = fun.sqrt(x1)
    assert x2.val == [2]
    assert x2.der == [0.25]

    x3 = fun.sqrt(4)
    assert x3 == 2


def test_logistic():
    # Scalar
    x = Variable(2)
    f = fun.logistic(x)
    assert np.round(f.val, 8) == [0.88079708]
    assert np.round(f.der, 8) == [0.10499359]
    # Vector
    x = Variable(1.5, [1, 0, 0])
    y = Variable(1.0, [0, 1, 0])
    z = Variable(2.0, [0, 0, 1])
    f = x + y + z
    f1 = fun.logistic(f)
    np.testing.assert_array_equal(np.round(f1.val, 3), np.array([0.989]))
    np.testing.assert_array_equal(np.round(f1.der, 3), np.array([0.011, 0.011, 0.011]))
    # 1 to n
    x = Variable(3.0, [1])
    f = Variable().f([2 * x, x + 1, x ** 2])
    f1 = fun.logistic(f)
    np.testing.assert_array_equal(np.round(f1.val, 3), np.array([0.998, 0.982, 1.000]))
    # np.testing.assert_array_equal(np.round(f1.der, 3), np.array([[0.005],
    # m to n
    x = Variable(3.0, [1, 0, 0])
    y = Variable(1.0, [0, 1, 0])
    z = Variable(2.0, [0, 0, 1])
    f = Variable().f([2 * x, y + 1, z ** 2])
    f1 = fun.logistic(f)
    np.testing.assert_array_equal(np.around(f1.val, 3), np.array([0.998, 0.881, 0.982]))
    # np.testing.assert_array_equal(np.around(f1.der, 3), np.array([[0.005, 0.000, 0.000],
    f2 = fun.logistic(5)
    assert np.round(f2, 9) == 0.993307149



test_exp()
test_log()
test_trigonometric()
test_hyberbolic()
test_inverse_trigonometric()
test_sqrt()
test_logistic()


