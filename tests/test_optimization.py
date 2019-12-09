
import sys
sys.path.append('..')
import autoDiff.element_func as fun
from autoDiff.operator import *
from autoDiff.optimization import *

def f(variables):
    x, y = variables
    return 4 * (y - (x ** 2)) ** 2 + (1 - x) ** 2
def test_GradientDescent():
	'''Test gradient descent optimization method to find local minima.'''

	def case_1():
		'''Simple quadratic function with minimum at x = 0.'''
		
		def f(x):
			return x ** 2
		for x_val in [-4, -2, 0, 2, 4]:
			x0 = Variable(x_val)
			solution, var_path,g_path = GradientDescent(f, x0)
			assert np.allclose(solution.val, [0])
			assert np.allclose(solution.der, [0])
	
	def case_2():
		def f(x):
			return (x - 1) ** 2 + 3
		for x_val in [-3, -1, 1, 3, 5]:
			# Test initial guess without using Variable type
			solution, x_path, f_path = GradientDescent(f, x_val)
			assert np.allclose(solution.val, [1])
			assert np.allclose(solution.der, [0])	
	
	def case_3():
		def f(x):
			return fun.sin(x)
		for x_val in [-3, -1, 1, 3]:
			x0 = [Variable(x_val)]
			solution, x_path, f_path = GradientDescent(f, x0)
			multiple_of_three_halves_pi = (solution.val  % (2 * np.pi))
			np.testing.assert_array_almost_equal(multiple_of_three_halves_pi, 1.5 * np.pi)				
			np.testing.assert_array_almost_equal(solution.der, [0])	
	
	def case_4():
		'''Minimize Rosenbrock function: f(x, y) = 4(y - x^2)^2 + (1 - x)^2.
		The global minimum is 0 when (x, y) = (1, 1).
		'''
		def f(variables):
			x, y = variables
			return 4 * (y - (x ** 2)) ** 2 + (1 - x) ** 2
		for x_val, y_val in [[-6., -6], [2., 3.], [-2., 5.]]:
			x0 = Variable(x_val, [1, 0])
			y0 = Variable(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = GradientDescent(f, init_vars, iters=25000, eta=0.002)
			xn, yn = solution.val
			minimum = [1, 1]
			assert np.allclose([xn, yn], minimum)
			# dfdx = 2(200x^3 - 200xy + x - 1)
			# dfdy = 200(y - x^2)
			der_x = 2 * (8 * (xn ** 3) - 8 * xn * yn + xn - 1)
			der_y = 8 * (yn - (xn ** 2))
			assert np.allclose(solution.der, [der_x, der_y])
	
	def case_5():
		'''Minimize bowl function: f(x, y) = x^2 + y^2 + 2.
		The global minimum is 2 when (x, y) = (0, 0).
		'''
		def f(variables):
			x, y = variables
			return x ** 2 + y ** 2 + 2
		for x_val, y_val in [[2., 3.], [-2., 5.]]:
			x0 = Variable(x_val, [1, 0])
			y0 = Variable(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = GradientDescent(f, init_vars)
			xn, yn = solution.val
			minimum = [0, 0]
			assert np.allclose([xn, yn], minimum)
			der = [0, 0]
			assert np.allclose(solution.der, der)
	case_1()
	case_2()
	case_3()
	case_4()
	case_5()


def test_BFGS():
	'''Test gradient descent optimization method to find local minima.'''

	def case_1():
		'''Simple quadratic function with stationary point at x = 0.'''

		def f(x):
			return x ** 2

		f_string = 'f(x) = x^2'

		for x_val in [-4, -2, 0, 2, 4]:
			x0 = Variable(x_val)
			solution, x_path, f_path = BFGS(f, x0)
			assert np.allclose(solution.val, [0])
			assert np.allclose(solution.der, [0])

	def case_2():
		'''Stationary point (minimum) at x = 1.'''
		def f(x):
			return (x - 1) ** 2 + 3
		for x_val in [-3, -1, 1, 3, 5]:
			# Test initial guess without using Variable type
			solution, x_path, f_path = BFGS(f, x_val)
			assert np.allclose(solution.val, [1])
			assert np.allclose(solution.der, [0])

	def case_3():
		'''Find stationary point of f(x) = sin(x).'''
		def f(x):
			return fun.sin(x)
		for x_val in [-3, -1, 1, 3]:
			# Test case when 1D input is a list
			x0 = [Variable(x_val)]
			solution, x_path, f_path = BFGS(f, x0)
			# BFGS finds a stationary point, which are every pi offset from pi/2
			first = solution.val - (np.pi / 2)
			multiple_of_one_half_pi = ((abs(solution.val) - (np.pi / 2)) % (np.pi))
			np.testing.assert_array_almost_equal(multiple_of_one_half_pi, [0])				
			np.testing.assert_array_almost_equal(solution.der, [0])

	def case_4():
		'''Find stationary point of Rosenbrock function: f(x, y) = 4(y - x^2)^2 + (1 - x)^2.
		The global minimum is 0 when (x, y) = (1, 1).
		'''
		def f(variables):
			x, y = variables
			return 4 * (y - (x ** 2)) ** 2 + (1 - x) ** 2
		for x_val, y_val in [[-6., -6], [2., 3.], [-2., 5.]]:
			x0 = Variable(x_val, [1, 0])
			y0 = Variable(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = BFGS(f, init_vars, iters=25000)
			xn, yn = solution.val
			minimum = [1, 1]
			assert np.allclose([xn, yn], minimum)
			# dfdx = 2(200x^3 - 200xy + x - 1)
			# dfdy = 200(y - x^2)
			der_x = 2 * (8 * (xn ** 3) - 8 * xn * yn + xn - 1)
			der_y = 8 * (yn - (xn ** 2))
			assert np.allclose(solution.der, [der_x, der_y])

	def case_5():
		'''Find stationary point of bowl function: f(x, y) = x^2 + y^2 + 2.
		The global minimum is 2 when (x, y) = (0, 0).
		'''
		def f(variables):
			x, y = variables
			return x ** 2 + y ** 2 + 2
		for x_val, y_val in [[2., 3.], [-2., 5.]]:
			x0 = x_val
			y0 = y_val
			init_vars = [x0, y0]
			solution, xy_path, f_path = BFGS(f, init_vars)
			xn, yn = solution.val
			minimum = [0, 0]
			assert np.allclose([xn, yn], minimum)
			der = [0, 0]
			assert np.allclose(solution.der, der)

	def case_6():
		'''Find stationary point of Easom's function. Stationary points include (pi, pi), (pi/2, pi/2).'''
		def f(variables):
			x, y = variables
			return -fun.cos(x) * fun.cos(y) * fun.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))
		x0 = 2.5
		y0 = 2.5
		init_vars = [x0, y0]
		solution, xy_path, f_path = BFGS(f, init_vars, iters=10000)
		xn, yn = solution.val
		assert (np.allclose(xn, np.pi) and np.allclose(yn, np.pi))

	def case_7():
		'''Find stationary point of Himmelblau's function.'''
		def f(variables):
			x, y = variables
			return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
		x0 = 5
		y0 = 5
		init_vars = [x0, y0]
		solution, xy_path, f_path = BFGS(f, init_vars, iters=10000)
		xn, yn = solution.val
		assert ((np.allclose(xn, 3) and np.allclose(yn, 2)) or (
				 np.allclose(xn, -2.805118) and np.allclose(yn, 3.131312)) or (
				 np.allclose(xn, -3.779310) and np.allclose(yn, -3.283186)) or (
				 np.allclose(xn, 3.584428) and np.allclose(yn, -1.848126)))
	case_1()
	case_2()
	case_3()
	case_4()
	case_5()
	case_6()
	case_7()

def test_NewtonRoot():
	'''Integration tests for scalar to scalar functions.'''
	def case_1():
		'''Find root of quadratic function that is also a global minimum.'''
		def f(x):
			return x ** 2
		x0 = [Variable(-2)]
		solution, x_path, y_path = NewtonRoot(f, x0)

		root = [0]
		der = [0]
		np.testing.assert_array_almost_equal(solution.val, root, decimal=4)
		np.testing.assert_array_almost_equal(solution.der, der, decimal=4)

	def case_2():
		'''Find root of concave up quadratic function.'''
		def f(x):
			return (x - 1) ** 2 - 1
		x0 = [Variable(3)]
		solution, x_path, y_path = NewtonRoot(f, x0)
		root_1, der_1 = [0], [-2]
		root_2, der_2 = [2], [2]
		assert ((np.allclose(solution.val, root_1) and np.allclose(solution.der, der_1)) or 
				(np.allclose(solution.val, root_2) and np.allclose(solution.der, der_2)))

	def case_3():
		'''Find roots of concave down quadratic function.'''
		def f(x):
			return -((x + 3) ** 2) + 2
		# Test non-Var input
		x0 = 1
		solution, x_path, y_path = NewtonRoot(f, x0)
		root_1 = [-3 + np.sqrt(2)]
		root_2 = [-3 - np.sqrt(2)]
		assert (np.allclose(solution.val, root_1) or np.allclose(solution.val, root_2))

	def case_4():
		'''Find roots of cubic function.'''
		def f(x):
			return (x - 4) ** 3 - 3
		x0 = 2.5
		solution, x_path, y_path = NewtonRoot(f, x0)
		root = [4 + np.cbrt(3)]
		np.testing.assert_array_almost_equal(solution.val, root)	

	def case_5():
		'''Find roots of sinusoidal wave.'''
		def f(x):
			return fun.sin(x)
		x0 = Variable(2 * np.pi - 0.25)
		solution, x_path, y_path = NewtonRoot(f, x0)
		root_multiple_of_pi = (solution.val / np.pi) % 1
		np.testing.assert_array_almost_equal(root_multiple_of_pi, [0.])	

	def case_6():
		'''Find roots of complicated scalar function.'''
		x_var = Variable(0.1)
		def f(x):
			return x - fun.exp(-2.0 * fun.sin(4.0 * x) * fun.sin(4.0 * x)) + 0.3
		x0 = 0.0
		solution, x_path, y_path = NewtonRoot(f, x0)
		root = [0.166402]
		der = [4.62465]
		np.testing.assert_array_almost_equal(solution.val, root, decimal=4)
		np.testing.assert_array_almost_equal(solution.der, der, decimal=4)	

	def case_7():
		'''Find a root of f(x, y) = x + y, i.e x = -y'''
		def f(variables):
			x, y = variables
			return x + y
		for x_val, y_val in [[-4, 3], [8, 1]]:
			x0 = Variable(x_val, [1, 0])
			y0 = Variable(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = NewtonRoot(f, init_vars)
			xn, yn = solution.val
			# root: x = -y
			der = [1, 1]
			np.testing.assert_array_almost_equal(xn, -yn)
			np.testing.assert_array_almost_equal(solution.der, der)

	def case_8():
		'''Find a root of z(x, y) = x^2 - y^2, i.e. x = +-y.'''

		def f(variables):
			x, y = variables
			return x ** 2 - y ** 2
		for x_val, y_val in [[-4, 2], [12, -1]]:
			x0 = Variable(x_val, [1, 0])
			y0 = Variable(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = NewtonRoot(f, init_vars)
			xn, yn = solution.val
			# root: x = +-y
			der = [2 * xn, -2 * yn]
			assert (np.allclose(xn, yn) or np.allclose(xn, -yn))
			assert np.allclose(solution.der, der)

	def case_9():
		'''Find global mininum and root at (0, 0) for f(x, y) = x^2 + y^2.'''
		def f(variables):
			x, y = variables
			return x ** 2 + y ** 2
		for x_val, y_val in [[2, 5], [-1, 3]]:
			x0 = Variable(x_val, [1, 0])
			y0 = Variable(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = NewtonRoot(f, init_vars, iters=4000)
			xn, yn = solution.val
			# root: x = y = 0
			der = [0, 0]
			assert (np.allclose(xn, 0) and np.allclose(xn, 0))
			assert np.allclose(solution.der, der)

	def case_10():
		'''Complicated function.'''
		def f(variables):
			x, y = variables
			return x ** 2 + 4 * y ** 2 - 2 * (x ** 2) * y + 4
		for x_val, y_val in [[-8, -5], [2, 12], [-5, -4]]:
			# Test initial guess without using Variable type
			init_vars = [x_val, y_val]
			solution, xy_path, f_path = NewtonRoot(f, init_vars)
			xn, yn = solution.val	
			# root: x = +- 2(sqrt(y^2 + 1))/sqrt(2y - 1)
			value = 2 * fun.sqrt(yn ** 2 + 1) / (fun.sqrt(2 * yn - 1))
			assert (np.allclose(xn, value) or np.allclose(xn, -value))
			# dfdx = x(2 - 4y)
			# dfdy = 8y - 2x^2
			der_x = xn * (2 - 4 * yn)
			der_y = 8 * yn - (2 * (xn ** 2))
			assert np.allclose(solution.der, [der_x, der_y])
		
	def case_11():
		def f(variables):
			x, y = variables
			return (x - 1) ** 2 + (y + 1) ** 2
		for x_val, y_val in [[2, 10], [-5, -4]]:
			x0 = Variable(x_val, [1, 0])
			y0 = Variable(y_val, [0, 1])
			init_vars = [x0, y0]
			solution, xy_path, f_path = NewtonRoot(f, init_vars)
			xn, yn = solution.val
			# root: x = 1 +- sqrt(-(y + 1)^2)
			inner = -(yn + 1) ** 2
			inner = 0 if abs(inner) < 1e-6 else inner
			value = np.sqrt(inner)
			root_1 = 1 + value
			root_2 = 1 - value
			assert (np.allclose(xn, root_1) or np.allclose(xn, root_2))
			# dfdx = 2(x - 1)
			# dfdy = 2(y + 1)
			der = [2 * (xn - 1), 2 * (yn + 1)]
			assert np.allclose(solution.der, der)

	def case_12():
		'''Find the roots of a function from R^3 to R^1.'''

		def f(variables):
			x, y, z = variables
			return x ** 2 + y ** 2 + z ** 2
		for x_val, y_val, z_val in [[1, -2, 5], [20, 15, -5]]:
			init_vars = [x_val, y_val, z_val]
			solution, xyz_path, f_path = NewtonRoot(f, init_vars)
			m = len(solution.val)
			xn, yn, zn = solution.val
			root = [0, 0, 0]
			assert np.allclose(solution.val, root)
			# dfdx = 2x
			# dfdy = 2y
			# dfdz = 2z
			der = [2 * xn, 2 * yn, 2 * zn]
			assert np.allclose(solution.der, der)

	def case_13():
		'''Find the roots of a more complicated function from R^3 to R^1.'''
		def f(variables):
			x, y, z = variables
			return (x + y + z) ** 2 - 3
		f_string = 'f(x, y, z) = (x + y + z)^2 - 3'
		for x_val, y_val, z_val in [[1, -2, 5], [20, 15, -5]]:
			init_vars = [x_val, y_val, z_val]
			solution, xyz_path, f_path = NewtonRoot(f, init_vars)
			m = len(solution.val)
			xn, yn, zn = solution.val
			# root: z = -y -z +- sqrt(3)
			root_1 = -yn - zn - np.sqrt(3)
			root_2 = -yn - zn + np.sqrt(3)
			assert (np.allclose(xn, root_1) or np.allclose(xn, root_2))
			# dfdx = 2(x + y + z)
			# dfdy = 2(x + y + z)
			# dfdz = 2(x + y + z)
			value = 2 * (xn + yn + zn)
			der = [value, value, value]
			assert np.allclose(solution.der, der)	

	case_1()
	case_2()
	case_3()
	case_4()
	case_5()
	case_6()
	case_7()
	case_8()
	case_9()
	case_10()
	case_11()
	case_12()
	case_13()

def test_spline():
    
    def f1(var):
        return 10**var

    xMin1 = -1
    xMax1 = 1
    nIntervals1 = 10
    nSplinePoints1 = 5

    y1, A1, coeffs1, ks1 = quad_spline_coeff(f1, xMin1, xMax1, nIntervals1)
    fig1 = quad_spline_plot(f1, coeffs1, ks1, nSplinePoints1)
    spline_points1 = spline_points(f1, coeffs1, ks1, nSplinePoints1)
    error = spline_error(f1, spline_points1)



test_GradientDescent()
test_BFGS()
test_NewtonRoot()
test_spline()
print("All test passed!")

