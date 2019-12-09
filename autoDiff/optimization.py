import sys
sys.path.append('..')
from autoDiff.operator import *
import numpy as np
import matplotlib.pyplot as plt


def init_normal(length, pos):
	'''Internal function to compute a unit vector, used for intializing derivatives.'''
	unit = np.zeros(length)
	unit[pos] = 1
	return unit



def GradientDescent(f, x, iters=10000, tol=1e-10, eta=0.01):
	""" Run the gradient descent algorithm to minimize a function f.
		
		Parameters
		----------
		f: callable (function).
		
		x: int, float, or `Variable` object
		   initial guess/starting point for minimum

		iters: integer
			   maximum number of iterations to run the algorithm.
		
		tol: float
			 this denotes the stopping criterion for the algorithm. 
		
		eta: float
			 learning rate
		
		Returns
		-------
		minimum: Variable object
				 contains the optimal scalar or vector solution and the derivative at that point.
		
		input_list: numpy.ndarray
				  contains the path of the input variable(s) throughout the min(iters, t) steps,
	
		f_list: numpy.ndarray 
				contains the path of the objective function 
		"""

	#If data is a vector
	if isinstance(x, list):
		# Number of variables
		m = len(x)
		# Convert to Variable type
		if m > 1:
			for i in range(len(x)):
				if not isinstance(x[i], Variable):
					x[i] = Variable(x[i], init_normal(m, i))
			# Initial step
			g = f(x)
			new_val = np.reshape(np.array([x_i.val for x_i in x]), [-1])
			input_list = [new_val]
			f_list = [g.val]

			for i in range(iters):
			# Take step in direction of steepest descent
				new_val = new_val - eta * g.der
				x = [Variable(v_i, init_normal(m, i)) for i, v_i in enumerate(new_val)]
				g = f(x)
				input_list.append(new_val)
				f_list.append(g.val)
				# If step size is below tolerance, no need to continue
				if np.linalg.norm(eta * g.der) < tol:
					break
			minimum = Variable(new_val, g.der)
			input_list = np.reshape(np.concatenate((input_list)), [-1, m])
			f_list = np.concatenate(f_list)
			return (minimum, input_list, f_list)
		x = x[0]
	if not isinstance(x, Variable):
		x = Variable(x)

	# If data is scalar
	# Initial step
	g = f(x)
	input_list = [x.val]
	f_list = [g.val]

	for i in range(iters):
		# Take step in direction of steepest descent
		step = eta * g.der
		x = x - step
		g = f(x)
		input_list.append(x.val)
		f_list.append(g.val)

		# If step size is below tolerance, no need to continue
		if np.abs(eta*g.der) < tol:
			break
	minimum = Variable(x.val, g.der)
	input_list = np.reshape(np.concatenate((input_list)), [-1])
	f_list = np.concatenate(f_list)
	return (minimum, input_list, f_list)


def BFGS(f, x, iters=10, tol=1e-10):
	""" Run the gradient descent algorithm to minimize a function f.
		
		Parameters
		----------
		f: callable (function)
		
		x: int, float, or `Variable` object
		   initial guess/starting point for minimum

		iters: integer
			   maximum number of iterations
		
		tol: float
			 this denotes the stopping criterion for the algorithm. 
		Returns
		-------
		minimum: `Variable` object
				 contains the optimal scalar or vector solution and the derivative at that point.
		
		input_list: numpy.ndarray
				  contains the path of the input variable(s) throughout the min(iters, t) steps,
		
		f_list: numpy.ndarray
				contains the path of the objective function :math:`f` throughout the min(iters, t) steps.
		
		"""
	if isinstance(x, list):
		# Number of variables
		m = len(x)
		# Convert to Variable type
		if m > 1:
			for i in range(len(x)):
				if not isinstance(x[i], Variable):
					x[i] = Variable(x[i], init_normal(m, i))
			m = len(x)
			g = f(x)
			values_flat = np.reshape(np.array([x_i.val for x_i in x]), [-1])
			input_list = [values_flat]
			f_list = [g.val]
			# Initialize inv_hessian guess
			inv_hessian = np.eye(m)
			for i in range(iters):
				step = -1 * np.matmul(inv_hessian, g.der)
				values_flat_update = values_flat + step
				# if step size is below tolerance, no need to continue
				cond = np.linalg.norm(step)
				if cond < tol:
					g = f(x)
					input_list.append(values_flat_update)
					f_list.append(g.val)
					break
				x_update = [Variable(v_i, init_normal(m, i)) for i, v_i in enumerate(values_flat_update)]
				y = f(x_update).der - f(x).der
				identity = np.eye(m)
				rho = 1 / np.matmul(np.transpose(y), step)
				A = identity - np.matmul(np.vstack(step), rho * y.reshape(1, 2))
				B = identity - np.matmul(rho * np.vstack(y), step.reshape(1, 2))
				C = rho * np.matmul(np.vstack(step), step.reshape(1, 2))
				delta_hessian = np.matmul(np.matmul(A, inv_hessian), B) + C
				x = [Variable(v_i, init_normal(m, i)) for i, v_i in enumerate(values_flat_update)]
				g = f(x)
				input_list.append(values_flat_update)
				f_list.append(g.val)
				values_flat = values_flat_update
				inv_hessian = delta_hessian
			minimum = Variable(values_flat_update, g.der)
			input_list = np.reshape(np.concatenate((input_list)), [-1, m])
			f_list = np.concatenate(f_list)
			return minimum, input_list, f_list
		x = x[0]
	if not isinstance(x, Variable):
		x = Variable(x)
	m = 1
	# Initial step
	g = f(x)
	input_list = [x.val]
	f_list = [g.val]
	# Initialize inv_hessian guess
	inv_hessian = np.eye(m)
	for i in range(iters):
		# Take step in direction of steepest descent
		step = (-1 * inv_hessian * g.der)[0]
		x_update = x + step

		# If step size is below tolerance, no need to continue
		cond = -step if step < 0 else step
		if cond < tol:
			# print ("Reached tol in {} iterations".format(i + 1))
			g = f(x_update)
			input_list.append(x_update.val.flatten())
			f_list.append(g.val.flatten())
			break

		y = f(x_update).der - f(x).der
		identity = np.eye(m)
		rho = 1 / (y * step)

		A = identity - step* rho * y
		B = identity - rho * y * step
		C = rho * step * step

		delta_hessian = A * inv_hessian * B + C

		g = f(x_update)
		input_list.append(x_update.val.flatten())
		f_list.append(g.val.flatten())

		x = x_update
		inv_hessian = delta_hessian
	minimum = Variable(x.val, g.der)
	input_list = np.reshape(np.concatenate((input_list)), [-1])
	f_list = np.concatenate(f_list)
	return minimum, input_list, f_list

def NewtonRoot(f, x, tol=1e-10, iters=2000, der_shift=1):
	""" Run the gradient descent algorithm to minimize a function f.
		
		Parameters
		----------
		f: callable (function), or string
		
		x: int, float, `Variable` object
		   initial guess/starting point for root.

		iters: integer
			   maximum number of iterations 
		
		tol: float, optional
			 this denotes the stopping criterion for the algorithm. 
			
		der_shift: int or float
				   this serves as a random restart value for the derivative in the case that a
				   root guess has derivative 0, since the iteration would require division by 0.
		Returns
		-------
		root: ``DeriveAlive.Var``
				 contains the optimal scalar or vector solution and the derivative at that point.
		
		input_list: numpy.ndarray (min(iters, t), m)
				  contains the path of the input variable(s) throughout the min(iters, t) steps,
				  where iters is defined above and t is the number of steps needed to satisfy tol.
		
		f_list: numpy.ndarray (min(iters, t),)
				contains the path of the evaluations of :math:`f` throughout the min(iters, t) steps.
		
		"""

	if isinstance(x, list):
		# Number of variables
		m = len(x)

		# Convert to Variable type
		if m > 1:
			for i in range(m):
				if not isinstance(x[i], Variable):
					x[i] = Variable(x[i], init_normal(m, i))
			#Initial step
			g = f(x)
			values = np.array([x_i.val for x_i in x])
			values_flat = np.reshape(values, [-1])
			input_list = [values_flat]
			f_list = [g.val]
			for i in range(iters):
			# Check if guess is a root
				if np.array_equal(g.val, np.zeros((g.val.shape))):
					break
				if np.linalg.norm(g.der) < tol:
					g.der = np.ones(g.der.shape) * (der_shift if np.random.random() < 0.5 else -der_shift)
			# Appropriate shape for taking pseudoinverse
				if len(g.der.shape) == 1:
					g_der_pinv = np.linalg.pinv(np.expand_dims(g.der, 0))
			# Take step in direction of steepest descent
				step = g_der_pinv * g.val
				values = values - step
				values_flat = np.reshape(values, [-1])
				x = [Variable(v_i, init_normal(m, i)) for i, v_i in enumerate(values_flat)]
				g = f(x)
				input_list.append(values_flat)
				f_list.append(g.val)
				# If step size is below tolerance, no need to continue
				cond = np.linalg.norm(step)
				if cond < tol:
					break
			root = Variable(values_flat, g.der)
			input_list = np.reshape(np.concatenate((input_list)), [-1, m])
			f_list = np.concatenate(f_list)
			return root, input_list, f_list
		x = x[0]
	if not isinstance(x, Variable):
		x = Variable(x)
	# Initial step	
	g = f(x)
	input_list = [x.val]
	f_list = [g.val]

	# Run Newton's root-finding method
	for i in range(iters):
		# Check if guess is a root
		if np.array_equal(g.val, np.zeros((g.val.shape))):
			break

		# If derivative is extremely close to 0, set to +1 or -1 as a form of random restart
		# This avoids making a new guess at, e.g., x + 1e10
		if np.linalg.norm(g.der) < tol:
			g.der = np.ones(g.der.shape) * (der_shift if np.random.random() < 0.5 else -der_shift)

		# Take step and include in path
		step = g.val / g.der
		x = x - step
		g = f(x)
		input_list.append(x.val)
		f_list.append(g.val)

		# Avoid using abs(step) in case guess is at 0, because derivative is not continuous
		cond = -step if step < 0 else step
		
		# If step size is below tolerance, no need to continue
		if cond < tol:
			break
	root = Variable(x.val, g.der)
	input_list = np.reshape(np.concatenate((input_list)), [-1])
	f_list = np.concatenate(f_list)
	return root, input_list, f_list


# Calculate the coefficients of the quadratic functions
def quad_spline_coeff(f, xMin, xMax, nint):
    """ Constructs the matrix for quadratic spline calculation
        and returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
           function specified by user.
        
        xMin: float
              left endpoint of the :math:`x` interval
        
        xMax: float
              right endpoint of the :math:`x` interval
        
        nint: integer
                    number of intervals that you want to slice the original function
            
        Returns
        -------
        y: list of floats
           the right hand side of Ax=y
        
        A: numpy.ndarray
            the sqaure matrix in the left hand side of Ax=y
        
        coeffs: list of floats
                coefficients of a_i, b_i, c_i
        
        ks: list of ``DeriveAlive.Var`` objects
            points of interest in the :math:`x` interval as ``DeriveAlive`` objects
        
        """
    
    h = 1/nint
    ks = []
    for i in np.linspace(xMin, xMax, nint+1):
        k = Variable(i)
        ks.append(k)
    
    # Construct the quadratic functions
    def a(var):
        return var ** 2
    def b(var):
        return var
    def c(var):
        return Variable(1)
    
    # Construct y
    y = []
    for i in range(nint):
        y.append(f(ks[i]).val)
        y.append(f(ks[i+1]).val)
    for i in range(nint):
        y.append([0])
    y = np.vstack(y)
    
    # Construct A
    A = np.zeros((3*nint, 3*nint))
    # Constraint 1:
    for i in range(nint):
        A[2*i, 3*i] = a(ks[i]).val
        A[2*i, 3*i+1] = b(ks[i]).val
        A[2*i, 3*i+2] = c(ks[i]).val
        A[2*i+1, 3*i] = a(ks[i+1]).val
        A[2*i+1, 3*i+1] = b(ks[i+1]).val
        A[2*i+1, 3*i+2] = c(ks[i+1]).val
    # Constraint 2:
    for i in range(nint-1):
        A[2*nint+i, 3*i] = a(ks[i+1]).der
        A[2*nint+i, 3*i+1] = b(ks[i+1]).der
        A[2*nint+i, 3*i+3] = -1*a(ks[i+1]).der
        A[2*nint+i, 3*i+4] = -1*b(ks[i+1]).der
    # Constraint 3:
    A[3*nint-1, 1] = 10*b(ks[0]).der
    A[3*nint-1, -3] = -1*a(ks[-1]).der
    A[3*nint-1, -2] = -1*b(ks[-1]).der
    
    coeffs = np.linalg.solve(A, y)
    
    return y, A, coeffs, ks

# Get the positions of the spline points
def spline_points(f, coeffs, ks, NSP):
    """ Returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
            function specified by user.
        
        coeffs: list of floats
                coefficients of a_i, b_i, c_i
        
        ks: list of ``DeriveAlive.Var`` objects
            points of interest in the :math:`x` interval as ``DeriveAlive`` objects
        
        NSP: integer
                       number of points to draw each spline
        
        Returns
        -------
        spline_points: list of numpy.darrays
                       a list of spline points (x,y) on each s_i
        
        """
    
    spline_points = []

    for i in range(len(ks)-1):
        a = coeffs[3*i]
        b = coeffs[3*i+1]
        c = coeffs[3*i+2]
        sx = np.linspace(ks[i].val, ks[i+1].val, NSP)
        sy = a*(sx**2) + b*sx + c
        spline_points.append([sx, sy])

    return spline_points

# Plot the spline and the orignal function
def quad_spline_plot(f, coeffs, ks, NSP):
    """ Returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
           function specified by user.
        
        coeffs: list of floats
                coefficients of a_i, b_i, c_i
        
        ks: list of ``DeriveAlive.Var`` objects
            points of interest in the :math:`x` interval as ``DeriveAlive`` objects
        
        NSP: integer
                       number of points to draw each spline
        
        Returns
        -------
        fig: matplotlib.figure
             the plot of :math:`f(x)` and splines
        
        """
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    # Plot the original function
    fx = []
    fy = []
    for k in ks:
        fx.append(k.val)
        fy.append(f(k).val)
    ax.plot(fx, fy, 'o-', linewidth=2, label='original')
    
    spline_points = []
    # Plot the splines
    for i in range(len(ks)-1):
        a = coeffs[3*i]
        b = coeffs[3*i+1]
        c = coeffs[3*i+2]
        sx = np.linspace(ks[i].val, ks[i+1].val, NSP)
        sy = a*(sx**2) + b*sx + c
        spline_points.append([sx, sy])
        ax.plot(sx, sy, label=r'$s_{%s}(x)$' % i)
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    box = ax.get_position()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return fig

# Calculate spline squared error
def spline_error(f, spline_points):
    """ Returns the coefficients of quadratic functions.
        
        Parameters
        ----------
        f: ``DeriveAlive.Var`` objects
           function specified by user.
        
        spline_points: list of numpy.darrays
                       a list of spline points (x,y) on each s_i
        
        Returns
        -------
        error: float
               average absolute error of the spline and the original function on one given interval
        
        """
    
    error = 0
    for spline_point in spline_points:
        xs = spline_point[0]
        original_ys = []
        for x in xs:
            original_y = f(Variable(x)).val
            original_ys.append(original_y)
        
        error += abs(sum((np.hstack(original_ys) - spline_point[1])) / len(spline_point[0])) ** 1
    
    return error / len(spline_points)
