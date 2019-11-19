import numpy as np
import sys
sys.path.append('..')
from autoDiff.autoDiff import Variable
# ELEMENTARY FUNCTIONS

# Exponential
def exp(obj):
	"""
		Returns e raised to the input object

		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:np.exp(pre_val)*pre_der.get(x,0) for x in set(pre_der)}
		val = np.exp(pre_val)
		return Variable(val,der= der)
	except:
		return np.exp(obj)
# Log
def log(obj):
	"""
		Returns the natrual log of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:np.divide(1,pre_val)*pre_der.get(x,0) for x in set(pre_der)}
		val = np.log(pre_val)
		return Variable(val,der=der)
	except:
		return np.log(obj)

# TGRIGONOMETRIC FUNCTIONS

def sin(obj):
	"""
		Returns the sine of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:np.cos(pre_val)*pre_der.get(x,0) for x in set(pre_der)}
		val = np.sin(pre_val)
		return Variable(val,der= der)
	except:
		return np.sin(obj)

def cos(obj):
	"""
		Returns the cosine of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:-np.sin(pre_val)*pre_der.get(x,0) for x in set(pre_der)}
		val = np.cos(pre_val)
		return Variable(val,der= der)
	except:
		return np.cos(obj)
def tan(obj):
	"""
		Returns the tangent of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:(1+np.tan(pre_val)**2)*pre_der.get(x,0) for x in set(pre_der)}
		val = np.tan(pre_val)
		return Variable(val,der= der)
	except:
		return np.tan(obj)

# HYPERBOLIC FUNCTIONS

def sinh(obj):
	"""
		Returns the hyperbolic sine of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:np.cosh(pre_val)*pre_der.get(x,0) for x in set(pre_der)}
		val = np.sinh(pre_val)
		return Variable(val,der= der)
	except:
		return np.sinh(obj)

def cosh(obj):
	"""
		Returns the hyperbolic cosine of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:np.sinh(pre_val)*pre_der.get(x,0) for x in set(pre_der)}
		val = np.cosh(pre_val)
		return Variable(val,der= der)
	except:
		return np.cosh(obj)

def tanh(obj):
	"""

		Returns hyperbolic tangent of the input object

		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:(1-np.tanh(pre_val)**2)*pre_der.get(x,0) for x in set(pre_der)}
		val = np.tanh(pre_val)
		return Variable(val,der= der)
	except:
		return np.tanh(obj)

# INVERSE TGRIGONOMETRIC FUNCTIONS

def arcsin(obj):
	"""
		Returns the inverse sine of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		if pre_val >= 1 or pre_val <= -1:
      		raise ValueError("arcsin does not exist beyond (-1,1)")
		der = {x:((1-(pre_val)**2)**(-0.5))*pre_der.get(x,0) for x in set(pre_der)}
		val = np.arcsin(pre_val)
		return Variable(val,der= der)
	except:
		if obj >= 1 or obj <= -1:
			raise ValueError("arcsin does not exist beyond (-1,1)")
		return np.arcsin(obj)


def arccos(obj):
	"""
		Returns the inverse cosine of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		if pre_val >= 1 or pre_val <= -1:
      		raise ValueError("arccos does not exist beyond (-1,1)")
		der = {x:(-(1-(pre_val)**2)**(-0.5))*pre_der.get(x,0) for x in set(pre_der)}
		val = np.arccos(pre_val)
		return Variable(val,der= der)
	except:
		if obj >= 1 or obj <= -1:
			raise ValueError("arccos does not exist beyond (-1,1)")
		return np.arccos(obj)

def arctan(obj):
	"""
		Returns the inverse tangent of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:np.divide(1,1+(pre_val)**2)*pre_der.get(x,0) for x in set(pre_der)}
		val = np.arctan(pre_val)
		return Variable(val,der= der)
	except:
		return np.arctan(obj)

# OTHER FUNCTION(S)

def sqrt(obj):
	"""
		Returns the square root of the input object
		
		Inputs:
			obj: Autodiff.Variable object or a number

		Output:
			Autodiff.Variable object or a scalar

	"""
	try:
		pre_der = obj.der
		pre_val = obj.val
		der = {x:(0.5*pre_val**(-0.5))*pre_der.get(x,0) for x in set(pre_der)}
		val = np.sqrt(pre_val)
		return Variable(val,der= der)
	except:
		return np.sqrt(obj)

