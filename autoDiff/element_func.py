import numpy as np
import sys
sys.path.append('..')
from autoDiff.operator import Variable
# ELEMENTARY FUNCTIONS

# Exponential
def exp(obj):
	"""
		Returns e raised to the input object

		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object or scalar

	"""
	if isinstance(obj, Variable):
		
		val = np.exp(obj.val)
		der = np.exp(obj.val)
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.exp(obj)
# Log natural
def log(obj):
	"""
		Returns the natrual log of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.log(obj.val)
		der = np.divide(1,obj.val)
		# print(obj.der)
		# print(der)
		# print(obj.der.shape)
		# print(der.shape)
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.log(obj)

# Log 2
def log2(obj):
	"""
		Returns the natrual log of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.log2(obj.val)
		der = np.divide(1,obj.val)
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.log2(obj)

# Log 10
def log10(obj):
	"""
		Returns the natrual log of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.log10(obj.val)
		der = np.divide(1,obj.val)
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.log10(obj)

# TGRIGONOMETRIC FUNCTIONS

def sin(obj):
	"""
		Returns the sine of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.sin(obj.val)
		der = np.cos(obj.val)
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.sin(obj)

def cos(obj):
	"""
		Returns the cosine of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.cos(obj.val)
		der = -np.sin(obj.val)
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.cos(obj)

def tan(obj):
	"""
		Returns the tangent of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.tan(obj.val)
		der = 1+np.tan(obj.val)**2
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.tan(obj)
	
# HYPERBOLIC FUNCTIONS

def sinh(obj):
	"""
		Returns the hyperbolic sine of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.sinh(obj.val)
		der = np.cosh(obj.val)
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.sinh(obj)

def cosh(obj):
	"""
		Returns the hyperbolic cosine of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.cosh(obj.val)
		der = np.sinh(obj.val)
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.cosh(obj)
def tanh(obj):
	"""

		Returns hyperbolic tangent of the input object

		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.tanh(obj.val)
		der = 1-np.tanh(obj.val)**2
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		return np.tanh(obj)

# INVERSE TGRIGONOMETRIC FUNCTIONS

def arcsin(obj):
	"""
		Returns the inverse sine of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		values = map(lambda x: -1 <= x <= 1, obj.val)
		if not all(values):
			raise ValueError("Domain of arcsin is [-1, 1].")	
		val = np.arcsin(obj.val)
		if obj.val == 1:
			der = np.nan
		elif obj.val == -1:
			der = np.nan
		else:
			der = 1 / np.sqrt(1 - (obj.val ** 2))
			if len(obj.der.shape)>len(der.shape):
				der = np.expand_dims(der,1)
			der = np.multiply(der, obj.der)
		return Variable(val, der)
	else:
		if obj >=1 or obj<= -1:
			raise ValueError("Domain of arcsin is [-1, 1].")	
		return np.arcsin(obj)

def arccos(obj):
	"""
		Returns the inverse cosine of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		values = map(lambda x: -1 <= x <= 1, obj.val)
		if not all(values):
			raise ValueError("Domain of arccos is [-1, 1].")	
		val = np.arccos(obj.val)
		if obj.val == 1:
			der = np.nan
		elif obj.val == -1:
			der = np.nan
		else:
			der = -1 / np.sqrt(1 - (obj.val ** 2))
			if len(obj.der.shape)>len(der.shape):
				der = np.expand_dims(der,1)
			der = np.multiply(der, obj.der)
		return Variable(val,der)
	else:
		if obj >=1 or obj<= -1:
			raise ValueError("Domain of arccos is [-1, 1].")	
		return np.arccos(obj)

def arctan(obj):
	"""
		Returns the inverse tangent of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object

	"""
	if isinstance(obj, Variable):
		val = np.arctan(obj.val)
		der = 1 / (1 + (obj.val) ** 2)
		if len(obj.der.shape)>len(der.shape):
			der = np.expand_dims(der,1)
		der = np.multiply(der, obj.der)
		return Variable(val,der)
	else:
		return np.arctan(obj)

# OTHER FUNCTION(S)

def sqrt(obj):
	"""
		Returns the square root of the input object
		
		Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object or a scalar

	"""
	if isinstance(obj, Variable):
  		new_Variable = Variable(obj.val, obj.der)
  		return new_Variable.__pow__(0.5)
	else:
		return np.sqrt(obj)

def logistic(obj):
	""" 
	Return the logistic function evaluation (sigmoid): f(x) = 1 / (1 + e^{-x})
	Inputs:
			obj: Autodiff.Variableiable object or a number

		Output:
			Autodiff.Variableiable object or a scalar

	"""

	# Case for constant scalar or vector with no derivative
	if isinstance(obj, Variable):
		val = 1 / (1 + np.exp(-obj.val))
		der = np.exp(obj.val) / ((1 + np.exp(obj.val)) ** 2)

		if len(obj.der.shape) > 1:
			new_der = np.reshape(der, [-1, 1])
		else:
			new_der = der
		final_der = new_der * obj.der
		return Variable(val, final_der)
	else:
		val = 1 / (1 + np.exp(-obj))
		return val
	

