import numpy as np

class Variable:
    def __init__(self, val, der=None):
        """
		Initialization of the Variable class, the default variable name in this milestone is x

		Inputs:
			val: variable value, int or float
            der: (optional) variable derivative

		Output:
			Variable class

	    """
        self.val = val
        self.name = 'x'
        if der == None:
            self.der = {'x': 1.0}
        else:
            self.der = der
    
    def __add__(self, other):
        """ 
        Returns  Variable object from addition
        
        Inputs:
            self: Variable object
            other: Variable object or scalar
        
        Output:
            Variable object: self + other
    
        """
        try:
            der = {var: self.der[var] + other.der[var] for var in self.der.keys()}
            return Variable(self.val + other.val, der=der)
        except AttributeError:
            return Variable(self.val + other, der=self.der)
        
    def __radd__(self, other):
        """ 
        Returns Variable object from addition (from __add__ method)
        
        Inputs:
            self: scalar
            other: Variable object
        
        Output:
            Variable object from __add__ method

        """
        return self.__add__(other)
   
    def __sub__(self, other):
        """ 
        Returns Variable object from subtraction
        
        Inputs:
            self: Variable object
            other: Variable object or scalar
        
        Output:
            Variable object: self - other
    
        """
        try:
            der = {var: self.der[var] - other.der[var] for var in self.der.keys()}
            return Variable(self.val - other.val, der=der)
        except AttributeError:
            return Variable(self.val - other, der=self.der)

    def __rsub__(self, other):
        """ 
        Returns Variable object from subtraction
        
        Inputs:
            self: scalar
            other: Variable object
        
        Output:
            Variable object: self - other
    
        """
        try:
            der = {var: other.der[var] - self.der[var] for var in self.der.keys()}
            return Variable(other.val - self.val, der=der)
        except AttributeError:
            der = {var: 0 - self.der[var] for var in self.der.keys()}
            return Variable(other - self.val, der= der)

    def __mul__(self, other):
        """ 
        Returns Variable object from multiplication
        
        Inputs:
            self: Variable object
            other: Variable object or scalar
        
        Output:
            Variable object: self * other
    
        """
        try:
            der = {var: self.der[var] * other.val + other.der[var] * self.val for var in self.der.keys()}
            return Variable(self.val * other.val, der=der)
        except AttributeError:
            der = {var: self.der[var] * other for var in self.der.keys()}
            return Variable(self.val * other, der=der)

    def __rmul__(self, other):
        """ 
        Returns Variable object from multiplication (from __mul__ method)
        
        Inputs:
            self: scalar
            other: Variable object
        
        Output:
            Variable object from __mul__ method

        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """ 
        Returns Variable object from division
        
        Inputs:
            self: Variable object
            other: Variable object or scalar
        
        Output:
            Variable object: self / other
    
        """
        try:
            if isinstance(other, int) or isinstance(other, float):
                if int(other) == 0:
                    raise ZeroDivisionError('Can not divide by 0')
            elif int(other.val) == 0:
                raise ZeroDivisionError('Can not divide by 0')
            else:
                der = {var: (self.der[var]*other.val-other.der[var]*self.val)/(other.val**2) for var in self.der.keys()}
                return Variable(self.val/other.val, der=der)
        except AttributeError:
            der = {var: self.der[var] / other for var in self.der.keys()}
            return Variable(self.val/other, der=der)

    def __rtruediv__(self, other):
        """ 
        Returns Variable object from division
        
        Inputs:
            self: scalar
            other: Variable object
        
        Output:
            Variable object: self / other
    
        """
        try:
            der = {var: (self.der[var]*other.val-other.der[var]*self.val)/(other.val**2) for var in self.der.keys()}
            return Variable(self.val/other.val, der=der)
        except AttributeError:
            if int(self.val) == 0:
                raise ZeroDivisionError('Can not divide by 0')
            der = {var: -1*other/self.der[var] for var in self.der.keys()}
            return Variable(other/self.val, der= der)
    
    def __pow__(self, other):
        #other is a scalar
        der = {var: other*self.val**(other-1)*self.der[var] for var in self.der.keys()}
        return Variable(self.val**other, der=der)

    def __rpow__(self,other):
        return self.__pow__(other)

    def __neg__(self):
        """ 
        Returns Variable object from negation
        
        Input:
            self: Variable object
        
        Output:
            Variable object: -self

        """
        der = {var: -self.der[var] for var in self.der.keys()}
        return Variable(-self.val, der=der)
 


