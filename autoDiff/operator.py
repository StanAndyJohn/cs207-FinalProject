import numpy as np
# import element_func as fun

class StartUp:
    def __init__(self, n):
        self.n = n
        self.independent_variable_dict = {}
    
    def create_variable(self, x, name):
        """ 
        Returns Variable object after the initialization
        
        Inputs:
            x: variable value
            name: variable name
        
        Output:
            Variable object after the initialization
    
        """
        if name in self.independent_variable_dict.keys():
            raise Exception('Duplicated Name!')
        x = np.array([x])
        der = np.array([0]*self.n)
        try:
            der[len(self.independent_variable_dict)] = 1
            var = Variable(x, der)
            if len(self.independent_variable_dict) < self.n: 
                self.independent_variable_dict[name] = str(var)
        except:
            raise Exception("More variables than declared") 
        return var

class Variable(object):
    def __init__(self, val=[1], der=[1]):
        if isinstance(val, float) or isinstance(val, int):
            self.val = np.array([val])
        else:
            self.val = val
        if isinstance(der, list):
            self.der = np.array(der)
        else:
            self.der = der

    def __str__(self):
        """ 
        Returns the string format of Variable object
        
        Inputs:
            self: Variable object
        
        Output:
            String: Variable object
    
        """
        return "Variable(" + str(self.val) + ", " + str(self.der) + ")"

    def f(self, values, der=[1]):
        """ 
        Returns Variable object from computing vector functions
        
        Inputs:
            values: Variable objects
            der: default is [1]
        
        Output:
            Variable object
    
        """
        new_vals = []
        new_ders = []
        for x in values:
            new_vals.append(x.val)
            new_ders.append(x.der)

        self.val = np.hstack((new_vals))
        self.der = np.vstack((new_ders))
        return Variable(self.val,self.der)


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
            val = self.val + other.val
            der = self.der + other.der
            return Variable(val, der)
        except AttributeError:
            val = self.val + other
            der = self.der
            return Variable(val, der)

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
            val = self.val - other.val
            der = self.der - other.der
            return Variable(val, der)
        except AttributeError:
            val = self.val - other
            der = self.der
            return Variable(val, der)

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
            raise AttributeError()
        except AttributeError:
            val = other - self.val
            der = 0 - self.der
            return Variable(val, der)

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
            val = self.val * other.val
            m1 = np.multiply(other.val, self.der) 
            m2 = np.multiply(self.val, other.der)
            der = m1 + m2
            return Variable(val, der)
        except AttributeError:
            val = self.val * other
            der = self.der * other
            return Variable(val, der)

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
            val = self.val / other.val
            
            len_other_der_shape = len(other.der.shape)
            len_self_der_shape = len(self.der.shape)

            self_val = np.expand_dims(self.val, 1) if len_other_der_shape > 1 else self.val
            other_val = np.expand_dims(other.val, 1) if len_self_der_shape > 1 else other.val
            
            num = self.der*other.val-other.der*self.val
            denom = other_val ** 2 if len(num.shape) > 1 else other.val ** 2
            der = num/denom
            return Variable(val, der)
        except AttributeError:
            val = self.val/other
            der = self.der / other
            return Variable(val, der)

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
            raise AttributeError()
        except AttributeError:
            val = other/self.val
            der = (-1)*other/self.val**(2)*self.der
            return Variable(val, der)
    
    def __pow__(self, other):
        """ 
        Returns Variable object from power
        
        Inputs:
            self: Variable object
            other: Variable object or scalar
        
        Output:
            Variable object: self ^ other
    
        """

        if isinstance(other, float) or isinstance(other, int):
            val = self.val**other
            der =other*self.val**(other-1)*self.der
            return Variable(val, der)
        else:
            val = np.power(self.val, other.val)
            der = val * ((other.val / self.val) * self.der + np.log(self.val) * other.der)
            return Variable(val, der)



    def __rpow__(self,other):
        """ 
        Returns Variable object from power
        
        Inputs:
            self: scalar
            other: Variable object
        
        Output:
            Variable object: self ^ other
    
        """
        val = other**self.val
        der = other**self.val*np.log(other)*self.der
        return Variable(val, der)

    def __neg__(self):
        """ 
        Returns Variable object from negation
        
        Input:
            self: Variable object
        
        Output:
            Variable object: -self

        """
        val = - self.val
        der = - self.der if len(self.der.shape) else None
        return Variable(val, der)

    def __eq__(self, other):
        """
        Returns True if two Variable objects are the same, otherwise, returns False
        
        Inputs:
            self: Variable object
            other: Variable object

        Output:
            Bool: True -> the same, False -> different
        
        """
        try:
            return (np.array_equal(self.val, other.val) and 
                    np.array_equal(self.der, other.der))
        except AttributeError:
            return False

    def __ne__(self, other):
        """
        Returns True if two Variable objects are different, otherwise, returns False
        
        Inputs:
            self: Variable object
            other: Variable object

        Output:
            Bool: True -> different, False -> the same
        
        """
        return not self.__eq__(other)



# a = StartUp(2)
# x1 = a.create_variable(4, 'x1')
# x2 = a.create_variable(4, 'x2')
# print(x1)
# print(x2)
# print(x1 != x2)
