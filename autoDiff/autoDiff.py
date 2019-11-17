import numpy as np

class Variable:
    def __init__(self, val, name=None, der=None):
            self.val = val
            if name == None and der == None:
                self.name = 'x'
                self.der = {'x': 1.0}
            else:
                self.der = der
    
    def __add__(self, other):
        try:
            der = {var: self.der[var] + other.der[var] for var in self.der.keys()}
            return Variable(self.val + other.val, der=der)
        except AttributeError:
            return Variable(self.val + other, der=self.der)
        
    def __radd__(self, other):
        return self.__add__(other)
   
    def __sub__(self, other):
        try:
            der = {var: self.der[var] - other.der[var] for var in self.der.keys()}
            return Variable(self.val - other.val, der=der)
        except AttributeError:
            return Variable(self.val - other, der=self.der)

    def __rsub__(self, other):
        try:
            der = {var: other.der[var] - self.der[var] for var in self.der.keys()}
            return Variable(other.val - self.val, der=der)
        except AttributeError:
            der = {var: 0 - self.der[var] for var in self.der.keys()}
            return Variable(other - self.val, der= der)

    def __mul__(self, other):
        try:
            der = {var: self.der[var] * other.val + other.der[var] * self.val for var in self.der.keys()}
            return Variable(self.val * other.val, der=der)
        except AttributeError:
            der = {var: self.der[var] * other for var in self.der.keys()}
            return Variable(self.val * other, der=der)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            der = {var: (self.der[var]*other.val-other.der[var]*self.val)/(other.val**2) for var in self.der.keys()}
            return Variable(self.val/other.val, der=der)
        except AttributeError:
            der = {var: self.der[var] / other for var in self.der.keys()}
            return Variable(self.val/other, der=der)

    def __rtruediv__(self, other):
        try:
            der = {var: (self.der[var]*other.val-other.der[var]*self.val)/(other.val**2) for var in self.der.keys()}
            return Variable(self.val/other.val, der=der)
        except AttributeError:
            der = {var: -other/self.der[var] for var in self.der.keys()}
            return Variable(other/self.val, der= der)
    
    def __pow__(self, other):
        #other is a scalar
        der = {var: other*self.val**(other-1)*self.der[var] for var in self.der.keys()}
        return Variable(self.val**other, der=der)

    def __rpow__(self,other):
        return self.__pow__(other)

    def __neg__(self):
        der = {var: -self.der[var] for var in self.der.keys()}
        return Variable(-self.val, der=der)
 



