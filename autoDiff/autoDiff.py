import numpy as np

class Variable:
    def __init__(self, val, name=None, der=None):
            self.val = val
            self.name = name
            if name != None:
                self.der = {name: 1.0} 
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
        return self.__sub__(other)

    def __mul__(self, other):
        try:
            der = {var: self.der[var] * other.val + other.der[var] * self.val for var in self.der.keys()}
            return Variable(self.val * other.val, der=der)
        except AttributeError:
            return Variable(self.val * other, der=self.der*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            der = {var: (self.der[var]*other.val-other.der[var]*self.val)/(other.val**2) for var in self.der.keys()}
            return Variable(self.val/other.val, der=der)
        except AttributeError:
            return Variable(self.val/other, der=self.der/other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __pow__(self, other):
        #other is a scalar
        der = {var: other*self.val**(other-1)*self.der[var] for var in self.der.keys()}
        return Variable(self.val**other, der=der)

    def __rpow__(self,other):
        return self.__pow__(other)

    def __neg__(self):
        der = {var: -self.der[var] for var in self.der.keys()}
        return Variable(-self.val, der=der)
 


# x1 = Variable(1, 'x1')
# x2 = x1+1
# x3 = x1+x2
# print(x1.val)
# print(x1.der)
# print(x2.val)
# print(x2.der)
# print(x3.val)
# print(x3.der)




