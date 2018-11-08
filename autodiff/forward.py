"""
This file contains the central data structure and functions related to the
forward mode auto differentiation. We may want to separate the code into 
multiple files later.
"""

import numpy as np

class Expression:
    def __init__(self, ele_func, sub_expr1, sub_expr2=None):
        self._ele_func  = ele_func
        self._sub_expr1 = sub_expr1
        self._sub_expr2 = sub_expr2
    
    def evaluation_at(self, val_dict):
        
        # self._sub_expr2 is None implies that self._ele_func is an unary operator
        if self._sub_expr2 is None: 
            return self._ele_func.evaluation_at(
                self._sub_expr1, val_dict)
        
        # self._sub_expr2 not None implies that self._ele_func is a binary operator
        else:
            return self._ele_func.evaluation_at(
                self._sub_expr1, self._sub_expr2, val_dict)
    
    def derivative_at(self, var, val_dict):
        
        if var is self: return 1.0
        
        # sub_expr2 is None implies that _ele_func is an unary operator
        if self._sub_expr2 is None:
            return self._ele_func.derivative_at(
                self._sub_expr1, var, val_dict)
        
        # sub_expr2 not None implies that _ele_func is a binary operator
        else:
            return self._ele_func.derivative_at(
                self._sub_expr1, self._sub_expr2, var, val_dict)
    
    def __neg__(self):
        return Expression(Neg, self)

                
    def __add__(self, another):
        if isinstance(another, Expression):
            return Expression(Add, self, another)
        # if the other operand is not an Expression, then it must be a number
        # the number then should be converted to a Constant
        else:
            return Expression(Add, self, Constant(another))
    
    
    def __radd__(self, another):
        if isinstance(another, Expression):
            return Expression(Add, another, self)
        else:
            return Expression(Add, Constant(another), self)
    
    def __sub__(self, another):
        if isinstance(another, Expression):
            return Expression(Sub, self, another)
        else:
            return Expression(Sub, self, Constant(another))
    
    def __rsub__(self, another):
        if isinstance(another, Expression):
            return Expression(Sub, another, self)
        else:
            return Expression(Sub, Constant(another), self)
        

    def __mul__(self, another):
        if isinstance(another, Expression):
            return Expression(Mul,self,another)
        else:
            return Expression(Mul, self, Constant(another))

    def __rmul__(self, another):
        if isinstance(another, Expression):
            return Expression(Mul,another,self)
        else:
            return Expression(Mul, Constant(another),self)
    
    def __truediv__(self, another):
        if isinstance(another, Expression):
            return Expression(Div,self,another)
        else:
            return Expression(Div, self, Constant(another))

    def __rtruediv__(self, another):
        if isinstance(another, Expression):
            return Expression(Div,another,self)
        else:
            return Expression(Div, Constant(another),self)
    
    def __pow__(self,another):
        if isinstance(another, Expression):
            return Expression(Pow,self,another)
        else:
            return Expression(Pow, self, Constant(another))
    
    def __rpow__(self,another):
        if isinstance(another, Expression):
            return Expression(Pow,another,self)
        else:
            return Expression(Pow, Constant(another),self)


class Variable(Expression):
    def __init__(self):
        return
    
    def evaluation_at(self, val_dict):
        return val_dict[self]
    
    def derivative_at(self, var, val_dict):
        return 1.0 if var is self else 0.0


class Constant(Expression):
    def __init__(self, val):
        self.val = val
        
    def evaluation_at(self, val_dict):
        return self.val
    
    def derivative_at(self, var, val_dict):
        return 0.0


class Add:
    @staticmethod
    def evaluation_at(sub_expr1, sub_expr2, val_dict):
        return sub_expr1.evaluation_at(val_dict) + \
               sub_expr2.evaluation_at(val_dict)
    @staticmethod
    def derivative_at(sub_expr1, sub_expr2, var, val_dict):
        return sub_expr1.derivative_at(var, val_dict) + \
               sub_expr2.derivative_at(var, val_dict)

class Sub:
    @staticmethod
    def evaluation_at(sub_expr1, sub_expr2, val_dict):
        return sub_expr1.evaluation_at(val_dict) - \
               sub_expr2.evaluation_at(val_dict)
    @staticmethod
    def derivative_at(sub_expr1, sub_expr2, var, val_dict):
        return sub_expr1.derivative_at(var, val_dict) - \
               sub_expr2.derivative_at(var, val_dict)

class Mul:
    @staticmethod
    def evaluation_at(sub_expr1, sub_expr2, val_dict):
        return sub_expr1.evaluation_at(val_dict) *\
               sub_expr2.evaluation_at(val_dict)
    @staticmethod
    def derivative_at(sub_expr1, sub_expr2, var, val_dict):
        return sub_expr1.derivative_at(var, val_dict) * \
               sub_expr2.evaluation_at(val_dict)+ \
               sub_expr1.evaluation_at(val_dict) *\
               sub_expr2.derivative_at(var, val_dict)
               
class Div:
    @staticmethod
    def evaluation_at(sub_expr1, sub_expr2, val_dict):
        return sub_expr1.evaluation_at(val_dict) /\
               sub_expr2.evaluation_at(val_dict)
    @staticmethod
    def derivative_at(sub_expr1, sub_expr2, var, val_dict):
        return  sub_expr1.derivative_at(var, val_dict) / \
                sub_expr2.evaluation_at(val_dict)- \
                sub_expr1.evaluation_at(val_dict) *\
                sub_expr2.derivative_at(var, val_dict)/\
                sub_expr2.evaluation_at(val_dict)/\
                sub_expr2.evaluation_at(val_dict)

class Pow:
    
    @staticmethod
    def evaluation_at(sub_expr1, sub_expr2, val_dict):
        return sub_expr1.evaluation_at(val_dict) **\
               sub_expr2.evaluation_at(val_dict)
    @staticmethod
    #f(x)^g(x) * g‘(x)  * ln( f(x) )+ f(x)^( g(x)-1 ) * g(x) * f’(x) 
    def derivative_at(sub_expr1, sub_expr2, var, val_dict):
        return  sub_expr1.evaluation_at(val_dict)** \
                sub_expr2.evaluation_at(val_dict)* \
                sub_expr2.derivative_at(var, val_dict)*\
                np.log(sub_expr1.evaluation_at(val_dict))+ \
                sub_expr1.evaluation_at(val_dict) **\
                (sub_expr2.evaluation_at(val_dict)-1)*\
                sub_expr2.evaluation_at(val_dict)*\
                sub_expr1.derivative_at(var, val_dict)

class Exp:
    @staticmethod
    def evaluation_at(sub_expr1, val_dict):
        return np.exp(sub_expr1.evaluation_at(val_dict))
    
    @staticmethod
    def derivative_at(sub_expr1, var, val_dict):
        return sub_expr1.derivative_at(var, val_dict) * \
               np.exp(sub_expr1.evaluation_at(val_dict))

class Neg:
    @staticmethod
    def evaluation_at(sub_expr1, val_dict):
        return -sub_expr1.evaluation_at(val_dict)
    
    @staticmethod
    def derivative_at(sub_expr1, var, val_dict):
        return -sub_expr1.derivative_at(var, val_dict)
             

def exp(expr):
    return Expression(Exp, expr)


class Sin:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        return np.sin(sub_expr1.evaluation_at(val_dict))
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        return sub_expr1.derivative_at(var, val_dict) * \
        np.cos(sub_expr1.evaluation_at(val_dict)) 

def sin(expr):
    return Expression(Sin, expr)

class Cos:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        return np.cos(sub_expr1.evaluation_at(val_dict))
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        return -sub_expr1.derivative_at(var, val_dict) * \
               np.sin(sub_expr1.evaluation_at(val_dict)) 

def cos(expr):
    return Expression(Cos, expr)
    
class Tan:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        return np.tan(sub_expr1.evaluation_at(val_dict))
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        return sub_expr1.derivative_at(var, val_dict) * \
               (1/np.cos(2*sub_expr1.evaluation_at(val_dict)))

def tan(expr):
    return Expression(Tan, expr)
    
class Cotan:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        return 1/np.tan(sub_expr1.evaluation_at(val_dict))
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):    
        return -sub_expr1.derivative_at(var, val_dict) * \
               (1/np.sin(sub_expr1.evaluation_at(val_dict))**2)

def cotan(expr):
    return Expression(Cotan, expr)
    
class Sec:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        return 1/np.cos(sub_expr1.evaluation_at(val_dict))
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return sub_expr1.derivative_at(var, val_dict) * \
               np.tan(x) * (1/np.cos(x))
def sec(expr):
    return Expression(Sec, expr) 

class Csc:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        return 1/np.sin(sub_expr1.evaluation_at(val_dict))
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return sub_expr1.derivative_at(var, val_dict) * \
               (1/np.tan(x)) * (1/np.sin(x))

def csc(expr):
    return Expression(Csc, expr) 

class Sinh:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        return np.sinh(sub_expr1.evaluation_at(val_dict))
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return sub_expr1.derivative_at(var, val_dict) * np.cosh(x)

def sinh(expr):
    return Expression(Sinh, expr) 

class Cosh:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        return np.cosh(sub_expr1.evaluation_at(val_dict))
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return sub_expr1.derivative_at(var, val_dict) * np.sinh(x)

def cosh(expr):
    return Expression(Cosh, expr) 
    
class Tanh:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return np.sinh(x)/np.cosh(x)
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return sub_expr1.derivative_at(var, val_dict) * (1/np.cosh(x)**2)

def tanh(expr):
    return Expression(Tanh,expr) 

class Csch:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return 1/np.sinh(x)
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        # d = -csch(x)*cot(x)
        d = -(1/np.sinh(x)) * (np.cosh(x)/np.sinh(x))
        return sub_expr1.derivative_at(var, val_dict) * d

def csch(expr):
    return Expression(Csch, expr) 

class Sech:
    def evaluation_at(sub_expr1,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return 1/np.cosh(x)
    
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        # d = -sech(x)tanh(x)
        d = -(1/np.cosh(x)) * (np.sinh(x)/np.cosh(x))
        return sub_expr1.derivative_at(var, val_dict)*d

def sech(expr):
    return Expression(Sech, expr) 

class Coth:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return np.cosh(x)/np.sinh(x)
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        # d = -csch^2(x)
        return -sub_expr1.derivative_at(var, val_dict) * (1/np.sinh(x)**2)

def coth(expr):
    return Expression(Coth, expr)    

class Arcsin:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return np.arcsin(x)
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        d = 1/np.sqrt(1-x**2)
        #1/sqrt(1-x^2)
        return sub_expr1.derivative_at(var, val_dict) * d

def arcsin(expr):
    return Expression(Arcsin, expr)
    
class Arccos:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return np.arccos(x)
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        d = 1/np.sqrt(1-x**2)
        #-1/sqrt(1-x^2)
        return -sub_expr1.derivative_at(var, val_dict) * d

def arccos(expr):
    return Expression(Arccos, expr)
    
class Arctan:
    @staticmethod
    def evaluation_at(sub_expr1,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        return np.arctan(x)
    
    @staticmethod
    def derivative_at(sub_expr1,var,val_dict):
        x = sub_expr1.evaluation_at(val_dict)
        d = 1/(1+x**2)
        # d =1/1-x^2
        return sub_expr1.derivative_at(var, val_dict) * d

def arctan(expr):
    return Expression(Arctan, expr)