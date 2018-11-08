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
                sub_expr2.evaluation_at(val_dict)+ \
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


def exp(expr):
    return Expression(Exp, expr)
