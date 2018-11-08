"""
This file contains the tests for the forward mode auto differentiation. We may 
want to separate the code into multiple files later.
"""
import pytest
import numpy as np
import autodiff.forward as fwd

def test_negation():
    x=fwd.Variable()
    f=-x
    assert f.evaluation_at({x: 3.0}) == -3.0
    assert f.derivative_at(x, {x: 3.0}) == -1.0

def test_adding_constant():
    a = fwd.Variable()
    assert (a+1).derivative_at(a, {a: 0.0}) == 1.0
    assert (1+a).derivative_at(a, {a: 0.0}) == 1.0
    
def test_subtracting_constant():
    a = fwd.Variable()
    assert (a-1).derivative_at(a, {a: 0.0}) == 1.0
    assert (1-a).derivative_at(a, {a: 0.0}) == -1.0

def test_adding_three_variables():
    a = fwd.Variable()
    b = fwd.Variable()
    c = fwd.Variable()
    f = fwd.exp(a-b+c)
    assert f.evaluation_at({a: 1.0, b: 2.0, c: 3.0})    == np.exp(2.0)
    assert f.derivative_at(b, {a: 1.0, b: 2.0, c: 3.0}) == -np.exp(2.0)
    assert f.derivative_at(a, {a: 1.0, b: 2.0, c: 3.0}) == np.exp(2.0)
    
def test_exp():
    x = fwd.Variable()
    y = fwd.Variable()
    g = x + fwd.exp(y-1)
    assert g.evaluation_at({x: 1.0, y: 2.0})    == 1.0 + np.exp(1.0)
    assert g.derivative_at(x, {x: 1.0, y: 2.0}) == 1.0
    assert g.derivative_at(y, {x: 1.0, y: 2.0}) == np.exp(1.0)

def test_multiply_constant():
    x = fwd.Variable()
    assert (2.0*x).derivative_at(x,{x:3.0}) == 2.0
    assert (x*2.0).derivative_at(x,{x:3.0}) == 2.0

def test_divide_constant():
    x = fwd.Variable()
    assert (x/2.0).derivative_at(x,{x:3.0}) == 0.5
    assert (2.0/x).derivative_at(x,{x:3.0}) == -2/9.0

def test_multiply():
    x = fwd.Variable()
    y = fwd.Variable()
    f = x*y
    assert f.evaluation_at({x: 3.0, y: 2.0}) == 6.0
    assert f.derivative_at(x, {x: 3.0, y: 2.0}) == 2.0
    assert f.derivative_at(y, {x: 3.0, y: 2.0}) == 3.0

def test_divide():
    x = fwd.Variable()
    y = fwd.Variable()
    f = x/y
    assert f.evaluation_at({x: 3.0, y: 2.0}) == 1.5
    assert f.derivative_at(x, {x: 3.0, y: 2.0}) == 1/2.0
    assert f.derivative_at(y, {x: 3.0, y: 2.0}) == -0.75

def test_power():
    x = fwd.Variable()
    y = fwd.Variable()
    f = x**y
    assert f.evaluation_at({x: 3.0, y: 2.0}) == 9.0
    assert f.derivative_at(x, {x: 3.0, y: 2.0}) == 6.0
    assert f.derivative_at(y, {x: 3.0, y: 2.0}) == np.log(3.)*3**2

def test_sin():
    a = fwd.Variable()
    b = fwd.Variable()
    f = fwd.sin(a*b)
    assert(f.derivative_at(a,{a:2,b:2}==np.cos(4)*2))
    assert(f.evaluation_at({a:1,b:2}) == np.sin(2))


def test_cos():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a+b
    f1 = fwd.cos(a+c)
    f2 = fwd.cos(a*b)
    assert(f1.evaluation_at({a:1.0, b: 2.0}) == np.cos(4))
    assert(f2.evaluation_at({a:1.0,b:2}) == np.cos(2))
    assert(f1.derivative_at(a,{a:1.0, b: 2.0}) == -np.sin(1+3)*2)
    assert(f2.derivative_at(a,{a:2,b:2}) == -np.sin(2*2)*2)

def test_tan():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.tan(c*b)
    assert f.evaluation_at({a:1,b:2}) == np.tan(4)
    assert f.derivative_at(c,{a:1,b:2}) == 2*(1/np.cos(4*2))

def test_cotan():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.cotan(c*b)
    assert f.evaluation_at({a:1,b:2}) == 1/np.tan(4)
    assert f.derivative_at(c,{a:1,b:2}) == -(1/(np.sin(4)**2))*2

def test_sec():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.sec(c*b)
    assert f.evaluation_at({a:1,b:2}) == 1/np.cos(4)
    assert f.derivative_at(c,{a:1,b:2}) == np.tan(4)*(1/np.cos(4))*2
    
def test_csc():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.csc(c*b)
    assert f.evaluation_at({a:1,b:2}) == 1/np.sin(4)
    assert f.derivative_at(c,{a:1,b:2}) == (1/np.tan(4))*(1/np.sin(4))*2

def test_sinh():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.sinh(c*b)
    assert f.evaluation_at({a:1,b:2})==np.sinh(4)
    assert f.derivative_at(c,{a:1,b:2})==np.cosh(4)*2


def test_cosh():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.cosh(c*b)
    assert(f.evaluation_at({a:3,b:2})==np.cosh(12))
    assert(f.derivative_at(c,{a:3,b:2})==np.sinh(12)*2)
    
def test_tanh():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.tan(c*b)
    assert(f.evaluation_at({a:3,b:2})==np.sin(12)/np.cos(12))
    assert(f.derivative_at(c,{a:3,b:2})==1/np.cos(24)*2)

def test_csch():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.csch(c*b)
    assert f.evaluation_at({a:3,b:2}) == 1/np.sinh(12)
    assert f.derivative_at(c,{a:3,b:2}) == \
     -(np.cosh(12)/np.sinh(12))*(1/np.sinh(12))*2

def test_sech():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.sech(c*b)
    assert f.evaluation_at({a:2,b:1}) == 1/np.cosh(2)
    assert f.derivative_at(c,{a:2,b:1}) == \
     -(np.sinh(2)/np.cosh(2))*(1/np.cosh(2))*1

def test_coth():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.coth(c*b)
    assert f.evaluation_at({a:3,b:2}) == np.cosh(12)/np.sinh(12)
    assert f.derivative_at(c,{a:3,b:2}) == -(1/np.sinh(12))**2*2

def test_arcsin():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.arcsin(c*b)
    assert f.evaluation_at({a:0.2,b:0.5}) == np.arcsin(0.05)
    assert f.derivative_at(c,{a:0.2,b:0.5})== \
    (1/np.sqrt(1-(0.2*0.5*0.5)**2))*0.5

def test_arccos():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.arccos(c*b)
    assert f.evaluation_at({a:0.2,b:0.5})== np.arccos(0.05)
    assert f.derivative_at(c,{a:0.2,b:0.5}) == \
     (-1/np.sqrt(1-(0.2*0.5*0.5)**2))*0.5

def test_arctan():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.arctan(c*b)
    assert(f.evaluation_at({a:2,b:3}) == np.arctan(18))
    assert(f.derivative_at(c,{a:2,b:3}) == (1/(18**2+1))*3)

    
