"""
This file contains the tests for the forward mode auto differentiation. We may 
want to separate the code into multiple files later.
"""
import pytest
import numpy as np
import autodiff.forward as fwd

def test_negation():
    x=fwd.Variable()
    assert x.evaluation_at({x: 3.0}) == -3.0
    assert x.derivative_at(x, {x: 3.0}) == -1.0

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
