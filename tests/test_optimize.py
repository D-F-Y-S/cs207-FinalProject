# -*- coding: utf-8 -*-

"""
This file contains the tests for newton's method and gradient descent.
"""
import pytest
import numpy as np
import autodiff.forward  as fwd
import autodiff.optimize as opt

def equals(a, b, tol=1e-10):
    """
    Function for comparing equal, accomodating precision differences
    """
    return np.abs(a-b) <= tol


def test_newton():
    """
    Function testing Newton's Method
    """
    x, y = fwd.Variable(), fwd.Variable()
    rosen = 100.0*(y - x**2)**2 + (1 - x)**2.0
    newton_returned = opt.newton(rosen, {x: 0.0, y: 1.0})
    assert equals(newton_returned[x], 1.0, tol=1e-6)
    assert equals(newton_returned[y], 1.0, tol=1e-6)
    

def test_bfgs():
    """
    Function testing Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm 
    """
    x, y = fwd.Variable(), fwd.Variable()
    rosen = 100.0*(y - x**2)**2 + (1 - x)**2.0
    bfgs_returned = opt.bfgs(rosen, {x: 0.0, y: 1.0})
    assert equals(bfgs_returned[x], 1.0, tol=1e-6)
    assert equals(bfgs_returned[y], 1.0, tol=1e-6)

def test_gradient_descent():
    """
    Function testing Gradient Descent
    """
    x, y = fwd.Variable(), fwd.Variable()
    f= x**2-2*x+1+y**2
    gradient_descent_returned = opt.gradient_descent(f, {x: -1.0, y: 2.0},learning_rate=0.1)
    assert equals(gradient_descent_returned[x], 1.0, tol=1e-3)
    assert equals(gradient_descent_returned[y], 0.0, tol=1e-3)
    