"""
This file contains the tests for root finding using Newont's method.
"""

import pytest
import numpy as np
import autodiff.forward     as fwd
import autodiff.rootfinding as rf


def equals(a, b, tol=1e-10):
    """
    Function comparing equal with accomodation to precision difference
    """
    return np.abs(a-b) <= tol

def test_newton_scalar():
    """
    Function testing Newton's Method on finding roots for 1-D or 2-D scalar 
    function using both forward and backward modes
    """ 
    # test 1-d scalar function
    x = fwd.Variable()
    f = x - fwd.sin(x)
    root_1d = rf.newton_scalar(f, {x: 3.0}, 100, tol=1e-6)
    root_x  = root_1d[x]
    assert equals((root_x)-np.sin(root_x), 0.0, tol=1e-6)
    root_1d = rf.newton_scalar(f, {x: 3.0}, 100, tol=1e-6, 
    method = 'backward')
    root_x  = root_1d[x]
    assert equals((root_x)-np.sin(root_x), 0.0, tol=1e-6)
    
    # test 2-d scalar function
    x, y = fwd.Variable(), fwd.Variable()
    g = x**2+y**2
    root_2d = rf.newton_scalar(g, {x: 1.0, y: 2.0}, 100, tol=1e-6)
    root_x, root_y = root_2d[x], root_2d[y]
    assert equals(root_x**2+root_y**2, 0.0, tol=1e-6)
    root_2d = rf.newton_scalar(g, {x: 1.0, y: 2.0}, 100, tol=1e-6),
    method = 'backward')
    root_x, root_y = root_2d[x], root_2d[y]
    assert equals(root_x**2+root_y**2, 0.0, tol = 1e-6)
    # test warning
    x = fwd.Variable()
    f = x - fwd.sin(x)
    root_warning = rf.newton_scalar(f, {x: 3.0}, 0, tol=1e-6)
    
