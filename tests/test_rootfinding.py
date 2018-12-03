"""
This file contains the tests for root finding.
"""

import pytest
import numpy as np
import autodiff.forward     as fwd
import autodiff.rootfinding as rf


def equals(a, b, tol=1e-10):
    return np.abs(a-b) <= tol

def test_newton_scalar():
    # test 1-d scalar function
    x = fwd.Variable()
    f = x - fwd.sin(x)
    root_1d = rf.newton_scalar(f, {x: 3.0}, 100, tol=1e-6)
    root_x  = root_1d[x]
    assert equals((root_x)-np.sin(root_x), 0.0, tol=1e-6)
    
    # test 2-d scalar function
    x, y = fwd.Variable(), fwd.Variable()
    g = x**2+y**2
    root_2d = rf.newton_scalar(g, {x: 1.0, y: 2.0}, 100, tol=1e-6)
    root_x, root_y = root_2d[x], root_2d[y]
    assert equals(root_x**2+root_y**2, 0.0, tol=1e-6)
    
#    # test warning (not sure how to do it...)
#    x, y = fwd.Variable(), fwd.Variable()
#    g = x**2+y**2
#    try:
#        root_2d = rf.newton_scalar(g, {x: 1.0, y: 2.0}, 10, tol=1e-6)
#    except:
#        assert True