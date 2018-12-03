# -*- coding: utf-8 -*-

import pytest
import numpy as np
import autodiff.forward  as fwd
import autodiff.optimize as opt

def equals(a, b, tol=1e-10):
    return np.abs(a-b) <= tol

def test_bfgs():
    x, y = fwd.Variable(), fwd.Variable()
    rosen = 100.0*(y - x**2)**2 + (1 - x)**2.0
    bfgs_returned = opt.bfgs(rosen, {x: 0.0, y: 1.0})
    assert equals(bfgs_returned[x], 1.0, tol=1e-6)
    assert equals(bfgs_returned[y], 1.0, tol=1e-6)
    