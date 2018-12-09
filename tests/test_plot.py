import pytest
import numpy as np
import autodiff.forward     as fwd
import autodiff.rootfinding as rf

def testplot():
    x, y = fwd.Variable(), fwd.Variable()
    g = x**2 + y**2
    plot_contour(g,{x: 1.0, y: 2.0},method = 'gradient_descent')
    plot_contour(g,{x: 1.0, y: 2.0},method = 'newton')
  