import pytest
import numpy as np
import autodiff.forward as fwd
from autodiff.plot import plot_contour

def testplot():
    """
    Function testing whether plot.py works
    """
    x, y = fwd.Variable(), fwd.Variable()
    f= 100.0*(y - x**2)**2 + (1 - x)**2.0
    plot_contour(f, {x:-2,y:-1}, x, y, plot_range=[-3,3],method = 'gradient_descent')
  