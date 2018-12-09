"""
This file contains the tests for the backward mode auto differentiation. 
"""
from test_forward import equals
import autodiff.forward as fw
import autodiff.backprop as bp
import pytest
def test_backward():
    """
    testing backward propagation with 3 examples
    """
    a = fw.Variable()
    b = fw.Variable()
    c = a+b
    d = fw.Variable()
    e = c*d
    f = a+e
    val_dict = {b:1,a:2,d:4}
    bp.back_propagation(f,val_dict)
    var_list = [a,b,c,d,e,f]
    for i in var_list:
        assert(equals(i.bder, f.derivative_at(i,val_dict)))
    
    a = fw.Variable()
    b = fw.Variable()
    e = b-a
    c = fw.cos(e)
    d = a+c
    val_dict = {b:1,a:2}
    bp.back_propagation(d,val_dict)
    var_list = [a,b,c,d]
    for i in var_list:
        assert(equals(i.bder, d.derivative_at(i,val_dict)))
        
    
    a = fw.Variable()
    b = fw.Variable()
    c = fw.csc(a)
    d = fw.sec(a)
    e = fw.tan(c)
    f = fw.cotan(d)
    g = fw.sinh(f-e)
    val_dict = {b:1,a:2}
    bp.back_propagation(g,val_dict)
    var_list = [a,b,c,d,e,f,g]
    for i in var_list:
        assert(equals(i.bder, g.derivative_at(i,val_dict)))


