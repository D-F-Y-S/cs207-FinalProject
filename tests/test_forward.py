"""
This file contains the tests for the forward mode auto differentiation. We may 
want to separate the code into multiple files later.
"""

import pytest
import numpy as np
import autodiff.forward as fwd

def equals(a, b, tol=1e-10):
    return np.abs(a-b) <= tol

def test_negation():
    x=fwd.Variable()
    f=-x
    assert equals(f.evaluation_at({x: 3.0}),    -3.0)
    assert equals(f.derivative_at(x, {x: 3.0}), -1.0)

def test_adding_constant():
    a = fwd.Variable()
    assert equals((a+1).derivative_at(a, {a: 0.0}), 1.0)
    assert equals((1+a).derivative_at(a, {a: 0.0}), 1.0)
    
def test_subtracting_constant():
    a = fwd.Variable()
    assert equals((a-1).derivative_at(a, {a: 0.0}),  1.0)
    assert equals((1-a).derivative_at(a, {a: 0.0}), -1.0)

def test_adding_three_variables():
    a = fwd.Variable()
    b = fwd.Variable()
    c = fwd.Variable()
    f = fwd.exp(a-b+c)
    assert equals(f.evaluation_at({a: 1.0, b: 2.0, c: 3.0}),     np.exp(2.0))
    assert equals(f.derivative_at(b, {a: 1.0, b: 2.0, c: 3.0}), -np.exp(2.0))
    assert equals(f.derivative_at(a, {a: 1.0, b: 2.0, c: 3.0}),  np.exp(2.0))
    
def test_exp():
    x = fwd.Variable()
    y = fwd.Variable()
    g = x + fwd.exp(y-1)
    assert equals(g.evaluation_at({x: 1.0, y: 2.0}),    1.0+np.exp(1.0))
    assert equals(g.derivative_at(x, {x: 1.0, y: 2.0}), 1.0)
    assert equals(g.derivative_at(y, {x: 1.0, y: 2.0}), np.exp(1.0))

def test_multiply_constant():
    x = fwd.Variable()
    assert equals((2.0*x).derivative_at(x,{x:3.0}), 2.0)
    assert equals((x*2.0).derivative_at(x,{x:3.0}), 2.0)

def test_divide_constant():
    x = fwd.Variable()
    assert equals((x/2.0).derivative_at(x,{x:3.0}), 0.5)
    assert equals((2.0/x).derivative_at(x,{x:3.0}), -2/9.0)

def test_multiply():
    x = fwd.Variable()
    y = fwd.Variable()
    f = x*y
    assert equals(f.evaluation_at({x: 3.0, y: 2.0}),    6.0)
    assert equals(f.derivative_at(x, {x: 3.0, y: 2.0}), 2.0)
    assert equals(f.derivative_at(y, {x: 3.0, y: 2.0}), 3.0)

def test_divide():
    x = fwd.Variable()
    y = fwd.Variable()
    f = x/y
    assert equals(f.evaluation_at({x: 3.0, y: 2.0}), 1.5)
    assert equals(f.derivative_at(x, {x: 3.0, y: 2.0}), 1/2.0)
    assert equals(f.derivative_at(y, {x: 3.0, y: 2.0}), -0.75)

def test_power():
    x = fwd.Variable()
    y = fwd.Variable()
    f = x**y
    assert equals(f.evaluation_at({x: 3.0, y: 2.0}),    9.0)
    assert equals(f.derivative_at(x, {x: 3.0, y: 2.0}), 6.0)
#    assert equals(f.derivative_at(y, {x: 3.0, y: 2.0}), np.log(3.)*3**2)

def test_sin():
    a = fwd.Variable()
    b = fwd.Variable()
    f = fwd.sin(a*b)
    assert equals(f.derivative_at(a,{a:2,b:2}), np.cos(4)*2)
    assert equals(f.evaluation_at({a:1,b:2}),   np.sin(2))


def test_cos():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a+b
    f1 = fwd.cos(a+c)
    f2 = fwd.cos(a*b)
    assert equals(f1.evaluation_at({a:1.0, b: 2.0}), np.cos(4))
    assert equals(f2.evaluation_at({a:1.0,b:2}), np.cos(2))
    assert equals(f1.derivative_at(a,{a:1.0, b: 2.0}), -np.sin(1+3)*2)
    assert equals(f2.derivative_at(a,{a:2,b:2}), -np.sin(2*2)*2)

def test_tan():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.tan(c*b)
    assert equals(f.evaluation_at({a:1,b:2}),   np.tan(4))
    assert equals(f.derivative_at(c,{a:1,b:2}), 2*(1/np.cos(4))**2)
    
def test_cotan():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.cotan(c*b)
    assert equals(f.evaluation_at({a:1,b:2}),   1/np.tan(4))
    assert equals(f.derivative_at(c,{a:1,b:2}), -(1/(np.sin(4)**2))*2)

def test_sec():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.sec(c*b)
    assert equals(f.evaluation_at({a:1,b:2}), 1/np.cos(4))
    assert equals(f.derivative_at(c,{a:1,b:2}), np.tan(4)*(1/np.cos(4))*2)
    
def test_csc():
    # -csc x cot x 
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.csc(c*b)
    assert equals(f.evaluation_at({a:1,b:2}),   1/np.sin(4))
    assert equals(f.derivative_at(c,{a:1,b:2}), -(1/np.tan(4))*(1/np.sin(4))*2)

def test_sinh():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.sinh(c*b)
    assert equals(f.evaluation_at({a:1,b:2}),   np.sinh(4))
    assert equals(f.derivative_at(c,{a:1,b:2}), np.cosh(4)*2)


def test_cosh():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.cosh(c*b)
    assert equals(f.evaluation_at({a:3,b:2}),   np.cosh(12))
    assert equals(f.derivative_at(c,{a:3,b:2}), np.sinh(12)*2)
    
def test_tanh():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.tanh(c)
    assert equals(f.evaluation_at({a:3,b:2}),   np.tanh(6))
    assert equals(f.derivative_at(c,{a:3,b:2}), 1-np.tanh(6)**2)

def test_csch():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.csch(c*b)
    assert equals(f.evaluation_at({a:3,b:2}),   1/np.sinh(12))
    assert equals(f.derivative_at(c,{a:3,b:2}), \
                  -2*np.cosh(12)/np.sinh(12)/np.sinh(12))

def test_sech():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.sech(c*b)
    # - tanh x sech x 
    assert equals(f.evaluation_at({a:2,b:1}),   1/np.cosh(2))
    assert equals(f.derivative_at(c,{a:2,b:1}), \
                  -(np.sinh(2)/np.cosh(2))*(1/np.cosh(2))*1)

def test_coth():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.coth(c*b)
    assert equals(f.evaluation_at({a:3,b:2}),   np.cosh(12)/np.sinh(12))
    assert equals(f.derivative_at(c,{a:3,b:2}), (-(1/np.sinh(12))**2*2))

def test_arcsin():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.arcsin(c*b)
    assert equals(f.evaluation_at({a:0.2,b:0.5}),   np.arcsin(0.05))
    assert equals(f.derivative_at(c,{a:0.2,b:0.5}), \
                  (1/np.sqrt(1-(0.2*0.5*0.5)**2))*0.5)

def test_arccos():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.arccos(c*b)
    assert equals(f.evaluation_at({a:0.2,b:0.5}), np.arccos(0.05))
    assert equals(f.derivative_at(c,{a:0.2,b:0.5}), \
                  (-1/np.sqrt(1-(0.2*0.5*0.5)**2))*0.5)

def test_arctan():
    a = fwd.Variable()
    b = fwd.Variable()
    c = a*b
    f = fwd.arctan(c*b)
    assert equals(f.evaluation_at({a:2,b:3}),   np.arctan(18))
    assert equals(f.derivative_at(c,{a:2,b:3}), (1/(18**2+1))*3)

def test_vectorfunction():
    x, y = fwd.Variable(), fwd.Variable()
    f = fwd.sin(x) + fwd.cos(y)
    g = x**2 - y**2
    vector = fwd.VectorFunction([f, g])
    # test evaluation_at
    evaluation_returned = vector.evaluation_at({x: np.pi/6, y: np.pi/6})
    evaluation_expected = np.array([np.sin(np.pi/6) + np.cos(np.pi/6),
                                    (np.pi/6)**2    - (np.pi/6)**2])
    for r, e in zip(evaluation_returned, evaluation_expected):
        assert equals(r, e)
    # test gradient_at
    gradient_returned = vector.gradient_at(x, {x: np.pi/6, y: np.pi/6})
    gradient_expected = np.array([np.cos(np.pi/6), np.pi/3])
    for r, e in zip(gradient_returned, gradient_expected):
        assert equals(r, e)
    #test jacobian_at
    jacobian_returned = vector.jacobian_at({x: np.pi/6, y: np.pi/6})
    jacobian_expected = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                                  [np.pi/3,         -np.pi/3]])
    for i in range(2): 
        for j in range(2):
            assert equals(jacobian_returned[i, j], jacobian_expected[i, j])

def test_sin_2ndord():
    # one variable
    x = fwd.Variable()
    f = fwd.sin(x)
    assert equals(f.derivative_at(x, {x: 1.0}, order=2), -np.sin(1.0))
    # two variables
    x, y = fwd.Variable(), fwd.Variable()
    g = fwd.sin(x*y)
    assert equals(g.derivative_at(x, {x:1.0, y: 2.0}, order=2), 
                  -2.0**2 * np.sin(2.0))
    # test error raising
    with pytest.raises(NotImplementedError):
        g.derivative_at(x, {x:1.0, y: 2.0}, order=3)

def test_cos_2ndord():
    # one variable
    x = fwd.Variable()
    f = fwd.cos(x)
    assert equals(f.derivative_at(x, {x: 1.0}, order=2), -np.cos(1.0))
    # two variables
    x, y = fwd.Variable(), fwd.Variable()
    g = fwd.cos(x*y)
    assert equals(g.derivative_at(x, {x:1.0, y: 2.0}, order=2), 
                  -2.0**2 * np.cos(2.0))
    # test error raising
    with pytest.raises(NotImplementedError):
        g.derivative_at(x, {x:1.0, y: 2.0}, order=3)

def test_pow_2ndord():
    # one variable
    x = fwd.Variable()
    f = (x+1)**3
    assert equals(f.derivative_at(x, {x: 2.0}, order=2), 18.0)
    # two variables
    x, y = fwd.Variable(), fwd.Variable()
    g = (x+y)**3
    assert equals(g.derivative_at(x, {x: 2.0, y: 1.0}, order=2), 18.0)
    # test error raising
    with pytest.raises(NotImplementedError):
        g.derivative_at(x, {x:1.0, y: 2.0}, order=3)
    
def test_exp_2ndord():
    # one variable
    x = fwd.Variable()
    f = fwd.exp(2.0*x + 3.0)
    assert equals(f.derivative_at(x, {x: 1.5}, order=2), 4.0*np.exp(2.0*1.5+3.0))
    # two variables
    x, y = fwd.Variable(), fwd.Variable()
    g = fwd.exp(2.0*x / y)
    assert equals(g.derivative_at(x, {x: 1.5, y: 2.5}, order=2), 
                  4.0*np.exp(2.0*1.5/2.5) / (2.5**2) )
    # test error raising
    with pytest.raises(NotImplementedError):
        g.derivative_at(x, {x:1.0, y: 2.0}, order=3)

def test_tan_2ndord():
    # one variable
    x = fwd.Variable()
    f = fwd.tan(2.0*x - 3.0)
    assert equals( f.derivative_at(x, {x: 1.5}, order=2), 
                   8.0*np.tan(2.0*1.5-3.0)/(np.cos(2.0*1.5-3.0))**2 )
    # two variables
    x, y = fwd.Variable(), fwd.Variable()
    g = fwd.tan(2.0*x / y)
    assert equals(g.derivative_at(x, {x: 1.5, y: 2.5}, order=2), 
                  8.0*np.tan(2.0*1.5/2.5) / (np.cos(2.0*1.5/2.5)**2 * (2.5**2)) )
    # test error raising
    with pytest.raises(NotImplementedError):
        g.derivative_at(x, {x:1.0, y: 2.0}, order=3)

def test_notimplemented():
    x = fwd.Variable()
    y = fwd.Variable()
    
    with pytest.raises(NotImplementedError):
        f = x * y
        f.derivative_at(x, {x:0.5, y:0.5}, order=3)
    with pytest.raises(NotImplementedError):
        f = x / y
        f.derivative_at(x, {x:0.5, y:0.5}, order=3)
    
    with pytest.raises(NotImplementedError):
        f = fwd.cotan(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.sec(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.csc(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.sinh(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.cosh(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.tanh(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.csch(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.sech(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.coth(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.arcsin(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.arccos(x)
        f.derivative_at(x, {x:0.5}, order=2)
    with pytest.raises(NotImplementedError):
        f = fwd.arctan(x)
        f.derivative_at(x, {x:0.5}, order=2)

def test_pow_2ndord_2vars():
    x, y = fwd.Variable(), fwd.Variable()
    f = x**3 + y**3
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 0.0)
    assert equals(f.derivative_at((x, x), {x: 1.5, y:2.5}, order=2), 9.0)
    assert equals(f.derivative_at((y, y), {x: 1.5, y:2.5}, order=2), 15.0)
    f = (x-y)**3
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at((y, x), {x: 1.5, y:2.5}, order=2))
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  -6.0*(1.5-2.5))

def test_mul_2ndord_2vars():
    x, y = fwd.Variable(), fwd.Variable()
    f = x**2 * y**2
    assert equals(f.derivative_at((x, x), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at( x,     {x: 1.5, y:2.5}, order=2))    
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at((y, x), {x: 1.5, y:2.5}, order=2))
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  4.0*1.5*2.5)

def test_div_2ndord_2vars():
    x, y = fwd.Variable(), fwd.Variable()
    f = x**2 / y**2
    assert equals(f.derivative_at((x, x), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at( x,     {x: 1.5, y:2.5}, order=2))    
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at((y, x), {x: 1.5, y:2.5}, order=2))
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  -4.0*1.5/2.5**3)

def test_sin_2ndord_2vars():
    x, y = fwd.Variable(), fwd.Variable()
    f = fwd.sin(x/y)
    df_dxdy = lambda x, y: -(y*np.cos(x/y) - x*np.sin(x/y))/y**3
    assert equals(f.derivative_at((x, x), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at( x,     {x: 1.5, y:2.5}, order=2))    
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at((y, x), {x: 1.5, y:2.5}, order=2))
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  df_dxdy(1.5, 2.5))
    
def test_cos_2ndord_2vars():
    x, y = fwd.Variable(), fwd.Variable()
    f = fwd.cos(x/y)
    df_dxdy = lambda x, y: (y*np.sin(x/y) + x*np.cos(x/y))/y**3
    assert equals(f.derivative_at((x, x), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at( x,     {x: 1.5, y:2.5}, order=2))    
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at((y, x), {x: 1.5, y:2.5}, order=2))
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  df_dxdy(1.5, 2.5))

def test_tan_2ndord_2vars():
    x, y = fwd.Variable(), fwd.Variable()
    f = fwd.tan(x/y)
    df_dxdy = lambda x, y: -(y/np.cos(x/y)**2 + 2*x*np.tan(x/y)/np.cos(x/y)**2) / y**3
    assert equals(f.derivative_at((x, x), {x: 1.5, y:2.5}, order=2),
                  f.derivative_at( x,     {x: 1.5, y:2.5}, order=2))    
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at((y, x), {x: 1.5, y:2.5}, order=2))
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  df_dxdy(1.5, 2.5))

def test_exp_2ndord_2vars():
    x, y = fwd.Variable(), fwd.Variable()
    f = fwd.exp(x/y)
    df_dxdy = lambda x, y: -(x*np.exp(x/y) + y*np.exp(x/y)) / y**3
    assert equals(f.derivative_at((x, x), {x: 1.5, y:2.5}, order=2),
                  f.derivative_at( x,     {x: 1.5, y:2.5}, order=2))    
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  f.derivative_at((y, x), {x: 1.5, y:2.5}, order=2))
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}, order=2), 
                  df_dxdy(1.5, 2.5))

def test_hessian():
    x, y = fwd.Variable(), fwd.Variable()
    rosen = 100.0*(y - x**2)**2 + (1 - x)**2.0
    rosen_hessian = lambda x, y: \
        np.array([[1200*x**2-400*x+2, -400*x],
                  [-400*x,             200]])
    rosen_hessian_returned = rosen.hessian_at({x: 1.0, y: 1.0})
    rosen_hessian_expected = rosen_hessian(1.0, 1.0)
    for i in range(2):
        for j in range(2):
            assert equals(rosen_hessian_returned[i, j],
                          rosen_hessian_expected[i, j])

def test_gradient():
    x, y = fwd.Variable(), fwd.Variable()
    f = fwd.sin(x) + fwd.cos(y)
    f_gradient_at = lambda x, y: np.array([np.cos(x), -np.sin(y)])
    gradient_expected = f_gradient_at(1.5, 2.5)
    gradient_returned = f.gradient_at({x: 1.5, y: 2.5})
    for i in range(2):
        assert equals(gradient_expected[i], gradient_returned[i])
    gradient_returned = f.gradient_at({x: 1.5, y: 2.5}, returns_dict=True)
    assert equals(gradient_returned[x], gradient_expected[0])
    assert equals(gradient_returned[y], gradient_expected[1])

def test_eq():
    x, y = fwd.Variable(), fwd.Variable()
    f = fwd.sin(x) + fwd.cos(y)
    g = fwd.sin(x) + fwd.cos(y)
    h = fwd.sin(y) + fwd.cos(x)
    assert f == g
    assert f != h

def test_sqrt():
    x, y = fwd.Variable(), fwd.Variable()
    f = fwd.sqrt(fwd.sin(x) + fwd.cos(y))
    dfdx = lambda x, y:  np.cos(x) / (2*np.sqrt(np.sin(x)+np.cos(y)))
    dfdy = lambda x, y: -np.sin(y) / (2*np.sqrt(np.sin(x)+np.cos(y)))
    d2fdxdy = lambda x, y: np.cos(x)*np.sin(y) / (4*(np.sin(x) + np.cos(y))**1.5)
    assert equals(f.evaluation_at({x: 1.5, y:2.5}), np.sqrt(np.sin(1.5)+np.cos(2.5)))
    assert equals(f.derivative_at(x, {x: 1.5, y:2.5}), dfdx(1.5, 2.5))
    assert equals(f.derivative_at(y, {x: 1.5, y:2.5}), dfdy(1.5, 2.5))
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}), d2fdxdy(1.5, 2.5))

def test_log():
    x, y = fwd.Variable(), fwd.Variable()
    f = fwd.log(fwd.sin(x)+y**2)
    dfdx = lambda x, y: np.cos(x) / (np.sin(x)+y**2)
    dfdy = lambda x, y: 2*y / (np.sin(x)+y**2)
    d2fdxdy = lambda x, y: -2*y*np.cos(x) / (np.sin(x)+y**2)**2
    assert equals(f.evaluation_at({x: 1.5, y:2.5}), np.log(np.sin(1.5)+2.5**2))
    assert equals(f.derivative_at(x, {x: 1.5, y:2.5}), dfdx(1.5, 2.5))
    assert equals(f.derivative_at(y, {x: 1.5, y:2.5}), dfdy(1.5, 2.5))
    assert equals(f.derivative_at((x, y), {x: 1.5, y:2.5}), d2fdxdy(1.5, 2.5))
    
def test_logit():
    x = fwd.Variable()
    f = fwd.logit(x)
    f_exact = lambda x: np.log(x/(1-x))
    dfdx = lambda x: 1/(x*(1-x))
    assert equals(f.evaluation_at({x: 0.8}), f_exact(0.8))
    assert equals(f.derivative_at(x, {x: 0.8}), dfdx(0.8))

def test_sigmoid():
    x = fwd.Variable()
    f = fwd.sigmoid(x)
    f_exact = lambda x: 1/(1+np.exp(-x))
    dfdx = lambda x: f_exact(x)*(1-f_exact(x))
    assert equals(f.evaluation_at({x: 0.8}), f_exact(0.8))
    assert equals(f.derivative_at(x, {x: 0.8}), dfdx(0.8))