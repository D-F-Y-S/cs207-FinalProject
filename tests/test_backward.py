from test_forward import equals
import autodiff.forward

def test_backward():
    a= Variable()
    b = Variable()
    c = a+b
    d = Variable()
    e = c*d
    f = a+e
    val_dict = {b:1,a:2,d:4}
    back_propagation(f,val_dict)
    var_list = [a,b,c,d,e,f]
    for i in var_list:
        assert(equals(i.bder == f.derivative_at(i,val_dict)))
    
    a = Variable()
    b = Variable()
    e = b-a
    c = cos(e)
    d = a+c
    val_dict = {b:1,a:2}
    back_propagation(d,val_dict)
    var_list = [a,b,c,d]
    for i in var_list:
        assert(equals(i.bder == d.derivative_at(i,val_dict)))
        
    
    a = Variable()
    b = Variable()
    c = csc(a)
    d = sec(a)
    e = tan(c)
    f = cotan(d)
    g = sinh(f-e)
    val_dict = {b:1,a:2}
    back_propagation(g,val_dict)
    var_list = [a,b,c,d,e,f,g]
    for i in var_list:
        assert(equals(i.bder == g.derivative_at(i,val_dict)))
   
