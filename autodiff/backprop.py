import autodiff.forward as fwd

def forward_pass(y,val_dict):
    # forward pass, store values
    if type(y) == fwd.Expression:
        y.val = y.evaluation_at(val_dict)
        if y._sub_expr1 != None:
            forward_pass(y._sub_expr1,val_dict)
        if y._sub_expr2!=None:
            forward_pass(y._sub_expr2,val_dict)
    elif isinstance(y,fwd.Variable):
        y.val = val_dict[y]
    return 

def initialize(top,y):
    #print(y.val)
    if y == top:
        y.bder = 1     
    return

def backward(y,val_dict,depth = 0):
    # val_dict stores the basic variables
    # (atomic,cannot be further decomposed)
    if type(y) == fwd.Expression:
        if y._sub_expr1 != None:
            y._sub_expr1.bder += y.bder*y.back_derivative(y._sub_expr1,val_dict)
            backward(y._sub_expr1,val_dict,depth+1)
        if y._sub_expr2 != None:
            y._sub_expr2.bder += y.bder*y.back_derivative(y._sub_expr2,val_dict)
            backward(y._sub_expr2,val_dict,depth+1)
    return 

def back_propagation(y,val_dict):
    # get backprop derivative with respect to y at every node lower than y
    forward_pass(y,val_dict)
    initialize(y,y)
    backward(y,val_dict)
