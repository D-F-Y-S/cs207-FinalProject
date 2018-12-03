import numpy as np
from numpy.linalg import multi_dot
from numpy.linalg import norm
from scipy.linalg import solve

def bfgs(f, init_val_dict, max_iter=2000, stop_stepsize=1e-8):
    
    variables  = [var for var in init_val_dict.keys()]
    curr_point = np.array([v for k, v in init_val_dict.items()])
    B          = np.eye(len(curr_point))
    
    for i in range(max_iter):
        
        # solve Bs = - (gradient of f at x)
        curr_val_dict = {var: val for var, val in zip(variables, curr_point)}
        f_grad = f.gradient_at(curr_val_dict)
        s = solve(B, -f_grad)
        if norm(s, ord=2) < stop_stepsize: break
            
        # x_next := x + s
        next_point = curr_point + s

        # y := (gradient of f at x_next) - (gradient of f at x)
        # x := x_next
        next_val_dict = {var: val for var, val in zip(variables, next_point)}
        y = f.gradient_at(next_val_dict) - f.gradient_at(curr_val_dict)
        curr_point = next_point
        
        # B := B + deltaB
        s, y = s.reshape(-1, 1), y.reshape(-1, 1)
        deltaB = multi_dot([y, y.T])/multi_dot([y.T, s]) \
                 - multi_dot([B, s, s.T, B])/multi_dot([s.T, B, s]) 
        B = B + deltaB
    
    return {var: val for var, val in zip(variables, curr_point)}