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


def newton(f,  init_val_dict, max_iter=10, stop_stepsize=1e-8):
    
    variables  = [var for var in init_val_dict.keys()]
    curr_point = np.array([v for k, v in init_val_dict.items()])
    f_grad = f.gradient_at(init_val_dict)
    f_hess = f.hessian_at(init_val_dict)
    
    for i in range(max_iter):
        
        curr_val_dict = {var: val for var, val in zip(variables, curr_point)}
        # solve (Hessian of f at x)s = - (gradient of f at x)
        f_grad =f.gradient_at(curr_val_dict)
        f_hess = f.hessian_at(curr_val_dict)

        step = np.linalg.solve(f_hess, -f_grad)
        if np.linalg.norm(step, ord=2) < stop_stepsize: break
        
        # x := x + s
        curr_point = curr_point + step
    
    return {var: val for var, val in zip(variables, curr_point)}






# def gradient_descent(f,init_val_dict, lambda_dict, step=0.001, maxsteps=0, precision=0.001):
#     costs = []
#     m = y.size # number of data points

#     lambda1 = lambda1_init
#     lambda2 = lambda2_init
#     history1 = [] # to store all thetas
#     history2 = []

#     counter = 0
#     oldcost = 0

#     currentcost = np.sum(0.000045*lambda2**2*y - 0.000098*lambda1**2*x +\
#                   0.003926*lambda1*x*np.exp((y**2 - x**2)*(lambda1**2 + lambda2**2)))

#     costs.append(currentcost)

#     history1.append(lambda1)
#     history2.append(lambda2)
#     counter+=1
#     while abs(currentcost - oldcost) > precision:
#         oldcost = currentcost
#         #gradient = x.T.dot(error)/m

#         start = timer()
#         gradient1 = np.sum(-2 * 0.000098 * lambda1 * x +\
#                     0.003926 * x * (np.exp((y**2 - x**2) * (lambda1**2 + lambda2**2)) +\
#                     2 * lambda1**2 * np.exp((y**2 - x**2)*(lambda1**2 + lambda2**2)) * (y**2 - x**2)))
#         gradient2 = np.sum(2 * 0.000045 * lambda2 * y +\
#                     0.007852 * lambda2 * np.exp((y**2 - x**2)*(lambda1**2 + lambda2**2)) * lambda1 * x * (y**2 - x**2))

#         lambda1 = lambda1 - step * gradient1
#         lambda2 = lambda2 - step * gradient2

#         end = timer()

#         avg_time.append(end - start)

#         history1.append(lambda1)
#         history2.append(lambda2)

#         currentcost = np.sum(0.000045*lambda2**2*y - 0.000098*lambda1**2*x +\
#                   0.003926*lambda1*x*np.exp((y**2 - x**2)*(lambda1**2 + lambda2**2)))
#         costs.append(currentcost)

#         #if counter % 25 == 0: preds.append(pred)
#         counter += 1
#         if maxsteps:
#             if counter == maxsteps:
#                 break

#     return history1, history2, costs, counter
