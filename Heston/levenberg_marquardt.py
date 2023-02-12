import numpy as np
from typing import Tuple, Callable, Dict

from tqdm import tqdm
import numba as nb
import matplotlib.pyplot as plt

nb.njit
def Levenberg_Marquardt(Niter:int, 
                          f:Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], 
                          proj:Callable[ [np.ndarray], np.ndarray ], 
                          x0:np.ndarray) -> Dict:
    ''' 
        Nonlinear least squares method, Levenberg-Marquardt Method
        
        Args:
            Niter(int): number of iteration
            f(Callable[ [np.ndarray], Tuple[np.ndarray, np.ndarray]]): 
                callable, gets vector of model parameters x as input, 
                returns tuple res, J, where res is numpy vector of residues, 
                J is jacobian of residues with respect to x 
            proj(Callable[ [np.ndarray], np.ndarray ]):
                callable, gets vector of model parameters x,
                returns vector of projected parameters 
            x0(np.ndarray): initial parameters
        Returns:
            result(dict): dictionary with results
            result['xs'] contains optimized parameters on each iteration
            result['objective'] contains norm of residuals on each iteration
            result['x'] is optimized parameters
    '''
    x = x0.copy()

    mu = 100.0
    nu1 = 2.0
    nu2 = 2.0

    # fs = []
    res, J = f(x)
    # сумма квадратов остатков
    F = np.linalg.norm(res)
    
    result = { "xs":[x], "objective":[F], "x":None }
    
    for i in range(Niter):
        # print(J, J.T)
        multipl = J @ J.T
        I = np.diag(np.diag(multipl)) + 1e-5 * np.eye(len(x))
        dx = np.linalg.solve( mu * I + multipl, J @ res )
        x_ = proj(x - dx)
        res_, J_ = f(x_)
        F_ = np.linalg.norm(res_)
        if F_ < F:
            x, F, res, J = x_, F_, res_, J_
            mu /= nu1
            result['xs'].append(x)
            result['objective'].append(F)
        else:
            i -= 1
            mu *= nu2
            continue        
        eps = 1e-10
        if F < eps:
            break
        result['x'] = x
    return result


def get_value(x, y):
    # тут обязательно массивы
    return 3*iv0, np.exp(iv0), K + x**2


def residals(params: np.array):
    x, y = params
    dx, dy , iv = get_value(x, y)
    res = iv - iv0
    J = np.asarray([dx, dy])
    weights = np.ones_like(iv0)
    weights = weights / np.sum(weights)
    return res * weights, J @ np.diag(weights)



def proj(params: np.array):
    x, y = params
    x = np.clip(x, -10, 10)
    y = np.clip(y, -10, 10)
    return np.asarray([x, y])

K = np.array([1, 1, 1, 1, 1])
iv0 = np.array([1, 1, 2, 1, 3])
x0 = np.array([10, 5])
Niter = 1000
result = Levenberg_Marquardt(Niter = Niter, f = residals, proj = proj, x0 = x0)

# print("x", result["x"])
# print("=====")
# print("objective", result["objective"])
# print("======")
# # print("xs", result["xs"])
# # print("======")
# # plt.plot(np.ones(len(result["objective"])), result["objective"])
