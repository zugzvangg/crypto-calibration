from heston import MarketParameters, ModelParameters, fHes, JacHes
import numpy as np
from levenberg_marquardt import Levenberg_Marquardt
from typing import Tuple

karr = np.array(
    [
        0.9371,
        0.8603,
        0.8112,
        0.7760,
        0.7470,
        0.7216,
        0.6699,
        0.6137,
        0.9956,
        0.9868,
        0.9728,
        0.9588,
        0.9464,
        0.9358,
        0.9175,
        0.9025,
        1.0427,
        1.0463,
        1.0499,
        1.0530,
        1.0562,
        1.0593,
        1.0663,
        1.0766,
        1.2287,
        1.2399,
        1.2485,
        1.2659,
        1.2646,
        1.2715,
        1.2859,
        1.3046,
        1.3939,
        1.4102,
        1.4291,
        1.4456,
        1.4603,
        1.4736,
        1.5005,
        1.5328,
    ],
    dtype=np.float64,
)

# цены берем с дерибита
carr = karr

tarr = np.array(
    [
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
    ],
    dtype=np.float64,
)
S_val = np.float64(1.0)
r_val = np.float64(0.02)

market = MarketParameters(K=karr, T=tarr, S=S_val, r=r_val, C = carr)

a = np.float64(3.0)  # kappa                           |  mean reversion rate
b = np.float64(0.10)  # v_infinity                      |  long term variance
c = np.float64(0.25)  # sigma                           |  variance of volatility
rho = np.float64(
    -0.8
)  # rho                             |  correlation between spot and volatility
v0 = np.float64(0.08)



def proj_heston( heston_params : np.ndarray )->np.ndarray:
    """
        This funciton project heston parameters into valid range
        Attributes:
            heston_params(np.ndarray): model parameters
        
        Returns:
            heston_params(np.ndarray): clipped parameters
    """
    eps = 1e-3
    for i in range(len(heston_params) // 5):
        v0, theta, rho, k, sig = heston_params[i * 5 : i * 5 + 5]
        v0 = np.clip(v0, eps, 5.0)
        theta = np.clip(theta, eps, 5.0)
        rho = np.clip(rho, -1 + eps, 1 - eps)
        k = np.clip(k, eps, 10.0)
        sig = np.clip(sig, eps, 5.0)
        heston_params[i * 5 : i * 5 + 5] = v0, theta, rho, k, sig
    return heston_params

def get_residuals( heston_params:np.ndarray ) -> Tuple[ np.ndarray, np.ndarray ]:
    '''
        This function calculates residuals and Jacobian matrix
        Args:
            heston_params(np.ndarray): model params
        Returns:
            res(np.ndarray) : vector or residuals
            J(np.ndarray)   : Jacobian
    '''
    # needed format to go
    model_parameters = ModelParameters(
            heston_params[0],
            heston_params[1],
            heston_params[2],
            heston_params[3],
            heston_params[4])
    # тут ок в целом, надо подогнать дальше и смотреть
    #  чтоб ваще те параметры подставлялись в якобиан
    C = fHes(
    model_parameters=model_parameters,
    market_parameters=market,
    )

    J = JacHes(
    model_parameters=model_parameters, 
    market_parameters=market)

    K = karr
    F = np.ones(len(K))*market.S
    weights = np.ones_like(K)
    weights = weights / np.sum(weights)
    typ = True
    P = C + np.exp(-market.r * market.T) * ( K - F )
    X_ = C
    X_[~typ] = P[~typ]
    res = X_ - market.C
    return res * weights,  J @ np.diag(weights)



start_params = ModelParameters(a=1.2, b=0.2, c=0.3, rho=-0.6, v0=0.2)
start_params = np.array([a, b, c, rho, v0])
res = Levenberg_Marquardt(100, get_residuals, proj_heston, start_params)
print(res["x"])


