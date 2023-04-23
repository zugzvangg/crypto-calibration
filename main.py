import numba as nb
import numpy as np
import pandas as pd
# from src.Wishart.WASC import get_iv_wishart, ModelParameters, MarketParameters, jacobian_wishart
from src.SABR.sabr_approx import vol_sabr, MarketParameters, ModelParameters

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
iv = karr
types = np.ones(len(karr), dtype=bool)
T = np.float64(0.5)
carr = karr
S_val = np.float64(1.0)
r_val = np.float64(0.02)


alpha = np.float64(1.0)
v = np.float64(0.5)
beta = np.float64(1.0)
rho = np.float64(0.01)

# E = np.array([[0.4, 0.0], [0, 0.35]], dtype = np.float64)
# Q = np.array([[0.3, 0.0], [0, 0.2]], dtype = np.float64)
# # M = np.array([[0.2, 0.1], [0.18, 0.2]], dtype = np.float64)
# R = np.array([[0.1, 0.3], [0.18, 0.5]], dtype = np.float64)

# E11, E12, E21, E22, Q11, Q12, Q21, Q22, R11, R12, R21, R22 = 0.4, 0.0, 0.0, 0.35, 0.3, 0.0, 0.0, 0.2, 0.1, 0.3, 0.12, 0.5

# model = ModelParameters(E11, E12, E21, E22, Q11, Q12, Q21, Q22, R11, R12, R21, R22)
market = MarketParameters(K=karr, T=T, S=S_val, r=r_val, C=carr, types=types, iv = iv)
model = ModelParameters(alpha, v, beta, rho)

if __name__ == "__main__":
    print(vol_sabr(model=model, market=market))
    # print(jacobian_sabr(model=model, market=market))
    # print(get_iv_wishart(model = model, market = market))
    # print(Gamma(model = model, market = market))
    # print(Theta(model = model, market = market))
    # print(jacobian_wishart(model = model, market = market))