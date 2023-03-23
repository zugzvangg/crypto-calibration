import numba as nb
import numpy as np
import pandas as pd

# from src.SABR.sabr import ModelParameters, MarketParameters, vol_sabr, jacobian_sabr
from src.SABR.sabr_approx import (
    ModelParameters,
    MarketParameters,
    vol_sabr,
    jacobian_sabr,
)


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
beta = np.float64(0.8)
rho = np.float64(0.01)

model = ModelParameters(alpha, v, beta, rho)
market = MarketParameters(K=karr, T=T, S=S_val, r=r_val, C=carr, types=types, iv=iv)

if __name__ == "__main__":
    # print(vol_sabr(model=model, market=market))
    # print(jacobian_sabr(model=model, market=market))
    print(vol_sabr(model=model, market=market))



