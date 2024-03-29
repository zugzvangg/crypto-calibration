import numba as nb
import numpy as np
from typing import Final, Tuple
import pandas as pd
from src.utils import get_tick, get_implied_volatility, get_bid_ask
from src.levenberg_marquardt import LevenbergMarquardt
from typing import Union
from scipy.integrate import quad
import warnings

warnings.filterwarnings("ignore")
from ..Heston.heston import ModelParameters as HestonModelParameters, JacHes, MarketParameters as HestonMarketParameters


_spec_market_params = [
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64),
    ("K", nb.float64[:]),
    ("C", nb.float64[:]),
    ("iv", nb.float64[:]),
    ("types", nb.boolean[:]),
]

_spec_model_params = [
    ("beta", nb.float64),
    ("SIGMA", nb.types.Array(nb.float64, 2, "A")),
    ("M", nb.types.Array(nb.float64, 2, "A")),
    ("Q", nb.types.Array(nb.float64, 2, "A")),
    ("alpha", nb.float64),
]


@nb.experimental.jitclass(_spec_market_params)
class MarketParameters(object):
    S: nb.float64
    r: nb.float64
    T: nb.float64
    K: nb.float64[:]
    C: nb.float64[:]
    iv: nb.float64[:]
    types: nb.boolean[:]

    def __init__(
        self,
        S: nb.float64,
        r: nb.float64,
        T: nb.float64,
        K: nb.float64[:],
        C: nb.float64[:],
        iv: nb.float64[:],
        types: nb.boolean[:],
    ):
        self.S = S
        self.r = r
        self.T = T
        self.K = K
        self.C = C
        self.iv = iv
        self.types = types


@nb.experimental.jitclass(_spec_model_params)
class WirsherModelParameters(object):
    beta: nb.float64
    SIGMA: nb.types.Array(nb.float64, 2, "A")
    M: nb.types.Array(nb.float64, 2, "A")
    Q: nb.types.Array(nb.float64, 2, "A")
    alpha: nb.float64

    def __init__(
        self,
        beta: nb.float64,
        SIGMA: nb.types.Array(nb.float64, 2, "A"),
        M: nb.types.Array(nb.float64, 2, "A"),
        Q: nb.types.Array(nb.float64, 2, "A"),
        alpha: nb.float64,
    ):
        self.beta = beta
        self.SIGMA = SIGMA
        self.Q = Q
        self.M = M
        self.alpha = alpha
        # self.R = alpha*np.eye(2)


_tmp_values_get_iv_wishart = {}

_tmp_values_get_iv_wishart = {
    "f": nb.float64,
    "Ks": nb.types.Array(nb.float64, 1, "A"),
    "sigmas": nb.types.Array(nb.float64, 1, "A"),
    "n": nb.int64,
    "SIGMA": nb.types.Array(nb.float64, 2, "A"),
    "R": nb.types.Array(nb.float64, 2, "A"),
    "Q": nb.types.Array(nb.float64, 2, "A"),
    "Q_T": nb.types.Array(nb.float64, 2, "A"),
    "R_T": nb.types.Array(nb.float64, 2, "A"),
    "trace_sigma": nb.float64,
    "trace_triple": nb.float64,
    "triple": nb.types.Array(nb.float64, 2, "A"),
    "iv_sqared": nb.float64,
    "K": nb.float64,
    "mf": nb.float64,
}


@nb.njit(locals=_tmp_values_get_iv_wishart)
def get_iv_wishart(market: MarketParameters, model: WirsherModelParameters):
    Ks = market.K
    f = market.S
    n = len(Ks)
    sigmas = np.zeros(n, dtype=np.float64)
    SIGMA, R, Q = model.SIGMA, model.R, model.Q
    trace_sigma = np.trace(SIGMA)
    triple = np.dot(R, np.dot(Q, SIGMA))
    trace_triple = np.trace(triple)
    Q_T = Q.T
    R_T = R.T
    for index in range(n):
        K = Ks[index]
        mf = np.log(f / K)
        # iv_sqared = (
        #     trace_sigma
        #     + trace_triple * mf / trace_sigma
        #     + mf**2
        #     / trace_sigma**2
        #     * (
        #         np.trace(Q.T @ Q @ SIGMA) / 3
        #         + np.trace(R @ Q @ (Q.T @ R.T + R @ Q) @ SIGMA) / 3
        #         - 5 / 4 * trace_triple**2 / trace_sigma
        #     )
        # )
        iv_sqared = (
            trace_sigma
            + trace_triple * mf / trace_sigma
            + mf**2
            / trace_sigma**2
            * (
                np.trace(np.dot(Q_T, np.dot(Q, SIGMA))) / 3
                + np.trace(
                    np.dot(
                        R,
                        np.dot(Q, np.dot(np.dot(Q_T, R_T) + np.dot(R, Q), SIGMA)),
                    )
                )
                / 3
                - 5 / 4 * trace_triple**2 / trace_sigma
            )
        )
        sigmas[index] = np.sqrt(iv_sqared)
    return sigmas


def Gamma(model: WirsherModelParameters, market: MarketParameters):
    # page 5 from orig article
    tau = market.T
    SIGMA = model.SIGMA
    M = model.M
    L1 = np.exp(tau * M)
    L2 = np.exp(tau * M.T)
    return np.dot(L1, np.dot(SIGMA, L2))


def Theta(model: WirsherModelParameters, market: MarketParameters):
    Q = model.Q
    M = model.M
    Q11, Q12, Q21, Q22 = Q[0][0], Q[0][1], Q[1][0], Q[1][1]
    M11, M12, M21, M22 = M[0][0], M[0][1], M[1][0], M[1][1]
    tau = market.T

    # can't avoid scipy here
    def M11_calc(u):
        return (
            Q11 * (Q11 * np.exp(M11 * u) + Q12 * np.exp(M12 * u))
            + Q21 * (Q21 * np.exp(M11 * u) + Q22 * np.exp(M12 * u))
        ) * np.exp(M11 * u) + (
            Q12 * (Q11 * np.exp(M11 * u) + Q12 * np.exp(M12 * u))
            + Q22 * (Q21 * np.exp(M11 * u) + Q22 * np.exp(M12 * u))
        ) * np.exp(
            M12 * u
        )

    def M12_calc(u):
        return (
            Q11 * (Q11 * np.exp(M11 * u) + Q12 * np.exp(M12 * u))
            + Q21 * (Q21 * np.exp(M11 * u) + Q22 * np.exp(M12 * u))
        ) * np.exp(M21 * u) + (
            Q12 * (Q11 * np.exp(M11 * u) + Q12 * np.exp(M12 * u))
            + Q22 * (Q21 * np.exp(M11 * u) + Q22 * np.exp(M12 * u))
        ) * np.exp(
            M22 * u
        )

    def M21_calc(u):
        return (
            Q11 * (Q11 * np.exp(M21 * u) + Q12 * np.exp(M22 * u))
            + Q21 * (Q21 * np.exp(M21 * u) + Q22 * np.exp(M22 * u))
        ) * np.exp(M11 * u) + (
            Q12 * (Q11 * np.exp(M21 * u) + Q12 * np.exp(M22 * u))
            + Q22 * (Q21 * np.exp(M21 * u) + Q22 * np.exp(M22 * u))
        ) * np.exp(
            M12 * u
        )

    def M22_calc(u):
        return (
            Q11 * (Q11 * np.exp(M21 * u) + Q12 * np.exp(M22 * u))
            + Q21 * (Q21 * np.exp(M21 * u) + Q22 * np.exp(M22 * u))
        ) * np.exp(M21 * u) + (
            Q12 * (Q11 * np.exp(M21 * u) + Q12 * np.exp(M22 * u))
            + Q22 * (Q21 * np.exp(M21 * u) + Q22 * np.exp(M22 * u))
        ) * np.exp(
            M22 * u
        )

    theta = np.array(
        [
            [quad(M11_calc, 0.0, tau)[0], quad(M12_calc, 0.0, tau)[0]],
            [quad(M21_calc, 0.0, tau)[0], quad(M22_calc, 0.0, tau)[0]],
        ],
        dtype=np.float64,
    )
    return theta


def E_nu_sigma(gamma: float, theta: float, model: WirsherModelParameters):
    return np.trace(gamma + model.beta * theta)


def VAR_nu_sigma(gamma: float, theta: float, model: WirsherModelParameters):
    return 2 * np.trace(np.dot((2 * gamma + model.beta * theta), theta))


def jacobian_wishart(market: MarketParameters, model: WirsherModelParameters):
    # heston_params = HestonModelParameters()
    # JacHes
    gamma = Gamma(model=model, market=market)
    theta = Theta(model=model, market=market)
    matoj = E_nu_sigma(gamma, theta, model)
    var = VAR_nu_sigma(gamma, theta, model)
    # find heston params
    beta, SIGMA, Q, M, alpha = model.beta, model.SIGMA, model.Q, model.M, model.alpha
    R = alpha * np.eye(2)
    eps = 0.1
    V0 = np.clip(np.trace(SIGMA), eps, 10.0)
    theta = np.clip(-beta / 2 * np.trace(np.dot(np.linalg.inv(M), np.dot(Q.T, Q))), eps, 10.0)
    # NOTE: V0 - theta а точно тут V0? Там должно быть NU(t)
    kappa = np.clip(-1 / market.T * np.log((matoj - theta) / (V0 - theta)), eps, 10.0)
    eta = np.clip(np.sqrt(var / (alpha * kappa * ((1 - alpha) * V0 + alpha / 2 * theta))), eps, 100.0)
    rho_num = np.trace(np.dot(R, np.dot(Q, SIGMA)))
    rho_div = np.sqrt(np.trace(SIGMA)) * np.sqrt(
        np.trace(np.dot(Q.T, np.dot(Q, SIGMA)))
    )
    rho = np.clip(rho_num / rho_div, -1.0 + eps, 1.0 - eps)
    heston_params = HestonModelParameters(v0=V0, a=kappa, b=theta, c=eta, rho=rho)
    print(V0, kappa, theta, eta, rho)
    heston_market = HestonMarketParameters(S = market.S,
                                           r = market.r,
                                           T = market.T*np.ones(len(market.K)), 
                                           C = market.C, 
                                           types = market.types,
                                           K = market.K
                                           )
    heston_jacobians = JacHes(model_parameters=heston_params, market_parameters=heston_market)
    print(heston_jacobians)
    # NOTE: look for mistake in theta
