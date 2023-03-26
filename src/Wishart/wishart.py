import numba as nb
import numpy as np
from typing import Final, Tuple
import pandas as pd
from src.utils import get_tick, get_implied_volatility, get_bid_ask
from src.levenberg_marquardt import LevenbergMarquardt
from typing import Union
import warnings 
warnings.filterwarnings("ignore")


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
    ("SIGMA", nb.types.Array(nb.float64, 2, "A")),
    ("R", nb.types.Array(nb.float64, 2, "A")),
    ("Q", nb.types.Array(nb.float64, 2, "A")),
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
    SIGMA: nb.types.Array(nb.float64, 2, "A")
    R: nb.types.Array(nb.float64, 2, "A")
    Q: nb.types.Array(nb.float64, 2, "A")

    def __init__(
        self,
        SIGMA: nb.types.Array(nb.float64, 2, "A"),
        R: nb.types.Array(nb.float64, 2, "A"),
        Q: nb.types.Array(nb.float64, 2, "A"),
    ):
        self.SIGMA = SIGMA
        self.R = R
        self.Q = Q


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
                        np.dot(
                            Q, np.dot(np.dot(Q_T, R_T) + np.dot(R, Q), SIGMA)
                        ),
                    )
                )
                / 3
                - 5 / 4 * trace_triple**2 / trace_sigma
            )
        )
        sigmas[index] = np.sqrt(iv_sqared)
    return sigmas
