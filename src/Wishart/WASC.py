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
from ..Heston.heston import (
    ModelParameters as HestonModelParameters,
    JacHes,
    MarketParameters as HestonMarketParameters,
)


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
    ("E", nb.types.Array(nb.float64, 2, "A")),
    ("Q", nb.types.Array(nb.float64, 2, "A")),
    ("R", nb.types.Array(nb.float64, 2, "A")),
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
class ModelParameters(object):
    beta: nb.float64
    E: nb.types.Array(nb.float64, 2, "A")
    Q: nb.types.Array(nb.float64, 2, "A")
    R: nb.types.Array(nb.float64, 2, "A")

    def __init__(
        self,
        E: nb.types.Array(nb.float64, 2, "A"),
        Q: nb.types.Array(nb.float64, 2, "A"),
        R: nb.types.Array(nb.float64, 2, "A"),
    ):
        self.E = E
        self.Q = Q
        self.R = R


_tmp_values_get_iv_wishart = {
    "f": nb.float64,
    "Ks": nb.types.Array(nb.float64, 1, "A"),
    "Es": nb.types.Array(nb.float64, 1, "A"),
    "n": nb.int64,
    "E": nb.types.Array(nb.float64, 2, "A"),
    "Q": nb.types.Array(nb.float64, 2, "A"),
    "R": nb.types.Array(nb.float64, 2, "A"),
    "Q_T": nb.types.Array(nb.float64, 2, "A"),
    "R_T": nb.types.Array(nb.float64, 2, "A"),
    "trace_E": nb.float64,
    "trace_triple": nb.float64,
    "triple": nb.types.Array(nb.float64, 2, "A"),
    "iv_sqared": nb.float64,
    "K": nb.float64,
    "mf": nb.float64,
}


@nb.njit(locals=_tmp_values_get_iv_wishart)
def get_iv_wishart(market: MarketParameters, model: ModelParameters):
    Ks = market.K
    f = market.S
    n = len(Ks)
    Es = np.zeros(n, dtype=np.float64)
    E, R, Q = model.E, model.R, model.Q
    trace_E = np.trace(E)
    triple = np.dot(R, np.dot(Q, E))
    trace_triple = np.trace(triple)
    Q_T = Q.T
    R_T = R.T
    for index in range(n):
        K = Ks[index]
        mf = np.log(f / K)
        # iv_sqared = (
        #     trace_E
        #     + trace_triple * mf / trace_E
        #     + mf**2
        #     / trace_E**2
        #     * (
        #         np.trace(Q.T @ Q @ E) / 3
        #         + np.trace(R @ Q @ (Q.T @ R.T + R @ Q) @ E) / 3
        #         - 5 / 4 * trace_triple**2 / trace_E
        #     )
        # )
        iv_sqared = (
            trace_E
            + trace_triple * mf / trace_E
            + mf**2
            / trace_E**2
            * (
                np.trace(np.dot(Q_T, np.dot(Q, E))) / 3
                + np.trace(
                    np.dot(
                        R,
                        np.dot(Q, np.dot(np.dot(Q_T, R_T) + np.dot(R, Q), E)),
                    )
                )
                / 3
                - 5 / 4 * trace_triple**2 / trace_E
            )
        )
        Es[index] = np.sqrt(iv_sqared)
    return Es


_tmp_values_jacobian_wishart = {
    "Q": nb.types.Array(nb.float64, 2, "A"),
    "E": nb.types.Array(nb.float64, 2, "A"),
    "R": nb.types.Array(nb.float64, 2, "A"),
    "mf": nb.float64[:],
    "Q11": nb.float64,
    "Q12": nb.float64,
    "Q21": nb.float64,
    "Q22": nb.float64,
    "E11": nb.float64,
    "E12": nb.float64,
    "E21": nb.float64,
    "E22": nb.float64,
    "R11": nb.float64,
    "R12": nb.float64,
    "R21": nb.float64,
    "R22": nb.float64,
    "iv": nb.float64[:],
    "dQ11": nb.float64[:],
    "dQ12": nb.float64[:],
    "dQ21": nb.float64[:],
    "dQ22": nb.float64[:],
    "dE11": nb.float64[:],
    "dE12": nb.float64[:],
    "dE21": nb.float64[:],
    "dE22": nb.float64[:],
    "dR11": nb.float64[:],
    "dR12": nb.float64[:],
    "dR21": nb.float64[:],
    "dR22": nb.float64[:],
}

# @nb.njit(locals=_tmp_values_jacobian_wishart)
def jacobian_wishart(market: MarketParameters, model: ModelParameters):
    Q = model.Q
    E = model.E
    R = model.R
    mf = np.log(market.K/ market.S)
    Q11, Q12, Q21, Q22 = Q[0][0], Q[0][1], Q[1][0], Q[1][1]
    E11, E12, E21, E22 = E[0][0], E[0][1], E[1][0], E[1][1]
    R11, R12, R21, R22 = R[0][0], R[0][1], R[1][0], R[1][1]
    iv = get_iv_wishart(market=market, model=model)

    dQ11 = mf*(mf*((E11 + E22)*(2*E11*Q11 + E11*(4*R11*(Q11*R11 + Q21*R12) + R21*(Q12*R11 + Q22*R12)) + E12*Q12 + E12*(2*R11*(Q11*R21 + Q21*R22) + 2*R21*(Q11*R11 + Q21*R12) + R21*(Q12*R21 + Q22*R22)) + E21*Q12 + E21*(R11*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + R21*(Q11*R11 + Q21*R12)) + E22*R21*(2*Q11*R21 + Q12*R11 + 2*Q21*R22 + Q22*R12)) - 7.5*(E11*R11 + E12*R21)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(E11*R11 + E12*R21))/(3*(E11 + E22)**3)
    dQ12 = mf*(mf*((E11 + E22)*(E11*R11*(Q11*R21 + 2*Q12*R11 + Q21*R22 + 2*Q22*R12) + E12*Q11 + E12*(R11*(Q12*R21 + Q22*R22) + R21*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E21*Q11 + E21*(R11*(Q11*R11 + Q21*R12) + 2*R11*(Q12*R21 + Q22*R22) + 2*R21*(Q12*R11 + Q22*R12)) + 2*E22*Q12 + E22*(R11*(Q11*R21 + Q21*R22) + 4*R21*(Q12*R21 + Q22*R22))) - 7.5*(E21*R11 + E22*R21)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(E21*R11 + E22*R21))/(3*(E11 + E22)**3)
    dQ21 = mf*(mf*((E11 + E22)*(2*E11*Q21 + E11*(4*R12*(Q11*R11 + Q21*R12) + R22*(Q12*R11 + Q22*R12)) + E12*Q22 + E12*(2*R12*(Q11*R21 + Q21*R22) + 2*R22*(Q11*R11 + Q21*R12) + R22*(Q12*R21 + Q22*R22)) + E21*Q22 + E21*(R12*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + R22*(Q11*R11 + Q21*R12)) + E22*R22*(2*Q11*R21 + Q12*R11 + 2*Q21*R22 + Q22*R12)) - 7.5*(E11*R12 + E12*R22)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(E11*R12 + E12*R22))/(3*(E11 + E22)**3)
    dQ22 = mf*(mf*((E11 + E22)*(E11*R12*(Q11*R21 + 2*Q12*R11 + Q21*R22 + 2*Q22*R12) + E12*Q21 + E12*(R12*(Q12*R21 + Q22*R22) + R22*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E21*Q21 + E21*(R12*(Q11*R11 + Q21*R12) + 2*R12*(Q12*R21 + Q22*R22) + 2*R22*(Q12*R11 + Q22*R12)) + 2*E22*Q22 + E22*(R12*(Q11*R21 + Q21*R22) + 4*R22*(Q12*R21 + Q22*R22))) - 7.5*(E21*R12 + E22*R22)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(E21*R12 + E22*R22))/(3*(E11 + E22)**3)

    dE11 = (mf**2*(-2*(E11 + E22)*(E11*(Q11**2 + Q21**2) + E11*(2*(Q11*R11 + Q21*R12)**2 + (Q12*R11 + Q22*R12)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E12*(Q11*Q12 + Q21*Q22) + E12*(2*(Q11*R11 + Q21*R12)*(Q11*R21 + Q21*R22) + (Q12*R21 + Q22*R22)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E21*(Q11*Q12 + Q21*Q22) + E21*((Q11*R11 + Q21*R12)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + 2*(Q12*R11 + Q22*R12)*(Q12*R21 + Q22*R22)) + E22*(Q12**2 + Q22**2) + E22*((Q11*R21 + Q21*R22)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + 2*(Q12*R21 + Q22*R22)**2)) + 7.5*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))**2)/3 + mf**2*((E11 + E22)**2*(Q11**2 + Q21**2 + 2*(Q11*R11 + Q21*R12)**2 + (Q12*R11 + Q22*R12)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) - 7.5*(E11 + E22)*(Q11*R11 + Q21*R12)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22)) + 3.75*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))**2)/3 + mf*(E11 + E22)**3*(Q11*R11 + Q21*R12) - mf*(E11 + E22)**2*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22)) + (E11 + E22)**4)/(E11 + E22)**4
    dE12 = mf*(mf*((E11 + E22)*(Q11*Q12 + Q21*Q22 + 2*(Q11*R11 + Q21*R12)*(Q11*R21 + Q21*R22) + (Q12*R21 + Q22*R22)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) - 7.5*(Q11*R21 + Q21*R22)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(Q11*R21 + Q21*R22))/(3*(E11 + E22)**3)
    dE21 = mf*(mf*((E11 + E22)*(Q11*Q12 + Q21*Q22 + (Q11*R11 + Q21*R12)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + 2*(Q12*R11 + Q22*R12)*(Q12*R21 + Q22*R22)) - 7.5*(Q12*R11 + Q22*R12)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(Q12*R11 + Q22*R12))/(3*(E11 + E22)**3)
    dE22 = (mf**2*(-2*(E11 + E22)*(E11*(Q11**2 + Q21**2) + E11*(2*(Q11*R11 + Q21*R12)**2 + (Q12*R11 + Q22*R12)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E12*(Q11*Q12 + Q21*Q22) + E12*(2*(Q11*R11 + Q21*R12)*(Q11*R21 + Q21*R22) + (Q12*R21 + Q22*R22)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E21*(Q11*Q12 + Q21*Q22) + E21*((Q11*R11 + Q21*R12)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + 2*(Q12*R11 + Q22*R12)*(Q12*R21 + Q22*R22)) + E22*(Q12**2 + Q22**2) + E22*((Q11*R21 + Q21*R22)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + 2*(Q12*R21 + Q22*R22)**2)) + 7.5*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))**2)/3 + mf**2*((E11 + E22)**2*(Q12**2 + Q22**2 + (Q11*R21 + Q21*R22)*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + 2*(Q12*R21 + Q22*R22)**2) - 7.5*(E11 + E22)*(Q12*R21 + Q22*R22)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22)) + 3.75*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))**2)/3 + mf*(E11 + E22)**3*(Q12*R21 + Q22*R22) - mf*(E11 + E22)**2*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22)) + (E11 + E22)**4)/(E11 + E22)**4

    dR11 = mf*(mf*((E11 + E22)*(E11*(4*Q11*(Q11*R11 + Q21*R12) + Q12*(Q12*R11 + Q22*R12) + Q12*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E12*(2*Q11*(Q11*R21 + Q21*R22) + Q12*(Q12*R21 + Q22*R22)) + E21*(Q11*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + Q12*(Q11*R11 + Q21*R12) + 2*Q12*(Q12*R21 + Q22*R22)) + E22*Q12*(Q11*R21 + Q21*R22)) - 7.5*(E11*Q11 + E21*Q12)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(E11*Q11 + E21*Q12))/(3*(E11 + E22)**3)
    dR12 = mf*(mf*((E11 + E22)*(E11*(4*Q21*(Q11*R11 + Q21*R12) + Q22*(Q12*R11 + Q22*R12) + Q22*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E12*(2*Q21*(Q11*R21 + Q21*R22) + Q22*(Q12*R21 + Q22*R22)) + E21*(Q21*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + Q22*(Q11*R11 + Q21*R12) + 2*Q22*(Q12*R21 + Q22*R22)) + E22*Q22*(Q11*R21 + Q21*R22)) - 7.5*(E11*Q21 + E21*Q22)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(E11*Q21 + E21*Q22))/(3*(E11 + E22)**3)
    dR21 = mf*(mf*((E11 + E22)*(E11*Q11*(Q12*R11 + Q22*R12) + E12*(2*Q11*(Q11*R11 + Q21*R12) + Q11*(Q12*R21 + Q22*R22) + Q12*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E21*(Q11*(Q11*R11 + Q21*R12) + 2*Q12*(Q12*R11 + Q22*R12)) + E22*(Q11*(Q11*R21 + Q21*R22) + Q11*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + 4*Q12*(Q12*R21 + Q22*R22))) - 7.5*(E12*Q11 + E22*Q12)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(E12*Q11 + E22*Q12))/(3*(E11 + E22)**3)
    dR22 = mf*(mf*((E11 + E22)*(E11*Q21*(Q12*R11 + Q22*R12) + E12*(2*Q21*(Q11*R11 + Q21*R12) + Q21*(Q12*R21 + Q22*R22) + Q22*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12)) + E21*(Q21*(Q11*R11 + Q21*R12) + 2*Q22*(Q12*R11 + Q22*R12)) + E22*(Q21*(Q11*R21 + Q21*R22) + Q21*(Q11*R21 + Q12*R11 + Q21*R22 + Q22*R12) + 4*Q22*(Q12*R21 + Q22*R22))) - 7.5*(E12*Q21 + E22*Q22)*(E11*(Q11*R11 + Q21*R12) + E12*(Q11*R21 + Q21*R22) + E21*(Q12*R11 + Q22*R12) + E22*(Q12*R21 + Q22*R22))) + 3*(E11 + E22)**2*(E12*Q21 + E22*Q22))/(3*(E11 + E22)**3)

    dQ11, dQ12, dQ21, dQ22 = dQ11/(2*iv), dQ12/(2*iv), dQ21/(2*iv), dQ22/(2*iv)
    dE11, dE12, dE21, dE22 = dE11/(2*iv), dE12/(2*iv), dE21/(2*iv), dE22/(2*iv)
    dR11, dR12, dR21, dR22 = dR11/(2*iv), dR12/(2*iv), dR21/(2*iv), dR22/(2*iv)

    return dQ11, dQ12, dQ21, dQ22, dE11, dE12, dE21, dE22, dR11, dR12, dR21, dR22


def calibrate_wasc(
    df: pd.DataFrame,
    start_params: np.array,
    timestamp: int = None,
    calibration_type: str = "all",
    beta: float = None,
):
        """
    Function to calibrate SABR model.
    Attributes:
        @param df (pd.DataFrame): Dataframe with history
        [
            timestamp(ns),
            type(put or call),
            strike_price(usd),
            expiration(ns),
            mark_price(etc/btc),
            underlying_price(usd)
        ]
        @param start_params (np.array): Params to start calibration via LM from
        @param timestamp (int): On which timestamp to calibrate the model.
            Should be in range of df timestamps.
        @param calibration_type(str): Type of calibration. Should be one of: ["all", "beta"]
        @param beta(float): Fix it to needed value if you don't want to calibrate it

    Return:
        calibrated_params (np.array): Array of optimal params on timestamp tick.
        error (float): Value of error on calibration.
    """




