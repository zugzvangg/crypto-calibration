import numba as nb
import numpy as np
from typing import Final, Tuple
import pandas as pd
from src.utils import get_tick, get_implied_volatility, get_bid_ask
from src.levenberg_marquardt import LevenbergMarquardt
from typing import Union

# from scipy.integrate import quad
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
    ("E11", nb.float64),
    ("Ed", nb.float64),
    # ("Ed", nb.float64),
    ("E22", nb.float64),
    ("Q11", nb.float64),
    ("Q12", nb.float64),
    ("Q21", nb.float64),
    ("Q22", nb.float64),
    ("R11", nb.float64),
    ("R12", nb.float64),
    ("R21", nb.float64),
    ("R22", nb.float64),
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
    E11: nb.float64
    Ed: nb.float64
    # Ed: nb.float64
    E22: nb.float64
    Q11: nb.float64
    Q12: nb.float64
    Q21: nb.float64
    Q22: nb.float64
    R11: nb.float64
    R12: nb.float64
    R21: nb.float64
    R22: nb.float64

    def __init__(
        self,
        E11: nb.float64,
        Ed: nb.float64,
        # Ed: nb.float64,
        E22: nb.float64,
        Q11: nb.float64,
        Q12: nb.float64,
        Q21: nb.float64,
        Q22: nb.float64,
        R11: nb.float64,
        R12: nb.float64,
        R21: nb.float64,
        R22: nb.float64,
    ):
        self.E11 = E11
        # self.Ed = Ed
        self.Ed = Ed
        self.E22 = E22
        self.Q11 = Q11
        self.Q12 = Q12
        self.Q21 = Q21
        self.Q22 = Q22
        self.R11 = R11
        self.R12 = R12
        self.R21 = R21
        self.R22 = R22


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


def get_iv_wishart(market: MarketParameters, model: ModelParameters):
    mf = np.log(market.K / market.S)
    Q11, Q12, Q21, Q22, E11, Ed, E22, R11, R12, R21, R22 = (
        model.Q11,
        model.Q12,
        model.Q21,
        model.Q22,
        model.E11,
        # model.Ed,
        model.Ed,
        model.E22,
        model.R11,
        model.R12,
        model.R21,
        model.R22,
    )

    sigmas = (
        E11
        + E22
        + mf**2
        * (
            E11 * (Q11**2 + Q21**2) / 3
            + E11
            * (
                (Q11 * R11 + Q21 * R12) * (2 * Q11 * R11 + 2 * Q21 * R12)
                + (Q12 * R11 + Q22 * R12)
                * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
            )
            / 3
            + Ed * (Q11 * Q12 + Q21 * Q22) / 3
            + Ed
            * (
                (2 * Q11 * R11 + 2 * Q21 * R12) * (Q11 * R21 + Q21 * R22)
                + (Q12 * R21 + Q22 * R22)
                * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
            )
            / 3
            + Ed * (Q11 * Q12 + Q21 * Q22) / 3
            + Ed
            * (
                (Q11 * R11 + Q21 * R12)
                * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                + (Q12 * R11 + Q22 * R12) * (2 * Q12 * R21 + 2 * Q22 * R22)
            )
            / 3
            + E22 * (Q12**2 + Q22**2) / 3
            + E22
            * (
                (Q11 * R21 + Q21 * R22)
                * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                + (Q12 * R21 + Q22 * R22) * (2 * Q12 * R21 + 2 * Q22 * R22)
            )
            / 3
            - 1.25
            * (
                E11 * (Q11 * R11 + Q21 * R12)
                + Ed * (Q11 * R21 + Q21 * R22)
                + Ed * (Q12 * R11 + Q22 * R12)
                + E22 * (Q12 * R21 + Q22 * R22)
            )
            ** 2
            / (E11 + E22)
        )
        / (E11 + E22) ** 2
        + mf
        * (
            E11 * (Q11 * R11 + Q21 * R12)
            + Ed * (Q11 * R21 + Q21 * R22)
            + Ed * (Q12 * R11 + Q22 * R12)
            + E22 * (Q12 * R21 + Q22 * R22)
        )
        / (E11 + E22)
    )
    sigmas = np.sqrt(sigmas)
    return sigmas


# @nb.njit(locals=_tmp_values_get_iv_wishart)
# def get_iv_wishart(market: MarketParameters, model: ModelParameters):
#     Ks = market.K
#     f = market.S
#     n = len(Ks)
#     Es = np.zeros(n, dtype=np.float64)
#     E, R, Q = model.E, model.R, model.Q
#     trace_E = np.trace(E)
#     triple = np.dot(R, np.dot(Q, E))
#     trace_triple = np.trace(triple)
#     Q_T = Q.T
#     R_T = R.T
#     for index in range(n):
#         K = Ks[index]
#         mf = np.log(f / K)
#         # iv_sqared = (
#         #     trace_E
#         #     + trace_triple * mf / trace_E
#         #     + mf**2
#         #     / trace_E**2
#         #     * (
#         #         np.trace(Q.T @ Q @ E) / 3
#         #         + np.trace(R @ Q @ (Q.T @ R.T + R @ Q) @ E) / 3
#         #         - 5 / 4 * trace_triple**2 / trace_E
#         #     )
#         # )
#         iv_sqared = (
#             trace_E
#             + trace_triple * mf / trace_E
#             + mf**2
#             / trace_E**2
#             * (
#                 np.trace(np.dot(Q_T, np.dot(Q, E))) / 3
#                 + np.trace(
#                     np.dot(
#                         R,
#                         np.dot(Q, np.dot(np.dot(Q_T, R_T) + np.dot(R, Q), E)),
#                     )
#                 )
#                 / 3
#                 - 5 / 4 * trace_triple**2 / trace_E
#             )
#         )
#         Es[index] = np.sqrt(iv_sqared)
#     return Es


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
    # "Ed": nb.float64,
    "Ed": nb.float64,
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
    "dEd": nb.float64[:],
    "dEd": nb.float64[:],
    "dE22": nb.float64[:],
    "dR11": nb.float64[:],
    "dR12": nb.float64[:],
    "dR21": nb.float64[:],
    "dR22": nb.float64[:],
}


# @nb.njit(locals=_tmp_values_jacobian_wishart)
def jacobian_wishart(market: MarketParameters, model: ModelParameters):
    mf = np.log(market.K / market.S)
    Q11, Q12, Q21, Q22, E11, Ed, E22, R11, R12, R21, R22 = (
        model.Q11,
        model.Q12,
        model.Q21,
        model.Q22,
        model.E11,
        # model.Ed,
        model.Ed,
        model.E22,
        model.R11,
        model.R12,
        model.R21,
        model.R22,
    )

    iv = get_iv_wishart(market=market, model=model)

    dQ11 = (
        mf
        * (
            mf
            * (
                (E11 + E22)
                * (
                    2 * E11 * Q11
                    + E11
                    * (
                        4 * R11 * (Q11 * R11 + Q21 * R12)
                        + R21 * (Q12 * R11 + Q22 * R12)
                    )
                    + Ed * Q12
                    + Ed
                    * (
                        2 * R11 * (Q11 * R21 + Q21 * R22)
                        + 2 * R21 * (Q11 * R11 + Q21 * R12)
                        + R21 * (Q12 * R21 + Q22 * R22)
                    )
                    + Ed * Q12
                    + Ed
                    * (
                        R11 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                        + R21 * (Q11 * R11 + Q21 * R12)
                    )
                    + E22
                    * R21
                    * (2 * Q11 * R21 + Q12 * R11 + 2 * Q21 * R22 + Q22 * R12)
                )
                - 7.5
                * (E11 * R11 + Ed * R21)
                * (
                    E11 * (Q11 * R11 + Q21 * R12)
                    + Ed * (Q11 * R21 + Q21 * R22)
                    + Ed * (Q12 * R11 + Q22 * R12)
                    + E22 * (Q12 * R21 + Q22 * R22)
                )
            )
            + 3 * (E11 + E22) ** 2 * (E11 * R11 + Ed * R21)
        )
        / (3 * (E11 + E22) ** 3)
    )
    dQ12 = (
        mf
        * (
            mf
            * (
                (E11 + E22)
                * (
                    E11 * R11 * (Q11 * R21 + 2 * Q12 * R11 + Q21 * R22 + 2 * Q22 * R12)
                    + Ed * Q11
                    + Ed
                    * (
                        R11 * (Q12 * R21 + Q22 * R22)
                        + R21 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    )
                    + Ed * Q11
                    + Ed
                    * (
                        R11 * (Q11 * R11 + Q21 * R12)
                        + 2 * R11 * (Q12 * R21 + Q22 * R22)
                        + 2 * R21 * (Q12 * R11 + Q22 * R12)
                    )
                    + 2 * E22 * Q12
                    + E22
                    * (
                        R11 * (Q11 * R21 + Q21 * R22)
                        + 4 * R21 * (Q12 * R21 + Q22 * R22)
                    )
                )
                - 7.5
                * (Ed * R11 + E22 * R21)
                * (
                    E11 * (Q11 * R11 + Q21 * R12)
                    + Ed * (Q11 * R21 + Q21 * R22)
                    + Ed * (Q12 * R11 + Q22 * R12)
                    + E22 * (Q12 * R21 + Q22 * R22)
                )
            )
            + 3 * (E11 + E22) ** 2 * (Ed * R11 + E22 * R21)
        )
        / (3 * (E11 + E22) ** 3)
    )
    dQ21 = (
        mf
        * (
            mf
            * (
                (E11 + E22)
                * (
                    2 * E11 * Q21
                    + E11
                    * (
                        4 * R12 * (Q11 * R11 + Q21 * R12)
                        + R22 * (Q12 * R11 + Q22 * R12)
                    )
                    + Ed * Q22
                    + Ed
                    * (
                        2 * R12 * (Q11 * R21 + Q21 * R22)
                        + 2 * R22 * (Q11 * R11 + Q21 * R12)
                        + R22 * (Q12 * R21 + Q22 * R22)
                    )
                    + Ed * Q22
                    + Ed
                    * (
                        R12 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                        + R22 * (Q11 * R11 + Q21 * R12)
                    )
                    + E22
                    * R22
                    * (2 * Q11 * R21 + Q12 * R11 + 2 * Q21 * R22 + Q22 * R12)
                )
                - 7.5
                * (E11 * R12 + Ed * R22)
                * (
                    E11 * (Q11 * R11 + Q21 * R12)
                    + Ed * (Q11 * R21 + Q21 * R22)
                    + Ed * (Q12 * R11 + Q22 * R12)
                    + E22 * (Q12 * R21 + Q22 * R22)
                )
            )
            + 3 * (E11 + E22) ** 2 * (E11 * R12 + Ed * R22)
        )
        / (3 * (E11 + E22) ** 3)
    )
    dQ22 = (
        mf
        * (
            mf
            * (
                (E11 + E22)
                * (
                    E11 * R12 * (Q11 * R21 + 2 * Q12 * R11 + Q21 * R22 + 2 * Q22 * R12)
                    + Ed * Q21
                    + Ed
                    * (
                        R12 * (Q12 * R21 + Q22 * R22)
                        + R22 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    )
                    + Ed * Q21
                    + Ed
                    * (
                        R12 * (Q11 * R11 + Q21 * R12)
                        + 2 * R12 * (Q12 * R21 + Q22 * R22)
                        + 2 * R22 * (Q12 * R11 + Q22 * R12)
                    )
                    + 2 * E22 * Q22
                    + E22
                    * (
                        R12 * (Q11 * R21 + Q21 * R22)
                        + 4 * R22 * (Q12 * R21 + Q22 * R22)
                    )
                )
                - 7.5
                * (Ed * R12 + E22 * R22)
                * (
                    E11 * (Q11 * R11 + Q21 * R12)
                    + Ed * (Q11 * R21 + Q21 * R22)
                    + Ed * (Q12 * R11 + Q22 * R12)
                    + E22 * (Q12 * R21 + Q22 * R22)
                )
            )
            + 3 * (E11 + E22) ** 2 * (Ed * R12 + E22 * R22)
        )
        / (3 * (E11 + E22) ** 3)
    )

    dE11 = (
        mf**2
        * (
            -2
            * (E11 + E22)
            * (
                E11 * (Q11**2 + Q21**2)
                + E11
                * (
                    2 * (Q11 * R11 + Q21 * R12) ** 2
                    + (Q12 * R11 + Q22 * R12)
                    * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                )
                + Ed * (Q11 * Q12 + Q21 * Q22)
                + Ed
                * (
                    2 * (Q11 * R11 + Q21 * R12) * (Q11 * R21 + Q21 * R22)
                    + (Q12 * R21 + Q22 * R22)
                    * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                )
                + Ed * (Q11 * Q12 + Q21 * Q22)
                + Ed
                * (
                    (Q11 * R11 + Q21 * R12)
                    * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    + 2 * (Q12 * R11 + Q22 * R12) * (Q12 * R21 + Q22 * R22)
                )
                + E22 * (Q12**2 + Q22**2)
                + E22
                * (
                    (Q11 * R21 + Q21 * R22)
                    * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    + 2 * (Q12 * R21 + Q22 * R22) ** 2
                )
            )
            + 7.5
            * (
                E11 * (Q11 * R11 + Q21 * R12)
                + Ed * (Q11 * R21 + Q21 * R22)
                + Ed * (Q12 * R11 + Q22 * R12)
                + E22 * (Q12 * R21 + Q22 * R22)
            )
            ** 2
        )
        / 3
        + mf**2
        * (
            (E11 + E22) ** 2
            * (
                Q11**2
                + Q21**2
                + 2 * (Q11 * R11 + Q21 * R12) ** 2
                + (Q12 * R11 + Q22 * R12)
                * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
            )
            - 7.5
            * (E11 + E22)
            * (Q11 * R11 + Q21 * R12)
            * (
                E11 * (Q11 * R11 + Q21 * R12)
                + Ed * (Q11 * R21 + Q21 * R22)
                + Ed * (Q12 * R11 + Q22 * R12)
                + E22 * (Q12 * R21 + Q22 * R22)
            )
            + 3.75
            * (
                E11 * (Q11 * R11 + Q21 * R12)
                + Ed * (Q11 * R21 + Q21 * R22)
                + Ed * (Q12 * R11 + Q22 * R12)
                + E22 * (Q12 * R21 + Q22 * R22)
            )
            ** 2
        )
        / 3
        + mf * (E11 + E22) ** 3 * (Q11 * R11 + Q21 * R12)
        - mf
        * (E11 + E22) ** 2
        * (
            E11 * (Q11 * R11 + Q21 * R12)
            + Ed * (Q11 * R21 + Q21 * R22)
            + Ed * (Q12 * R11 + Q22 * R12)
            + E22 * (Q12 * R21 + Q22 * R22)
        )
        + (E11 + E22) ** 4
    ) / (E11 + E22) ** 4
    # dEd = (
    #     mf
    #     * (
    #         mf
    #         * (
    #             (E11 + E22)
    #             * (
    #                 Q11 * Q12
    #                 + Q21 * Q22
    #                 + 2 * (Q11 * R11 + Q21 * R12) * (Q11 * R21 + Q21 * R22)
    #                 + (Q12 * R21 + Q22 * R22)
    #                 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
    #             )
    #             - 7.5
    #             * (Q11 * R21 + Q21 * R22)
    #             * (
    #                 E11 * (Q11 * R11 + Q21 * R12)
    #                 + Ed * (Q11 * R21 + Q21 * R22)
    #                 + Ed * (Q12 * R11 + Q22 * R12)
    #                 + E22 * (Q12 * R21 + Q22 * R22)
    #             )
    #         )
    #         + 3 * (E11 + E22) ** 2 * (Q11 * R21 + Q21 * R22)
    #     )
    #     / (3 * (E11 + E22) ** 3)
    # )
    dEd = (
        mf
        * (
            mf
            * (
                (E11 + E22)
                * (
                    Q11 * Q12
                    + Q21 * Q22
                    + (Q11 * R11 + Q21 * R12)
                    * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    + 2 * (Q12 * R11 + Q22 * R12) * (Q12 * R21 + Q22 * R22)
                )
                - 7.5
                * (Q12 * R11 + Q22 * R12)
                * (
                    E11 * (Q11 * R11 + Q21 * R12)
                    + Ed * (Q11 * R21 + Q21 * R22)
                    + Ed * (Q12 * R11 + Q22 * R12)
                    + E22 * (Q12 * R21 + Q22 * R22)
                )
            )
            + 3 * (E11 + E22) ** 2 * (Q12 * R11 + Q22 * R12)
        )
        / (3 * (E11 + E22) ** 3)
    )
    dE22 = (
        mf**2
        * (
            -2
            * (E11 + E22)
            * (
                E11 * (Q11**2 + Q21**2)
                + E11
                * (
                    2 * (Q11 * R11 + Q21 * R12) ** 2
                    + (Q12 * R11 + Q22 * R12)
                    * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                )
                + Ed * (Q11 * Q12 + Q21 * Q22)
                + Ed
                * (
                    2 * (Q11 * R11 + Q21 * R12) * (Q11 * R21 + Q21 * R22)
                    + (Q12 * R21 + Q22 * R22)
                    * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                )
                + Ed * (Q11 * Q12 + Q21 * Q22)
                + Ed
                * (
                    (Q11 * R11 + Q21 * R12)
                    * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    + 2 * (Q12 * R11 + Q22 * R12) * (Q12 * R21 + Q22 * R22)
                )
                + E22 * (Q12**2 + Q22**2)
                + E22
                * (
                    (Q11 * R21 + Q21 * R22)
                    * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    + 2 * (Q12 * R21 + Q22 * R22) ** 2
                )
            )
            + 7.5
            * (
                E11 * (Q11 * R11 + Q21 * R12)
                + Ed * (Q11 * R21 + Q21 * R22)
                + Ed * (Q12 * R11 + Q22 * R12)
                + E22 * (Q12 * R21 + Q22 * R22)
            )
            ** 2
        )
        / 3
        + mf**2
        * (
            (E11 + E22) ** 2
            * (
                Q12**2
                + Q22**2
                + (Q11 * R21 + Q21 * R22)
                * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                + 2 * (Q12 * R21 + Q22 * R22) ** 2
            )
            - 7.5
            * (E11 + E22)
            * (Q12 * R21 + Q22 * R22)
            * (
                E11 * (Q11 * R11 + Q21 * R12)
                + Ed * (Q11 * R21 + Q21 * R22)
                + Ed * (Q12 * R11 + Q22 * R12)
                + E22 * (Q12 * R21 + Q22 * R22)
            )
            + 3.75
            * (
                E11 * (Q11 * R11 + Q21 * R12)
                + Ed * (Q11 * R21 + Q21 * R22)
                + Ed * (Q12 * R11 + Q22 * R12)
                + E22 * (Q12 * R21 + Q22 * R22)
            )
            ** 2
        )
        / 3
        + mf * (E11 + E22) ** 3 * (Q12 * R21 + Q22 * R22)
        - mf
        * (E11 + E22) ** 2
        * (
            E11 * (Q11 * R11 + Q21 * R12)
            + Ed * (Q11 * R21 + Q21 * R22)
            + Ed * (Q12 * R11 + Q22 * R12)
            + E22 * (Q12 * R21 + Q22 * R22)
        )
        + (E11 + E22) ** 4
    ) / (E11 + E22) ** 4

    dR11 = (
        mf
        * (
            mf
            * (
                (E11 + E22)
                * (
                    E11
                    * (
                        4 * Q11 * (Q11 * R11 + Q21 * R12)
                        + Q12 * (Q12 * R11 + Q22 * R12)
                        + Q12 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    )
                    + Ed
                    * (
                        2 * Q11 * (Q11 * R21 + Q21 * R22)
                        + Q12 * (Q12 * R21 + Q22 * R22)
                    )
                    + Ed
                    * (
                        Q11 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                        + Q12 * (Q11 * R11 + Q21 * R12)
                        + 2 * Q12 * (Q12 * R21 + Q22 * R22)
                    )
                    + E22 * Q12 * (Q11 * R21 + Q21 * R22)
                )
                - 7.5
                * (E11 * Q11 + Ed * Q12)
                * (
                    E11 * (Q11 * R11 + Q21 * R12)
                    + Ed * (Q11 * R21 + Q21 * R22)
                    + Ed * (Q12 * R11 + Q22 * R12)
                    + E22 * (Q12 * R21 + Q22 * R22)
                )
            )
            + 3 * (E11 + E22) ** 2 * (E11 * Q11 + Ed * Q12)
        )
        / (3 * (E11 + E22) ** 3)
    )
    dR12 = (
        mf
        * (
            mf
            * (
                (E11 + E22)
                * (
                    E11
                    * (
                        4 * Q21 * (Q11 * R11 + Q21 * R12)
                        + Q22 * (Q12 * R11 + Q22 * R12)
                        + Q22 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    )
                    + Ed
                    * (
                        2 * Q21 * (Q11 * R21 + Q21 * R22)
                        + Q22 * (Q12 * R21 + Q22 * R22)
                    )
                    + Ed
                    * (
                        Q21 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                        + Q22 * (Q11 * R11 + Q21 * R12)
                        + 2 * Q22 * (Q12 * R21 + Q22 * R22)
                    )
                    + E22 * Q22 * (Q11 * R21 + Q21 * R22)
                )
                - 7.5
                * (E11 * Q21 + Ed * Q22)
                * (
                    E11 * (Q11 * R11 + Q21 * R12)
                    + Ed * (Q11 * R21 + Q21 * R22)
                    + Ed * (Q12 * R11 + Q22 * R12)
                    + E22 * (Q12 * R21 + Q22 * R22)
                )
            )
            + 3 * (E11 + E22) ** 2 * (E11 * Q21 + Ed * Q22)
        )
        / (3 * (E11 + E22) ** 3)
    )
    dR21 = (
        mf
        * (
            mf
            * (
                (E11 + E22)
                * (
                    E11 * Q11 * (Q12 * R11 + Q22 * R12)
                    + Ed
                    * (
                        2 * Q11 * (Q11 * R11 + Q21 * R12)
                        + Q11 * (Q12 * R21 + Q22 * R22)
                        + Q12 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    )
                    + Ed
                    * (
                        Q11 * (Q11 * R11 + Q21 * R12)
                        + 2 * Q12 * (Q12 * R11 + Q22 * R12)
                    )
                    + E22
                    * (
                        Q11 * (Q11 * R21 + Q21 * R22)
                        + Q11 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                        + 4 * Q12 * (Q12 * R21 + Q22 * R22)
                    )
                )
                - 7.5
                * (Ed * Q11 + E22 * Q12)
                * (
                    E11 * (Q11 * R11 + Q21 * R12)
                    + Ed * (Q11 * R21 + Q21 * R22)
                    + Ed * (Q12 * R11 + Q22 * R12)
                    + E22 * (Q12 * R21 + Q22 * R22)
                )
            )
            + 3 * (E11 + E22) ** 2 * (Ed * Q11 + E22 * Q12)
        )
        / (3 * (E11 + E22) ** 3)
    )
    dR22 = (
        mf
        * (
            mf
            * (
                (E11 + E22)
                * (
                    E11 * Q21 * (Q12 * R11 + Q22 * R12)
                    + Ed
                    * (
                        2 * Q21 * (Q11 * R11 + Q21 * R12)
                        + Q21 * (Q12 * R21 + Q22 * R22)
                        + Q22 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                    )
                    + Ed
                    * (
                        Q21 * (Q11 * R11 + Q21 * R12)
                        + 2 * Q22 * (Q12 * R11 + Q22 * R12)
                    )
                    + E22
                    * (
                        Q21 * (Q11 * R21 + Q21 * R22)
                        + Q21 * (Q11 * R21 + Q12 * R11 + Q21 * R22 + Q22 * R12)
                        + 4 * Q22 * (Q12 * R21 + Q22 * R22)
                    )
                )
                - 7.5
                * (Ed * Q21 + E22 * Q22)
                * (
                    E11 * (Q11 * R11 + Q21 * R12)
                    + Ed * (Q11 * R21 + Q21 * R22)
                    + Ed * (Q12 * R11 + Q22 * R12)
                    + E22 * (Q12 * R21 + Q22 * R22)
                )
            )
            + 3 * (E11 + E22) ** 2 * (Ed * Q21 + E22 * Q22)
        )
        / (3 * (E11 + E22) ** 3)
    )

    dQ11, dQ12, dQ21, dQ22 = (
        dQ11 / (2 * iv),
        dQ12 / (2 * iv),
        dQ21 / (2 * iv),
        dQ22 / (2 * iv),
    )
    dE11, dEd, dE22 = (
        dE11 / (2 * iv),
        dEd / (2 * iv),
        dE22 / (2 * iv),
    )
    dR11, dR12, dR21, dR22 = (
        dR11 / (2 * iv),
        dR12 / (2 * iv),
        dR21 / (2 * iv),
        dR22 / (2 * iv),
    )

    return dQ11, dQ12, dQ21, dQ22, dE11, dEd, dE22, dR11, dR12, dR21, dR22


def calibrate_wasc(
    df: pd.DataFrame,
    start_params: np.array,
    timestamp: int = None,
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

    Return:
        calibrated_params (np.array): Array of optimal params on timestamp tick.
        error (float): Value of error on calibration.
    """
    tick = get_tick(df=df, timestamp=timestamp)
    iv = []
    # count ivs
    for index, t in tick.iterrows():
        market_iv = get_implied_volatility(
            option_type=t["type"],
            C=t["mark_price_usd"],
            K=t["strike_price"],
            T=t["tau"],
            F=t["underlying_price"],
            r=0.0,
            error=0.001,
        )
        iv.append(market_iv)

    # drop if iv is None
    tick["iv"] = iv
    tick = tick[~tick["iv"].isin([None, np.nan])]
    karr = tick.strike_price.to_numpy(dtype=np.float64)
    carr = tick.mark_price_usd.to_numpy(dtype=np.float64)
    iv = tick.iv.to_numpy(dtype=np.float64)
    T = np.float64(tick.tau.mean())
    types = np.where(tick["type"] == "call", True, False)
    # take it zero as on deribit
    r_val = np.float64(0.0)
    # tick dataframes may have not similar timestamps -->
    # not equal value if underlying --> take mean
    S_val = np.float64(tick.underlying_price.mean())
    market = MarketParameters(K=karr, T=T, S=S_val, r=r_val, C=carr, types=types, iv=iv)

    def clip_params(wasc_params: np.ndarray) -> np.ndarray:
        """
        This funciton project WASC matrix parameters into valid range
        Attributes:
            wasc_params(np.ndarray): model parameters
        Returns:
            wasc_params(np.ndarray): clipped parameters
        """
        eps = 1e-4

        def clip_all(params):
            return params

        return wasc_params

    def get_residuals(wasc_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function calculates residuals of market ivd and calibrated ones
        and Jacobian matrix
        Args:
            sabr_params(np.ndarray): model params
        Returns:
            res(np.ndarray) : vector or residuals
            J(np.ndarray)   : Jacobian
        """
        # function supports several calibration types

        model_parameters = ModelParameters(
            wasc_params[0],
            wasc_params[1],
            wasc_params[2],
            wasc_params[3],
            wasc_params[4],
            wasc_params[5],
            wasc_params[6],
            wasc_params[7],
            wasc_params[8],
            wasc_params[9],
            wasc_params[10],
            # wasc_params[11],
        )

        J = jacobian_wishart(model=model_parameters, market=market)
        iv = get_iv_wishart(model=model_parameters, market=market)
        weights = np.ones_like(market.K)
        weights = weights / np.sum(weights)
        res = iv - market.iv
        return res * weights, J @ np.diag(weights)

    res = LevenbergMarquardt(500, get_residuals, clip_params, start_params)
    calibrated_params = np.array(res["x"], dtype=np.float64)
    error = res["objective"][-1]
    final_params = ModelParameters(
        calibrated_params[0],
        calibrated_params[1],
        calibrated_params[2],
        calibrated_params[3],
        calibrated_params[4],
        calibrated_params[5],
        calibrated_params[6],
        calibrated_params[7],
        calibrated_params[8],
        calibrated_params[9],
        calibrated_params[10],
        # calibrated_params[11],
    )
    final_vols = get_iv_wishart(model=final_params, market=market)
    tick["calibrated_iv"] = final_vols
    result = tick[
        [
            "type",
            "strike_price",
            "expiration",
            "underlying_price",
            "iv",
            "calibrated_iv",
        ]
    ]
    result["iv"] = 100 * result["iv"]
    result["calibrated_iv"] = 100 * result["calibrated_iv"]
    return calibrated_params, error, result
