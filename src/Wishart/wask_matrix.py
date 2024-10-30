import numba as nb
import numpy as np


@nb.experimental.jitclass([("Q", nb.float64[:, ::1])])
class Q:
    def __init__(self, Q: nb.float64[:, ::1]):
        if not len(Q) == len(Q[0]):
            raise ValueError("Matrix not square")
        self.Q = Q


@nb.experimental.jitclass([("R", nb.float64[:, ::1])])
class R:
    def __init__(self, R: nb.float64[:, ::1]):
        if not len(R) == len(R[0]):
            raise ValueError("Matrix not square")
        self.R = R


@nb.experimental.jitclass([("sigma", nb.float64[:, ::1])])
class Sigma:
    def __init__(self, sigma: nb.float64[:, ::1]):
        if not len(sigma) == len(sigma[0]):
            raise ValueError("Matrix not square")
        self.sigma = sigma


@nb.experimental.jitclass(
    [
        ("Q", nb.float64[:, ::1]),
        ("R", nb.float64[:, ::1]),
        ("sigma", nb.float64[:, ::1]),
    ]
)
class WASCParams:
    def __init__(self, Q: Q, R: R, sigma: Sigma) -> None:
        if not len(Q.Q) == len(R.R) == len(sigma.sigma):
            raise ValueError("Matrixes are not of equal dimension")
        self.Q = Q.Q
        self.R = R.R
        self.sigma = sigma.sigma

    def array(self) -> nb.float64[:]:
        return np.concatenate(
            (self.R.reshape(-1), self.Q.reshape(-1), self.sigma.reshape(-1))
        )


class WASC:
    def __init__(self):
        pass


Q = Q(np.array([[1.0, 2.0], [3.0, 4.0]]))
R = R(np.array([[3.0, 4.0], [-3.0, -4.0]]))
sigma = Sigma(np.array([[-2.0, -4.0], [8.0, -9.0]]))
wask_params = WASCParams(Q, R, sigma)
print(wask_params.array())
