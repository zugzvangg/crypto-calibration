from heston import MarketParameters, ModelParameters, GLAW, fHes, JacHes
import numpy as np
from levenberg_marquardt import Levenberg_Marquardt
from typing import Tuple

m = np.int32(5)
n = np.int32(40)  # actually should be equal len of karr
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

# iv?
X = np.ones(len(karr))

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

market = MarketParameters(K=karr, T=tarr, S=S_val, r=r_val)

a = np.float64(3.0)  # kappa                           |  mean reversion rate
b = np.float64(0.10)  # v_infinity                      |  long term variance
c = np.float64(0.25)  # sigma                           |  variance of volatility
rho = np.float64(
    -0.8
)  # rho                             |  correlation between spot and volatility
v0 = np.float64(0.08)

# model = ModelParameters(a=a, b=b, c=c, rho=rho, v0=v0)


x64 = np.array(
    [
        0.0243502926634244325089558,
        0.0729931217877990394495429,
        0.1214628192961205544703765,
        0.1696444204239928180373136,
        0.2174236437400070841496487,
        0.2646871622087674163739642,
        0.3113228719902109561575127,
        0.3572201583376681159504426,
        0.4022701579639916036957668,
        0.4463660172534640879849477,
        0.4894031457070529574785263,
        0.5312794640198945456580139,
        0.5718956462026340342838781,
        0.6111553551723932502488530,
        0.6489654712546573398577612,
        0.6852363130542332425635584,
        0.7198818501716108268489402,
        0.7528199072605318966118638,
        0.7839723589433414076102205,
        0.8132653151227975597419233,
        0.8406292962525803627516915,
        0.8659993981540928197607834,
        0.8893154459951141058534040,
        0.9105221370785028057563807,
        0.9295691721319395758214902,
        0.9464113748584028160624815,
        0.9610087996520537189186141,
        0.9733268277899109637418535,
        0.9833362538846259569312993,
        0.9910133714767443207393824,
        0.9963401167719552793469245,
        0.9993050417357721394569056,
    ],
    dtype=np.float64,
)
w64 = np.array(
    [
        0.0486909570091397203833654,
        0.0485754674415034269347991,
        0.0483447622348029571697695,
        0.0479993885964583077281262,
        0.0475401657148303086622822,
        0.0469681828162100173253263,
        0.0462847965813144172959532,
        0.0454916279274181444797710,
        0.0445905581637565630601347,
        0.0435837245293234533768279,
        0.0424735151236535890073398,
        0.0412625632426235286101563,
        0.0399537411327203413866569,
        0.0385501531786156291289625,
        0.0370551285402400460404151,
        0.0354722132568823838106931,
        0.0338051618371416093915655,
        0.0320579283548515535854675,
        0.0302346570724024788679741,
        0.0283396726142594832275113,
        0.0263774697150546586716918,
        0.0243527025687108733381776,
        0.0222701738083832541592983,
        0.0201348231535302093723403,
        0.0179517157756973430850453,
        0.0157260304760247193219660,
        0.0134630478967186425980608,
        0.0111681394601311288185905,
        0.0088467598263639477230309,
        0.0065044579689783628561174,
        0.0041470332605624676352875,
        0.0017832807216964329472961,
    ],
    dtype=np.float64,
)
r = 0.0
T = tarr
numgrid = np.int32(60)
integration_settings = GLAW(numgrid=numgrid, u=x64, w=w64)

# x = fHes(
#     model_parameters=model,
#     market_parameters=market,
#     integration_settings=integration_settings,
#     m=m,
#     n=n,
# )
# print(x)

# hes = JacHes(
#     glaw=integration_settings,
#     model_parameters=model,
#     market_parameters=market,
#     market_pointer=np.int32(1),
#     n=n,
# )
# print(hes)

# print("Worked out")

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
    return 

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
    integration_settings=integration_settings,
    m=m,
    n=n,
    )

    J = JacHes(
    glaw=integration_settings,
    model_parameters=model_parameters, ###
    market_parameters=market,
    market_pointer=np.int32(1),
    n=n,)
    
    K = karr
    F = np.ones(len(K))*market.S
    weights = np.ones_like(K)
    weights = weights / np.sum(weights)

    typ = True
    P = C + np.exp(-r * T) * ( K - F )
    X_ = C
    X_[~typ] = P[~typ]
    res = X_ - X
    return res * weights, J @ np.diag(weights)



# start_params = ModelParameters(a=1.2, b=0.2, c=0.3, rho=-0.6, v0=0.2)
start_params = np.array([a, b, c, rho, v0])
res = Levenberg_Marquardt(100, get_residuals, proj_heston, start_params)

