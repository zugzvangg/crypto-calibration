import pandas as pd
from scipy import stats as sps
import numpy as np
import datetime
import numba as nb
import math


def get_tick(df: pd.DataFrame, timestamp: int = None):
    """Function gets tick for each expiration and strike
    from closest timestamp from given. If timestamp is None, it takes last one."""
    if timestamp:
        data = df[df["timestamp"] <= timestamp].copy()
        # only not expired on curret tick
        data = data[data["expiration"] > timestamp].copy()
    else:
        data = df.copy()
        # only not expired on max available tick
        data = data[data["expiration"] > data["timestamp"].max()].copy()
    # tau is time before expiration in years
    data["tau"] = (data.expiration - data.timestamp) / 1e6 / 3600 / 24 / 365

    data_grouped = data.loc[
        data.groupby(["type", "expiration", "strike_price"])["timestamp"].idxmax()
    ]

    data_grouped = data_grouped[data_grouped["tau"] > 0.0]
    # We need Only out of the money to calibrate
    data_grouped = data_grouped[
        (
            (data_grouped["type"] == "call")
            & (data_grouped["underlying_price"] <= data_grouped["strike_price"])
        )
        | (
            (data_grouped["type"] == "put")
            & (data_grouped["underlying_price"] >= data_grouped["strike_price"])
        )
    ]
    data_grouped["mark_price_usd"] = (
        data_grouped["mark_price"] * data_grouped["underlying_price"]
    )
    data_grouped = data_grouped[data_grouped["strike_price"] <= 10_000]
    return data_grouped


# Newton-Raphsen
# nb.njit


def get_implied_volatility(
    option_type: str,
    C: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
    error: float = 0.001,
) -> float:
    """
    Function to count implied volatility via given params of option, using Newton-Raphson method :

    Args:
        C (float): Option market price(USD).
        K (float): Strike(USD).
        T (float): Time to expiration in years.
        F (float): Underlying price.
        r (float): Risk-free rate.
        error (float): Given threshhold of error.

    Returns:
        float: Implied volatility in percent.
    """
    vol = 1.0
    dv = error + 1
    while abs(dv) > error:
        d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        D = np.exp(-r * T)
        if option_type.lower() == "call":
            price = F * sps.norm.cdf(d1) - K * sps.norm.cdf(d2) * D
        elif option_type.lower() == "put":
            price = -F * sps.norm.cdf(-d1) + K * sps.norm.cdf(-d2) * D
        else:
            raise ValueError("Wrong option type, must be 'call' or 'put' ")
        Vega = F * np.sqrt(T / np.pi / 2) * np.exp(-0.5 * d1**2)
        PriceError = price - C
        dv = PriceError / Vega
        vol = vol - dv
    return vol

def cdf(x) -> float:
    return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0

def pdf(x):
    probability = 1.0 / np.sqrt(2 * np.pi)
    probability *= np.exp(-0.5 * x**2)
    return probability

def get_price_bsm(
    option_type: str,
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
    )->float:
    d1 = (np.log(F/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    p = 1 if option_type else -1
    return p*F*cdf(p*d1) - p*K*np.exp(-r*T)*cdf(p*d2)

def process_data(data, granularity: int = 5) -> pd.DataFrame:
    """
    Args:
        data (pd.DataFrame): dataframe of .h5 format to preprocess.
        granularity (int): In which granularity in minutes to resample.
    Return:
        df (pd.DataFrame): Preprocessed dataframe
    """
    # only options
    time_granularity = f"{granularity}min"
    df = (
        data[
            (data["instrument"].str.endswith("C"))
            | (data["instrument"].str.endswith("P"))
        ]
        .copy()
        .set_index("dt")
    )
    df = (
        df.groupby("instrument")
        .resample(time_granularity)["price"]
        .ohlc()
        .ffill()
        .reset_index()[["instrument", "dt", "close"]]
    )
    df = df.rename(columns={"close": "price"})
    df["type"] = np.where(df["instrument"].str.endswith("C"), "call", "put")

    perpetual = (
        data[data["instrument"] == "ETH-PERPETUAL"][["dt", "price"]]
        .copy()
        .set_index("dt")
    )
    perpetual = (
        perpetual.resample(time_granularity)
        .agg({"price": "mean"})
        .ffill()
        .reset_index()
    )
    perpetual = perpetual.rename(columns={"price": "underlying_price"})

    def get_strike(x):
        return int(x.split("-")[2])

    def get_expiration(x):
        return x.split("-")[1]

    df["strike_price"] = df["instrument"].apply(get_strike)
    df["expiration"] = df["instrument"].apply(get_expiration)

    def unix_time_millis(dt):
        epoch = datetime.datetime.utcfromtimestamp(0)
        return int((dt - epoch).total_seconds() * 1000_000)

    def get_normal_date(s):
        """Function to convert date to find years to maturity"""
        monthToNum = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }

        full_date = s.split("-")[1]
        try:
            day = int(full_date[:2])
            month = monthToNum[full_date[2:5]]
        except:
            day = int(full_date[:1])
            month = monthToNum[full_date[1:4]]

        year = int("20" + full_date[-2:])
        exp_date = datetime.datetime(year, month, day)
        return unix_time_millis(exp_date)

    df = df.merge(perpetual, on="dt")
    df["timestamp"] = df["dt"].apply(unix_time_millis)
    df["expiration"] = df["instrument"].apply(get_normal_date)
    df["tau"] = (df.expiration - df.timestamp) / 1e6 / 3600 / 24 / 365
    df = df.rename(columns={"price": "mark_price"})
    df["mark_price_usd"] = df["mark_price"] * df["underlying_price"]

    # filter only not expired and out of the money
    df = df[df["tau"] > 0.0]

    df = df[
        ((df["type"] == "call") & (df["underlying_price"] <= df["strike_price"]))
        | ((df["type"] == "put") & (df["underlying_price"] >= df["strike_price"]))
    ]
    # drop very big put strikes
    df = df[df["strike_price"] <= df["underlying_price"].max()*5]
    return df


bid_ask_approx = {100: 1.1365374283822347,
 150: 1.1086183598196304,
 200: 1.0860836588964222,
 250: 1.0679171230221263,
 300: 1.0532943462322986,
 350: 1.0415465197692093,
 400: 1.0321310648888606,
 450: 1.024607808389764,
 500: 1.0186196547381465,
 550: 1.013876906108444,
 600: 1.0101445418368042,
 650: 1.0072318987322313,
 700: 1.0049842991109657,
 750: 1.0032762589436128,
 800: 1.0020059778868116,
 850: 1.0010908692584026,
 900: 1.0004639336786534,
 950: 1.0000708171452195,
 1000: 1.0,
 1050: 1.0,
 1100: 1.0,
 1150: 1.0000703447837442,
 1200: 1.000329505130653,
 1250: 1.0006554564669683,
 1275: 1.0008554564669683,
 1300: 1.0010356281976733,
 1350: 1.0014598222946316,
 1400: 1.0019197655017247,
 1450: 1.0024087460561573,
 1500: 1.0029213189744555,
 1550: 1.0034530669623627,
 1600: 1.004000406450258,
 1650: 1.0045604302371722,
 1700: 1.0051307798339444,
 1750: 1.0057095419001474,
 1800: 1.006295164227358,
 1850: 1.0068863875796281,
 1900: 1.0074821903982834,
 1950: 1.008081743943059,
 2000: 1.0086843758998336,
 2050: 1.0092895408569897,
 2100: 1.0098967963540186,
 2150: 1.01050578345068,
 2200: 1.0111162109635101,
 2250: 1.0117278426775091,
 2300: 1.0123404869714752,
 2350: 1.012953988401438,
 2400: 1.0135682208726249,
 2450: 1.0141830821001365,
 2500: 1.0147984891151114,
 2550: 1.0154143746190472,
 2600: 1.016030684026203,
 2650: 1.0166473730642127,
 2700: 1.01726440582756,
 2750: 1.0178817531984325,
 2800: 1.0184993915656237,
 2850: 1.019117301785228,
 2900: 1.0197354683374917,
 2950: 1.0203538786428,
 3000: 1.0209725225067605,
 3050: 1.021591391670027,
 3100: 1.022210479443081,
 3150: 1.0228297804099502,
 3200: 1.0234492901878447,
 3250: 1.0240690052321573,
 3300: 1.0246889226782727,
 3350: 1.0253090402132252,
 3400: 1.0259293559715896,
 3450: 1.0265498684510148,
 3500: 1.0271705764436976,
 3550: 1.0277914789807996,
 3600: 1.028412575287343,
 3650: 1.029033864745629,
 3700: 1.0296553468655554,
 3750: 1.0302770212605354,
 3800: 1.030898887627969,
 3850: 1.0315209457333863,
 3900: 1.0321431953976017,
 3950: 1.0327656364862765,
 4000: 1.0333882689014597,
 4050: 1.034011092574724,
 4100: 1.0346341074615966,
 4150: 1.0352573135370418,
 4200: 1.0358807107917962,
 4250: 1.0365042992293936,
 4300: 1.0371280788637534,
 4350: 1.0377520497172275,
 4400: 1.0383762118190054,
 4450: 1.039000565203834,
 4500: 1.039625109910967,
 4550: 1.0402498459833205,
 4600: 1.0408747734667836,
 4650: 1.0414998924096621,
 4700: 1.0421252028622252,
 4750: 1.0427507048763358,
 4800: 1.043376398505159,
 4850: 1.044002283802914,
 4900: 1.0446283608246802,
 4950: 1.0452546296262408,
 5000: 1.0458810902639497,
 5050: 1.046507742794629,
 5100: 1.0471345872754845,
 5150: 1.0477616237640346,
 5200: 1.0483888523180585,
 5250: 1.0490162729955461,
 5300: 1.049643885854664,
 5350: 1.0502716909537253,
 5400: 1.0508996883511648,
 5450: 1.05152787810552,
 5500: 1.0521562602754155,
 5550: 1.0527848349195477,
 5600: 1.0534136020966787,
 5650: 1.0540425618656233,
 5700: 1.0546717142852442,
 5750: 1.0553010594144476,
 5800: 1.0559305973121758,
 5850: 1.056560328037405,
 5900: 1.0571902516491432,
 5950: 1.0578203682064267,
 6000: 1.0584506777683185,
 6050: 1.059081180393906,
 6100: 1.059711876142301,
 6150: 1.0603427650726367,
 6200: 1.0609738472440693,
 6250: 1.061605122715776,
 6300: 1.0622365915469536,
 6350: 1.0628682537968204,
 6400: 1.0635001095246128,
 6450: 1.0641321587895876,
 6500: 1.0647644016510212,
 6550: 1.0653968381682075,
 6600: 1.0660294684004619,
 6650: 1.0666622924071154,
 6700: 1.0672953102475198,
 6750: 1.0679285219810435,
 6800: 1.0685619276670764,
 6850: 1.0691955273650242,
 6900: 1.069829321134311,
 6950: 1.0704633090343814,
 7000: 1.0710974911246964,
 7050: 1.0717318674647363,
 7100: 1.0723664381139986,
 7150: 1.073001203132001,
 7200: 1.0736361625782775,
 7250: 1.0742713165123812,
 7300: 1.0749066649938834,
 7350: 1.0755422080823744,
 7400: 1.0761779458374616,
 7450: 1.076813878318772,
 7500: 1.0774500055859488,
 7550: 1.0780863276986552,
 7600: 1.078722844716573,
 7650: 1.0793595566994019,
 7700: 1.0799964637068569,
 7750: 1.080633565798677,
 7800: 1.0812708630346148,
 7850: 1.081908355474444,
 7900: 1.0825460431779534,
 7950: 1.0831839262049545,
 8000: 1.0838220046152736,
 8050: 1.0844602784687565,
 8100: 1.0850987478252674,
 8150: 1.0857374127446895,
 8200: 1.0863762732869229,
 8250: 1.0870153295118872,
 8300: 1.0876545814795193,
 8350: 1.0882940292497754,
 8400: 1.0889336728826304,
 8450: 1.0895735124380765,
 8500: 1.0902135479761244,
 8550: 1.090853779556804,
 8600: 1.0914942072401623,
 8650: 1.0921348310862666,
 8700: 1.0927756511552005,
 8750: 1.0934166675070673,
 8800: 1.094057880201989,
 8850: 1.0946992893001042,
 8900: 1.0953408948615722,
 8950: 1.0959826969465687,
 9000: 1.0966246956152892,
 9050: 1.0972668909279473,
 9100: 1.0979092829447756,
 9150: 1.0985518717260234,
 9200: 1.0991946573319602,
 9250: 1.0998376398228729,
 9300: 1.1004808192590674,
 9350: 1.1011241957008688,
 9400: 1.1017677692086183,
 9450: 1.1024115398426786,
 9500: 1.1030555076634285,
 9550: 1.1036996727312665,
 9600: 1.104344035106609,
 9650: 1.104988594849892,
 9700: 1.1056333520215689,
 9750: 1.106278306682111,
 9800: 1.1069234588920098,
 9850: 1.107568808711775,
 9900: 1.108214356201934,
 9950: 1.108860101423033}


def get_bid_ask(strike):
    return bid_ask_approx[strike]


def round_params(params, n_signs=3):
    return [round(x, n_signs) for x in params]
