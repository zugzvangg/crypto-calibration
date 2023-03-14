import pandas as pd
from scipy import stats as sps
import numpy as np
import datetime


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
    # print(data_grouped)
    return data_grouped

# Newton-Raphsen
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

def process_data(data):
    # only options
    df = data.copy()
    df = df[(df["instrument"].str.endswith("C")) | (df["instrument"].str.endswith("P"))].sort_values("dt")
    df["type"] = np.where(df["instrument"].str.endswith("C"), "call", "put")
    
    perpetuals = data[data["instrument"].str.endswith("PERPETUAL")][["dt", "price"]].copy()
    perpetuals = perpetuals.rename(columns = {"price": "underlying_price"}).sort_values("dt")
    
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
    
    df["dt"] = pd.to_datetime(df["dt"])
    perpetuals["dt"] = pd.to_datetime(perpetuals["dt"])
    
    df = pd.merge_asof(df, perpetuals, on="dt",
                       tolerance=pd.Timedelta('7 minutes'),
                       direction='nearest',)
    
    df["timestamp"] = df["dt"].apply(unix_time_millis)
    df["expiration"] = df["instrument"].apply(get_normal_date)
    df = df.rename(columns = {"price": "mark_price"})
    
    
    return df
