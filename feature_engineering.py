import pandas as pd 
import numpy as np 
import os 

STOCK_DICT = {
    "NIFTY 50":           "^NSEI",
    "SBI":                "SBIN.NS",
    "Reliance":           "RELIANCE.NS",
    "TCS":                "TCS.NS",
    "HDFC Bank":          "HDFCBANK.NS",
    "Infosys":            "INFY.NS",
    "Wipro":              "WIPRO.NS",
    "ICICI Bank":         "ICICIBANK.NS",
    "ITC":                "ITC.NS",
    "L&T":                "LT.NS",
    "HUL":                "HINDUNILVR.NS",
    "Axis Bank":          "AXISBANK.NS",
    "ONGC":               "ONGC.NS",
    "Dr Reddys":          "DRREDDY.NS",
    "Sun Pharma":         "SUNPHARMA.NS",
    "Tata Steel":         "TATASTEEL.NS",
    "HCL Tech":           "HCLTECH.NS",
    "M&M":                "M&M.NS",
    "NTPC":               "NTPC.NS",
    "Asian Paints":       "ASIANPAINT.NS",
    "Hero MotoCorp":      "HEROMOTOCO.NS",
    "Cipla":              "CIPLA.NS",
    "Grasim":             "GRASIM.NS",
    "UltraTech Cement":   "ULTRACEMCO.NS",
    "Hindalco":           "HINDALCO.NS",
    "JSW Steel":          "JSWSTEEL.NS",
    "BPCL":               "BPCL.NS",
    "IOC":                "IOC.NS",
    "Maruti Suzuki":      "MARUTI.NS",
    "Kotak Mahindra":     "KOTAKBANK.NS",
    "Titan":              "TITAN.NS",
}

DATA_DIR     = "data"
FEATURES_DIR = "data/features"
os.makedirs(FEATURES_DIR, exist_ok=True)


# FIX 1 — .where() needs 0 as second argument, not nothing
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss  = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def load_external():
    vix   = pd.read_csv("data/INDIAVIX.csv", header=[0,1,2], index_col=0)
    crude = pd.read_csv("data/CL_F.csv",     header=[0,1,2], index_col=0)
    vix.columns   = vix.columns.get_level_values(0)
    crude.columns = crude.columns.get_level_values(0)
    vix.index   = pd.to_datetime(vix.index,   errors="coerce")
    crude.index = pd.to_datetime(crude.index, errors="coerce")
    vix   = vix[["Close"]].rename(columns={"Close": "VIX"})
    crude = crude[["Close"]].rename(columns={"Close": "CRUDE"})
    return vix, crude


def build_features(name, ticker):

    filename = ticker.replace("^", "").replace("=", "_")
    filepath = os.path.join(DATA_DIR, f"{filename}.csv")

    if not os.path.exists(filepath):
        print(f"❌ {name} — File not found")
        return None

    df = pd.read_csv(filepath, header=[0,1,2], index_col=0)
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    df = df[["Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(inplace=True)

    # FIX 2 — start from 2008 because VIX data starts from 2008
    df = df[df.index >= "2008-01-01"]

    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_7d"]  = df["Close"].pct_change(7)
    df["return_30d"] = df["Close"].pct_change(30)

    df["MA_50"]  = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()

    df["RSI"] = compute_rsi(df["Close"])

    vix, crude = load_external()
    df = df.join(vix,   how="left")
    df = df.join(crude, how="left")
    df["VIX"]   = df["VIX"].ffill()
    df["CRUDE"] = df["CRUDE"].ffill()

    df["VIX_lag1"]   = df["VIX"].shift(1)
    df["VIX_lag7"]   = df["VIX"].shift(7)
    df["CRUDE_lag7"] = df["CRUDE"].shift(7)

    df["target"] = (df["Close"].shift(-30) > df["Close"]).astype(int)

    df.drop(columns=["VIX", "CRUDE"], inplace=True)

    feature_cols = ["return_1d", "return_7d", "return_30d",
                    "MA_50", "MA_200", "RSI",
                    "VIX_lag1", "VIX_lag7", "CRUDE_lag7", "target"]
    df.dropna(subset=feature_cols, inplace=True)

    out_path = os.path.join(FEATURES_DIR, f"{filename}_features.csv")
    df.to_csv(out_path)
    print(f"✅ {name} — {len(df)} rows saved")


if __name__ == "__main__":
    print("=" * 55)
    print("Building Features for All Stocks...")
    print("=" * 55)
    for name, ticker in STOCK_DICT.items():
        build_features(name, ticker)
    print("\n✅ Feature engineering complete.")