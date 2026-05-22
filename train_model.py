import pandas as pd 
import numpy as np
import joblib
import os 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


FEATURES_DIR = "data/features"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR , exist_ok=True)

FEATURE_COLS = ["return_1d" , "return_7d" , "return_30d" , "MA_50" , "MA_200" , "RSI" , "VIX_lag1" , "VIX_lag7" , "CRUDE_lag7"]

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

def train_stock(name, ticker):
    filename = ticker.replace("^", "").replace("=", "_")
    filepath = os.path.join(FEATURES_DIR, f"{filename}_features.csv")

    if not os.path.exists(filepath):
        print(f"❌ {name} — features file not found")
        return None, None

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    X = df[FEATURE_COLS]
    y = df["target"]
    X_train , X_test , y_train , y_test = train_test_split(X , y  , random_state= 42 , test_size=0.2 , shuffle=False)

    model = RandomForestClassifier(
        n_estimators= 100 ,
        max_depth=10 , 
        random_state= 42 
    )
    model.fit(X_train , y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test , pred)

    print("Accuracy : " , accuracy)
    return model , accuracy

if __name__ == "__main__":
    all_models  = {}
    all_metrics = {}

    for name, ticker in STOCK_DICT.items():
        model, accuracy = train_stock(name, ticker)
        if model is not None:
            all_models[name]  = model
            all_metrics[name] = accuracy

    joblib.dump(all_models,  "models/all_models.pkl")
    joblib.dump(all_metrics, "models/all_metrics.pkl")

    print(f"\n✅ {len(all_models)} models saved.")