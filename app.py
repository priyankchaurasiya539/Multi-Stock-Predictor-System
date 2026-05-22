import pandas as pd 
import joblib 
import os 
import numpy as np 
import yfinance as yf 
import streamlit as st

st.set_page_config(
    page_title="Multi Stock Predictor",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_models():
    models  = joblib.load("models/all_models.pkl")
    metrics = joblib.load("models/all_metrics.pkl")
    return models, metrics

all_models, all_metrics = load_models()

st.sidebar.title("📈 Multi-Stock Predictor")
page = st.sidebar.radio("Navigate", ["About", "Predict", "Model Info"])

# ── About ─────────────────────────────────────────────────────
if page == "About":
    st.title("Multi Stock Predictor System")
    st.markdown("""
    ### What does this app do?
    This app predicts whether a stock will go **UP** or **DOWN** 
    in the next 30 days using Machine Learning.
    
    ### Stocks Covered
    31 major Indian stocks including NIFTY 50, SBI, Reliance, TCS, and more.
    
    ### How it works
    - Fetches live data from Yahoo Finance
    - Calculates technical indicators (RSI, Moving Averages, Returns)
    - Uses a trained Random Forest model per stock
    - Predicts direction with confidence score
    
    ### Disclaimer
    ⚠️ This is a learning project. Not financial advice.
    """)

# ── Predict ───────────────────────────────────────────────────
elif page == "Predict":
    st.title("🔮 Stock Movement Predictor")

    STOCK_DICT = {
        "NIFTY 50":         "^NSEI",
        "SBI":              "SBIN.NS",
        "Reliance":         "RELIANCE.NS",
        "TCS":              "TCS.NS",
        "HDFC Bank":        "HDFCBANK.NS",
        "Infosys":          "INFY.NS",
        "Wipro":            "WIPRO.NS",
        "ICICI Bank":       "ICICIBANK.NS",
        "ITC":              "ITC.NS",
        "L&T":              "LT.NS",
        "HUL":              "HINDUNILVR.NS",
        "Axis Bank":        "AXISBANK.NS",
        "ONGC":             "ONGC.NS",
        "Dr Reddys":        "DRREDDY.NS",
        "Sun Pharma":       "SUNPHARMA.NS",
        "Tata Steel":       "TATASTEEL.NS",
        "HCL Tech":         "HCLTECH.NS",
        "M&M":              "M&M.NS",
        "NTPC":             "NTPC.NS",
        "Asian Paints":     "ASIANPAINT.NS",
        "Hero MotoCorp":    "HEROMOTOCO.NS",
        "Cipla":            "CIPLA.NS",
        "Grasim":           "GRASIM.NS",
        "UltraTech Cement": "ULTRACEMCO.NS",
        "Hindalco":         "HINDALCO.NS",
        "JSW Steel":        "JSWSTEEL.NS",
        "BPCL":             "BPCL.NS",
        "IOC":              "IOC.NS",
        "Maruti Suzuki":    "MARUTI.NS",
        "Kotak Mahindra":   "KOTAKBANK.NS",
        "Titan":            "TITAN.NS",
    }

    selected_stock = st.selectbox("Select a Stock", list(STOCK_DICT.keys()))
    ticker = STOCK_DICT[selected_stock]

    if st.button("Predict"):
        with st.spinner("Fetching live data and predicting..."):
            try:
                # fetch live data
                df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df[["Close"]].copy()
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                df.dropna(inplace=True)

                # returns
                df["return_1d"]  = df["Close"].pct_change(1)
                df["return_7d"]  = df["Close"].pct_change(7)
                df["return_30d"] = df["Close"].pct_change(30)

                # moving averages
                df["MA_50"]  = df["Close"].rolling(window=50).mean()
                df["MA_200"] = df["Close"].rolling(window=200).mean()

                # RSI
                delta = df["Close"].diff()
                gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss  = -delta.where(delta < 0, 0).rolling(window=14).mean()
                df["RSI"] = 100 - (100 / (1 + gain / loss))

                # VIX and CRUDE
                vix   = yf.download("^INDIAVIX", period="2y", auto_adjust=True, progress=False)
                crude = yf.download("CL=F",      period="2y", auto_adjust=True, progress=False)

                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                if isinstance(crude.columns, pd.MultiIndex):
                    crude.columns = crude.columns.get_level_values(0)

                vix   = vix[["Close"]].rename(columns={"Close": "VIX"})
                crude = crude[["Close"]].rename(columns={"Close": "CRUDE"})

                df = df.join(vix,   how="left")
                df = df.join(crude, how="left")
                df["VIX"]   = df["VIX"].ffill()
                df["CRUDE"] = df["CRUDE"].ffill()

                df["VIX_lag1"]   = df["VIX"].shift(1)
                df["VIX_lag7"]   = df["VIX"].shift(7)
                df["CRUDE_lag7"] = df["CRUDE"].shift(7)

                df.drop(columns=["VIX", "CRUDE"], inplace=True)
                df.dropna(inplace=True)

                # predict
                FEATURE_COLS = ["return_1d", "return_7d", "return_30d",
                                "MA_50", "MA_200", "RSI",
                                "VIX_lag1", "VIX_lag7", "CRUDE_lag7"]

                latest     = df[FEATURE_COLS].iloc[[-1]]
                model      = all_models[selected_stock]
                prediction = model.predict(latest)[0]
                confidence = model.predict_proba(latest)[0][prediction]

                # result
                st.subheader(f"Prediction for {selected_stock}")
                if prediction == 1:
                    st.success(f"📈 UP — Confidence: {confidence:.1%}")
                else:
                    st.error(f"📉 DOWN — Confidence: {confidence:.1%}")

                if confidence < 0.55:
                    st.warning("⚠️ Low confidence — prediction may be unreliable")

                # indicators
                st.subheader("Current Indicators")
                col1, col2, col3 = st.columns(3)
                col1.metric("Last Close", f"₹{df['Close'].iloc[-1]:.2f}")
                col2.metric("RSI",        f"{df['RSI'].iloc[-1]:.1f}")
                col3.metric("MA 50",      f"₹{df['MA_50'].iloc[-1]:.2f}")

            except Exception as e:
                st.error(f"Error: {e}")

# ── Model Info ────────────────────────────────────────────────
elif page == "Model Info":
    st.title("📊 Model Performance")

    st.markdown("""
    Each stock has its own trained Random Forest model.
    Training data: 2008 - 2024 | Test size: 20%
    """)

    metrics_df = pd.DataFrame({
        "Stock":    list(all_metrics.keys()),
        "Accuracy": [f"{v:.1%}" for v in all_metrics.values()]
    })

    st.dataframe(metrics_df, use_container_width=True)

    avg = sum(all_metrics.values()) / len(all_metrics)
    st.metric("Average Accuracy", f"{avg:.1%}")

st.info("ℹ️ Accuracy below 55% indicates this stock is harder to predict with technical indicators alone.")