# Multi-Stock-Predictor-System

# Multi-Stock Movement Predictor

A machine learning app that predicts whether Indian stocks will move 
UP or DOWN in the next 30 days.

## Live Demo
[Click here](https://multi-stock-predictor-system-ikncetnsrcrzm25szumhdp.streamlit.app/)

## Stocks Covered
31 major Indian stocks including NIFTY 50, SBI, Reliance, TCS, and more.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Random Forest)
- yfinance (live data)
- Streamlit (deployment)
- Joblib (model persistence)

## Features Used
- return_1d, return_7d, return_30d
- MA_50, MA_200
- RSI (14 day)
- VIX_lag1, VIX_lag7
- CRUDE_lag7

## How to Run Locally
pip install -r requirements.txt
python data_fetch.py
python feature_engineering.py
python train_model.py
streamlit run app.py

## Disclaimer
This is a learning project. Not financial advice.
