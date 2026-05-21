import yfinance as yf
import os

# ── Stock Dictionary ──────────────────────────────────────────
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

# ── External Feature Tickers ──────────────────────────────────
EXTRA_DICT = {
    "VIX":   "^INDIAVIX",
    "CRUDE": "CL=F",
}

# ── Settings ──────────────────────────────────────────────────
START_DATE = "2005-01-01"
DATA_DIR   = "data"

os.makedirs(DATA_DIR, exist_ok=True)

# ── Download Function ─────────────────────────────────────────
def download_stock(name, ticker):
    try:
        df = yf.download(ticker, start=START_DATE, auto_adjust=True, progress=False)
        if df.empty:
            print(f"❌ {name} ({ticker}) — No data found")
            return
        filename = ticker.replace("^", "").replace("=", "_")
        filepath = os.path.join(DATA_DIR, f"{filename}.csv")
        df.to_csv(filepath)
        print(f"✅ {name} ({ticker}) — {len(df)} rows saved")
    except Exception as e:
        print(f"❌ {name} ({ticker}) — Error: {e}")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Downloading Stock Data...")
    print("=" * 50)

    for name, ticker in STOCK_DICT.items():
        download_stock(name, ticker)

    print("\nDownloading External Features (VIX, Crude)...")
    print("=" * 50)

    for name, ticker in EXTRA_DICT.items():
        download_stock(name, ticker)

    print("\n✅ All downloads complete. Check data/ folder.")