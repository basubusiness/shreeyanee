import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy.optimize import minimize

st.set_page_config(page_title="Shreeyanee Edge System", layout="wide")

st.title("🧠 Shreeyanee Edge System")
st.caption("Allocation + Timing + Deployment → Transparent investing framework")

# =========================================================
# ------------------- CONFIG -------------------------------
# =========================================================

ASSETS = {
    "World Equity": {"return": 0.075, "vol": 0.16, "cat": "Equity"},
    "US Equity": {"return": 0.078, "vol": 0.17, "cat": "Equity"},
    "Emerging Markets": {"return": 0.080, "vol": 0.22, "cat": "Equity"},
    "Global Small Cap": {"return": 0.082, "vol": 0.19, "cat": "Equity"},
    "Global REIT": {"return": 0.060, "vol": 0.19, "cat": "Real"},
    "Gold": {"return": 0.065, "vol": 0.17, "cat": "Real"},
    "Euro Gov Bonds": {"return": 0.030, "vol": 0.06, "cat": "Bond"},
    "Corp Bonds": {"return": 0.035, "vol": 0.07, "cat": "Bond"},
    "Cash": {"return": 0.020, "vol": 0.01, "cat": "Cash"},
}

# =========================================================
# ------------------- HELPERS ------------------------------
# =========================================================

def build_corr(names):
    size = len(names)
    mat = np.full((size, size), 0.3)
    np.fill_diagonal(mat, 1.0)
    return pd.DataFrame(mat, index=names, columns=names)

def optimize_portfolio(names, target_r):
    df = pd.DataFrame(ASSETS).T.loc[names]
    rets = df["return"].values
    vols = df["vol"].values

    corr = build_corr(names).values
    cov = np.diag(vols) @ corr @ np.diag(vols)

    def objective(w):
        port_r = w @ rets
        port_v = np.sqrt(w.T @ cov @ w) + 1e-6
        return - (port_r / port_v)

    res = minimize(
        objective,
        np.ones(len(names))/len(names),
        bounds=[(0, 0.5)]*len(names),
        constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}]
    )

    w = res.x
    return w, w @ rets, np.sqrt(w.T @ cov @ w)

def get_data(symbol):
    try:
        df = yf.download(symbol, period="1y", progress=False, timeout=10)
        if df is None or df.empty:
            raise ValueError("No data")
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None

def get_fear_greed():
    try:
        url = "https://production.dataviz.cnn.io/index/feargreed/graphdata"
        res = requests.get(url, timeout=3)
        return float(res.json()["fear_and_greed"]["score"])
    except:
        return 50

# =========================================================
# ------------------- SIDEBAR ------------------------------
# =========================================================

with st.sidebar:
    st.header("Inputs")

    ticker = st.text_input("Ticker", "VOO")
    monthly = st.number_input("Monthly Investment", 500, 5000, 1000)
    target = st.slider("Target Return %", 3, 12, 7) / 100

# =========================================================
# ------------------- MAIN --------------------------------
# =========================================================
if "run" not in st.session_state:
    st.session_state.run = False

if st.button("Run System"):
    st.session_state.run = True

if st.session_state.run:

if st.button("Run System"):

    # ---------------- Allocation ----------------
    selected = list(ASSETS.keys())
    weights, port_r, port_v = optimize_portfolio(selected, target)

    st.subheader("📊 Strategic Allocation")
    df_alloc = pd.DataFrame({
        "Asset": selected,
        "Weight": weights
    })
    st.dataframe(df_alloc.style.format({"Weight":"{:.1%}"}))

    # ---------------- Market Data ----------------
    df = get_data(ticker)
    if df is None or "Close" not in df:
    st.error("Failed to load market data")
    st.stop()
    close = df["Close"]

    ma200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    rsi = float(100 - (100 / (1 + rs))).iloc[-1]

    price_change = close.pct_change().iloc[-1] * 100
    volatility = close.pct_change().rolling(20).std().iloc[-1] * 100

    fg = get_fear_greed()

    # ---------------- Timing ----------------
    if rsi < 35:
        state = "WAIT"
    elif price_change > volatility:
        state = "TRIGGER"
    else:
        state = "WATCH"

    st.subheader("⚡ Market Timing")
    st.write(f"State: **{state}**")
    st.caption("RSI + Price vs Volatility used for timing")

    # ---------------- Deployment ----------------
    if state == "WAIT":
        deploy = 0.5
    elif state == "WATCH":
        deploy = 1.0
    else:
        deploy = 1.75

    st.subheader("🎯 Deployment")
    st.write(f"Multiplier: **{deploy}x**")
    st.caption("Controls how fast you invest based on market conditions")

    # ---------------- Final Action ----------------
    invest_amount = monthly * deploy

    st.subheader("💡 Action")
    st.success(f"Invest this month: €{invest_amount:.0f}")

    # ---------------- Explanation ----------------
    with st.expander("🔍 How this works"):
        st.write("""
### System Layers

1. Allocation → decides where money should go  
2. Timing → detects market condition  
3. Deployment → adjusts how fast to invest  

---

### Key Principle
We do NOT change portfolio frequently.

We only change:
→ how aggressively we deploy capital

---

### Why this works

- Avoids buying during falling markets  
- Avoids missing recoveries  
- Keeps decisions simple and explainable  
        """)

    with st.expander("📚 References"):
        st.write("""
- Modern Portfolio Theory (Markowitz)
- RSI (Relative Strength Index)
- Volatility breakout logic
- Behavioral investing principles
        """)
