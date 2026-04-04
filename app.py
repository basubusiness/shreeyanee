# ═══════════════════════════════════════════════════════════════════
# Market Decision Engine v11.0 — Streamlit Port
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import threading
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta

try:
    import justetf_scraping
    JUSTETF_AVAILABLE = True
except ImportError:
    JUSTETF_AVAILABLE = False

try:
    import financedatabase as fd
    FD_AVAILABLE = True
except ImportError:
    FD_AVAILABLE = False

try:
    import pandas_datareader as pdr
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False

APP_VERSION = "v11.0"
BUILD_TIME = datetime.utcnow().strftime("%d %b %H:%M UTC")

# ───────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Market Decision Engine",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
  --sig-buy:   #0d9488;
  --sig-watch: #0284c7;
  --sig-sell:  #dc2626;
  --sig-avoid: #d97706;
  --sig-wait:  #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid #e2e6ed;
    border-radius: 6px;
    padding: 12px 16px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8f9fc !important;
    border-right: 1px solid #e2e6ed;
}

/* Signal badges */
.badge-buy   { background:#f0fdfa; color:#0d9488; border:1px solid #99f6e4; padding:2px 8px; border-radius:4px; font-weight:600; font-size:12px; }
.badge-watch { background:#f0f9ff; color:#0284c7; border:1px solid #bae6fd; padding:2px 8px; border-radius:4px; font-weight:600; font-size:12px; }
.badge-sell  { background:#fff5f5; color:#dc2626; border:1px solid #fecaca; padding:2px 8px; border-radius:4px; font-weight:600; font-size:12px; }
.badge-avoid { background:#fffbeb; color:#d97706; border:1px solid #fde68a; padding:2px 8px; border-radius:4px; font-weight:600; font-size:12px; }
.badge-wait  { background:#f8fafc; color:#64748b; border:1px solid #e2e8f0; padding:2px 8px; border-radius:4px; font-size:12px; }

.stDataFrame { font-family: 'DM Mono', monospace !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────
# IN-MEMORY CACHE  (st.cache_resource = global, shared across sessions)
# ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_cache_store():
    """Global TTL cache dict shared across all sessions."""
    return {}

@st.cache_resource
def _get_cache_lock():
    return threading.Lock()

_DISK_CACHE_FILE = "/tmp/mde_fmp_cache.json"

def cache_get(key):
    store = _get_cache_store()
    lock  = _get_cache_lock()
    with lock:
        entry = store.get(key)
        if entry is None:
            return None
        val, expires = entry
        if expires and time.time() > expires:
            del store[key]
            return None
        return val

def cache_set(key, val, ttl=None):
    store = _get_cache_store()
    lock  = _get_cache_lock()
    expires = (time.time() + ttl) if ttl else None
    with lock:
        store[key] = (val, expires)
    if str(key).startswith("fmp_"):
        threading.Thread(target=_save_disk_cache, daemon=True).start()

def _save_disk_cache():
    store = _get_cache_store()
    lock  = _get_cache_lock()
    try:
        with lock:
            fmp = {k: v for k, v in store.items() if str(k).startswith("fmp_")}
        with open(_DISK_CACHE_FILE, "w") as f:
            json.dump(fmp, f)
    except Exception:
        pass

def _load_disk_cache():
    store = _get_cache_store()
    lock  = _get_cache_lock()
    try:
        if os.path.exists(_DISK_CACHE_FILE):
            with open(_DISK_CACHE_FILE) as f:
                data = json.load(f)
            now = time.time()
            with lock:
                for k, (v, exp) in data.items():
                    if exp is None or exp > now:
                        store[k] = (v, exp)
    except Exception:
        pass


# ───────────────────────────────────────────────────────────────────
# UNIVERSE LOADERS  (cached globally — loaded once)
# ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_base_universe():
    csv_path = os.path.join(os.path.dirname(__file__), "universe.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for col in ["country","sector","name","category_group","category",
                        "currency","yf_symbol","yf_suffix","isin"]:
                if col not in df.columns:
                    df[col] = ""
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].fillna("").astype(str).str.strip()
            df = df[df["ticker"].str.match(r"^[A-Z0-9]{2,6}$")]
            df = df.drop_duplicates(subset=["ticker","type"], keep="first")
            return df
        except Exception as e:
            st.warning(f"universe.csv load failed: {e}")
    if FD_AVAILABLE:
        try:
            etfs     = fd.ETFs().select().copy()
            equities = fd.Equities().select().copy()
            etfs["type"]     = "ETF"
            equities["type"] = "Stock"
            df = pd.concat([etfs, equities]).reset_index()
            df.rename(columns={df.columns[0]: "ticker"}, inplace=True)
            for col in ["country","sector","name","category_group","category",
                        "currency","yf_symbol","yf_suffix","isin"]:
                if col not in df.columns:
                    df[col] = ""
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].fillna("").astype(str).str.strip()
            df = df[df["ticker"].str.match(r"^[A-Z]{2,5}$")]
            return df.drop_duplicates(subset=["ticker","type"], keep="first")
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_resource
def load_justetf():
    if not JUSTETF_AVAILABLE:
        return pd.DataFrame()
    try:
        df = justetf_scraping.load_overview().reset_index()
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "ticker":                             col_map[c] = "ticker"
            elif cl == "isin":                             col_map[c] = "isin"
            elif "name" in cl and "index" not in cl:      col_map[c] = "jname"
            elif "domicile" in cl:                         col_map[c] = "domicile"
            elif cl in ("ter","expense_ratio"):            col_map[c] = "ter"
            elif "distribution" in cl or "policy" in cl:  col_map[c] = "dist_policy"
            elif "fund_size" in cl or "aum" in cl:        col_map[c] = "fund_size_eur"
            elif "replication" in cl:                      col_map[c] = "replication"
            elif "strategy" in cl:                         col_map[c] = "strategy"
        df = df.rename(columns=col_map)
        keep = [c for c in ["ticker","isin","jname","domicile","ter",
                             "dist_policy","fund_size_eur","replication","strategy"]
                if c in df.columns]
        df = df[keep].copy()
        if "dist_policy" not in df.columns and "jname" in df.columns:
            def _infer_dist(name):
                n = str(name).lower()
                if any(x in n for x in [" acc","(acc)","accumulating","capitalisation"]):
                    return "Accumulating"
                if any(x in n for x in [" dist","(dist)"," inc","distributing","dividend"]):
                    return "Distributing"
                return "Accumulating"
            df["dist_policy"] = df["jname"].apply(_infer_dist)
        df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
        df = df[df["ticker"].str.match(r"^[A-Z0-9]{1,6}$")]
        return df.drop_duplicates(subset=["ticker"], keep="first")
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource
def load_signals():
    path = os.path.join(os.path.dirname(__file__), "signals.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        def _bad_ticker(t):
            t = str(t).strip()
            if len(t) < 2: return True
            if t[-1] in ('W','U','R') and len(t) >= 4: return True
            if t.endswith(('WS','WT','WI','WD')): return True
            return False
        df = df[~df["ticker"].apply(_bad_ticker)]
        if "name" in df.columns:
            spac_kw = ["acquisition corp","blank check","special purpose","spac"]
            lev_kw  = ["2x","3x","4x","-1x","-2x","-3x","leveraged","inverse daily","ultrashort"]
            df = df[~df["name"].str.lower().apply(lambda n: any(k in n for k in spac_kw + lev_kw))]
        if "price" in df.columns:
            df = df[(df["price"].isna()) | (df["price"] >= 1.00)]
        if "dist_ma200" in df.columns:
            df = df[(df["dist_ma200"].isna()) | (df["dist_ma200"] > -95)]
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False).drop_duplicates(
                subset=["ticker"], keep="first").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


# Load globals once at startup
_load_disk_cache()
universe  = load_base_universe()
jetf_df   = load_justetf()
signals_df_global = load_signals()   # use getter below so session can override

def get_signals_df():
    """Return signals_df, preferring session-level live-updated version."""
    return st.session_state.get("signals_df", signals_df_global)

def update_signals_df(new_row_dict):
    """Write a fresh signal row into session-level signals_df."""
    sdf = get_signals_df().copy()
    if not sdf.empty and "ticker" in sdf.columns:
        sdf = sdf[sdf["ticker"] != new_row_dict["ticker"]]
    sdf = pd.concat([pd.DataFrame([new_row_dict]), sdf], ignore_index=True)
    st.session_state["signals_df"] = sdf

# Build name/ISIN lookup
_name_lookup = {}
if not universe.empty and "name" in universe.columns:
    for _, row in universe.iterrows():
        t = str(row.get("ticker","")).strip()
        n = str(row.get("name","")).strip()
        if t and n not in ("","nan","None"):
            _name_lookup[t] = n
if not jetf_df.empty:
    for _, row in jetf_df.iterrows():
        t = str(row.get("ticker","")).strip()
        n = str(row.get("jname","")).strip()
        i = str(row.get("isin","")).strip() if pd.notna(row.get("isin","")) else ""
        if t and n not in ("","nan","None"):
            _name_lookup[t] = (n, i)


# ───────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ───────────────────────────────────────────────────────────────────

def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs    = gain / loss
    rs[loss == 0] = np.inf
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema12  = prices.ewm(span=12, adjust=False).mean()
    ema26  = prices.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def linear_slope(series, window=10):
    y = series.tail(window).values
    if len(y) < window:
        return 0.0
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope / (abs(y.mean()) + 1e-9))


# ───────────────────────────────────────────────────────────────────
# LIVE DATA
# ───────────────────────────────────────────────────────────────────

def get_live_vix():
    cached = cache_get("vix")
    if cached is not None:
        return cached
    try:
        d = flatten_df(yf.Ticker("^VIX").history(period="5d", auto_adjust=True, timeout=15))
        if not d.empty and "Close" in d.columns:
            val = float(d["Close"].dropna().iloc[-1])
            cache_set("vix", val, ttl=300)
            return val
    except Exception:
        pass
    return 20.0

def get_fg_index():
    cached = cache_get("fg")
    if cached is not None:
        return cached
    headers = {"User-Agent": "Mozilla/5.0"}
    today = date.today().strftime("%Y-%m-%d")
    for url in [
        "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
        f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}",
    ]:
        try:
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                fg = r.json().get("fear_and_greed", {})
                if "score" in fg:
                    s     = round(float(fg["score"]))
                    label = fg.get("rating","").replace("_"," ").title()
                    result = (s, f"{label} (CNN)")
                    cache_set("fg", result, ttl=1800)
                    return result
        except Exception:
            continue
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", headers=headers, timeout=6)
        if r.status_code == 200:
            e = r.json().get("data", [{}])[0]
            s     = round(float(e.get("value", 50)))
            label = e.get("value_classification","").replace("_"," ").title()
            result = (s, f"{label} (alt.me)")
            cache_set("fg", result, ttl=1800)
            return result
    except Exception:
        pass
    return (50, "Unavailable")


# ───────────────────────────────────────────────────────────────────
# PRICE FETCH
# ───────────────────────────────────────────────────────────────────

def _fetch_stooq(symbol):
    if not PDR_AVAILABLE:
        return pd.DataFrame()
    try:
        stooq_sym = symbol
        if symbol.endswith(".L"):    stooq_sym = symbol.replace(".L", ".UK")
        elif symbol.endswith(".AS"): stooq_sym = symbol.replace(".AS", ".NL")
        elif symbol.endswith(".PA"): stooq_sym = symbol.replace(".PA", ".FR")
        elif symbol.endswith(".MI"): stooq_sym = symbol.replace(".MI", ".IT")
        elif symbol.endswith(".SW"): stooq_sym = symbol.replace(".SW", ".CH")
        elif "." not in symbol:      stooq_sym = symbol + ".US"
        end   = pd.Timestamp.today()
        start = end - pd.Timedelta(days=800)
        df = pdr.get_data_stooq(stooq_sym, start=start, end=end)
        if df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        df = df.sort_index()
        return flatten_df(df) if len(df["Close"].dropna()) >= 30 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_justetf_chart(isin):
    if not isin or len(str(isin)) != 12 or not JUSTETF_AVAILABLE:
        return pd.DataFrame()
    try:
        df = justetf_scraping.load_chart(str(isin), unclosed=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if "quote" in df.columns:
            df = df.rename(columns={"quote": "Close"})
        elif "quote_with_dividends" in df.columns:
            df = df.rename(columns={"quote_with_dividends": "Close"})
        if "Close" not in df.columns:
            return pd.DataFrame()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=800)
        df = df[df.index >= cutoff]
        return df if len(df["Close"].dropna()) >= 30 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_ticker_data(ticker, isin=None, force_refresh=False):
    if not ticker or ticker.startswith("$") or len(ticker) < 2:
        return None
    key = f"tick_{ticker}"
    if force_refresh:
        store = _get_cache_store()
        lock  = _get_cache_lock()
        base  = ticker.split(".")[0]
        with lock:
            for k in list(store.keys()):
                if k in (key, f"sfx_{ticker}", f"resolved_{ticker}") \
                        or k.startswith(f"yfund_{ticker}") \
                        or k.startswith(f"conv_{ticker}") \
                        or (k.startswith("fmp_") and f"/{base}" in k):
                    store.pop(k, None)
    else:
        cached = cache_get(key)
        if cached is not None:
            return None if cached == "FAILED" else cached

    try:
        known_sfx    = cache_get(f"sfx_{ticker}")
        resolved_sym = cache_get(f"resolved_{ticker}")
        universe_sfx = None

        if known_sfx is None and resolved_sym is None and not universe.empty and "yf_symbol" in universe.columns:
            rows = universe[universe["ticker"] == ticker]
            if not rows.empty:
                yf_sym = str(rows.iloc[0].get("yf_symbol","")).strip()
                yf_sfx = str(rows.iloc[0].get("yf_suffix","")).strip()
                if yf_sym and yf_sym not in ("","nan","None"):
                    if yf_sfx.startswith("→"):
                        resolved_sym = yf_sym
                    else:
                        universe_sfx = "" if yf_sfx in ("","nan","None") else yf_sfx

        known_sfx    = cache_get(f"sfx_{ticker}")
        resolved_sym = resolved_sym or cache_get(f"resolved_{ticker}")

        def _fetch_history(sym):
            try:
                df = flatten_df(yf.Ticker(sym).history(period="1y", auto_adjust=True, timeout=15))
                if not df.empty and "Close" in df.columns and len(df["Close"].dropna()) >= 30:
                    return df
            except Exception:
                pass
            try:
                end   = date.today()
                start = end - timedelta(days=400)
                df = flatten_df(yf.Ticker(sym).history(start=str(start), end=str(end), auto_adjust=True, timeout=15))
                if not df.empty and "Close" in df.columns and len(df["Close"].dropna()) >= 30:
                    return df
            except Exception:
                pass
            return pd.DataFrame()

        def _valid(df):
            return not df.empty and "Close" in df.columns and len(df["Close"].dropna()) >= 30

        _attempts = []
        if resolved_sym:
            _attempts.append(("resolved", resolved_sym))
        elif known_sfx is not None:
            _attempts.append(("cached_sfx", ticker + known_sfx))
        if universe_sfx is not None and universe_sfx != "":
            _univ_sym = ticker + universe_sfx
            if not any(s == _univ_sym for _, s in _attempts):
                _attempts.append(("universe", _univ_sym))
        if not any(s == ticker for _, s in _attempts):
            _attempts.append(("bare", ticker))

        df = pd.DataFrame()
        used_sym = ticker
        for _src, _sym in _attempts:
            df = _fetch_stooq(_sym)
            if _valid(df):
                used_sym = _sym
                break
            df = _fetch_history(_sym)
            if _valid(df):
                used_sym = _sym
                break

        if _valid(df):
            cache_set(f"sfx_{ticker}", used_sym[len(ticker):], ttl=86400*7)

        # Suffix scan for ETFs
        if not _valid(df):
            sfx_list = [".DE",".DU",".L",".AS",".SG"] if isin else [".DE",".DU",".L",".AS",".PA",".MI",".SW",".F"]
            for sfx in sfx_list:
                df2 = _fetch_history(ticker + sfx)
                if _valid(df2):
                    df = df2
                    used_sym = ticker + sfx
                    cache_set(f"sfx_{ticker}", sfx, ttl=86400*7)
                    break

        # ISIN search / justETF fallback
        if not _valid(df) and isin:
            df = fetch_justetf_chart(isin)
        if not _valid(df):
            cache_set(key, "FAILED", ttl=300)
            return None

        close = df["Close"].dropna()
        if len(close) < 30:
            cache_set(key, "FAILED", ttl=3600)
            return None

        price = float(close.iloc[-1])
        if price < 0.10:
            return None

        ma50_raw  = close.rolling(50).mean().iloc[-1]
        ma200_raw = close.rolling(200).mean().iloc[-1]
        ma50  = float(ma50_raw)  if pd.notna(ma50_raw)  else float(price)
        ma200 = float(ma200_raw) if pd.notna(ma200_raw) else float(price)

        rsi_s     = calculate_rsi(close)
        rsi       = float(rsi_s.iloc[-1])
        if not (1 < rsi < 99):
            return None

        macd_l, macd_sig, macd_h = calculate_macd(close)
        rsi_slope    = linear_slope(rsi_s.dropna(), window=5)
        dist_ma      = ((price - ma200) / ma200) * 100
        dist_52h     = ((price - float(close.tail(252).max())) / float(close.tail(252).max())) * 100
        vol_pct      = float(close.pct_change().rolling(20).std().iloc[-1] * 100)
        price_slope  = linear_slope(close, window=10)
        conf  = min(abs(dist_ma)/20, 1)*0.5 + min(abs(50-rsi)/50, 1)*0.5
        conf *= 1 - min(vol_pct/5, 0.5)

        ma200_s_full = close.rolling(200).mean()
        _tail = ma200_s_full.dropna().tail(60)
        if len(_tail) >= 20:
            _y = _tail.values
            _slope_raw = np.polyfit(np.arange(len(_y)), _y, 1)[0]
            ma200_slope = float(_slope_raw / (abs(_y.mean()) + 1e-9))
        else:
            ma200_slope = 0.0

        result = dict(
            price=price, ma50=ma50, ma200=ma200,
            rsi=rsi, rsi_slope=rsi_slope,
            macd=float(macd_l.iloc[-1]), macd_signal=float(macd_sig.iloc[-1]),
            macd_hist=float(macd_h.iloc[-1]),
            dist_ma=dist_ma, dist_52h=dist_52h,
            vol=vol_pct, price_slope=price_slope,
            trend_down_strong=(price_slope < -0.003),
            confidence=conf,
            ma200_slope=ma200_slope,
            close=close, ma50_s=close.rolling(50).mean(),
            ma200_s=ma200_s_full,
            rsi_s=rsi_s, macd_l=macd_l, macd_sig=macd_sig, macd_h=macd_h,
        )
        cache_set(key, result, ttl=3600)
        return result
    except Exception:
        return None


# ───────────────────────────────────────────────────────────────────
# SIGNAL SCORING
# ───────────────────────────────────────────────────────────────────

def compute_action_score(dist_ma, rsi, rsi_slope, macd, macd_signal, macd_hist,
                         conf, vol, price_slope, ma200_slope, fund_delta=0.0, is_etf=False):
    """
    Unified action + score computation used by both scanner and Deep Dive.
    Mirrors build_signals.py compute_technicals logic + MA200 slope from app.py.
    """
    macd_bull  = macd > macd_signal
    macd_accel = macd_hist > 0
    rsi_rising = rsi_slope > 0
    trend_down = price_slope < -0.003

    ma200_trend = (
        "rising"  if ma200_slope >  0.001 else
        "falling" if ma200_slope < -0.001 else
        "flat"
    )
    ma200_steep_down = ma200_slope < -0.003

    # Structural collapse guard
    if dist_ma < -55:
        return "AVOID", -2.0, macd_bull, macd_accel, rsi_rising, ma200_trend, ma200_steep_down

    knife_thr  = -25
    is_knife   = (dist_ma < knife_thr) and (15 < rsi < 35) and trend_down
    reversal   = is_knife and macd_bull and rsi_rising and rsi > 25 and not trend_down

    if is_knife and not reversal:
        action = "AVOID"
    elif (-40 < dist_ma < -10) and (30 <= rsi < 48) and macd_bull and macd_accel:
        action = "BUY"
    elif (-40 < dist_ma < -5)  and rsi < 48 and (macd_bull or rsi_rising):
        action = "WATCH"
    elif dist_ma < -40 and rsi > 28 and macd_bull and rsi_rising:
        action = "WATCH"
    elif dist_ma > 10  and rsi > 70:
        action = "SELL"
    else:
        action = "WAIT"

    sweet2    = -20
    dist_pen  = max(0, (-dist_ma - 40) / 30)
    dist_s    = max(0, 1 - abs(dist_ma - sweet2) / 25) * (1 - min(dist_pen, 1))
    rsi_s2    = max((50 - rsi) / 50, 0) if rsi >= 25 else 0
    score     = (dist_s*0.35 + rsi_s2*0.25 +
                 (0.20 if macd_bull  else 0) +
                 (0.10 if macd_accel else 0) +
                 conf*0.10)
    if action == "AVOID": score -= 2
    if is_knife:          score -= 0.3

    score += fund_delta

    # Fundamental-driven action adjustments
    if action == "WATCH" and fund_delta >= 0.08:  action = "BUY"
    if action == "BUY"   and fund_delta <= -0.08: action = "WATCH"

    # MA200 slope adjustments
    if ma200_steep_down:
        score -= 0.10
        if action == "BUY":   action = "WATCH"
        if action == "WATCH": action = "WAIT"
    elif ma200_trend == "falling":
        score -= 0.05
        if action == "BUY":   action = "WATCH"
    elif ma200_trend == "rising":
        score += 0.05

    return action, round(score, 4), macd_bull, macd_accel, rsi_rising, ma200_trend, ma200_steep_down

def analyse_ticker(ticker, risk_mult=1.0, isin=None):
    raw = fetch_ticker_data(ticker, isin=isin)
    if raw is None:
        return None
    action, score, macd_bull, macd_accel, rsi_rising, ma200_trend, ma200_steep = compute_action_score(
        dist_ma=raw["dist_ma"], rsi=raw["rsi"], rsi_slope=raw["rsi_slope"],
        macd=raw["macd"], macd_signal=raw["macd_signal"], macd_hist=raw["macd_hist"],
        conf=raw["confidence"], vol=raw["vol"], price_slope=raw["price_slope"],
        ma200_slope=raw["ma200_slope"]
    )
    strength = "Strong" if raw["confidence"] > 0.7 else "Medium" if raw["confidence"] > 0.4 else "Weak"
    is_knife  = (raw["dist_ma"] < -25) and (raw["rsi"] < 35) and raw["trend_down_strong"]
    reversal  = is_knife and macd_bull and rsi_rising

    row = {
        "Ticker": ticker,
        "Price":  round(raw["price"], 2),
        "MA200":  round(raw["ma200"], 2),
        "Dist%":  round(raw["dist_ma"], 1),
        "52W%":   round(raw["dist_52h"], 1),
        "RSI":    round(raw["rsi"], 1),
        "RSI↗":   "↗" if rsi_rising else "↘",
        "MACD":   "▲ Bull" if macd_bull else "▼ Bear",
        "MACD⚡": "⚡" if macd_accel else "—",
        "Vol%":   round(raw["vol"], 2),
        "Conf":   round(raw["confidence"], 2),
        "Action": action,
        "Signal": f"{action} ({strength})",
        "Knife":  "⚠️" if (is_knife and not reversal) else ("✅Rev" if reversal else ""),
        "Score":  round(score, 4),
    }
    # Write back to session signals_df
    update_signals_df({
        "ticker": ticker, "data_source": "live_scan",
        "price": raw["price"], "ma200": raw["ma200"],
        "dist_ma200": raw["dist_ma"], "rsi": raw["rsi"],
        "rsi_rising": int(rsi_rising), "macd_bull": int(macd_bull),
        "macd_accel": int(macd_accel), "vol_pct": raw["vol"],
        "conf": raw["confidence"], "action": action, "score": score,
        "is_knife": int(is_knife), "reversal": int(reversal),
        "computed_at": date.today().isoformat(),
    })
    return row


# ───────────────────────────────────────────────────────────────────
# FUNDAMENTALS
# ───────────────────────────────────────────────────────────────────

def _safe_float(val):
    try:
        v = float(val)
        return round(v, 4) if not np.isnan(v) else None
    except Exception:
        return None

def fetch_yf_fundamentals(ticker, timeout=8):
    cache_key = f"yfund_{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        import concurrent.futures as _cf
        def _fetch():
            t    = yf.Ticker(ticker)
            info = {}
            try:    info = t.info or {}
            except: pass
            def _get(*keys):
                for k in keys:
                    v = _safe_float(info.get(k))
                    if v is not None and v > 0: return v
                return None
            pe       = _get("trailingPE","forwardPE")
            fwd_pe   = _get("forwardPE")
            div      = _get("dividendYield","trailingAnnualDividendYield")
            mcap     = _get("marketCap")
            beta     = _get("beta")
            pb       = _get("priceToBook")
            roe      = _safe_float(info.get("returnOnEquity"))
            de_raw   = _safe_float(info.get("debtToEquity"))
            rev_gr   = _safe_float(info.get("revenueGrowth"))
            eps_gr   = _safe_float(info.get("earningsGrowth"))
            fcf_raw  = _safe_float(info.get("freeCashflow"))
            fcf_yield = (fcf_raw / mcap) if fcf_raw and mcap and mcap > 0 else None
            peg      = _safe_float(info.get("trailingPegRatio"))
            if peg is None and pe and eps_gr and 0.001 < eps_gr < 5:
                try: peg = round(pe / (eps_gr * 100), 2)
                except: pass
            # Normalise D/E: yfinance returns as % (150 = 1.5x)
            de_ratio = (de_raw / 100) if de_raw and de_raw > 10 else de_raw
            return {
                "fmp_pe_ttm":     pe or fwd_pe,
                "fmp_peg":        peg,
                "fmp_pb":         pb,
                "fmp_fcf_yield":  fcf_yield,
                "fmp_roe":        roe,
                "fmp_debt_eq":    de_ratio,
                "fmp_rev_growth": rev_gr,
                "fmp_eps_growth": eps_gr,
                "fmp_div_yield":  div,
                "fmp_mcap":       mcap,
                "fmp_beta":       beta,
                "_source": "yfinance",
            }
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            result = ex.submit(_fetch).result(timeout=timeout)
        cache_set(cache_key, result, ttl=3600)
        return result
    except Exception:
        return {}

def fetch_yf_fundamentals_batch(tickers, max_workers=8, timeout=6):
    results, timed_out = {}, []
    to_fetch = []
    for t in tickers:
        c = cache_get(f"yfund_{t}")
        if c is not None:
            results[t] = c
        else:
            to_fetch.append(t)
    if to_fetch:
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(fetch_yf_fundamentals, t, timeout): t for t in to_fetch}
            for fut in _cf.as_completed(futs, timeout=timeout + 2):
                t = futs[fut]
                try:
                    results[t] = fut.result()
                except Exception:
                    results[t] = {}
                    timed_out.append(t)
        for fut, t in futs.items():
            if t not in results:
                results[t] = {}
                timed_out.append(t)
    return results, timed_out

def _get_fmp_key():
    return os.environ.get("FMP_API_KEY","").strip() or ""

def fetch_fmp_fundamentals(ticker):
    key = _get_fmp_key()
    if not key:
        return {}
    cache_key = f"fmp_fund_{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        result = {}
        for endpoint in [
            f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={key}",
            f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={key}",
        ]:
            r = requests.get(endpoint, timeout=8)
            if r.status_code == 200:
                data = r.json()
                if data and isinstance(data, list):
                    d = data[0]
                    result.update({
                        "fmp_pe_ttm":     _safe_float(d.get("peRatioTTM")),
                        "fmp_peg":        _safe_float(d.get("pegRatioTTM")),
                        "fmp_pb":         _safe_float(d.get("pbRatioTTM") or d.get("priceToBookRatioTTM")),
                        "fmp_fcf_yield":  _safe_float(d.get("freeCashFlowYieldTTM")),
                        "fmp_roe":        _safe_float(d.get("roeTTM") or d.get("returnOnEquityTTM")),
                        "fmp_debt_eq":    _safe_float(d.get("debtToEquityTTM")),
                        "fmp_rev_growth": _safe_float(d.get("revenueGrowthTTM")),
                        "fmp_div_yield":  _safe_float(d.get("dividendYieldTTM") or d.get("dividendYieldPercentageTTM")),
                        "fmp_mcap":       _safe_float(d.get("marketCapTTM")),
                        "_source": "fmp",
                    })
        cache_set(cache_key, result, ttl=86400)
        return result
    except Exception:
        return {}

def compute_value_score(fund):
    """Returns (score 0-100, grade A-D, breakdown_dict, coverage_count)."""
    if not fund:
        return 0, "—", {}, 0

    def _f(v):
        try: return float(v)
        except: return None

    pe        = _f(fund.get("fmp_pe_ttm"))
    peg       = _f(fund.get("fmp_peg"))
    pb        = _f(fund.get("fmp_pb"))
    fcf_yield = _f(fund.get("fmp_fcf_yield"))
    roe       = _f(fund.get("fmp_roe"))
    debt_eq   = _f(fund.get("fmp_debt_eq"))
    rev_growth= _f(fund.get("fmp_rev_growth"))
    div_yield = _f(fund.get("fmp_div_yield"))

    total_weight = 0
    weighted_sum = 0
    breakdown    = {}
    coverage     = 0

    def sm(value, weight, thresholds, label):
        nonlocal total_weight, weighted_sum, coverage
        if value is None: return
        coverage += 1
        s = thresholds[-1][1]
        for thresh, pts in thresholds:
            if value <= thresh: s = pts; break
        weighted_sum += s * weight
        total_weight += weight
        breakdown[label] = round(s * 100)

    if pe is not None and pe > 0:
        sm(pe, 25, [(10,1.0),(15,0.85),(20,0.70),(25,0.55),(35,0.35),(50,0.15),(999,0.0)], "PE")
    elif pe is not None:
        total_weight += 25

    if peg is not None and peg > 0:
        sm(peg, 20, [(0.5,1.0),(0.8,0.85),(1.0,0.70),(1.5,0.45),(2.0,0.20),(999,0.0)], "PEG")

    if pb is not None and pb > 0:
        sm(pb, 15, [(1.0,1.0),(1.5,0.80),(2.5,0.55),(4.0,0.25),(7.0,0.10),(999,0.0)], "P/B")

    if fcf_yield is not None:
        sm(fcf_yield*100, 15, [(-999,0.0),(0,0.05),(2,0.30),(4,0.55),(6,0.75),(8,0.90),(999,1.0)], "FCF Yield")

    if roe is not None:
        sm(roe*100, 10, [(0,0.0),(5,0.20),(10,0.45),(15,0.65),(20,0.80),(30,0.95),(999,1.0)], "ROE")

    if debt_eq is not None and debt_eq >= 0:
        sm(debt_eq, 10, [(0.1,1.0),(0.3,0.85),(0.7,0.65),(1.5,0.35),(3.0,0.10),(999,0.0)], "D/E")

    if rev_growth is not None:
        sm(rev_growth*100, 5, [(-999,0.0),(0,0.10),(5,0.40),(10,0.65),(15,0.85),(999,1.0)], "Rev Growth")

    div_bonus = min(5, round(div_yield*100)) if div_yield and div_yield > 0.02 else 0

    if total_weight == 0:
        return 0, "—", {}, 0

    score = min(100, round((weighted_sum / total_weight) * 100 + div_bonus))
    grade = "A" if score >= 75 else "B" if score >= 55 else "C" if score >= 35 else "D"
    return score, grade, breakdown, coverage

def _fundamental_score_adjustment(pe=None, pb=None, div=None, eps=None, mc=None, beta=None, asset_type="Stock"):
    if asset_type == "ETF":
        return 0.0
    delta = 0.0
    if pe is not None:
        if   pe <= 0:    delta -= 0.06
        elif pe < 12:    delta += 0.04
        elif pe < 25:    delta += 0.02
        elif pe >= 50:   delta -= 0.04
    if pb is not None:
        if   pb <= 0:    delta -= 0.02
        elif pb < 1.5:   delta += 0.02
        elif pb > 4:     delta -= 0.03
    if eps is not None:
        if   eps > 0:    delta += 0.02
        elif eps < 0:    delta -= 0.03
    if div is not None:
        if   div > 0.04: delta += 0.02
        elif div > 0.01: delta += 0.01
    if mc is not None:
        if   mc < 50e6:  delta -= 0.04
        elif mc < 300e6: delta -= 0.02
        elif mc > 10e9:  delta += 0.01
    if beta is not None:
        if   beta > 3.0:           delta -= 0.03
        elif beta > 2.0:           delta -= 0.01
        elif 0.4 < beta < 1.5:     delta += 0.01
    return max(-0.15, min(0.15, round(delta, 4)))

def fetch_conviction_signals(ticker, timeout=6):
    cache_key = f"conv_{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    import concurrent.futures as _cf
    def _fetch():
        result = {}
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
            rec  = info.get("recommendationMean")
            n_analysts = info.get("numberOfAnalystOpinions", 0)
            if rec and n_analysts:
                result["analyst_mean"]  = float(rec)
                result["analyst_count"] = int(n_analysts)
        except Exception:
            pass
        try:
            si = (t.info or {}).get("shortPercentOfFloat")
            if si:
                result["short_pct"] = float(si) * 100 if float(si) < 1 else float(si)
        except Exception:
            pass
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty and "surprisePercent" in eh.columns:
                surprises = eh["surprisePercent"].dropna().tail(4).tolist()
                result["earnings_beat_count"] = sum(1 for s in surprises if s > 0)
        except Exception:
            pass
        try:
            ins = t.insider_purchases
            if ins is not None and not ins.empty and "Shares" in ins.columns:
                buys  = ins[ins.get("Transaction","").str.contains("Purchase", na=False)]["Shares"].sum() if "Transaction" in ins.columns else 0
                sells = ins[ins.get("Transaction","").str.contains("Sale",     na=False)]["Shares"].sum() if "Transaction" in ins.columns else 0
                result["insider_buying"] = buys > sells
        except Exception:
            pass
        return result
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            raw = ex.submit(_fetch).result(timeout=timeout)
    except Exception:
        raw = {}

    score  = 50
    labels = []
    am = raw.get("analyst_mean")
    if am is not None:
        if   am <= 1.5: score += 20; labels.append("Strong analyst Buy")
        elif am <= 2.2: score += 12; labels.append("Analyst Buy")
        elif am <= 2.8: score +=  4
        elif am <= 3.5: score -=  8; labels.append("Analyst Hold")
        else:           score -= 20; labels.append("Analyst Sell")
    si = raw.get("short_pct")
    if si is not None:
        if   si > 30: score -= 15; labels.append(f"High short {si:.0f}%")
        elif si > 15: score -=  8; labels.append(f"Elevated short {si:.0f}%")
        elif si <  3: score +=  5
    beats = raw.get("earnings_beat_count", 0)
    if beats == 4:   score += 15; labels.append("4/4 earnings beats")
    elif beats == 3: score += 10; labels.append("3/4 earnings beats")
    elif beats <= 1: score -=  8; labels.append("Poor earnings history")
    if raw.get("insider_buying") is True:
        score += 15; labels.append("Insider buying")

    score = max(0, min(100, score))
    grade = "🟢 HIGH" if score >= 70 else "🟡 MED" if score >= 45 else "🔴 LOW"
    conviction = {**raw, "conviction_score": score, "conviction_grade": grade, "conviction_labels": labels}
    cache_set(cache_key, conviction, ttl=86400)
    return conviction


# ───────────────────────────────────────────────────────────────────
# UNIVERSE / PRESET BUILDER
# ───────────────────────────────────────────────────────────────────

PRESETS = {
    "🌍 All ETFs":          {"type":"ETF"},
    "🇪🇺 UCITS ETFs":       {"type":"ETF","domicile":["Ireland","Luxembourg"]},
    "🇺🇸 US ETFs":          {"type":"ETF","domicile":["United States"]},
    "📈 US Stocks":         {"type":"Stock","country":["United States"]},
    "🇬🇧 UK Stocks":        {"type":"Stock","country":["United Kingdom"]},
    "🇩🇪 German Stocks":    {"type":"Stock","country":["Germany"]},
    "🌐 Global Stocks":     {"type":"Stock","country":["United States","United Kingdom","Germany","France","Japan","Canada"]},
    "💰 Equity ETFs":       {"type":"ETF","category_group":["Equities"]},
    "🏦 Fixed Income ETFs": {"type":"ETF","category_group":["Fixed Income"]},
    "🥇 Commodity ETFs":    {"type":"ETF","category_group":["Commodities"]},
    "🔧 Custom":            {"type":"custom"},
}

def build_tickers(preset_key, filters):
    if universe.empty:
        return []
    preset = PRESETS.get(preset_key, {"type":"ETF"})
    ptype  = preset.get("type","ETF")
    tickers = []

    if ptype in ("ETF","custom"):
        src_df = jetf_df if not jetf_df.empty else universe[universe["type"]=="ETF"]
        if not src_df.empty:
            mask = pd.Series([True]*len(src_df), index=src_df.index)
            if "domicile" in preset and "domicile" in src_df.columns:
                mask &= src_df["domicile"].isin(preset["domicile"])
            for col, fkey in [("domicile","domicile"),("dist_policy","dist_policy"),
                               ("replication","replication"),("strategy","strategy")]:
                if filters.get(fkey) and col in src_df.columns:
                    mask &= src_df[col].isin(filters[fkey])
            if filters.get("min_size",0) > 0 and "fund_size_eur" in src_df.columns:
                mask &= src_df["fund_size_eur"].fillna(0) >= filters["min_size"]
            if filters.get("max_ter",2.0) < 2.0 and "ter" in src_df.columns:
                mask &= src_df["ter"].fillna(99) <= filters["max_ter"]
            tcol = "ticker" if "ticker" in src_df.columns else src_df.columns[0]
            tickers += src_df[mask][tcol].dropna().str.upper().tolist()

    if ptype in ("Stock","custom"):
        if not universe.empty:
            mask = universe["type"] == "Stock"
            if "country" in preset and "country" in universe.columns:
                mask &= universe["country"].isin(preset["country"])
            if filters.get("country") and "country" in universe.columns:
                mask &= universe["country"].isin(filters["country"])
            if filters.get("sector") and "sector" in universe.columns:
                mask &= universe["sector"].isin(filters["sector"])
            tickers += universe[mask]["ticker"].dropna().str.upper().tolist()

    # Filter leveraged/SPAC/warrants
    def _is_bad(t):
        t = str(t).strip()
        if len(t) < 2: return True
        if t[-1] in ('W','U','R') and len(t) >= 4: return True
        return False
    tickers = [t for t in dict.fromkeys(tickers) if not _is_bad(t)]
    return tickers[:5000]

def get_name_isin(ticker):
    entry = _name_lookup.get(ticker)
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return entry or ticker, ""


# ───────────────────────────────────────────────────────────────────
# SIGNAL BADGE HELPER
# ───────────────────────────────────────────────────────────────────

def action_badge(action):
    cls = {"BUY":"buy","WATCH":"watch","SELL":"sell","AVOID":"avoid","WAIT":"wait"}.get(action,"wait")
    return f'<span class="badge-{cls}">{action}</span>'

def action_color(action):
    return {"BUY":"#0d9488","WATCH":"#0284c7","SELL":"#dc2626",
            "AVOID":"#d97706","WAIT":"#64748b"}.get(action,"#64748b")


# ───────────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown(f"**📡 Market Decision Engine** `{APP_VERSION}`")
    st.sidebar.caption(f"deployed {BUILD_TIME}")

    # Live indicators
    vix      = get_live_vix()
    fg, lbl  = get_fg_index()
    rm       = 1.0 + ((vix/20) * ((100-fg)/50))
    gauge    = "🔴" if fg<=25 else "🟠" if fg<=44 else "🟡" if fg<=55 else "🟢" if fg<=75 else "💚"

    c1, c2 = st.sidebar.columns(2)
    c1.metric("VIX", f"{vix:.1f}")
    c2.metric("Fear & Greed", f"{gauge} {fg}")
    st.sidebar.caption(lbl)

    if rm > 2.5:
        st.sidebar.error(f"🔥 Extreme Fear — {rm:.1f}x")
    elif rm > 2.0:
        st.sidebar.error(f"😰 High Fear — {rm:.1f}x")
    elif rm < 1.2:
        st.sidebar.info(f"😌 Calm — {rm:.1f}x")
    else:
        st.sidebar.warning(f"⚖️ Normal — {rm:.1f}x")

    st.sidebar.divider()

    # Preset
    preset = st.sidebar.selectbox("🎯 Universe preset", list(PRESETS.keys()), index=0)
    ptype  = PRESETS[preset].get("type","ETF")

    # Filters
    filters = {}
    with st.sidebar.expander("🔧 Optional Filters", expanded=False):
        if ptype == "custom":
            types = st.multiselect("Asset Type", ["ETF","Stock"], default=["ETF"])
            filters["types"] = types
        if ptype in ("ETF","custom"):
            dom_opts  = sorted(jetf_df["domicile"].dropna().unique().tolist()) if not jetf_df.empty and "domicile" in jetf_df.columns else ["Ireland","Luxembourg","Germany","United States"]
            dist_opts = sorted(jetf_df["dist_policy"].dropna().unique().tolist()) if not jetf_df.empty and "dist_policy" in jetf_df.columns else ["Accumulating","Distributing"]
            repl_opts = sorted(jetf_df["replication"].dropna().unique().tolist()) if not jetf_df.empty and "replication" in jetf_df.columns else ["Physical (Full)","Physical (Sampling)","Swap-based"]
            st.markdown("**📦 ETF Filters**")
            filters["domicile"]     = st.multiselect("Domicile",     dom_opts)
            filters["dist_policy"]  = st.multiselect("Distribution", dist_opts)
            filters["replication"]  = st.multiselect("Replication",  repl_opts)
            filters["min_size"]     = st.number_input("Min Size €m", min_value=0, value=0, step=50)
            filters["max_ter"]      = st.number_input("Max TER %",   min_value=0.0, max_value=5.0, value=2.0, step=0.05)
        if ptype in ("Stock","custom"):
            st.markdown("**📈 Stock Filters**")
            if not universe.empty:
                ctry_opts = sorted(universe[universe["type"]=="Stock"]["country"].dropna().unique().tolist())
                sect_opts = sorted(universe[universe["type"]=="Stock"]["sector"].dropna().unique().tolist())
            else:
                ctry_opts, sect_opts = [], []
            filters["country"] = st.multiselect("Country", ctry_opts)
            filters["sector"]  = st.multiselect("Sector",  sect_opts)

    st.sidebar.divider()

    budget  = st.sidebar.number_input("💰 Monthly Budget (EUR)", min_value=100, value=1000, step=100)
    workers = st.sidebar.slider("⚡ Parallel Workers", min_value=2, max_value=12, value=6)
    fetch_pe = st.sidebar.checkbox("Fetch PE / Fundamentals (slower)")

    tickers = build_tickers(preset, filters)
    st.sidebar.caption(f"{len(tickers):,} tickers in scope")

    with st.sidebar.expander("📖 VIX & F&G Guide", expanded=False):
        st.markdown("""
**VIX** (expected S&P 500 volatility):
- < 15 → Complacency  · 15–25 → Normal
- 25–30 → Elevated  · > 30 → Fear  · > 40 → Crisis

**Fear & Greed** (CNN composite):
- 0–25 → Extreme Fear 🔴 *(best entries)*
- 25–55 → Fear/Neutral 🟠🟡
- 55–100 → Greed/Extreme Greed 🟢💚 *(caution)*
        """)

    return preset, filters, budget, workers, fetch_pe, tickers, vix, fg, rm


# ───────────────────────────────────────────────────────────────────
# TAB 1 — MARKET SCANNER
# ───────────────────────────────────────────────────────────────────

def render_scanner(tickers, budget, workers, fetch_pe, vix, fg, rm):
    st.subheader("🔭 Market Scanner")

    c1, c2 = st.columns([5,1])
    run_scan  = c1.button("🔄 Run Scan", type="primary", use_container_width=True)
    clear_btn = c2.button("🗑️", use_container_width=True)

    if clear_btn:
        for k in ["scan_results","scan_status"]:
            st.session_state.pop(k, None)
        st.rerun()

    sdf = get_signals_df()

    # ── FAST PATH ────────────────────────────────────────────────────
    if run_scan:
        st.session_state.pop("scan_results", None)
        if not sdf.empty:
            sig = sdf[sdf["ticker"].isin(tickers)].copy()
            if len(sig) >= 10:
                with st.spinner(f"⚡ Loading {len(sig)} pre-computed signals…"):
                    budget_val = budget or 1000
                    rows = []
                    for _, r in sig.iterrows():
                        action = str(r.get("action","WAIT"))
                        conf   = float(r.get("conf", 0.5) or 0.5)
                        vol    = float(r.get("vol_pct", 2) or 2)
                        dm     = float(r.get("dist_ma200", 0) or 0)
                        rsi_v  = float(r.get("rsi", 50) or 50)
                        if action == "AVOID":  alloc = "⛔ Skip"
                        elif action == "WAIT": alloc = "—"
                        elif action == "WATCH":
                            alloc = f"👀 €{budget_val*0.25*(0.5+conf):,.0f}"
                        else:
                            ctx_s = (40 if fg<35 else 20 if fg<50 else 0)/100
                            sig_s = min(-dm/20,1)*0.5 + min((50-rsi_v)/50,1)*0.5 if dm<0 else 0
                            amt   = budget_val*(ctx_s+sig_s)*(0.5+conf)*max(0.5,1-vol/20)*rm
                            amt   = max(budget_val*0.25, min(amt, budget_val*3))
                            tier  = "🔥" if amt>=budget_val*1.5 else "⚖️" if amt>=budget_val*0.8 else "🔍"
                            alloc = f"{tier} €{amt:,.0f}"
                        pe   = r.get("pe_ratio")
                        div  = r.get("div_yield")
                        mcap = r.get("market_cap")
                        beta = r.get("beta")
                        dist_display = r.get("dist_ma200") if pd.notna(r.get("dist_ma200")) else r.get("ret_1m")
                        name, isin = get_name_isin(str(r.get("ticker","")))
                        rows.append({
                            "Ticker":  r["ticker"],
                            "Name":    r.get("name", name) or name,
                            "Price":   round(float(r["price"]),2) if pd.notna(r.get("price")) else "—",
                            "Dist%":   round(float(dist_display),2) if dist_display is not None and pd.notna(dist_display) else 0,
                            "52W%":    r.get("dist_52w",0) if pd.notna(r.get("dist_52w")) else "—",
                            "RSI":     round(float(r.get("rsi",50)),1) if pd.notna(r.get("rsi")) else "—",
                            "MACD":    ("▲ Bull" if r.get("macd_bull",0) else "▼ Bear") if pd.notna(r.get("macd_bull")) else "—",
                            "Vol%":    r.get("vol_pct",0) if pd.notna(r.get("vol_pct")) else "—",
                            "Action":  action,
                            "Score":   round(float(r.get("score",0) or 0),3),
                            "Knife":   "⚠️" if r.get("is_knife",0) and not r.get("reversal",0) else "",
                            "Alloc":   alloc,
                            "PE":      f"{pe:.1f}" if pe and str(pe) != 'nan' else "—",
                            "Beta":    f"{beta:.2f}" if beta and str(beta) != 'nan' else "—",
                            "Div%":    f"{div*100:.1f}%" if div and str(div) != 'nan' else "—",
                            "MCap":    f"${mcap/1e9:.1f}B" if mcap and str(mcap) != 'nan' else "—",
                            "VGrade":  str(r.get("value_grade","")) if pd.notna(r.get("value_grade")) and str(r.get("value_grade","")) not in ("","nan","None") else "—",
                            "Source":  r.get("data_source",""),
                        })
                    result_df = pd.DataFrame(rows)
                    action_order = {"BUY":0,"WATCH":1,"SELL":2,"AVOID":3,"WAIT":4}
                    result_df["_an"] = result_df["Action"].map(action_order).fillna(5)
                    result_df = result_df.sort_values(["_an","Score"],ascending=[True,False]).drop(columns=["_an"]).reset_index(drop=True)
                    result_df.insert(0,"#",result_df.index+1)
                    computed = sig["computed_at"].iloc[0] if "computed_at" in sig.columns else "unknown"
                    st.session_state["scan_results"] = result_df
                    st.session_state["scan_status"]  = f"⚡ {len(result_df)} pre-computed · updated: {computed}"
                    st.rerun()

        # ── LIVE SCAN ─────────────────────────────────────────────────
        MAX = 500
        scan_tickers = tickers[:MAX]
        progress_bar = st.progress(0, text="Starting live scan…")
        results = []
        isin_map = {}
        if not jetf_df.empty and "ticker" in jetf_df.columns and "isin" in jetf_df.columns:
            isin_map = dict(zip(jetf_df["ticker"], jetf_df["isin"].fillna("")))

        done, total = 0, len(scan_tickers)
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futs = {ex.submit(analyse_ticker, t, rm, isin_map.get(t,"")): t for t in scan_tickers}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if r:
                        results.append(r)
                except Exception:
                    pass
                done += 1
                progress_bar.progress(done/total, text=f"Scanning… {done}/{total} ({len(results)} signals)")

        progress_bar.empty()
        if results:
            result_df = pd.DataFrame(results)
            action_order = {"BUY":0,"WATCH":1,"SELL":2,"AVOID":3,"WAIT":4}
            result_df["_an"] = result_df["Action"].map(action_order).fillna(5)
            result_df = result_df.sort_values(["_an","Score"],ascending=[True,False]).drop(columns=["_an"]).reset_index(drop=True)
            result_df.insert(0,"#",result_df.index+1)
            st.session_state["scan_results"] = result_df
            st.session_state["scan_status"]  = f"✅ Live scan: {len(result_df)} signals from {total} tickers"
        else:
            st.warning("No signals found. Try a different preset or wider filters.")
        st.rerun()

    # ── Display results ───────────────────────────────────────────────
    if "scan_results" not in st.session_state:
        st.info("Click **Run Scan** to start. Pre-computed signals load instantly; live scan takes 1–3 minutes.")
        return

    result_df = st.session_state["scan_results"]
    st.caption(st.session_state.get("scan_status",""))

    # KPI row
    n_buy   = int((result_df["Action"]=="BUY").sum())
    n_watch = int((result_df["Action"]=="WATCH").sum())
    n_sell  = int((result_df["Action"]=="SELL").sum())
    n_avoid = int((result_df["Action"]=="AVOID").sum())
    n_wait  = int((result_df["Action"]=="WAIT").sum())

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("🟢 BUY",   n_buy)
    k2.metric("👀 WATCH", n_watch)
    k3.metric("🔴 SELL",  n_sell)
    k4.metric("⛔ AVOID", n_avoid)
    k5.metric("🟡 WAIT",  n_wait)

    # Tab filter
    tab_names = ["All", "🟢 BUY", "👀 WATCH", "⛔ AVOID", "🔴 SELL", "🟡 WAIT"]
    if "VGrade" in result_df.columns:
        tab_names += ["💎 Value A", "🔷 Value B"]

    tabs = st.tabs(tab_names)
    action_map = {
        "All": None,
        "🟢 BUY": "BUY",
        "👀 WATCH": "WATCH",
        "⛔ AVOID": "AVOID",
        "🔴 SELL": "SELL",
        "🟡 WAIT": "WAIT",
        "💎 Value A": "VALUE_A",
        "🔷 Value B": "VALUE_B",
    }

    display_cols = [c for c in result_df.columns
                    if c not in ("Source","Score","Conf","RSI↗","MACD⚡")]

    def _style_action(val):
        colors = {"BUY":"#0d9488","WATCH":"#0284c7","SELL":"#dc2626","AVOID":"#d97706","WAIT":"#64748b"}
        return f"color: {colors.get(val,'#64748b')}; font-weight: 600;"

    for tab, tname in zip(tabs, tab_names):
        with tab:
            key = action_map[tname]
            if key is None:
                df_show = result_df
            elif key == "VALUE_A":
                df_show = result_df[result_df.get("VGrade","—") == "A"] if "VGrade" in result_df.columns else result_df
            elif key == "VALUE_B":
                df_show = result_df[result_df.get("VGrade","—").isin(["A","B"])] if "VGrade" in result_df.columns else result_df
            else:
                df_show = result_df[result_df["Action"] == key]

            if df_show.empty:
                st.info("No results in this category.")
                continue

            show_cols = [c for c in display_cols if c in df_show.columns]
            st.dataframe(
                df_show[show_cols].style.applymap(_style_action, subset=["Action"]),
                use_container_width=True,
                height=min(600, 40 + len(df_show)*35),
                hide_index=True,
            )

    # Download
    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv_data, "signals.csv", "text/csv")


# ───────────────────────────────────────────────────────────────────
# TAB 2 — DEEP DIVE
# ───────────────────────────────────────────────────────────────────

def render_deepdive(budget):
    st.subheader("🔬 Deep Dive")

    c1, c2, c3 = st.columns([4, 1, 1])
    ticker_input = c1.text_input("Ticker or ISIN", placeholder="e.g. VWRA or IE00B3RBWM25")
    budget_dd    = c2.number_input("Budget (EUR)", min_value=100, value=budget, step=100)
    analyse_btn  = c3.button("🔍 Analyse", type="primary")

    # Allow clicking from scanner
    if "dd_ticker" in st.session_state and not ticker_input:
        ticker_input = st.session_state["dd_ticker"]

    if not analyse_btn or not ticker_input:
        st.info("Enter a ticker or ISIN and click **Analyse**.")
        return

    ticker = ticker_input.strip().upper()

    # ISIN → ticker lookup
    isin = None
    if len(ticker) == 12 and ticker[:2].isalpha():
        isin = ticker
        # Look up in justETF
        if not jetf_df.empty and "isin" in jetf_df.columns:
            match = jetf_df[jetf_df["isin"] == isin]
            if not match.empty:
                ticker = match.iloc[0]["ticker"]

    with st.spinner(f"Fetching {ticker}…"):
        raw = fetch_ticker_data(ticker, isin=isin, force_refresh=True)

    if raw is None:
        st.error(f"Could not fetch data for **{ticker}**. Check the ticker and try again.")
        return

    close    = raw["close"]
    ma50_s   = raw["ma50_s"]
    ma200_s  = raw["ma200_s"]
    rsi_s    = raw["rsi_s"]
    macd_l   = raw["macd_l"]
    macd_sig = raw["macd_sig"]
    macd_h   = raw["macd_h"]

    name, isin_found = get_name_isin(ticker)
    isin = isin or isin_found
    is_etf = not universe.empty and ticker in universe[universe["type"]=="ETF"]["ticker"].values

    # Fundamentals
    fund_data = {}
    if not is_etf:
        with st.spinner("Fetching fundamentals…"):
            fund_data = fetch_yf_fundamentals(ticker, timeout=8)
            if _get_fmp_key():
                fmp_data = fetch_fmp_fundamentals(ticker)
                fund_data = {**fund_data, **fmp_data}

    pe   = fund_data.get("fmp_pe_ttm")
    div  = fund_data.get("fmp_div_yield")
    mc   = fund_data.get("fmp_mcap")
    beta = fund_data.get("fmp_beta") or fund_data.get("beta")

    def _pf(v): 
        try: return float(v)
        except: return None

    fund_delta = _fundamental_score_adjustment(
        pe=_pf(pe), pb=_pf(fund_data.get("fmp_pb")),
        div=_pf(div), eps=None, mc=_pf(mc), beta=_pf(beta),
        asset_type="ETF" if is_etf else "Stock",
    )

    action, score, macd_bull, macd_accel, rsi_rising, ma200_trend, ma200_steep = compute_action_score(
        dist_ma=raw["dist_ma"], rsi=raw["rsi"], rsi_slope=raw["rsi_slope"],
        macd=raw["macd"], macd_signal=raw["macd_signal"], macd_hist=raw["macd_hist"],
        conf=raw["confidence"], vol=raw["vol"], price_slope=raw["price_slope"],
        ma200_slope=raw["ma200_slope"], fund_delta=fund_delta,
        is_etf=is_etf,
    )

    value_score, value_grade, value_bdown, value_cov = compute_value_score(fund_data)
    value_available = value_score > 0 and not is_etf

    conviction = {} if is_etf else fetch_conviction_signals(ticker, timeout=6)
    conv_grade  = conviction.get("conviction_grade", "—")
    conv_labels = conviction.get("conviction_labels", [])

    # justETF metadata
    jetf_meta = {}
    if not jetf_df.empty and "ticker" in jetf_df.columns and ticker in jetf_df["ticker"].values:
        jetf_meta = jetf_df[jetf_df["ticker"]==ticker].iloc[0].to_dict()

    # ETF category flags
    HARD_FLAG_KW = ["leveraged","2x ","3x ","4x ","-2x","-3x"," short ","inverse"]
    SOFT_FLAG_KW = ["carry","volatility","vix","futures","roll","enhanced"]
    combined = (name or "").lower() + " " + str(jetf_meta.get("strategy","")).lower()
    hard_flagged = is_etf and any(kw in combined for kw in HARD_FLAG_KW)
    soft_flagged = is_etf and not hard_flagged and any(kw in combined for kw in SOFT_FLAG_KW)

    is_knife  = raw["dist_ma"] < -25 and raw["rsi"] < 35 and raw["trend_down_strong"]
    reversal  = is_knife and macd_bull and rsi_rising
    dist_ma   = raw["dist_ma"]
    rsi_val   = raw["rsi"]
    cur_p     = raw["price"]

    buy_score  = (40 if (fg:=get_fg_index()[0])<35 else 0)+(30 if rsi_val<40 else 0)+(30 if dist_ma<0 else 0)
    sell_score = (40 if (fg:=get_fg_index()[0])>65 else 0)+(30 if rsi_val>65 else 0)+(30 if dist_ma>0 else 0)

    # ── KPI row ───────────────────────────────────────────────────────
    st.markdown(f"### {ticker} — {name}")
    if isin:
        st.caption(f"ISIN: {isin}")

    ma200_label = {"rising":"↗ Rising","falling":"↘ Falling","flat":"→ Flat"}.get(ma200_trend,"→ Flat")
    value_label = f"{value_score}/100 ({value_grade})" if value_available else "N/A"

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Price",       f"€{cur_p:.2f}")
    k2.metric("vs MA200",    f"{dist_ma:+.1f}%")
    k3.metric("MA200 Trend", ma200_label)
    k4.metric("RSI",         f"{rsi_val:.1f} {'↗' if rsi_rising else '↘'}")
    k5.metric("MACD",        "▲ Bull" if macd_bull else "▼ Bear")
    k6.metric("Value Score", value_label)

    # ── Warning banners ───────────────────────────────────────────────
    if hard_flagged:
        st.error("⚠️ **Strategy warning:** Leveraged/inverse ETF — dip signals unreliable.")
    elif soft_flagged:
        st.warning("⚠️ **Strategy note:** Complex strategy (carry/futures/VIX) — verify thesis before acting.")
    if ma200_steep:
        st.warning(f"📉 **MA200 in steep decline** (slope {raw['ma200_slope']:+.4f}) — structural downtrend, not a cyclical dip.")
    elif ma200_trend == "falling":
        st.warning(f"⚠️ **MA200 declining** (slope {raw['ma200_slope']:+.4f}) — signal downgraded.")

    # justETF metadata strip
    meta_parts = []
    if jetf_meta.get("domicile"):     meta_parts.append(f"🏳️ {jetf_meta['domicile']}")
    if jetf_meta.get("ter"):          meta_parts.append(f"💸 TER {float(jetf_meta['ter']):.2f}%")
    if jetf_meta.get("fund_size_eur"):meta_parts.append(f"📦 €{float(jetf_meta['fund_size_eur']):,.0f}m")
    if jetf_meta.get("dist_policy"):  meta_parts.append(f"💰 {jetf_meta['dist_policy']}")
    if meta_parts:
        st.caption("  ·  ".join(meta_parts))

    # ── Buy / Sell decision ───────────────────────────────────────────
    def _val_suffix():
        if not value_available: return ""
        if value_grade == "A": return f" · 🏆 Value A ({value_score}/100)"
        if value_grade == "B": return f" · ✅ Value B ({value_score}/100)"
        if value_grade == "D": return f" · ⚠️ Expensive ({value_score}/100)"
        return ""

    entry = "TRIGGER" if action == "BUY" else ("WATCH" if action == "WATCH" else "WAIT")

    col_buy, col_sell = st.columns(2)
    with col_buy:
        st.markdown("**📥 BUY DECISION**")
        if hard_flagged:
            st.error("⛔ AVOID — Leveraged/inverse. Not suitable for dip buying.")
        elif is_knife and not reversal:
            st.error("⛔ AVOID — Falling knife. Wait for reversal.")
        elif entry == "WAIT":
            st.warning("⏳ WAIT — Still falling. Let it stabilise.")
        elif entry == "WATCH":
            if buy_score >= 70:
                st.info(f"⚖️ PREPARE — Good setup (~€{budget_dd:,.0f}){_val_suffix()}")
            else:
                st.info(f"🔍 WATCH — No strong entry yet.{_val_suffix()}")
        else:
            if value_available and value_grade == "A" and buy_score >= 60:
                st.success(f"🏆 CONVICTION BUY — Technically ready + fundamentally cheap · €{budget_dd*2:,.0f}")
            elif buy_score >= 70:
                st.success(f"🔥 BUY — €{budget_dd*2:,.0f}{_val_suffix()}")
            elif buy_score >= 40:
                st.success(f"⚖️ BUY — €{budget_dd:,.0f}{_val_suffix()}")
            else:
                st.warning(f"⚠️ LIGHT — €{budget_dd*0.5:,.0f}{_val_suffix()}")
        if reversal:
            st.success("✅ Reversal confirmed (MACD + RSI turned bullish)")

    with col_sell:
        st.markdown("**📤 SELL DECISION**")
        if sell_score >= 70:
            st.error("🚨 STRONG SELL — Reduce significantly.")
        elif sell_score >= 40:
            st.warning("⚠️ TRIM — Partial reduction.")
        else:
            st.success("🟢 HOLD — No sell signal.")

    # Conviction
    if conviction:
        st.markdown(f"**Conviction:** {conv_grade}")
        if conv_labels:
            st.caption(" · ".join(conv_labels))

    # ── Charts ────────────────────────────────────────────────────────
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=close.index, y=close, name="Price", line=dict(color="#00bcd4",width=2)))
    fig_price.add_trace(go.Scatter(x=ma50_s.index, y=ma50_s, name="MA50", line=dict(color="#ffd600",width=1,dash="dash")))
    fig_price.add_trace(go.Scatter(x=ma200_s.index, y=ma200_s, name="MA200", line=dict(color="#ff6d00",width=1,dash="dash")))
    fig_price.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        margin=dict(l=0,r=0,t=30,b=0), height=280,
        title=f"{ticker} — {name}", legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig_price, use_container_width=True)

    fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5,0.5], vertical_spacing=0.05)
    fig_ind.add_trace(go.Scatter(x=macd_l.index,   y=macd_l,   name="MACD",   line=dict(color="#00e676")), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=macd_sig.index, y=macd_sig, name="Signal", line=dict(color="#ff6d00")), row=1, col=1)
    fig_ind.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="Hist",
        marker_color=["#00e676" if v>=0 else "#ff1744" for v in macd_h]), row=1, col=1)
    fig_ind.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s, name="RSI", line=dict(color="#00bcd4")), row=2, col=1)
    fig_ind.add_hline(y=30, line_color="#00e676", line_dash="dash", row=2, col=1)
    fig_ind.add_hline(y=70, line_color="#ff1744", line_dash="dash", row=2, col=1)
    fig_ind.update_layout(template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        margin=dict(l=0,r=0,t=10,b=0), height=260, legend=dict(orientation="h",y=1.05))
    st.plotly_chart(fig_ind, use_container_width=True)

    # ── Fundamentals panel ────────────────────────────────────────────
    if not is_etf and fund_data:
        st.markdown("**📊 Fundamentals**")
        def _fmt(v, mult=1, pct=False, suffix=""):
            try:
                f = float(v) * mult
                return f"{f:.1f}%" if pct else f"{f:.2f}{suffix}"
            except: return "—"

        f1,f2,f3,f4 = st.columns(4)
        f1.metric("PE",          _fmt(pe))
        f1.metric("P/B",         _fmt(fund_data.get("fmp_pb")))
        f2.metric("PEG",         _fmt(fund_data.get("fmp_peg")))
        f2.metric("FCF Yield",   _fmt(fund_data.get("fmp_fcf_yield"), mult=100, pct=True))
        f3.metric("ROE",         _fmt(fund_data.get("fmp_roe"), mult=100, pct=True))
        f3.metric("D/E",         _fmt(fund_data.get("fmp_debt_eq")))
        f4.metric("Div Yield",   _fmt(div, mult=100, pct=True))
        f4.metric("Rev Growth",  _fmt(fund_data.get("fmp_rev_growth"), mult=100, pct=True))

        if value_available:
            st.markdown(f"**Value Score: {value_score}/100 (Grade {value_grade})**")
            if value_bdown:
                bdown_str = " · ".join(f"{k}: {v}/100" for k,v in value_bdown.items())
                st.caption(bdown_str)

    # Write back to session signals_df
    update_signals_df({
        "ticker": ticker, "data_source": "live_deepdive",
        "price": cur_p, "ma200": raw["ma200"],
        "dist_ma200": dist_ma, "rsi": rsi_val,
        "rsi_rising": int(rsi_rising), "macd_bull": int(macd_bull),
        "macd_accel": int(macd_accel), "vol_pct": raw["vol"],
        "conf": raw["confidence"], "action": action, "score": score,
        "is_knife": int(is_knife), "reversal": int(reversal),
        "computed_at": date.today().isoformat(),
    })


# ───────────────────────────────────────────────────────────────────
# TAB 3 — VALUE SCREEN
# ───────────────────────────────────────────────────────────────────

SP500_TOP50 = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","BRK-B","LLY","AVGO","TSLA",
    "WMT","JPM","V","UNH","XOM","ORCL","MA","HD","PG","COST","JNJ","ABBV",
    "BAC","NFLX","KO","CRM","CVX","MRK","AMD","PEP","ADBE","TMO","ACN","LIN",
    "MCD","ABT","CSCO","TXN","DIS","DHR","NEE","INTC","PM","IBM","RTX","INTU",
    "NOW","QCOM","GE","HON",
]

def render_value_screen():
    st.subheader("💎 Value Screen")

    # Load candidates from scanner
    col_a, col_b = st.columns([3,1])
    with col_a:
        default_tickers = ""
        sdf = get_signals_df()
        if st.button("📡 Load stock candidates (BUY/WATCH) from scanner"):
            if not sdf.empty:
                cands = sdf[
                    (sdf["action"].isin(["BUY","WATCH"])) &
                    (sdf.get("type","Stock") != "ETF" if "type" in sdf.columns else True)
                ]["ticker"].dropna().unique().tolist()[:100]
                st.session_state["vs_tickers"] = ", ".join(cands)
        if st.button("🇺🇸 Load S&P 500 Top 50"):
            st.session_state["vs_tickers"] = ", ".join(SP500_TOP50)

    tickers_raw = st.text_area(
        "Tickers to screen (comma-separated)",
        value=st.session_state.get("vs_tickers",""),
        height=80,
        placeholder="AAPL, MSFT, GOOGL, META, AMZN, NVDA …",
    )

    c1,c2,c3 = st.columns([2,2,2])
    min_score   = c1.number_input("Min Value Score", min_value=0, max_value=100, value=20, step=5)
    tech_filter = c2.selectbox("Require tech signal", ["Any","BUY only","BUY or WATCH"])
    run_vs      = c3.button("🔍 Run Value Screen", type="primary")

    if not run_vs:
        st.info("Enter tickers and click **Run Value Screen**.")
        return

    tickers = [t.strip().upper() for t in tickers_raw.replace("\n"," ").split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:100]
    if not tickers:
        st.warning("No tickers entered.")
        return

    with st.spinner(f"Fetching fundamentals for {len(tickers)} tickers…"):
        fund_batch, timed_out = fetch_yf_fundamentals_batch(tickers, max_workers=8, timeout=5)

    if timed_out:
        st.caption(f"⚠️ Timed out: {', '.join(timed_out[:10])}")

    # Refresh tech signals
    with st.spinner("Refreshing technical signals…"):
        def _refresh(t):
            try: fetch_ticker_data(t, force_refresh=True)
            except: pass
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(_refresh, tickers[:30]))

    tech_map = {}
    sdf = get_signals_df()
    if not sdf.empty and "ticker" in sdf.columns:
        tech_map = dict(zip(sdf["ticker"].str.upper(), sdf["action"].fillna("WAIT")))

    # Score + filter
    scored = []
    for t in tickers:
        fund = fund_batch.get(t, {})
        score, grade, bdown, cov = compute_value_score(fund)
        if score < min_score:
            continue
        tech_sig = tech_map.get(t, "—")
        if tech_filter == "BUY only"    and tech_sig != "BUY":               continue
        if tech_filter == "BUY or WATCH" and tech_sig not in ("BUY","WATCH"): continue
        scored.append((t, fund, score, grade, bdown, cov, tech_sig))

    if not scored:
        st.warning("No tickers passed the filters. Try lowering Min Value Score or relaxing the tech filter.")
        return

    # Conviction layer
    passing = [t for t,*_ in scored]
    with st.spinner("Fetching conviction signals…"):
        import concurrent.futures as _cf
        conv_batch = {}
        with _cf.ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(fetch_conviction_signals, t, 5): t for t in passing}
            for fut in _cf.as_completed(futs, timeout=10):
                t2 = futs[fut]
                try:    conv_batch[t2] = fut.result()
                except: conv_batch[t2] = {}

    def _fmt(v, pct=False, mult=1):
        if v is None: return "—"
        try:
            f = float(v) * mult
            return f"{f:.1f}%" if pct else f"{f:.2f}"
        except: return "—"

    rows = []
    for t, fund, score, grade, bdown, cov, tech_sig in scored:
        conv   = conv_batch.get(t, {})
        cgrade = conv.get("conviction_grade","—")
        mcap   = fund.get("fmp_mcap")
        mcap_s = (f"${mcap/1e9:.1f}B" if mcap and mcap>=1e9
                  else f"${mcap/1e6:.0f}M" if mcap else "—")
        rows.append({
            "Ticker":     t,
            "Score":      score,
            "Grade":      grade,
            "Coverage":   f"{cov}/7",
            "Conviction": cgrade,
            "Tech":       tech_sig,
            "PE":         _fmt(fund.get("fmp_pe_ttm")),
            "PEG":        _fmt(fund.get("fmp_peg")),
            "P/B":        _fmt(fund.get("fmp_pb")),
            "FCF Yield":  _fmt(fund.get("fmp_fcf_yield"), pct=True, mult=100),
            "ROE":        _fmt(fund.get("fmp_roe"),        pct=True, mult=100),
            "D/E":        _fmt(fund.get("fmp_debt_eq")),
            "Rev Growth": _fmt(fund.get("fmp_rev_growth"), pct=True, mult=100),
            "MCap":       mcap_s,
        })

    df_out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df_out.insert(0, "#", range(1, len(df_out)+1))

    n_a  = sum(1 for r in rows if r["Grade"]=="A")
    n_b  = sum(1 for r in rows if r["Grade"]=="B")
    n_hi = sum(1 for r in rows if "HIGH" in str(r.get("Conviction","")))
    st.success(f"✅ {len(rows)} passed · Grade A: {n_a} · Grade B: {n_b} · High conviction: {n_hi}")

    def _style_vs(val):
        if val == "A":   return "color: #0d9488; font-weight: 700;"
        if val == "B":   return "color: #0284c7; font-weight: 600;"
        if val == "C":   return "color: #d97706;"
        if val == "D":   return "color: #dc2626;"
        if val == "BUY": return "color: #0d9488; font-weight: 700;"
        if val == "WATCH": return "color: #0284c7; font-weight: 600;"
        if val == "SELL":  return "color: #dc2626; font-weight: 700;"
        return ""

    st.dataframe(
        df_out.style.applymap(_style_vs, subset=["Grade","Tech"]),
        use_container_width=True,
        height=min(600, 40 + len(df_out)*35),
        hide_index=True,
    )

    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "value_screen.csv", "text/csv")


# ───────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────

def main():
    preset, filters, budget, workers, fetch_pe, tickers, vix, fg, rm = render_sidebar()

    tab_scanner, tab_deepdive, tab_value = st.tabs([
        "🔭 Market Scanner",
        "🔬 Deep Dive",
        "💎 Value Screen",
    ])

    with tab_scanner:
        render_scanner(tickers, budget, workers, fetch_pe, vix, fg, rm)

    with tab_deepdive:
        render_deepdive(budget)

    with tab_value:
        render_value_screen()

if __name__ == "__main__":
    main()
