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
from datetime import datetime, date, timedelta, timezone

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


APP_VERSION = "v11.0"
BUILD_TIME = datetime.now(timezone.utc).strftime("%d %b %H:%M UTC")

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
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* Hide header text but keep the sidebar toggle button visible */
header { visibility: hidden; }
header [data-testid="stToolbar"] { visibility: visible !important; }
.block-container { padding-top: 1rem !important; }
/* Always show sidebar collapse/expand toggle */
[data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; }
section[data-testid="stSidebar"] { display: block !important; }

/* Fix tab click area — Streamlit tabs have an inner p tag that blocks clicks */
[data-baseweb="tab"] { cursor: pointer !important; }
[data-baseweb="tab"] * { pointer-events: none !important; }
[data-baseweb="tab-list"] button { cursor: pointer !important; }
/* Ensure the top-level tab container catches all clicks */
[data-testid="stTabs"] [role="tab"] { cursor: pointer !important; position: relative !important; z-index: 1 !important; }

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

_DISK_CACHE_FILE = "/tmp/mde_fmp_cache.json"  # kept for compatibility but not used

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

def _save_disk_cache():
    pass  # no-op — Supabase write-back handles persistence

def _load_disk_cache():
    pass  # no-op


# ───────────────────────────────────────────────────────────────────
# UNIVERSE LOADERS  (cached globally — loaded once)
# ───────────────────────────────────────────────────────────────────

def _get_supabase():
    """Return a Supabase client using Streamlit secrets."""
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_base_universe():
    """Load universe from Supabase."""
    try:
        sb = _get_supabase()
        if sb is None:
            return pd.DataFrame()
        rows = []
        page, page_size = 0, 1000
        while True:
            res = sb.table("universe").select("*").range(page*page_size, (page+1)*page_size-1).execute()
            if not res.data:
                break
            rows.extend(res.data)
            if len(res.data) < page_size:
                break
            page += 1
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ["country","sector","name","category_group","category",
                    "currency","yf_symbol","yf_suffix","isin"]:
            if col not in df.columns:
                df[col] = ""
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("").astype(str).str.strip()
        df = df[df["ticker"].str.match(r"^[A-Z0-9]{2,6}$")]
        return df.drop_duplicates(subset=["ticker","type"], keep="first")
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_justetf():
    """Build justETF-style df from universe table (domicile/dist_policy/ter/replication/strategy)."""
    try:
        df = load_base_universe()
        if df.empty:
            return pd.DataFrame()
        etfs = df[df["type"]=="ETF"].copy()
        # Rename name -> jname for compatibility
        if "name" in etfs.columns:
            etfs = etfs.rename(columns={"name":"jname"})
        keep = [c for c in ["ticker","isin","jname","domicile","ter",
                             "dist_policy","fund_size_eur","replication","strategy"]
                if c in etfs.columns]
        return etfs[keep].drop_duplicates(subset=["ticker"], keep="first")
    except Exception:
        return pd.DataFrame()

SIGNALS_COLS = (
    "ticker,action,score,price,ma200,dist_ma200,rsi,rsi_rising,"
    "macd_bull,macd_accel,vol_pct,conf,is_knife,reversal,"
    "dist_52w,pe_ratio,div_yield,market_cap,beta,value_score,"
    "value_grade,name,isin,type,country,sector,domicile,"
    "dist_policy,ter,fund_size_eur,replication,strategy,"
    "roe,rev_growth,debt_equity,fcf_yield,peg,"
    "data_source,computed_at"
)

@st.cache_data(ttl=3600)
def load_signals():
    """Load all signals from Supabase with pagination."""
    try:
        sb = _get_supabase()
        if sb is None:
            return pd.DataFrame()
        rows = []
        page_size = 1000
        page = 0
        while True:
            res = (sb.table("signals")
                     .select(SIGNALS_COLS)
                     .range(page * page_size, (page + 1) * page_size - 1)
                     .execute())
            if not res.data:
                break
            rows.extend(res.data)
            if len(res.data) < page_size:
                break
            page += 1
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False).drop_duplicates(
                subset=["ticker"], keep="first").reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame()


# Load disk cache at startup (safe — no st calls)
_load_disk_cache()

# Lazy loaders — called inside Streamlit execution context, not at import time
def get_universe():
    if "universe" not in st.session_state:
        st.session_state["universe"] = load_base_universe()
    return st.session_state["universe"]

def get_jetf_df():
    if "jetf_df" not in st.session_state:
        st.session_state["jetf_df"] = load_justetf()
    return st.session_state["jetf_df"]

def get_signals_df():
    """Return signals_df — load from CSV on first call, prefer live session updates."""
    if "signals_df" not in st.session_state:
        st.session_state["signals_df"] = load_signals()
    return st.session_state["signals_df"]

def _push_signal_to_supabase(row_dict):
    """
    Persist an enriched signal row back to Supabase in a background thread.
    Called after Deep Dive fetches live fundamentals — so subsequent app loads
    show updated score/grade/fundamentals without re-fetching from yfinance.
    Fire-and-forget: never blocks the UI.
    """
    import math, threading

    def _do_push():
        try:
            sb = _get_supabase()
            if sb is None:
                return
            INT_COLS = {"rsi_rising","macd_bull","macd_accel","is_knife","reversal","momentum_up"}
            SIGNAL_COLS = {
                "ticker","action","score","price","ma50","ma200","dist_ma200","dist_52w",
                "rsi","rsi_rising","macd","macd_signal","macd_hist","macd_bull","macd_accel",
                "vol_pct","conf","is_knife","reversal","ret_1w","ret_1m","ret_3m","ret_6m","ret_1y",
                "momentum_up","pe_ratio","div_yield","market_cap","beta","pb_ratio","roe",
                "debt_equity","fcf_yield","rev_growth","peg","value_score","value_grade","fund_delta",
                "name","isin","type","country","sector","currency","domicile","dist_policy",
                "ter","fund_size_eur","replication","strategy","data_source","last_source",
                "computed_at","time_updated",
            }
            clean = {}
            for k, v in row_dict.items():
                if k not in SIGNAL_COLS:
                    continue
                if v is None:
                    clean[k] = None
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean[k] = None
                elif isinstance(v, str) and v in ("nan","None","NaN",""):
                    clean[k] = None
                elif k in INT_COLS:
                    try:    clean[k] = int(v)
                    except: clean[k] = None
                else:
                    clean[k] = v
            if "ticker" not in clean:
                return
            sb.table("signals").upsert(clean, on_conflict="ticker").execute()
        except Exception:
            pass  # never crash the UI for a background sync

    threading.Thread(target=_do_push, daemon=True).start()


def update_signals_df(new_row_dict, source_tab=None):
    """Write a fresh signal row into session-level signals_df and persist to Supabase."""
    sdf = get_signals_df().copy()
    if not sdf.empty and "ticker" in sdf.columns:
        sdf = sdf[sdf["ticker"] != new_row_dict["ticker"]]
    sdf = pd.concat([pd.DataFrame([new_row_dict]), sdf], ignore_index=True)
    st.session_state["signals_df"] = sdf
    if source_tab:
        st.session_state["_signals_updated_by"] = source_tab
    # Persist enriched signal back to Supabase in background (fire-and-forget)
    _push_signal_to_supabase(new_row_dict)


def _build_name_lookup():
    lookup = {}
    u = get_universe()
    j = get_jetf_df()
    if not u.empty and "name" in u.columns:
        for _, row in u.iterrows():
            t = str(row.get("ticker","")).strip()
            n = str(row.get("name","")).strip()
            if t and n not in ("","nan","None"):
                lookup[t] = n
    if not j.empty:
        for _, row in j.iterrows():
            t = str(row.get("ticker","")).strip()
            n = str(row.get("jname","")).strip()
            i = str(row.get("isin","")).strip() if pd.notna(row.get("isin","")) else ""
            if t and n not in ("","nan","None"):
                lookup[t] = (n, i)
    return lookup

def get_name_isin(ticker):
    lookup = _build_name_lookup()
    entry = lookup.get(ticker)
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return entry or ticker, ""


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

    # Primary: fear-greed PyPI package — uses CNN internal API directly
    try:
        import fear_greed
        data  = fear_greed.get()
        s     = round(float(data["score"]))
        label = str(data.get("rating","")).replace("_"," ").title()
        result = (s, f"{label} (CNN)")
        cache_set("fg", result, ttl=1800)
        return result
    except Exception:
        pass

    # Fallback: alt.me — crypto F&G, clearly labelled as such
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get("https://api.alternative.me/fng/?limit=1", headers=headers, timeout=6)
        if r.status_code == 200:
            e     = r.json().get("data", [{}])[0]
            s     = round(float(e.get("value", 50)))
            label = e.get("value_classification","").replace("_"," ").title()
            result = (s, f"{label} (alt.me crypto — CNN unavailable)")
            cache_set("fg", result, ttl=1800)
            return result
    except Exception:
        pass

    return (50, "Unavailable")


# ───────────────────────────────────────────────────────────────────
# PRICE FETCH
# ───────────────────────────────────────────────────────────────────

def _fetch_stooq(symbol):
    # Stooq blocks all automated requests — removed 2026-04
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

        if known_sfx is None and resolved_sym is None and not get_universe().empty and "yf_symbol" in get_universe().columns:
            rows = get_universe()[get_universe()["ticker"] == ticker]
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

def fetch_yf_fundamentals(ticker, timeout=20):
    cache_key = f"yfund_{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    last_error = None
    try:
        t = yf.Ticker(ticker)
        info = {}
        for method in ["get_info", "info"]:
            try:
                raw = getattr(t, method)
                info = raw() if callable(raw) else raw
                if info and len(info) > 5:
                    break
            except Exception as e:
                last_error = f"{method}: {e}"
                continue
        if not info:
            # Store the error so debug UI can surface it
            cache_set(f"yfund_err_{ticker}", str(last_error), ttl=300)
            return {}

        def _get(*keys):
            for k in keys:
                v = _safe_float(info.get(k))
                if v is not None and v > 0: return v
            return None

        pe        = _get("trailingPE","forwardPE")
        fwd_pe    = _get("forwardPE")
        div       = _get("dividendYield","trailingAnnualDividendYield")
        mcap      = _get("marketCap")
        beta      = _get("beta")
        pb        = _get("priceToBook")
        roe       = _safe_float(info.get("returnOnEquity"))
        de_raw    = _safe_float(info.get("debtToEquity"))
        rev_gr    = _safe_float(info.get("revenueGrowth"))
        eps_gr    = _safe_float(info.get("earningsGrowth"))
        fcf_raw   = _safe_float(info.get("freeCashflow"))
        fcf_yield = (fcf_raw / mcap) if fcf_raw and mcap and mcap > 0 else None
        peg       = _safe_float(info.get("trailingPegRatio"))
        if peg is None and pe and eps_gr and 0.001 < eps_gr < 5:
            try: peg = round(pe / (eps_gr * 100), 2)
            except: pass
        de_ratio = (de_raw / 100) if de_raw and de_raw > 10 else de_raw

        result = {
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
            "_source":        "yfinance",
        }
        cache_set(cache_key, result, ttl=3600)
        return result
    except Exception as e:
        cache_set(f"yfund_err_{ticker}", str(e), ttl=300)
        return {}

def fetch_yf_fundamentals_batch(tickers, max_workers=8, timeout=20):
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
    try:
        return st.secrets.get("FMP_API_KEY", "").strip()
    except Exception:
        return os.environ.get("FMP_API_KEY", "").strip()

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
    """
    Build ticker list entirely from signals_df — no universe fetch needed.
    Universe is only loaded when filters expander is opened (country/sector dropdowns).
    """
    preset = PRESETS.get(preset_key, {"type":"ETF"})
    ptype  = preset.get("type","ETF")
    sdf    = get_signals_df()

    if sdf.empty:
        return []

    # Start from all signals rows matching the asset type
    if ptype == "ETF":
        mask = sdf["type"] == "ETF" if "type" in sdf.columns else pd.Series([True]*len(sdf))
    elif ptype == "Stock":
        mask = sdf["type"] == "Stock" if "type" in sdf.columns else pd.Series([True]*len(sdf))
    else:  # custom
        mask = pd.Series([True]*len(sdf), index=sdf.index)

    df = sdf[mask].copy()

    # Preset-level filters (country for stock presets, domicile for ETF presets)
    if "country" in preset and "country" in df.columns:
        df = df[df["country"].isin(preset["country"])]
    if "domicile" in preset and "domicile" in df.columns:
        df = df[df["domicile"].isin(preset["domicile"])]

    # User filter overrides (only applied when filters expander is used)
    for col, fkey in [("domicile","domicile"),("dist_policy","dist_policy"),
                      ("replication","replication"),("strategy","strategy"),
                      ("country","country"),("sector","sector")]:
        if filters.get(fkey) and col in df.columns:
            df = df[df[col].isin(filters[fkey])]
    if filters.get("min_size",0) > 0 and "fund_size_eur" in df.columns:
        df = df[df["fund_size_eur"].fillna(0) >= filters["min_size"]]
    if filters.get("max_ter",2.0) < 2.0 and "ter" in df.columns:
        df = df[df["ter"].fillna(99) <= filters["max_ter"]]

    def _is_bad(t):
        t = str(t).strip()
        if len(t) < 2: return True
        if t[-1] in ("W","U","R") and len(t) >= 4: return True
        return False

    tickers = [t for t in df["ticker"].dropna().str.upper().tolist() if not _is_bad(t)]
    return list(dict.fromkeys(tickers))




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

    # Recently viewed tickers
    viewed = st.session_state.get("_viewed_tickers", [])
    if viewed:
        st.sidebar.markdown("**🕐 Recently viewed**")
        cols = st.sidebar.columns(min(len(viewed), 3))
        for i, t in enumerate(viewed[-3:]):
            if cols[i].button(t, key=f"rv_{t}", use_container_width=True):
                st.session_state["dd_ticker"]   = t
                st.session_state["dd_auto"]     = True
                st.session_state["_active_tab"] = 1
                st.rerun()
        st.sidebar.divider()

    # (2) Refresh button — busts VIX + F&G cache so re-fetch is live
    if st.sidebar.button("🔄 Refresh indicators", use_container_width=True):
        store = _get_cache_store()
        lock  = _get_cache_lock()
        with lock:
            store.pop("vix", None)
            store.pop("fg",  None)

    # Live indicators
    vix      = get_live_vix()
    fg, lbl  = get_fg_index()
    rm       = 1.0 + ((vix/20) * ((100-fg)/50))
    gauge    = "🔴" if fg<=25 else "🟠" if fg<=44 else "🟡" if fg<=55 else "🟢" if fg<=75 else "💚"

    c1, c2 = st.sidebar.columns(2)
    c1.metric("VIX", f"{vix:.1f}")
    c2.metric("Fear & Greed", f"{gauge} {fg}")
    st.sidebar.caption(lbl)

    # (1) Validation links
    st.sidebar.markdown(
        "[📈 Live VIX](https://finance.yahoo.com/quote/%5EVIX/)  ·  "
        "[🧠 CNN F&G](https://edition.cnn.com/markets/fear-and-greed)  ·  "
        "[🌐 Alt.me](https://alternative.me/crypto/fear-and-greed-index/)"
    )

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

    # Filters — all options derived from signals_df (no universe fetch on page load)
    filters = {}
    with st.sidebar.expander("🔧 Optional Filters", expanded=False):
        _sdf = get_signals_df()  # already loaded — no extra fetch

        if ptype == "custom":
            types = st.multiselect("Asset Type", ["ETF","Stock"], default=["ETF"])
            filters["types"] = types

        if ptype in ("ETF","custom"):
            _etfs = _sdf[_sdf["type"]=="ETF"] if "type" in _sdf.columns else _sdf
            def _opts(col, fallback):
                if col in _etfs.columns:
                    vals = sorted(_etfs[col].dropna().astype(str).unique().tolist())
                    vals = [v for v in vals if v not in ("","nan","None")]
                    if vals: return vals
                return fallback
            st.markdown("**📦 ETF Filters**")
            filters["domicile"]    = st.multiselect("Domicile",     _opts("domicile",    ["Ireland","Luxembourg","Germany","United States"]))
            filters["dist_policy"] = st.multiselect("Distribution", _opts("dist_policy", ["Accumulating","Distributing"]))
            filters["replication"] = st.multiselect("Replication",  _opts("replication", ["Physical (Full)","Physical (Sampling)","Swap-based"]))
            filters["min_size"]    = st.number_input("Min Size €m", min_value=0, value=0, step=50)
            filters["max_ter"]     = st.number_input("Max TER %",   min_value=0.0, max_value=5.0, value=2.0, step=0.05)

        if ptype in ("Stock","custom"):
            _stocks = _sdf[_sdf["type"]=="Stock"] if "type" in _sdf.columns else _sdf
            def _sopts(col):
                if col in _stocks.columns:
                    vals = sorted(_stocks[col].dropna().astype(str).unique().tolist())
                    return [v for v in vals if v not in ("","nan","None")]
                return []
            st.markdown("**📈 Stock Filters**")
            filters["country"] = st.multiselect("Country", _sopts("country"))
            filters["sector"]  = st.multiselect("Sector",  _sopts("sector"))

    st.sidebar.divider()

    budget  = st.sidebar.number_input("💰 Monthly Budget (EUR)", min_value=100, value=1000, step=100)

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

    return preset, filters, budget, tickers, vix, fg, rm


# ───────────────────────────────────────────────────────────────────
# STRATEGY CLASSIFIER
# ───────────────────────────────────────────────────────────────────

def classify_strategies(s):
    """
    Tag each row with one or more strategy flags.
    Returns DataFrame with added boolean columns:
      is_core, is_value, is_momentum, is_darkhorse
    All vectorised — no Python loops.
    """
    s = s.reset_index(drop=True)

    # Numeric columns with safe fallbacks
    dm    = pd.to_numeric(s.get("dist_ma200"),  errors="coerce").fillna(0)
    rsi   = pd.to_numeric(s.get("rsi"),         errors="coerce").fillna(50)
    mb    = pd.to_numeric(s.get("macd_bull",0), errors="coerce").fillna(0)
    vg    = s.get("value_grade","—").fillna("—").astype(str) if "value_grade" in s.columns else pd.Series(["—"]*len(s))
    roe   = pd.to_numeric(s.get("roe",   pd.Series([np.nan]*len(s))), errors="coerce")
    rev_g = pd.to_numeric(s.get("rev_growth", pd.Series([np.nan]*len(s))), errors="coerce")
    atype = s.get("type","").fillna("").astype(str) if "type" in s.columns else pd.Series([""]*len(s))
    action= s.get("action","WAIT").fillna("WAIT").astype(str)

    # 🟡 CORE — ETF + stock timing signals, risk-tiered
    is_etf  = atype.str.upper() == "ETF"

    # ── CORE risk tier classification ─────────────────────────────────
    # Keywords that indicate speculative / volatile ETFs
    name_col = s.get("name", pd.Series([""]*len(s))).fillna("").str.lower() if "name" in s.columns else pd.Series([""]*len(s))
    strat_col = s.get("strategy", pd.Series([""]*len(s))).fillna("").str.lower() if "strategy" in s.columns else pd.Series([""]*len(s))
    combined_name = name_col + " " + strat_col

    speculative_kw = ["crypto","bitcoin","blockchain","ethereum","nuclear","uranium",
                      "leverage","inverse","short","3x","2x","volatil","ark ",
                      "cannabis","marijuana","gaming","esport","space","metaverse"]
    sector_kw      = ["tech","cyber","security","health","pharma","biotech","energy",
                      "clean","solar","wind","real estate","reit","financial","bank",
                      "india","china","emerging","asia","latin","africa","korea"]

    ter_col  = pd.to_numeric(s.get("ter",  pd.Series([np.nan]*len(s))), errors="coerce")
    aum_col  = pd.to_numeric(s.get("fund_size_eur", pd.Series([np.nan]*len(s))), errors="coerce")

    is_speculative_etf = is_etf & (
        combined_name.apply(lambda n: any(kw in n for kw in speculative_kw)) |
        (ter_col > 0.50) |
        (aum_col < 100)
    )
    is_sector_etf = is_etf & ~is_speculative_etf & (
        combined_name.apply(lambda n: any(kw in n for kw in sector_kw))
    )
    is_steady_etf = is_etf & ~is_speculative_etf & ~is_sector_etf

    # Stocks in CORE = always speculative (pure technical, no fundamental check)
    is_stock_core = (~is_etf) & action.isin(["BUY","WATCH"]) & (dm < -5)

    # Risk tier labels
    core_risk = np.where(is_steady_etf,      "🟢 Steady",
                np.where(is_sector_etf,      "🟡 Sector",
                np.where(is_speculative_etf, "🔴 Speculative",
                np.where(is_stock_core,      "🔴 Speculative (Stock)",
                ""))))

    is_core = (is_etf | is_stock_core) & action.isin(["BUY","WATCH"]) & (dm < -5)

    # ── Has fundamental data? ─────────────────────────────────────────
    has_fund  = rev_g.notna() | roe.notna()   # at least one fundamental present
    has_rev   = rev_g.notna()
    has_roe   = roe.notna()

    is_stock  = ~is_etf
    grade_ok  = vg.isin(["A","B"])

    # 🔵 VALUE — two tiers:
    # Tier 1 (full): oversold + Grade A/B + ROE >10% + positive revenue growth
    # Tier 2 (tech-only): oversold + strong technicals when fundamentals missing
    roe_ok   = has_roe & (roe > 0.10)
    rev_ok   = has_rev & (rev_g > 0)
    value_full = (is_stock & (dm < -10) & (rsi < 48) & (mb > 0)
                  & grade_ok & roe_ok & rev_ok & action.isin(["BUY","WATCH"]))
    value_tech = (is_stock & (dm < -15) & (rsi < 40) & (mb > 0)
                  & ~has_fund & action.isin(["BUY","WATCH"]))
    is_value   = value_full | value_tech

    # 🔴 MOMENTUM — require actual revenue growth >20%
    rev_high    = has_rev & (rev_g > 0.20)
    is_momentum = (is_stock
                   & (dm > -5)
                   & (rsi >= 50) & (rsi <= 72)
                   & (mb > 0)
                   & rev_high
                   & action.isin(["BUY","WATCH","SELL"]))

    # ⚡ DARK HORSE — require actual revenue growth >15%
    rev_pos      = has_rev & (rev_g > 0.15)
    rsi_rec      = (rsi >= 28) & (rsi < 48)
    is_darkhorse = (is_stock
                    & (dm < -15)
                    & rsi_rec
                    & (mb > 0)
                    & rev_pos
                    & ~grade_ok
                    & action.isin(["BUY","WATCH"]))

    # De-duplicate: if stock qualifies for both value and darkhorse → value wins
    is_darkhorse = is_darkhorse & ~is_value

    s = s.copy()
    s["is_core"]      = is_core.values
    s["core_risk"]    = core_risk
    s["is_value"]     = is_value.values
    s["is_momentum"]  = is_momentum.values
    s["is_darkhorse"] = is_darkhorse.values
    return s


def build_result_df(sig, budget, fg, rm):
    """Vectorised result DataFrame builder."""
    s = sig.copy()
    s["action"]     = s["action"].fillna("WAIT").astype(str)
    s["conf"]       = pd.to_numeric(s.get("conf"),      errors="coerce").fillna(0.5)
    s["vol_pct"]    = pd.to_numeric(s.get("vol_pct"),   errors="coerce").fillna(2.0)
    s["dist_ma200"] = pd.to_numeric(s.get("dist_ma200"),errors="coerce").fillna(0.0)
    s["rsi"]        = pd.to_numeric(s.get("rsi"),       errors="coerce").fillna(50.0)
    s["score"]      = pd.to_numeric(s.get("score"),     errors="coerce").fillna(0.0)

    budget_val = budget or 1000
    ctx_s = (40 if fg < 35 else 20 if fg < 50 else 0) / 100
    dm    = s["dist_ma200"]
    rsi_v = s["rsi"]
    conf  = s["conf"]
    vol   = s["vol_pct"]
    sig_s = (np.minimum(-dm/20, 1)*0.5 + np.minimum((50-rsi_v)/50, 1)*0.5).where(dm < 0, 0)
    amt   = (budget_val*(ctx_s+sig_s)*(0.5+conf)*np.maximum(0.5,1-vol/20)*rm
             ).clip(budget_val*0.25, budget_val*3)
    tier  = np.where(amt>=budget_val*1.5,"🔥",np.where(amt>=budget_val*0.8,"⚖️","🔍"))
    alloc_buy   = tier + " €" + amt.round(0).astype(int).astype(str)
    alloc_watch = "👀 €" + (budget_val*0.25*(0.5+conf)).round(0).astype(int).astype(str)
    s["Alloc"] = np.where(s["action"]=="AVOID","⛔ Skip",
                 np.where(s["action"]=="WAIT","—",
                 np.where(s["action"]=="WATCH",alloc_watch,alloc_buy)))

    # Key driver label
    vg  = s.get("value_grade","—").fillna("—").astype(str) if "value_grade" in s.columns else pd.Series(["—"]*len(s))
    rg  = pd.to_numeric(s.get("rev_growth", pd.Series([np.nan]*len(s))), errors="coerce")
    rg_str = (rg*100).round(0).astype("Int64").astype(str).replace("<NA>","—") + "% rev"
    core_risk_col = s.get("core_risk", pd.Series([""]*len(s))).fillna("").astype(str)
    driver = np.where(s.get("is_core",  False), "Dip · " + core_risk_col,
             np.where(s.get("is_value", False), "Value · Gr." + vg.values,
             np.where(s.get("is_momentum",False), "Momentum · " + rg_str.values,
             np.where(s.get("is_darkhorse",False), "Dark horse · " + rg_str.values,
             "—"))))

    result_df = pd.DataFrame({
        "Ticker":   s["ticker"],
        "Name":     s["name"].fillna("") if "name" in s.columns else "",
        "Signal":   s["action"],
        "Score":    s["score"].round(2),
        "Price":    pd.to_numeric(s.get("price"),    errors="coerce").round(2),
        "Dist%":    dm.round(2).fillna(0),
        "RSI":      s["rsi"].round(1),
        "MACD":     np.where(pd.to_numeric(s.get("macd_bull",0),errors="coerce").fillna(0)>0,"▲","▼"),
        "Grade":    vg.values,
        "RevGr%":   (rg*100).round(1).astype(str).replace("nan","—") if rg.notna().any() else "—",
        "Alloc":    s["Alloc"],
        "Driver":   driver,
        "Knife":    np.where((pd.to_numeric(s.get("is_knife",0),errors="coerce").fillna(0)>0)&
                             (pd.to_numeric(s.get("reversal",0),errors="coerce").fillna(0)==0),"⚠️",""),
        "Risk":     s.get("core_risk", pd.Series([""]*len(s))).fillna("").astype(str),
        # strategy flags (hidden, used for filtering)
        "_core":      s.get("is_core",      pd.Series([False]*len(s))).values,
        "_value":     s.get("is_value",     pd.Series([False]*len(s))).values,
        "_momentum":  s.get("is_momentum",  pd.Series([False]*len(s))).values,
        "_darkhorse": s.get("is_darkhorse", pd.Series([False]*len(s))).values,
    })
    return result_df


def _show_strategy_table(df, label, color, empty_msg):
    """Render a strategy sub-table with Deep Dive button above table."""
    if df.empty:
        st.info(empty_msg)
        return

    display = [c for c in df.columns if not c.startswith("_")]
    df_edit = df[display].copy()
    df_edit.insert(0, "🔬", False)

    def _style(val):
        colors = {"BUY":"#0d9488","WATCH":"#0284c7","SELL":"#dc2626","AVOID":"#d97706"}
        if val in colors: return f"color:{colors[val]};font-weight:600;"
        if val in ("A","B"): return "color:#0d9488;font-weight:700;"
        if str(val).startswith("🟢"): return "color:#0d9488;font-weight:700;"
        if str(val).startswith("🟡"): return "color:#d97706;font-weight:600;"
        if str(val).startswith("🔴"): return "color:#dc2626;font-weight:600;"
        if val == "▲": return "color:#0d9488;"
        if val == "▼": return "color:#dc2626;"
        return ""

    style_cols = [c for c in ["Signal","Grade","MACD","Risk"] if c in display]

    # ── Dive button placeholder sits ABOVE the table ──────────────────
    dive_area = st.empty()
    dive_area.caption("💡 Tick the 🔬 checkbox on any row then click Deep Dive.")

    edited = st.data_editor(
        df_edit,
        column_config={"🔬": st.column_config.CheckboxColumn("🔬", help="Tick to Deep Dive", width="small")},
        disabled=[c for c in df_edit.columns if c != "🔬"],
        use_container_width=True,
        height=min(500, 45 + len(df_edit)*35),
        hide_index=True,
        key=f"tbl_{label}",
    )

    selected = edited[edited["🔬"] == True]
    if not selected.empty:
        ticker = selected.iloc[0]["Ticker"]
        with dive_area:
            if st.button(f"🔬 Deep Dive: {ticker}", type="primary",
                         key=f"dive_{label}_{ticker}", use_container_width=True):
                st.session_state["dd_ticker"]   = ticker
                st.session_state["dd_auto"]     = True
                st.session_state["_active_tab"] = 1  # switch to Deep Dive tab
                st.rerun()

    csv = df[display].to_csv(index=False).encode("utf-8")
    st.download_button(f"⬇️ Download {label}", csv,
                       f"{label.lower().replace(' ','_')}.csv", "text/csv",
                       key=f"dl_{label}")


# ───────────────────────────────────────────────────────────────────
# TAB 1 — MARKET SCANNER
# ───────────────────────────────────────────────────────────────────

def render_scanner(tickers, budget, vix, fg, rm):
    st.subheader("🔭 Market Scanner")

    c1, c2 = st.columns([5,1])
    run_scan  = c1.button("🔄 Run Scan", type="primary", use_container_width=True)
    clear_btn = c2.button("🗑️", use_container_width=True)

    if clear_btn:
        for k in ["scan_results","scan_status"]:
            st.session_state.pop(k, None)
        st.rerun()

    sdf = get_signals_df()

    if run_scan:
        st.session_state.pop("scan_results", None)
        if sdf.empty:
            st.warning("No signals loaded from Supabase.")
            return
        sig = sdf[sdf["ticker"].isin(tickers)].copy() if tickers else sdf.copy()
        if len(sig) == 0 and not sdf.empty:
            sig = sdf.copy()
            st.caption("⚠️ No preset filter matches — showing all signals.")
        if len(sig) == 0:
            st.warning("No signals found.")
            return
        with st.spinner(f"⚡ Classifying {len(sig):,} signals…"):
            # Filters are always respected — classify only what the preset returns
            sig_classified = classify_strategies(sig)
            result_df = build_result_df(sig_classified, budget, fg, rm)
            computed = sig["computed_at"].iloc[0] if "computed_at" in sig.columns else "unknown"
            st.session_state["scan_results"] = result_df
            st.session_state["scan_status"]  = (
                f"⚡ {len(result_df):,} signals · "
                f"Core:{result_df['_core'].sum()} · "
                f"Value:{result_df['_value'].sum()} · "
                f"Momentum:{result_df['_momentum'].sum()} · "
                f"Dark Horse:{result_df['_darkhorse'].sum()} · "
                f"updated: {computed}")
        st.rerun()

    if "scan_results" not in st.session_state:
        st.info("Click **Run Scan** to load signals.")
        return

    result_df = st.session_state["scan_results"]

    # ── Guard: stale cache from old format
    if "_core" not in result_df.columns:
        st.session_state.pop("scan_results", None)
        st.info("Strategy view updated — click **Run Scan** to reload.")
        return

    # ── Derive context from the actual data, not the preset name ─────
    has_etfs   = result_df["_core"].any()
    has_stocks = result_df["_value"].any() or result_df["_momentum"].any() or result_df["_darkhorse"].any()
    n_total    = len(result_df)
    computed   = st.session_state.get("scan_status","").split("updated:")[-1].strip() if "updated:" in st.session_state.get("scan_status","") else ""

    # ── Summary bar — one source of truth ────────────────────────────
    today = datetime.now().strftime("%d %b %Y")
    age_note = f" · built: {computed.strip()}" if computed else ""
    st.caption(f"{n_total:,} signals · as of {today}{age_note}")

    n_core = int(result_df["_core"].sum())
    n_val  = int(result_df["_value"].sum())
    n_mom  = int(result_df["_momentum"].sum())
    n_dh   = int(result_df["_darkhorse"].sum())

    # Delta vs previous scan
    prev = st.session_state.get("_prev_scan_counts", {})
    d_core = n_core - prev.get("core", n_core)
    d_val  = n_val  - prev.get("val",  n_val)
    d_mom  = n_mom  - prev.get("mom",  n_mom)
    d_dh   = n_dh   - prev.get("dh",   n_dh)
    st.session_state["_prev_scan_counts"] = {"core":n_core,"val":n_val,"mom":n_mom,"dh":n_dh}

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("🟡 Timing",     n_core, delta=d_core if d_core else None, help="Dip signals — ETFs sorted by risk, stocks flagged as speculative")
    k2.metric("🔵 Value",      n_val,  delta=d_val  if d_val  else None, help="Oversold quality stocks with solid fundamentals")
    k3.metric("🔴 Momentum",   n_mom,  delta=d_mom  if d_mom  else None, help="Stocks running with strong revenue growth")
    k4.metric("⚡ Dark Horse", n_dh,   delta=d_dh   if d_dh   else None, help="Beaten-down stocks with high growth potential")

    # ── Dynamic tab labels — no hardcoded asset class ────────────────
    tab_core, tab_value, tab_mom, tab_dh, tab_live = st.tabs([
        f"🟡 Timing ({n_core})",
        f"🔵 Value ({n_val})",
        f"🔴 Momentum ({n_mom})",
        f"⚡ Dark Horse ({n_dh})",
        "🎯 Best Picks",
    ])

    with tab_core:
        st.caption("Dip timing signals — sorted by risk. ETFs are most suitable; stocks included with caution warnings.")
        with st.expander("ℹ️ How does this work?", expanded=False):
            st.markdown(
                "**What this tab shows:** Assets that have dipped and are showing early recovery signs, "
                "sorted by risk so you know what you're getting into.\n\n"
                "**The Risk column:**\n"
                "- 🟢 **Steady** — broad diversified ETFs (world index, S&P 500, bonds). "
                "Dips here are almost always temporary. Lowest risk.\n"
                "- 🟡 **Sector** — sector or country ETFs (tech, healthcare, India etc). "
                "More concentrated, so dips can be deeper. Moderate risk.\n"
                "- 🔴 **Speculative** — niche ETFs (crypto, nuclear, ARK-style) and individual stocks. "
                "Same technical dip signal, but these can stay down or go further. Tread carefully.\n\n"
                "**To qualify:** Price 5%+ below 200-day average · RSI under 48 · MACD turning bullish.\n\n"
                "**What to do:** Focus on 🟢 Steady first for capital preservation. "
                "Use 🟡 Sector for tactical bets. Size 🔴 Speculative positions smaller."
            )
        df_core = result_df[result_df["_core"]].copy()
        if not df_core.empty:
            risk_order = {"🟢 Steady": 0, "🟡 Sector": 1, "🔴 Speculative": 2, "🔴 Speculative (Stock)": 3, "": 4}
            df_core["_risk_n"] = df_core["Risk"].map(risk_order).fillna(4)
            df_core = df_core.sort_values(["_risk_n","Score"], ascending=[True,False]).drop(columns=["_risk_n"]).reset_index(drop=True)
            df_core.insert(0,"#",range(1,len(df_core)+1))
            n_steady = (df_core["Risk"]=="🟢 Steady").sum()
            n_sector = (df_core["Risk"]=="🟡 Sector").sum()
            n_spec   = df_core["Risk"].str.startswith("🔴").sum()
            c1,c2,c3 = st.columns(3)
            c1.metric("🟢 Steady",     n_steady)
            c2.metric("🟡 Sector",     n_sector)
            c3.metric("🔴 Speculative",n_spec)
            if n_spec > 0:
                st.warning("⚠️ Speculative signals at the bottom include individual stocks and niche ETFs. "
                           "The dip signal is the same, but recovery is not guaranteed. Size these positions smaller.")
            def _style_core(val):
                if val == "🟢 Steady":           return "color:#0d9488;font-weight:700;"
                if val == "🟡 Sector":           return "color:#d97706;font-weight:600;"
                if str(val).startswith("🔴"):    return "color:#dc2626;font-weight:600;"
                c = {"BUY":"#0d9488","WATCH":"#0284c7","SELL":"#dc2626","AVOID":"#d97706"}
                if val in c: return f"color:{c[val]};font-weight:600;"
                if val == "▲": return "color:#0d9488;"
                if val == "▼": return "color:#dc2626;"
                return ""
            display = [c for c in df_core.columns if not c.startswith("_")]

            # ── Visual risk grouping with section headers ─────────────
            steady_df = df_core[df_core["Risk"] == "🟢 Steady"].copy()
            sector_df = df_core[df_core["Risk"] == "🟡 Sector"].copy()
            spec_df   = df_core[df_core["Risk"].str.startswith("🔴", na=False)].copy()

            if not steady_df.empty:
                st.markdown("#### 🟢 Steady — Broad diversified ETFs · Lowest risk")
                st.caption("World index, S&P 500, bonds. Dips here are almost always mean-reversion opportunities.")
                _show_strategy_table(steady_df.reset_index(drop=True), "Timing_Steady", "#0d9488",
                    "No steady ETF signals.")

            if not sector_df.empty:
                st.markdown("#### 🟡 Sector — Sector & country ETFs · Moderate risk")
                st.caption("Tech, healthcare, India, emerging markets etc. More concentrated — dips can be deeper.")
                _show_strategy_table(sector_df.reset_index(drop=True), "Timing_Sector", "#d97706",
                    "No sector ETF signals.")

            if not spec_df.empty:
                st.markdown("#### 🔴 Speculative — Niche ETFs & stocks · Higher risk")
                st.caption("Crypto, nuclear, ARK-style, individual stocks. Same dip signal — recovery not guaranteed. Size smaller.")
                _show_strategy_table(spec_df.reset_index(drop=True), "Timing_Spec", "#dc2626",
                    "No speculative signals.")
        else:
            st.info("No timing signals in this filter set. Switch to an ETF preset (e.g. All ETFs or UCITS ETFs) to see dip opportunities.")

    with tab_value:
        st.caption("Oversold quality stocks with solid fundamentals — medium-term mean reversion plays.")
        with st.expander("ℹ️ How does this work?", expanded=False):
            st.markdown(
                "**What this tab shows:** Good businesses whose price has temporarily fallen — a sale on a stock you'd want to own anyway.\n\n"
                "**To qualify (ALL must pass):** 10%+ below 200-day average · RSI under 48 · MACD bullish · Grade A or B · ROE above 10% · Revenue growing.\n\n"
                "**Grade** = quality score from PE, P/B, Free Cash Flow, ROE, Debt, and Revenue growth. A = top quality. D = weak or expensive.\n\n"
                "**What to do:** Medium-term holds (6–18 months). The idea is the price recovers toward fair value as the market recognises the quality."
            )
        df_val = result_df[result_df["_value"]].sort_values("Score", ascending=False).reset_index(drop=True)
        df_val.insert(0,"#",range(1,len(df_val)+1))
        _show_strategy_table(df_val, "Value", "#0284c7",
            "No value signals in this filter set. Needs Grade A/B + oversold technicals + fundamental data. Try Global Stocks or US Stocks preset.")

    with tab_mom:
        st.caption("Stocks already running — strong revenue growth confirmed by the price action.")
        with st.expander("ℹ️ How does this work?", expanded=False):
            st.markdown(
                "**What this tab shows:** Companies growing fast where the stock price is already reflecting it. We buy strength here, not weakness.\n\n"
                "**To qualify (ALL must pass):** Price near or above 200-day average · RSI 50–72 · MACD bullish · Revenue growing 20%+.\n\n"
                "**Key difference from Value:** Value buys dips. Momentum buys strength. A stock appearing here would likely fail the Value screen — that's intentional.\n\n"
                "**Risk:** Momentum reverses sharply. Needs more active monitoring than Value picks."
            )
        df_mom = result_df[result_df["_momentum"]].sort_values("Score", ascending=False).reset_index(drop=True)
        df_mom.insert(0,"#",range(1,len(df_mom)+1))
        _show_strategy_table(df_mom, "Momentum", "#dc2626",
            "No momentum signals yet. This strategy requires revenue growth data (>20%) which needs the fundamentals rebuild. "
            "Once fundamentals are loaded, try the US Stocks or Global Stocks preset.")

    with tab_dh:
        st.caption("Beaten-down stocks with high revenue growth — the market is wrong, or early.")
        with st.expander("ℹ️ How does this work?", expanded=False):
            st.markdown(
                "**What this tab shows:** Companies the market has punished, but whose revenue is still growing fast.\n\n"
                "**To qualify (ALL must pass):** 15%+ below 200-day average · RSI 28–48 recovering · MACD turning bullish · Revenue growing 15%+ · NOT Grade A/B.\n\n"
                "**Why not Grade A/B?** Those go to Value. Dark Horses are the riskier version — growth story may be real, financials aren't as clean.\n\n"
                "**Be realistic:** Most dark horses don't come good. Use smaller position sizes. RevGr% is the critical column — no growth = not a dark horse, just a broken stock."
            )
        df_dh = result_df[result_df["_darkhorse"]].sort_values("Score", ascending=False).reset_index(drop=True)
        df_dh.insert(0,"#",range(1,len(df_dh)+1))
        _show_strategy_table(df_dh, "Dark Horse", "#7c3aed",
            "No dark horse signals yet. This strategy requires revenue growth data (>15%) from the fundamentals rebuild. "
            "Once fundamentals are loaded, beaten-down high-growth companies will appear here.")

    with tab_live:
        st.caption("Best picks across all strategies — one view, ranked and interleaved.")
        with st.expander("ℹ️ How does this work?", expanded=False):
            st.markdown(
                "**What this tab shows:** Top picks from each strategy combined — so you don't need to read all four tabs.\n\n"
                "**How it's built:** Top 10 from each strategy, interleaved round-robin. No single strategy dominates.\n\n"
                "**Strategy column:** 🟡 Timing = dip signal · 🔵 Value = quality on sale · 🔴 Momentum = running strong · ⚡ Dark Horse = hidden growth.\n\n"
                "**How to use it:** Pick the strategies that fit your risk appetite and focus on those rows. Always Deep Dive before committing capital."
            )
        N = 10
        frames = []
        for strat, label, flag in [
            ("🟡 Timing",     "Timing",     "_core"),
            ("🔵 Value",      "Value",      "_value"),
            ("🔴 Momentum",   "Momentum",   "_momentum"),
            ("⚡ Dark Horse", "Dark Horse", "_darkhorse"),
        ]:
            sub = result_df[result_df[flag]].sort_values("Score", ascending=False).head(N).copy()
            if not sub.empty:
                sub.insert(0, "Strategy", strat)
                frames.append(sub)
        if frames:
            live_df = pd.concat(frames, ignore_index=True)
            live_df["_sr"] = live_df.groupby("Strategy").cumcount()
            live_df = live_df.sort_values(["_sr","Score"], ascending=[True,False]).drop(columns=["_sr"]).reset_index(drop=True)
            live_df.insert(0,"#",range(1,len(live_df)+1))
            _show_strategy_table(live_df, "Best Picks", "#1a56db", "No signals across any strategy.")
        else:
            st.info("No signals found across any strategy. Try a broader preset or rebuild signals with fundamentals.")


# ───────────────────────────────────────────────────────────────────
# TAB 2 — DEEP DIVE
# ───────────────────────────────────────────────────────────────────

def render_deepdive(budget):
    st.subheader("🔬 Deep Dive")

    c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
    ticker_input = c1.text_input("Ticker or ISIN", placeholder="e.g. VWRA or IE00B3RBWM25")
    budget_dd    = c2.number_input("Budget (EUR)", min_value=100, value=budget, step=100)
    analyse_btn  = c3.button("🔍 Analyse", type="primary")
    refresh_btn  = c4.button("🔄 Refresh", help="Force live re-fetch, busting all caches for this ticker")

    # Allow clicking from scanner
    if "dd_ticker" in st.session_state and not ticker_input:
        ticker_input = st.session_state["dd_ticker"]

    # Auto-trigger from scanner checkbox
    auto_dive = st.session_state.pop("dd_auto", False)
    if not ticker_input and "dd_ticker" in st.session_state:
        ticker_input = st.session_state["dd_ticker"]

    if not (analyse_btn or refresh_btn or auto_dive) or not ticker_input:
        st.info("Enter a ticker or ISIN and click **Analyse**.")
        return

    ticker       = ticker_input.strip().upper()
    force_refr   = refresh_btn  # always force-refresh on Refresh button

    # Track viewed tickers
    viewed = st.session_state.get("_viewed_tickers", [])
    if ticker not in viewed:
        viewed.append(ticker)
    st.session_state["_viewed_tickers"] = viewed[-10:]  # keep last 10

    # ISIN → ticker lookup
    isin = None
    if len(ticker) == 12 and ticker[:2].isalpha():
        isin = ticker
        # Look up in justETF
        if not get_jetf_df().empty and "isin" in get_jetf_df().columns:
            match = get_jetf_df()[get_jetf_df()["isin"] == isin]
            if not match.empty:
                ticker = match.iloc[0]["ticker"]

    with st.spinner(f"{'🔄 Force-refreshing' if force_refr else 'Fetching'} {ticker}…"):
        raw = fetch_ticker_data(ticker, isin=isin, force_refresh=True)  # always live in Deep Dive

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
    is_etf = not get_universe().empty and ticker in get_universe()[get_universe()["type"]=="ETF"]["ticker"].values

    # Fundamentals
    fund_data = {}
    if not is_etf:
        with st.spinner("Fetching fundamentals…"):
            fund_data = fetch_yf_fundamentals(ticker, timeout=20)
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
    if not get_jetf_df().empty and "ticker" in get_jetf_df().columns and ticker in get_jetf_df()["ticker"].values:
        jetf_meta = get_jetf_df()[get_jetf_df()["ticker"]==ticker].iloc[0].to_dict()

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
    caption_parts = []
    if isin: caption_parts.append(f"ISIN: {isin}")
    if is_etf: caption_parts.append("ETF")
    else: caption_parts.append("Stock")
    caption_parts.append(f"Updated: {date.today().strftime('%d %b %Y')}")
    st.caption("  ·  ".join(caption_parts))

    ma200_label = {"rising":"↗ Rising","falling":"↘ Falling","flat":"→ Flat"}.get(ma200_trend,"→ Flat")
    value_label = f"{value_score}/100 (Grade {value_grade})" if value_available else "—"

    # 52w high/low
    hi52 = float(close.tail(252).max()) if len(close) >= 50 else cur_p
    lo52 = float(close.tail(252).min()) if len(close) >= 50 else cur_p
    dist_hi = (cur_p - hi52) / hi52 * 100 if hi52 else 0
    dist_lo = (cur_p - lo52) / lo52 * 100 if lo52 else 0

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Price",        f"€{cur_p:.2f}", delta=f"{dist_ma:+.1f}% vs MA200")
    k2.metric("52w High",     f"€{hi52:.2f}",  delta=f"{dist_hi:+.1f}%")
    k3.metric("52w Low",      f"€{lo52:.2f}",  delta=f"{dist_lo:+.1f}%")
    k4.metric("RSI",          f"{rsi_val:.1f}", delta="↗ Rising" if rsi_rising else "↘ Falling",
              delta_color="normal" if rsi_rising else "inverse")
    k5.metric("MACD",         "▲ Bullish" if macd_bull else "▼ Bearish",
              delta="Accelerating" if macd_accel else None)
    k6.metric("Value Score",  value_label)

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

    # ── Plain-English signal explanation ─────────────────────────────
    with st.expander("🧠 Why is the system saying this? (plain English)", expanded=False):
        reasons = []
        if dist_ma < -20:
            reasons.append(f"📉 **Price is {abs(dist_ma):.0f}% below its long-term average** — a significant dip. Assets this far below their 200-day average tend to mean-revert upward over time.")
        elif dist_ma < -5:
            reasons.append(f"📉 **Price is {abs(dist_ma):.0f}% below its long-term average** — a moderate pullback that may represent a better-than-average entry point.")
        elif dist_ma > 10:
            reasons.append(f"📈 **Price is {dist_ma:.0f}% above its long-term average** — running hot. This is why the system may flag SELL or TRIM.")
        if rsi_val < 30:
            reasons.append(f"📊 **RSI is {rsi_val:.0f} — deeply oversold.** When RSI drops this low, sellers have often exhausted themselves and a bounce becomes more likely.")
        elif rsi_val < 45:
            reasons.append(f"📊 **RSI is {rsi_val:.0f} — oversold territory.** More sellers than buyers recently, which often precedes a recovery.")
        elif rsi_val > 70:
            reasons.append(f"📊 **RSI is {rsi_val:.0f} — overbought.** The price may be getting ahead of itself.")
        if macd_bull:
            reasons.append("📈 **MACD is bullish** — short-term momentum has crossed above the signal line. Think of this as the selling pressure easing indicator turning green.")
        else:
            reasons.append("📉 **MACD is bearish** — momentum is still to the downside. Sellers appear to still be in control.")
        if ma200_steep:
            reasons.append(f"⚠️ **The 200-day average itself is falling steeply** — even if the price bounces, you'd be buying into a declining long-term trend. The system has downgraded the signal because of this.")
        elif ma200_trend == "rising":
            reasons.append("✅ **The 200-day average is rising** — the long-term trend is healthy. A dip into a rising trend is one of the best possible setups.")
        if value_available:
            if value_grade == "A":
                reasons.append(f"🏆 **Business quality: Grade A ({value_score}/100)** — scores well on PE, cash flow, debt, and profitability. The dip looks like a market overreaction rather than a business problem.")
            elif value_grade == "B":
                reasons.append(f"✅ **Business quality: Grade B ({value_score}/100)** — solid fundamentals. A well-run business at a reasonable price.")
            elif value_grade == "D":
                reasons.append(f"⚠️ **Business quality: Grade D ({value_score}/100)** — weak fundamentals or expensive valuation. Be cautious even if the chart looks interesting.")
        if conv_labels:
            reasons.append(f"👥 **External signals:** {' · '.join(conv_labels)} — these are analyst ratings, short interest, insider activity, and earnings history.")
        if is_knife and not reversal:
            reasons.append("🔪 **Falling knife warning** — price is dropping fast with RSI still declining. The system says AVOID until MACD turns and RSI stabilises.")
        elif reversal:
            reasons.append("✅ **Reversal confirmed** — was a falling knife but MACD has turned bullish and RSI is rising. The worst may be over.")
        fg_now = get_fg_index()[0]
        if fg_now < 25:
            reasons.append(f"🔴 **Fear & Greed is {fg_now} (Extreme Fear)** — the whole market is fearful. Historically, extreme fear is when the best long-term entries happen.")
        elif fg_now > 75:
            reasons.append(f"💚 **Fear & Greed is {fg_now} (Extreme Greed)** — the market is euphoric. The system is more conservative with buy signals in this environment.")
        if not reasons:
            reasons.append("Not enough data to generate an explanation for this signal.")
        for r in reasons:
            st.markdown(f"- {r}")
        st.caption("This explanation is generated automatically from the same signals the system uses. It is not financial advice.")

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
    if not is_etf:
        st.divider()
        st.markdown("**📊 Fundamentals**")
        def _fmt(v, mult=1, pct=False, suffix=""):
            try:
                f = float(v) * mult
                return f"{f:.1f}%" if pct else f"{f:.2f}{suffix}"
            except: return "—"

        fund_fields = ["fmp_pe_ttm","fmp_pb","fmp_peg","fmp_fcf_yield",
                       "fmp_roe","fmp_debt_eq","fmp_rev_growth","fmp_div_yield"]
        fund_coverage = sum(1 for k in fund_fields
                            if fund_data.get(k) is not None
                            and str(fund_data.get(k)) not in ("nan","None",""))

        # Debug expander — remove once fundamentals are confirmed working
        with st.expander("🔧 Debug: fundamentals fetch", expanded=fund_coverage < 2):
            st.write(f"**is_etf:** {is_etf}")
            st.write(f"**fund_coverage:** {fund_coverage}/8")
            st.write(f"**fund_data keys:** {list(fund_data.keys())}")
            err = cache_get(f"yfund_err_{ticker}")
            if err:
                st.error(f"**Last fetch error:** {err}")
            else:
                st.success("No fetch errors recorded")
            st.json({k: str(v) for k, v in fund_data.items()})

        if fund_coverage < 2:
            st.warning(
                "⚠️ **Limited fundamental data.** "
                "yfinance may be rate-limited or this stock has limited coverage. "
                "Technical signals above are still valid."
            )
            st.caption("Try clicking **🔄 Refresh** in a few minutes, or use the ⚖️ Compare tab for multi-stock comparison.")
        else:
            if fund_coverage < 4:
                st.caption(f"⚠️ Partial data — {fund_coverage}/8 fields available.")

            # Valuation row
            st.caption("**Valuation**")
            v1,v2,v3,v4 = st.columns(4)
            v1.metric("P/E (TTM)",   _fmt(pe),
                      delta="Cheap" if _pf(pe) and 0 < _pf(pe) < 15 else ("Expensive" if _pf(pe) and _pf(pe) > 40 else None),
                      delta_color="normal" if _pf(pe) and 0 < _pf(pe) < 15 else "inverse")
            v2.metric("P/B",         _fmt(fund_data.get("fmp_pb")))
            v3.metric("PEG",         _fmt(fund_data.get("fmp_peg")))
            v4.metric("FCF Yield",   _fmt(fund_data.get("fmp_fcf_yield"), mult=100, pct=True))

            # Quality row
            st.caption("**Quality**")
            q1,q2,q3,q4 = st.columns(4)
            q1.metric("ROE",         _fmt(fund_data.get("fmp_roe"), mult=100, pct=True))
            q2.metric("D/E Ratio",   _fmt(fund_data.get("fmp_debt_eq")))
            q3.metric("Rev Growth",  _fmt(fund_data.get("fmp_rev_growth"), mult=100, pct=True))
            q4.metric("Div Yield",   _fmt(div, mult=100, pct=True))

        if value_available:
            grade_emoji = {"A":"🏆","B":"✅","C":"⚖️","D":"⚠️"}.get(value_grade,"")
            st.markdown(f"**Overall Quality: {grade_emoji} Grade {value_grade} ({value_score}/100)**")
            if value_bdown:
                bdown_str = " · ".join(f"{k}: {v}/100" for k,v in value_bdown.items())
                st.caption(bdown_str)

    # Write back to session signals_df (tagged so value screen doesn't re-trigger)
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
# TAB 3 — COMPARE
# ───────────────────────────────────────────────────────────────────

SP500_TOP50 = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","BRK-B","LLY","AVGO","TSLA",
    "WMT","JPM","V","UNH","XOM","ORCL","MA","HD","PG","COST","JNJ","ABBV",
    "BAC","NFLX","KO","CRM","CVX","MRK","AMD","PEP","ADBE","TMO","ACN","LIN",
    "MCD","ABT","CSCO","TXN","DIS","DHR","NEE","INTC","PM","IBM","RTX","INTU",
    "NOW","QCOM","GE","HON",
]

def render_compare():
    st.subheader("⚖️ Compare")
    st.caption("Side-by-side fundamental comparison of multiple stocks. "
               "Use this to validate candidates from the scanner before committing capital.")

    with st.expander("ℹ️ How to use this", expanded=False):
        st.markdown(
            "**What this does:** Fetches live fundamentals for multiple stocks at once and ranks them "
            "side by side — so you can compare PE, ROE, revenue growth, FCF yield and more in one view.\n\n"
            "**Workflow:** Run a scan → identify interesting candidates → enter their tickers here → "
            "compare fundamentals → Deep Dive on the best ones before buying.\n\n"
            "**Columns explained:**\n"
            "- **Score** = overall value quality 0–100 (higher = better fundamentals)\n"
            "- **Grade** = A (top quality) · B (solid) · C (average) · D (weak or expensive)\n"
            "- **Coverage** = how many of 7 fundamental fields had data (higher = more reliable score)\n"
            "- **Conviction** = external signals: analyst ratings, insider activity, short interest\n"
            "- **Tech** = current technical signal from the scanner (BUY/WATCH/WAIT etc.)\n\n"
            "**Tip:** Tickers can be loaded automatically from the scanner in a future update. "
            "For now, paste them manually or use the preset buttons."
        )

    col_a, col_b = st.columns([3,1])
    with col_a:
        sdf = get_signals_df()
        if st.button("📡 Load BUY/WATCH stocks from scanner"):
            if not sdf.empty:
                cands = sdf[
                    (sdf["action"].isin(["BUY","WATCH"])) &
                    (sdf.get("type","Stock") != "ETF" if "type" in sdf.columns else True)
                ]["ticker"].dropna().unique().tolist()[:100]
                st.session_state["vs_tickers"] = ", ".join(cands)
    with col_b:
        if st.button("🇺🇸 S&P 500 Top 50"):
            st.session_state["vs_tickers"] = ", ".join(SP500_TOP50)

    tickers_raw = st.text_area(
        "Tickers to compare (comma-separated, max 100)",
        value=st.session_state.get("vs_tickers",""),
        height=70,
        placeholder="e.g. AAPL, MSFT, ASML, SAP, HMC …",
    )

    c1,c2,c3,c4 = st.columns([2,2,1,1])
    min_score   = c1.number_input("Min quality score", min_value=0, max_value=100, value=20, step=5,
                                   help="0 = show all · 50 = solid quality · 75 = Grade A only")
    tech_filter = c2.selectbox("Technical signal filter", ["Any","BUY only","BUY or WATCH"],
                                help="Only show stocks with a matching signal from the scanner")
    run_vs      = c3.button("⚖️ Compare", type="primary")
    refresh_vs  = c4.button("🔄 Refresh", help="Re-fetch live data for all tickers")

    if not run_vs and not refresh_vs:
        st.info(
            "Enter tickers above and click **⚖️ Compare** to fetch live fundamentals.\n\n"
            "💡 **Tip:** You can also paste tickers directly from the scanner — just copy the ticker column values."
        )
        return

    tickers = [t.strip().upper() for t in tickers_raw.replace("\n"," ").split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:100]
    if not tickers:
        st.warning("No tickers entered.")
        return

    # Refresh Live: bust all cached fundamentals + tech for listed tickers
    if refresh_vs:
        store = _get_cache_store()
        lock  = _get_cache_lock()
        with lock:
            for t in tickers:
                for k in list(store.keys()):
                    if k in (f"tick_{t}", f"yfund_{t}", f"conv_{t}") or \
                       (k.startswith("fmp_") and t in k):
                        store.pop(k, None)
        st.toast(f"🔄 Cache busted for {len(tickers)} tickers — re-fetching…")

    with st.spinner(f"Fetching live fundamentals for {len(tickers)} tickers — this may take 15–30 seconds…"):
        fund_batch, timed_out = fetch_yf_fundamentals_batch(tickers, max_workers=8, timeout=5)

    if timed_out:
        st.caption(f"⚠️ Timed out: {', '.join(timed_out[:10])}")

    # Tech signals — read from session signals_df (written by scanner/deep dive).
    # We do NOT force-refresh here to avoid triggering reruns in other tabs.
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
    st.success(f"⚖️ {len(rows)} stocks compared · Grade A: {n_a} · Grade B: {n_b} · High conviction: {n_hi}")
    st.caption("Sorted by quality score. Click 🔬 Deep Dive on any row for detailed analysis and buy/sell decision.")

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
    preset, filters, budget, tickers, vix, fg, rm = render_sidebar()

    # Auto-switch to Deep Dive tab when triggered from scanner
    default_tab = st.session_state.pop("_active_tab", 0)

    tab_scanner, tab_deepdive, tab_value = st.tabs([
        "🔭 Market Scanner",
        "🔬 Deep Dive",
        "⚖️ Compare",
    ])

    with tab_scanner:
        render_scanner(tickers, budget, vix, fg, rm)

    with tab_deepdive:
        render_deepdive(budget)

    with tab_value:
        render_compare()

if __name__ == "__main__":
    main()
