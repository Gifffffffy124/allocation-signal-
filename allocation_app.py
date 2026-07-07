import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

def send_telegram(message):
    token = st.secrets["TELEGRAM_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})

# ── Phantom Flow Config ──────────────────────────────────────
PF_ASSETS        = ["SPY", "TLT", "GLD", "GSG", "TIP", "SHY", "EEM"]
PF_CASH_ASSET    = "SHY"
PF_TOP_N         = 3
PF_LOOKBACK_DAYS = 120
PF_VWAP_WINDOW   = 20

PF_CIMI_FAST, PF_CIMI_SLOW, PF_CIMI_SIGNAL, PF_CIMI_MULT = 8, 21, 5, 1.5
PF_DP_PERIOD = 13
PF_OFI_PERIOD, PF_OFI_ALPHA = 10, 0.3
PF_VWAZE_LEN = 20

def pf_fetch_ohlcv(tickers, lookback_days):
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    data = {}
    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                               progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            data[ticker] = raw[["open", "high", "low", "close", "volume"]].dropna()
        except Exception:
            continue
    return data

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _sma(s, w): return s.rolling(window=w, min_periods=1).mean()
def _stdev(s, w): return s.rolling(window=w, min_periods=2).std()
def _tanh_approx(x):
    x2 = x * x
    return x * (27.0 + x2) / (27.0 + 9.0 * x2)
def _rsi(s, period):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    ag = gain.ewm(com=period - 1, adjust=False).mean()
    al = loss.ewm(com=period - 1, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
def _atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()
def _vwap(close, volume, window=PF_VWAP_WINDOW):
    cv = close * volume
    return cv.rolling(window=window, min_periods=1).sum() / volume.rolling(window=window, min_periods=1).sum().replace(0, np.nan)

def pf_compute_cimi(df):
    c, v = df["close"], df["volume"]
    vol_sma = _sma(v, PF_CIMI_SLOW)
    vol_weight = v / vol_sma.replace(0, np.nan)
    vol_adj = c * np.sqrt(vol_weight.clip(lower=0))
    rsi_norm = (_rsi(vol_adj, PF_CIMI_FAST) - 50) / 50
    roc_comp = c.pct_change(PF_CIMI_FAST) * 0.6 + c.pct_change(PF_CIMI_SLOW) * 0.4
    vwap_val = _vwap(c, v)
    vwap_dev = (c - vwap_val) / vwap_val.replace(0, np.nan) * 100
    vwap_norm = _tanh_approx(vwap_dev * 0.5)
    cimi_raw = (rsi_norm * 0.40 + roc_comp * 0.35 + vwap_norm * 0.25) * PF_CIMI_MULT * 100
    return _ema(cimi_raw, PF_CIMI_SIGNAL)

def pf_compute_dppo(df):
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    atr = _atr(h, l, c)
    ha_close = (o + h + l + c) / 4
    ha_open = ha_close.copy()
    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_body_norm = (ha_close - ha_open) / atr.replace(0, c.median() * 0.0001)
    hl_range = (h - l).replace(0, c.median() * 0.0001)
    net_press = (c - l) / hl_range - (h - c) / hl_range
    return _ema(ha_body_norm * 0.5 + net_press * 0.5, PF_DP_PERIOD)

def pf_compute_ofit(df):
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    hl_range = (h - l).replace(0, c.median() * 0.0001)
    delta_ema = _ema((2 * c - h - l) / hl_range * v, PF_OFI_PERIOD)
    roll_std = delta_ema.rolling(window=PF_OFI_PERIOD * 4, min_periods=PF_OFI_PERIOD).std()
    ofi_norm = _tanh_approx((delta_ema / roll_std.replace(0, np.nan)) * 0.5)
    return (ofi_norm.ewm(alpha=(1 - PF_OFI_ALPHA), adjust=False).mean() * 100).clip(-100, 100)

def pf_compute_vwaze(df):
    h, l, v = df["high"], df["low"], df["volume"]
    n = PF_VWAZE_LEN
    vol_z = (v - _sma(v, n)) / _stdev(v, n).replace(0, np.nan)
    pr = h - l
    range_z = (pr - _sma(pr, n)) / _stdev(pr, n).replace(0, np.nan)
    return _ema(vol_z - range_z, 3)

def pf_compute_score(df):
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    if len(df) < 40:
        return np.nan
    cimi  = pf_compute_cimi(df)
    dppo  = pf_compute_dppo(df)
    ofi   = pf_compute_ofit(df)
    vwaze = pf_compute_vwaze(df)
    pf_raw = cimi * 0.35 + dppo * 40 * 0.25 + ofi * 0.20 + vwaze * 10 * 0.20
    return float(_ema(pf_raw, 3).clip(-100, 100).iloc[-1])

# ── New conviction-based allocation ──────────────────────────
def pf_compute_allocations_new(pf_scores):
    scores = pd.Series(pf_scores)
    positive = scores[scores > 0].sort_values(ascending=False)
    index = scores.index if PF_CASH_ASSET in scores.index else scores.index.append(pd.Index([PF_CASH_ASSET]))
    alloc = pd.Series(0.0, index=index)

    if positive.empty:
        alloc[PF_CASH_ASSET] = 1.0
        return alloc.to_dict()

    top = positive.head(PF_TOP_N)
    conviction = float(np.clip(top.mean() / 100.0, 0.0, 1.0))
    total_pf = top.sum()
    for asset in top.index:
        alloc[asset] = conviction * (top[asset] / total_pf)

    alloc[PF_CASH_ASSET]
    
