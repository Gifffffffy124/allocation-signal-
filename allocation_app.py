import streamlit as st
import yfinance as yf
import pandas as pd
import requests  # ← add this
from datetime import datetime, timedelta
import numpy as np 

def send_telegram(message):
    token = st.secrets["TELEGRAM_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})

# ── Phantom Flow Config ──────────────────────────────────────
PF_ASSETS      = ["SPY", "TLT", "GLD", "GSG", "TIP", "SHY", "EEM"]
PF_CASH_ASSET  = "SHY"
PF_TOP_N       = 3
PF_LOOKBACK_DAYS = 120

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
def _vwap(close, volume):
    cv = close * volume
    return cv.expanding().mean() / volume.expanding().mean()

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
    cimi, dppo, ofi, vwaze = pf_compute_cimi(df), pf_compute_dppo(df), pf_compute_ofit(df), pf_compute_vwaze(df)
    pf_raw = cimi * 0.35 + dppo * 40 * 0.25 + ofi * 0.20 + vwaze * 10 * 0.20
    return float(_ema(pf_raw, 3).clip(-100, 100).iloc[-1])

def pf_compute_allocations(pf_scores):
    scores = pd.Series(pf_scores)
    positive = scores[scores > 0].sort_values(ascending=False)
    n_excluded = (scores <= 0).sum()
    alloc = pd.Series(0.0, index=scores.index)

    if positive.empty:
        alloc[PF_CASH_ASSET] = 1.0
        return alloc.to_dict()

    top = positive.head(PF_TOP_N)
    total_pf = top.sum()
    for asset in top.index:
        alloc[asset] = top[asset] / total_pf

    excluded_weight = n_excluded / len(scores)
    if n_excluded > 0:
        scale = 1.0 - excluded_weight * (1 - alloc.get(PF_CASH_ASSET, 0))
        for asset in top.index:
            if asset != PF_CASH_ASSET:
                alloc[asset] *= scale
        alloc[PF_CASH_ASSET] = 1.0 - alloc[top.index[top.index != PF_CASH_ASSET]].sum()

    total = alloc.sum()
    return (alloc / total).to_dict() if total > 0 else alloc.to_dict()
# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Allocation Signal",
    page_icon="📊",
    layout="centered"
)

# ── Password gate ─────────────────────────────
password = st.text_input("Enter password", type="password")
if password != st.secrets["PASSWORD"]:
    st.stop()

# ── Sidebar config ───────────────────────────
st.sidebar.header("Configuration")

ASSETS = st.sidebar.multiselect(
    "Assets",
    options=["SPY", "TLT", "GLD", "DBC", "IEF", "TIP", "SHY", "QQQ", "VNQ", "LQD"],
    default=["SPY", "TLT", "GLD", "DBC", "IEF", "TIP", "SHY"]
)
LOOKBACK_MONTHS = st.sidebar.slider("Lookback (months)", min_value=1, max_value=12, value=3)
TOP_N = st.sidebar.slider("Top N assets to hold", min_value=1, max_value=5, value=2)
POWER_STRENGTH = st.sidebar.slider("Power exponent", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
CASH_ASSET = st.sidebar.text_input("Cash / safe-haven asset", value="SHY")

# ── Main ─────────────────────────────────────
st.title("📊 Monthly Allocation Signal")
st.caption("Momentum strategy — ranks assets by return, holds top N with power-weighted allocation.")

tab1, tab2, tab3 = st.tabs(["🇺🇸 US Signal", "🇬🇧 London Equivalents", "👻 Phantom Flow"])

with tab1:
    # all your existing code goes here (just indent it)
    if not ASSETS:
        st.warning("Please select at least one asset in the sidebar.")
        st.stop()
    
    if st.button("▶ Run Signal", type="primary", use_container_width=True):
        with st.spinner("Downloading price data..."):
            try:
                end_date   = datetime.today()
                start_date = end_date - timedelta(days=(LOOKBACK_MONTHS + 2) * 31)
    
                raw          = yf.download(ASSETS, start=start_date, end=end_date, auto_adjust=True, progress=False)
                prices_daily = raw["Close"] if len(ASSETS) > 1 else raw["Close"].to_frame(ASSETS[0])
                prices_monthly = prices_daily.resample("ME").last()
    
                latest = prices_daily.iloc[[-1]]
                if latest.index[0] > prices_monthly.index[-1]:
                    prices_monthly = pd.concat([prices_monthly, latest])
    
                if len(prices_monthly) < LOOKBACK_MONTHS + 1:
                    st.error("Not enough monthly data. Try reducing the lookback period.")
                    st.stop()
    
                returns_3m = prices_monthly.pct_change(LOOKBACK_MONTHS).iloc[-1]
    
                ranked = returns_3m.sort_values(ascending=False).reset_index()
                ranked.columns = ["Asset", "3M Return"]
                ranked["Rank"] = range(1, len(ranked) + 1)
    
                top_n    = ranked.head(TOP_N).copy()
                positive = top_n[top_n["3M Return"] > 0].copy()
    
                note = None
                if len(positive) == 0:
                    top_assets = [CASH_ASSET]
                    weights    = [1.0]
                    note = f"All signals negative — 100% cash ({CASH_ASSET})"
                elif len(positive) < TOP_N:
                    asset       = positive.iloc[0]["Asset"]
                    top_assets  = [asset, CASH_ASSET]
                    weights     = [0.5, 0.5]
                    note = f"Only {len(positive)} positive signal(s) — splitting remainder into {CASH_ASSET}"
                else:
                    vals       = positive["3M Return"].values ** POWER_STRENGTH
                    allocs     = vals / vals.sum()
                    top_assets = positive["Asset"].tolist()
                    weights    = allocs.tolist()
    
                signal_date = prices_daily.index[-1].strftime("%Y-%m-%d")
    
                # ── Rankings table ───────────────────────
                st.subheader("Asset rankings")
    
                display_df = ranked.copy()
                display_df["3M Return"] = display_df["3M Return"] * 100
                display_df["Top pick"] = display_df["Rank"] <= TOP_N
    
                st.dataframe(
                    display_df.style
                        .format({"3M Return": "{:+.2f}%"})
                        .map(
                            lambda v: "color: #2a9d5c; font-weight: 600" if isinstance(v, float) and v > 0
                                      else ("color: #d64c4c; font-weight: 600" if isinstance(v, float) and v < 0 else ""),
                            subset=["3M Return"]
                        )
                        .map(
                            lambda v: "background-color: #eaf3de" if v is True else "",
                            subset=["Top pick"]
                        ),
                    use_container_width=True,
                    hide_index=True
                )
    
                # ── Allocation ───────────────────────────
                st.subheader("Allocation")
    
                if note:
                    st.warning(note)
    
                alloc_df = pd.DataFrame({"Asset": top_assets, "Weight": weights})
                alloc_df["Weight %"] = alloc_df["Weight"] * 100
    
                for _, row in alloc_df.iterrows():
                    col1, col2 = st.columns([1, 5])
                    col1.metric(row["Asset"], f"{row['Weight %']:.1f}%")
                    col2.progress(row["Weight"])
    
                # ── Summary ──────────────────────────────
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Signal date", signal_date)
                c2.metric("Lookback", f"{LOOKBACK_MONTHS} months")
                c3.metric("Weight method", f"Power ^{POWER_STRENGTH}")
    
                # ── Telegram notification ─────────────────
                signal_lines = "\n".join([f"  {row['Asset']}: {row['Weight %']:.1f}%" for _, row in alloc_df.iterrows()])
                message = f"📊 *Allocation Signal — {signal_date}*\n{signal_lines}\n\nLookback: {LOOKBACK_MONTHS} months"
                send_telegram(message)
    
            except Exception as e:
                st.error(f"Something went wrong: {e}")

with tab2:
    st.subheader("LSE Equivalent ETFs")
    st.caption("London Stock Exchange equivalents for your US assets.")

    lse_map = {
        "SPY": ("iShares Core S&P 500",            "CSPX.L", "S&P 500 US Equities"),
        "TLT": ("iShares $ Treasury Bond 20yr+",   "IDTL.L", "Long US Treasuries"),
        "GLD": ("iShares Physical Gold",            "IGLN.L", "Gold"),
        "DBC": ("iShares Diversified Commodity",   "ICOM.L", "Commodities"),
        "IEF": ("iShares $ Treasury Bond 7-10yr",  "IBTM.L", "Medium US Treasuries"),
        "TIP": ("iShares $ TIPS",                  "ITPS.L", "US Inflation Bonds"),
        "SHY": ("iShares $ Treasury Bond 1-3yr",   "IBTS.L", "Short US Treasuries"),
    }

    lse_df = pd.DataFrame([
        {"US Ticker": us, "LSE Ticker": lse[1], "Fund Name": lse[0], "Exposure": lse[2]}
        for us, lse in lse_map.items()
        if us in ASSETS
    ])

    st.dataframe(lse_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("LSE Signal")

    if st.button("▶ Run LSE Signal", type="primary", use_container_width=True):
        with st.spinner("Downloading LSE price data..."):
            try:
                lse_tickers = [lse_map[a][1] for a in ASSETS if a in lse_map]

                end_date   = datetime.today()
                start_date = end_date - timedelta(days=(LOOKBACK_MONTHS + 2) * 31)

                raw = yf.download(lse_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
                prices_daily = raw["Close"] if len(lse_tickers) > 1 else raw["Close"].to_frame(lse_tickers[0])
                prices_monthly = prices_daily.resample("ME").last()

                latest = prices_daily.iloc[[-1]]
                if latest.index[0] > prices_monthly.index[-1]:
                    prices_monthly = pd.concat([prices_monthly, latest])

                if len(prices_monthly) < LOOKBACK_MONTHS + 1:
                    st.error("Not enough monthly data. Try reducing the lookback period.")
                    st.stop()

                returns_3m = prices_monthly.pct_change(LOOKBACK_MONTHS).iloc[-1]
                ranked = returns_3m.sort_values(ascending=False).reset_index()
                ranked.columns = ["Asset", "3M Return"]
                ranked["Rank"] = range(1, len(ranked) + 1)

                top_n    = ranked.head(TOP_N).copy()
                positive = top_n[top_n["3M Return"] > 0].copy()

                cash_lse = lse_map.get(CASH_ASSET, ("", "IBTS.L", ""))[1]

                note = None
                if len(positive) == 0:
                    top_assets = [cash_lse]
                    weights    = [1.0]
                    note = f"All signals negative — 100% cash ({cash_lse})"
                elif len(positive) < TOP_N:
                    asset      = positive.iloc[0]["Asset"]
                    top_assets = [asset, cash_lse]
                    weights    = [0.5, 0.5]
                    note = f"Only {len(positive)} positive signal(s) — splitting remainder into {cash_lse}"
                else:
                    vals       = positive["3M Return"].values ** POWER_STRENGTH
                    allocs     = vals / vals.sum()
                    top_assets = positive["Asset"].tolist()
                    weights    = allocs.tolist()

                st.subheader("LSE Asset Rankings")
                display_df = ranked.copy()
                display_df["3M Return"] = display_df["3M Return"] * 100
                display_df["Top pick"] = display_df["Rank"] <= TOP_N

                st.dataframe(
                    display_df.style
                        .format({"3M Return": "{:+.2f}%"})
                        .map(
                            lambda v: "color: #2a9d5c; font-weight: 600" if isinstance(v, float) and v > 0
                                      else ("color: #d64c4c; font-weight: 600" if isinstance(v, float) and v < 0 else ""),
                            subset=["3M Return"]
                        )
                        .map(
                            lambda v: "background-color: #eaf3de" if v is True else "",
                            subset=["Top pick"]
                        ),
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("LSE Allocation")
                if note:
                    st.warning(note)

                alloc_df = pd.DataFrame({"Asset": top_assets, "Weight": weights})
                alloc_df["Weight %"] = alloc_df["Weight"] * 100

                for _, row in alloc_df.iterrows():
                    col1, col2 = st.columns([1, 5])
                    col1.metric(row["Asset"], f"{row['Weight %']:.1f}%")
                    col2.progress(row["Weight"])

                lse_lines = "\n".join([f"  {row['Asset']}: {row['Weight %']:.1f}%" for _, row in alloc_df.iterrows()])
                lse_message = f"🇬🇧 *LSE Signal — {prices_monthly.index[-1].strftime('%Y-%m-%d')}*\n{lse_lines}\n\nLookback: {LOOKBACK_MONTHS} months"
                send_telegram(lse_message)


            except Exception as e:
                st.error(f"Something went wrong: {e}")

with tab3:
    st.subheader("Phantom Flow Signal")
    st.caption("Custom composite score (CIMI, DPPO, OFIT, VWAZE) — experimental, not backtested.")

    if st.button("▶ Run Phantom Flow Signal", type="primary", use_container_width=True):
        with st.spinner("Fetching data and computing scores..."):
            try:
                ohlcv = pf_fetch_ohlcv(PF_ASSETS, PF_LOOKBACK_DAYS)
                pf_scores = {}
                for ticker in PF_ASSETS:
                    if ticker in ohlcv:
                        score = pf_compute_score(ohlcv[ticker])
                        if not np.isnan(score):
                            pf_scores[ticker] = score

                if not pf_scores:
                    st.error("No data could be computed for any asset.")
                    st.stop()

                allocations = pf_compute_allocations(pf_scores)

                score_df = pd.DataFrame(
                    sorted(pf_scores.items(), key=lambda x: x[1], reverse=True),
                    columns=["Asset", "PF Score"]
                )
                top_assets = [a for a, s in sorted(pf_scores.items(), key=lambda x: x[1], reverse=True) if s > 0][:PF_TOP_N]
                score_df["Status"] = score_df.apply(
                    lambda r: "EXCLUDED" if r["PF Score"] <= 0 else (f"TOP {top_assets.index(r['Asset'])+1}" if r["Asset"] in top_assets else "—"),
                    axis=1
                )

                st.subheader("PF Scores")
                st.dataframe(
                    score_df.style.format({"PF Score": "{:+.2f}"}),
                    use_container_width=True, hide_index=True
                )

                st.subheader("Allocation")
                alloc_df = pd.DataFrame(
                    [(a, w) for a, w in sorted(allocations.items(), key=lambda x: x[1], reverse=True) if w > 0.0001],
                    columns=["Asset", "Weight"]
                )
                alloc_df["Weight %"] = alloc_df["Weight"] * 100

                for _, row in alloc_df.iterrows():
                    col1, col2 = st.columns([1, 5])
                    col1.metric(row["Asset"], f"{row['Weight %']:.1f}%")
                    col2.progress(row["Weight"])

                pf_lines = "\n".join([f"  {row['Asset']}: {row['Weight %']:.1f}%" for _, row in alloc_df.iterrows()])
                pf_message = f"👻 *Phantom Flow Signal — {datetime.today().strftime('%Y-%m-%d')}*\n{pf_lines}"
                send_telegram(pf_message)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
