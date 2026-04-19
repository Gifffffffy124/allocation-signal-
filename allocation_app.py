import streamlit as st
import yfinance as yf
import pandas as pd
import requests  # ← add this
from datetime import datetime, timedelta

def send_telegram(message):
    token = st.secrets["TELEGRAM_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})

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

tab1, tab2 = st.tabs(["🇺🇸 US Signal", "🇬🇧 London Equivalents"])

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
    
                today = datetime.today()
                last_month_end = prices_monthly.index[-1]
                if prices_monthly.index[-1].month == today.month:
                    next_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
                    st.warning(f"⚠️ Running mid-month — {today.strftime('%B')} not yet complete. For a clean signal, run on or after {next_month.strftime('%b 1')}.")
                    prices_monthly = prices_monthly.iloc[:-1]
    
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
        "SPY": ("iShares Core S&P 500",            "CSPX", "S&P 500 US Equities"),
        "TLT": ("iShares $ Treasury Bond 20yr+",   "IDTL", "Long US Treasuries"),
        "GLD": ("iShares Physical Gold",            "IGLN", "Gold"),
        "DBC": ("iShares Diversified Commodity",   "ICOM", "Commodities"),
        "IEF": ("iShares $ Treasury Bond 7-10yr",  "IBTM", "Medium US Treasuries"),
        "TIP": ("iShares $ TIPS",                  "ITPS", "US Inflation Bonds"),
        "SHY": ("iShares $ Treasury Bond 1-3yr",   "IBTS", "Short US Treasuries"),
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

                if prices_monthly.index[-1].month == datetime.today().month:
                    next_month = (end_date.replace(day=1) + timedelta(days=32)).replace(day=1)
                    st.warning(f"⚠️ Running mid-month — {end_date.strftime('%B')} not yet complete. For a clean signal, run on or after {next_month.strftime('%b 1')}.")
                    prices_monthly = prices_monthly.iloc[:-1]

                if len(prices_monthly) < LOOKBACK_MONTHS + 1:
                    st.error("Not enough monthly data. Try reducing the lookback period.")
                    st.stop()

                returns_3m = prices_monthly.pct_change(LOOKBACK_MONTHS).iloc[-1]
                ranked = returns_3m.sort_values(ascending=False).reset_index()
                ranked.columns = ["Asset", "3M Return"]
                ranked["Rank"] = range(1, len(ranked) + 1)

                top_n    = ranked.head(TOP_N).copy()
                positive = top_n[top_n["3M Return"] > 0].copy()

                cash_lse = lse_map.get(CASH_ASSET, ("", "IBTS", ""))[1]

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
