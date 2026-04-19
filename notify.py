import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

ASSETS = ["SPY", "TLT", "GLD", "DBC", "IEF", "TIP", "SHY"]
LOOKBACK_MONTHS = 3
TOP_N = 2
POWER_STRENGTH = 2.0
CASH_ASSET = "SHY"

end_date = datetime.today()
start_date = end_date - timedelta(days=(LOOKBACK_MONTHS + 2) * 31)

raw = yf.download(ASSETS, start=start_date, end=end_date, auto_adjust=True, progress=False)
prices_daily = raw["Close"]
prices_monthly = prices_daily.resample("ME").last()

# Drop partial month
if prices_monthly.index[-1].month == end_date.month:
    prices_monthly = prices_monthly.iloc[:-1]

returns_3m = prices_monthly.pct_change(LOOKBACK_MONTHS).iloc[-1]
ranked = returns_3m.sort_values(ascending=False)

top_n = ranked.head(TOP_N)
positive = top_n[top_n > 0]

if len(positive) == 0:
    top_assets = [CASH_ASSET]
    weights = [1.0]
elif len(positive) < TOP_N:
    top_assets = [positive.index[0], CASH_ASSET]
    weights = [0.5, 0.5]
else:
    vals = positive.values ** POWER_STRENGTH
    weights = (vals / vals.sum()).tolist()
    top_assets = positive.index.tolist()

signal_date = prices_monthly.index[-1].strftime("%Y-%m-%d")
lines = "\n".join([f"  {a}: {w*100:.1f}%" for a, w in zip(top_assets, weights)])
message = f"📊 *Allocation Signal — {signal_date}*\n{lines}\n\nLookback: {LOOKBACK_MONTHS} months"

token = os.environ["TELEGRAM_TOKEN"]
chat_id = os.environ["TELEGRAM_CHAT_ID"]
requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
              data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})

print("Sent:", message)

# After the US message is sent, calculate LSE signal
lse_map = {
    "SPY": "CSPX", "TLT": "IDTL", "GLD": "IGLN",
    "DBC": "ICOM", "IEF": "IBTM", "TIP": "ITPS", "SHY": "IBTS"
}
lse_tickers = [lse_map[a] for a in ASSETS if a in lse_map]

raw_lse = yf.download(lse_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
prices_lse = raw_lse["Close"].resample("ME").last()

if prices_lse.index[-1].month == end_date.month:
    prices_lse = prices_lse.iloc[:-1]

returns_lse = prices_lse.pct_change(LOOKBACK_MONTHS).iloc[-1]
ranked_lse = returns_lse.sort_values(ascending=False)

top_n_lse = ranked_lse.head(TOP_N)
positive_lse = top_n_lse[top_n_lse > 0]
cash_lse = lse_map.get(CASH_ASSET, "IBTS")

if len(positive_lse) == 0:
    top_assets_lse = [cash_lse]
    weights_lse = [1.0]
elif len(positive_lse) < TOP_N:
    top_assets_lse = [positive_lse.index[0], cash_lse]
    weights_lse = [0.5, 0.5]
else:
    vals_lse = positive_lse.values ** POWER_STRENGTH
    weights_lse = (vals_lse / vals_lse.sum()).tolist()
    top_assets_lse = positive_lse.index.tolist()

lse_lines = "\n".join([f"  {a}: {w*100:.1f}%" for a, w in zip(top_assets_lse, weights_lse)])
lse_message = f"🇬🇧 *LSE Signal — {signal_date}*\n{lse_lines}\n\nLookback: {LOOKBACK_MONTHS} months"

requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
              data={"chat_id": chat_id, "text": lse_message, "parse_mode": "Markdown"})
