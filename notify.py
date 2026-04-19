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
