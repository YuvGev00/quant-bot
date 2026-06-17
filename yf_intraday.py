"""
Instant free intraday pull via yfinance (no signup, no broker).
LIMIT: yfinance only serves ~60 days of intraday history -> ~42 trading days.
That's a REAL-DATA SMOKE TEST, not a verdict. For a multi-year backtest use ib_fetch.py (IBKR/TWS).

Writes the canonical files the engines read:
    bars_SPY_5m.parquet   (orb.py)
    bars_SPY_30m.parquet  (intraday_momentum.py)
"""
from __future__ import annotations
import pandas as pd
import yfinance as yf

SYMBOLS = ["SPY"]
SIZES = {"5m": (300, "5m"), "30m": (1800, "30m")}   # interval -> (bar_seconds, suffix)

def pull(sym: str, interval: str, secs: int) -> pd.DataFrame:
    df = yf.download(sym, period="60d", interval=interval, auto_adjust=True,
                     prepost=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[tcol], utc=True),
        "symbol": sym,
        "open": df["Open"], "high": df["High"], "low": df["Low"], "close": df["Close"],
        "volume": df["Volume"], "vwap": pd.NA, "bar_seconds": secs,
    })
    return out.dropna(subset=["open"]).sort_values("ts").reset_index(drop=True)

if __name__ == "__main__":
    for sym in SYMBOLS:
        for interval, (secs, suffix) in SIZES.items():
            d = pull(sym, interval, secs)
            if d.empty:
                print(f"  (empty) {sym} {interval}")
                continue
            path = f"bars_{sym}_{suffix}.parquet"
            d.to_parquet(path, index=False)
            print(f"  wrote {len(d):>5} rows  {d['ts'].min().date()} -> {d['ts'].max().date()}  {path}")
