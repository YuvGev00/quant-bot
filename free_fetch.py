"""
Run LOCALLY. Zero account needed. Pulls free DIVIDEND-ADJUSTED daily bars from Yahoo
(auto_adjust=True -> 'close' is total-return adjusted; essential for momentum on
bond/REIT/dividend ETFs) into the canonical schema, as Parquet.

Prereq:  pip install yfinance pyarrow pandas

Output: bars_<SYMBOL>_<interval>.parquet : ts(UTC), symbol, open, high, low, close, volume, vwap, bar_seconds
"""
from __future__ import annotations
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

_SECS = {"1d": 86400, "1h": 3600, "30m": 1800, "15m": 900, "5m": 300, "1m": 60}


def fetch(symbol: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    vol = df["Volume"] if "Volume" in df.columns else 0
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[ts_col], utc=True),
        "symbol": symbol,
        "open": df["Open"], "high": df["High"], "low": df["Low"], "close": df["Close"],
        "volume": pd.Series(vol).fillna(0), "vwap": pd.NA,
        "bar_seconds": _SECS.get(interval),
    })
    return out


def save(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        print(f"  (empty) {path}")
        return
    df.to_parquet(path, index=False)
    print(f"  wrote {len(df):>6} rows -> {path}  ({df['ts'].min().date()} -> {df['ts'].max().date()})")


if __name__ == "__main__":
    if yf is None:
        raise SystemExit("yfinance not installed. Run: pip install yfinance")
    # Build-vs-buy hurdle universe: 8 sleeve ETFs + safe asset + managed-futures overlays + FX.
    UNIVERSE = ["SPY", "EFA", "EEM", "VNQ", "GLD", "DBC", "TLT", "IEF",  # sleeves
                "QQQ", "IWM", "XLK", "XLF", "XLV", "XLE", "XLP", "XLY", "XLI", "XLU", "XLB",  # swing-basket equities (high mean-reversion, long histories)
                "BIL",                                                    # safe asset ("cash")
                "DBMF", "KMLM",                                           # managed-futures overlays (the "buy")
                "USDILS=X", "^VIX", "^IRX"]                                               # FX, for ILS-based evaluation
    interval, period = "1d", "max"   # max history -> covers 2008 for the long-lived ETFs
    for sym in UNIVERSE:
        df = fetch(sym, period=period, interval=interval)
        save(df, f"bars_{sym.replace('=','').replace('^','')}_{interval}.parquet")
    print("\nPulled hurdle universe. Send me the bars_*.parquet files; I'll run benchmark_eval.py.")
