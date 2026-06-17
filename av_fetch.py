"""
No-broker, no-desktop-app intraday fetcher using Alpha Vantage (free tier).
Get a free key in ~10 seconds: https://www.alphavantage.co/support/#api-key
Then either set it below (KEY = "...") or as an env var:  export ALPHAVANTAGE_KEY=yourkey

Pulls 5-min RTH bars month-by-month and writes the canonical files the engines read:
    bars_SPY_5m.parquet   (for orb.py)
    bars_SPY_30m.parquet  (resampled locally from the 5-min bars; for intraday_momentum.py)

Free tier = 25 requests/day. MONTHS below is how many 30-day slices to pull (1 req each),
so keep MONTHS <= ~24. Start with 18 (~1.5y); bump later if you want more history.
"""
from __future__ import annotations
import io, os, time, urllib.request
import pandas as pd

KEY = os.environ.get("ALPHAVANTAGE_KEY", "PASTE_YOUR_FREE_KEY_HERE")
SYMBOLS = ["SPY"]
MONTHS = 18
SLEEP = 1.5  # seconds between requests; raise to ~15 if you get throttled


def av_intraday(symbol: str, month: str, interval: str = "5min") -> pd.DataFrame:
    url = ("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
           f"&symbol={symbol}&interval={interval}&month={month}&outputsize=full"
           "&extended_hours=false&adjusted=true&datatype=csv&apikey=" + KEY)
    txt = urllib.request.urlopen(url, timeout=60).read().decode()
    head = txt[:300]
    if txt.lstrip().startswith("{") or any(w in head for w in ("Error", "Note", "Information", "Invalid")):
        raise RuntimeError(head.strip())
    return pd.read_csv(io.StringIO(txt))


def to_canonical(raw: pd.DataFrame, symbol: str, secs: int) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": pd.to_datetime(raw["timestamp"]),      # US/Eastern, naive (RTH only)
        "symbol": symbol,
        "open": raw["open"], "high": raw["high"], "low": raw["low"], "close": raw["close"],
        "volume": raw["volume"], "vwap": pd.NA, "bar_seconds": secs,
    }).sort_values("ts").reset_index(drop=True)


def resample_30m(df5: pd.DataFrame, symbol: str) -> pd.DataFrame:
    g = (df5.set_index("ts")[["open", "high", "low", "close", "volume"]]
         .resample("30min", label="left", closed="left")
         .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
              close=("close", "last"), volume=("volume", "sum"))
         .dropna(subset=["open"]).reset_index())
    g["symbol"] = symbol; g["vwap"] = pd.NA; g["bar_seconds"] = 1800
    return g[["ts", "symbol", "open", "high", "low", "close", "volume", "vwap", "bar_seconds"]]


if __name__ == "__main__":
    if KEY == "PASTE_YOUR_FREE_KEY_HERE":
        raise SystemExit("Set your free Alpha Vantage key: edit KEY=... or export ALPHAVANTAGE_KEY=...")
    months = [(pd.Timestamp.now() - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(MONTHS)]
    for sym in SYMBOLS:
        frames = []
        for m in months:
            try:
                frames.append(av_intraday(sym, m))
                print(f"  {sym} {m}: ok")
            except Exception as e:
                print(f"  {sym} {m}: STOP ({e})")
                break
            time.sleep(SLEEP)
        if not frames:
            continue
        df5 = to_canonical(pd.concat(frames).drop_duplicates(subset="timestamp"), sym, 300)
        df5.to_parquet(f"bars_{sym}_5m.parquet", index=False)
        print(f"  wrote {len(df5)} rows  {df5['ts'].min().date()} -> {df5['ts'].max().date()}  bars_{sym}_5m.parquet")
        df30 = resample_30m(df5, sym)
        df30.to_parquet(f"bars_{sym}_30m.parquet", index=False)
        print(f"  wrote {len(df30)} rows  bars_{sym}_30m.parquet (resampled)")
