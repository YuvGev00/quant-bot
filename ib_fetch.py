"""
Run LOCALLY (not in the sandbox). Pulls IBKR historical bars into the canonical
Parquet schema the engines read:
    bars_<SYMBOL>_<SIZE>.parquet  ->  ts(UTC), symbol, open, high, low, close, volume, vwap, bar_seconds
e.g. bars_SPY_5m.parquet (for orb.py), bars_SPY_30m.parquet (for intraday_momentum.py)

Prereqs:
  pip install ib_insync pyarrow pandas
  TWS or IB Gateway running, API enabled.
  Ports: 7497 paper / 7496 live (TWS),  4002 paper / 4001 live (Gateway).
  You need a US market-data subscription for SPY historical TRADES.
"""
from __future__ import annotations
import pandas as pd

try:
    from ib_insync import IB, Stock, util
except ImportError:
    IB = None

SIZE_SECS = {"1 min": 60, "5 mins": 300, "15 mins": 900, "30 mins": 1800, "1 hour": 3600}
SIZE_SUFFIX = {"1 min": "1m", "5 mins": "5m", "15 mins": "15m", "30 mins": "30m", "1 hour": "1h"}


def connect(host: str = "127.0.0.1", port: int = 7497, client_id: int = 17) -> "IB":
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib


def fetch_intraday_history(ib: "IB", symbol: str, bar_size: str = "5 mins", years: float = 2.0,
                           chunk: str = "30 D", exchange: str = "SMART", currency: str = "USD",
                           what: str = "TRADES", rth: bool = True) -> pd.DataFrame:
    """Paginate backwards in `chunk`-sized requests until `years` of history is covered."""
    contract = Stock(symbol, exchange, currency)
    ib.qualifyContracts(contract)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=int(years), months=int((years % 1) * 12))
    end, frames = "", []
    while True:
        bars = ib.reqHistoricalData(contract, endDateTime=end, durationStr=chunk,
                                    barSizeSetting=bar_size, whatToShow=what,
                                    useRTH=rth, formatDate=2)
        df = util.df(bars)
        if df is None or df.empty:
            break
        frames.append(df)
        earliest = pd.to_datetime(df["date"], utc=True).min()
        print(f"    {symbol} {bar_size}: got {len(df):>5} bars back to {earliest.date()}")
        if earliest <= cutoff:
            break
        end = earliest.to_pydatetime()
        ib.sleep(1.0)  # respect pacing limits
    if not frames:
        return pd.DataFrame()
    raw = (pd.concat(frames).drop_duplicates(subset="date").sort_values("date")
           .reset_index(drop=True))
    return pd.DataFrame({
        "ts": pd.to_datetime(raw["date"], utc=True),
        "symbol": symbol,
        "open": raw["open"], "high": raw["high"], "low": raw["low"], "close": raw["close"],
        "volume": raw["volume"], "vwap": raw.get("average"),
        "bar_seconds": SIZE_SECS.get(bar_size),
    })


def save(df: pd.DataFrame, path: str) -> None:
    if df is None or df.empty:
        print(f"  (empty) {path}")
        return
    df.to_parquet(path, index=False)
    print(f"  wrote {len(df):>6} rows  {df['ts'].min().date()} -> {df['ts'].max().date()}  {path}")


if __name__ == "__main__":
    if IB is None:
        raise SystemExit("ib_insync not installed. Run: pip install ib_insync")

    SYMBOLS = ["SPY"]            # add "QQQ", "TQQQ", etc. if you want
    BAR_SIZES = ["5 mins", "30 mins"]   # 5m -> orb.py ; 30m -> intraday_momentum.py
    YEARS = 2.0                 # how far back to pull

    ib = connect(port=7496)     # 7496 live TWS (read-only: only pulls historical data, never trades)
    try:
        for sym in SYMBOLS:
            for bs in BAR_SIZES:
                print(f"Pulling {sym} {bs} ({YEARS}y)...")
                df = fetch_intraday_history(ib, sym, bar_size=bs, years=YEARS)
                save(df, f"bars_{sym}_{SIZE_SUFFIX[bs]}.parquet")
    finally:
        ib.disconnect()
