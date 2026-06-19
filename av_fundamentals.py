#!/usr/bin/env python3
"""av_fundamentals.py — Alpha Vantage fundamentals + analyst ratings, as a cleaner/cross-check
source vs flaky yfinance for the value & news layers.

Get a free key: https://www.alphavantage.co/support/#api-key  ->  export ALPHAVANTAGE_KEY=...

⚠ FREE-TIER LIMIT: 25 requests/day. So DON'T call this for all 188 names — call it only for the
small weekly pick-list (~10-15). yfinance stays the bulk source; AV is the higher-quality
cross-check on the handful of names that actually matter each week.

OVERVIEW endpoint gives: PERatio, ForwardPE, ProfitMargin, ReturnOnEquityTTM, RevenueGrowth-ish
(QuarterlyRevenueGrowthYOY), AnalystTargetPrice, AND the analyst rating breakdown
(AnalystRatingStrongBuy/Buy/Hold/Sell/StrongSell) — the real Google-Finance-style consensus.
"""
from __future__ import annotations
import os, json, urllib.request
import pandas as pd

KEY = os.environ.get("ALPHAVANTAGE_KEY", "")


def av_overview(symbol: str) -> dict:
    """Fetch the OVERVIEW (fundamentals + analyst ratings) for one symbol. Returns {} on any issue."""
    if not KEY:
        return {"_error": "no ALPHAVANTAGE_KEY set"}
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={KEY}"
    try:
        d = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())
    except Exception as e:
        return {"_error": str(e)[:120]}
    if not d or "Symbol" not in d:
        # AV returns {"Information": "...rate limit..."} when throttled
        return {"_error": d.get("Information") or d.get("Note") or "empty/rate-limited"}
    def f(k):
        v = d.get(k)
        try: return float(v) if v not in (None, "None", "-", "") else None
        except: return None
    sb, b, h, s, ss = (int(d.get(k, 0) or 0) for k in
                       ("AnalystRatingStrongBuy","AnalystRatingBuy","AnalystRatingHold","AnalystRatingSell","AnalystRatingStrongSell"))
    n = sb+b+h+s+ss
    # consensus mean 1=strong buy .. 5=sell (same scale yfinance uses)
    mean = ((sb*1+b*2+h*3+s*4+ss*5)/n) if n else None
    consensus = (None if mean is None else "strong buy" if mean<1.7 else "buy" if mean<2.6 else "hold" if mean<3.4 else "sell")
    px = f("AnalystTargetPrice")
    return {"symbol": symbol, "forwardPE": f("ForwardPE") or f("PERatio"),
            "profitMargins": f("ProfitMargin"), "returnOnEquity": f("ReturnOnEquityTTM"),
            "revenueGrowth": f("QuarterlyRevenueGrowthYOY"), "pegRatio": f("PEGRatio"),
            "targetMeanPrice": px, "dividendYield": (f("DividendYield") or 0),
            "analyst_consensus": consensus, "analyst_mean": (round(mean,2) if mean else None),
            "n_analysts": n, "sector": d.get("Sector"), "name": d.get("Name")}


def screen(tickers: list[str]) -> pd.DataFrame:
    """Pull AV fundamentals+analyst for a SMALL list (respect 25/day). Stops on rate-limit."""
    rows = []
    for t in tickers:
        o = av_overview(t)
        if o.get("_error"):
            print(f"  {t}: {o['_error']}")
            if "rate" in o["_error"].lower() or "limit" in o["_error"].lower() or "Information" in str(o["_error"]):
                print("  -> hit AV daily limit (25/day). Stopping."); break
            continue
        rows.append(o)
        print(f"  {t}: PE {o['forwardPE']} | margin {o['profitMargins']} | analysts: {o['analyst_consensus']} ({o['n_analysts']})")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    if not KEY:
        raise SystemExit("Set your free Alpha Vantage key:\n  https://www.alphavantage.co/support/#api-key\n  then: export ALPHAVANTAGE_KEY=yourkey")
    tickers = sys.argv[1:] or ["NFLX", "SCHW", "PFE", "CMCSA"]   # demo: a few value picks
    print(f"Alpha Vantage fundamentals + analyst ratings for {len(tickers)} names (free tier: 25/day):\n")
    df = screen(tickers)
    if len(df):
        print("\n" + df[["symbol","forwardPE","profitMargins","analyst_consensus","analyst_mean","n_analysts","targetMeanPrice"]]
              .to_string(index=False))
        df.to_json("av_fundamentals_snapshot.json", orient="records", indent=1)
        print("\nsaved av_fundamentals_snapshot.json (the value agent can read this as a higher-quality cross-check)")
