#!/usr/bin/env python3
"""fundamentals.py — pull CURRENT fundamentals per ticker and turn them into a health score that
feeds the CONFIRM/CAUTION/VETO verdict, so the news layer reads NUMBERS (P/E, margins, debt,
analyst target vs price), not just sentiment.

HONEST LIMITS:
- These are CURRENT fundamentals (today's snapshot via yfinance), NOT point-in-time — fine for a
  live "should I buy this now" screen, but you cannot backtest a fundamental strategy with them
  (that needs paid point-in-time data we don't have). So: live screen only.
- yfinance fundamentals can be stale/missing for some tickers/ETFs; ETFs return mostly None
  (they have no P/E/margins) — for ETFs we fall back to the news+price verdict.
"""
from __future__ import annotations
import sys
import yfinance as yf

FIELDS = ["forwardPE", "trailingPE", "pegRatio", "revenueGrowth", "profitMargins",
          "returnOnEquity", "debtToEquity", "targetMeanPrice", "currentPrice", "dividendYield",
          "recommendationKey", "recommendationMean", "numberOfAnalystOpinions"]


def fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {k: info.get(k) for k in FIELDS}
    except Exception as e:
        return {"error": str(e)[:120]}


def health_score(f: dict) -> dict:
    """Turn raw fundamentals into a transparent 0-100 health score + flags. Higher = healthier.
    Rules are deliberately simple and explainable (no black box). ETFs (no fundamentals) -> None."""
    if not f or f.get("profitMargins") is None and f.get("forwardPE") is None:
        return {"score": None, "note": "no fundamentals (ETF or missing) — use price+news only", "flags": []}
    score, flags = 50, []   # start neutral
    fpe = f.get("forwardPE"); pm = f.get("profitMargins"); roe = f.get("returnOnEquity")
    rg = f.get("revenueGrowth"); d2e = f.get("debtToEquity")
    tgt = f.get("targetMeanPrice"); px = f.get("currentPrice")

    # profitability
    if pm is not None:
        if pm < 0: score -= 25; flags.append(f"UNPROFITABLE (margin {pm*100:.0f}%)")
        elif pm > 0.20: score += 12
        elif pm > 0.10: score += 6
    if roe is not None and roe < 0: score -= 8; flags.append("negative ROE")
    # valuation
    if fpe is not None:
        if fpe > 50: score -= 18; flags.append(f"very expensive (fwd P/E {fpe:.0f})")
        elif fpe > 30: score -= 8; flags.append(f"pricey (fwd P/E {fpe:.0f})")
        elif 0 < fpe < 20: score += 8
    # growth
    if rg is not None:
        if rg > 0.20: score += 10
        elif rg > 0.05: score += 4
        elif rg < 0: score -= 8; flags.append("shrinking revenue")
    # leverage
    if d2e is not None and d2e > 150: score -= 8; flags.append(f"high debt (D/E {d2e:.0f})")
    # analyst target vs price — the killer signal
    if tgt and px:
        upside = tgt / px - 1
        if upside < -0.10: score -= 22; flags.append(f"analyst target {upside*100:.0f}% BELOW price")
        elif upside < 0: score -= 8; flags.append(f"target {upside*100:.0f}% below price")
        elif upside > 0.20: score += 10
    score = max(0, min(100, score))
    return {"score": score, "upside_to_target%": (round((tgt/px-1)*100, 1) if (tgt and px) else None),
            "fwd_PE": (round(fpe, 1) if fpe else None),
            "margin%": (round(pm*100, 1) if pm is not None else None),
            "rev_growth%": (round(rg*100, 1) if rg is not None else None),
            "flags": flags}


def fundamental_verdict(score) -> str:
    """Map health score -> a vote that can pull a momentum pick toward VETO."""
    if score is None: return "NEUTRAL"          # ETF / no data — don't override news+price
    if score >= 60: return "CONFIRM"
    if score >= 40: return "CAUTION"
    return "VETO"


def screen(tickers: list[str]) -> list[dict]:
    rows = []
    for t in tickers:
        f = fetch_fundamentals(t)
        h = health_score(f)
        rows.append({"ticker": t, "fund_score": h["score"], "fund_verdict": fundamental_verdict(h["score"]),
                     "fwd_PE": h.get("fwd_PE"), "margin%": h.get("margin%"),
                     "upside_to_target%": h.get("upside_to_target%"), "flags": "; ".join(h["flags"]) or "—"})
    return rows


if __name__ == "__main__":
    tickers = sys.argv[1:] or ["INTC", "AMD", "AVGO", "TXN", "CSCO", "CAT", "SLB", "NEM", "JPM",
                               "EWY", "EWT"]   # mix of stocks + ETFs
    import pandas as pd
    rows = screen(tickers)
    df = pd.DataFrame(rows)
    print("=== FUNDAMENTAL HEALTH SCREEN (current snapshot) ===")
    print("score 0-100 (higher=healthier). ETFs show blank score -> judged on price+news only.\n")
    print(df.to_string(index=False))
    print("\nfund_verdict feeds the combined CONFIRM/CAUTION/VETO in news_check.py.")
    print("Note: CURRENT fundamentals only (not point-in-time) -> live screen, NOT backtestable.")
