#!/usr/bin/env python3
"""reversal_agent.py — the SECOND, independent agent: the VALIDATED edge.

While the momentum agent (the_bot.py / cloud_bot_runner.py) shows "what's hot", THIS agent surfaces
the statistically-proven signal: VOLUME-CONFIRMED SHORT-TERM REVERSAL — buy the ETFs that fell hardest
over ~10 days ON HEAVY VOLUME (capitulation), which tend to bounce. (Validated: permutation p=0.000,
Sharpe 0.70, survives OOS + costs to 40bps + every sub-period. See finding-meanrev-hardened.)

It runs the SAME pipeline shape as the momentum agent but is fully independent:
  detect (storm-gate) → rank reversal candidates → fundamentals → REVERSAL-specific news read → manage.

The news question is INVERTED vs momentum: momentum asks "is this froth?"; reversal asks
"is this drop a TEMPORARY OVERREACTION that bounces, or a JUSTIFIED COLLAPSE / falling knife?"

Run:  python3 reversal_agent.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, hysteresis, LAG
from early_scanner import panels
from fundamentals import fetch_fundamentals, health_score, fundamental_verdict
from universe import all_symbols, diversify, SECTOR, LEVERAGED

TOP_N = 5
MAX_PER_SECTOR = 2          # diversity cap: no more than 2 bounce candidates from one theme
HOLDINGS_FILE = "holdings_reversal.json"          # separate book from the momentum agent
NEWS_FILE = "news_verdicts_reversal.json"


def reversal_score(px, vol, lookback=10):
    """Volume-confirmed reversal: biggest N-day LOSERS, weighted UP if they fell on heavy volume
    (relative to their 63d average). Higher score = stronger bounce candidate."""
    drop = -(px / px.shift(lookback) - 1)                 # positive = it fell (we want fallers)
    relvol = (vol / vol.rolling(63).mean()).clip(0.5, 4.0)
    return drop * relvol


def load_json(path, default):
    if os.path.exists(path):
        try: return json.load(open(path))
        except Exception: pass
    return default


def main():
    print("=" * 64)
    print("  REVERSAL AGENT  —  validated edge (buy heavy-volume losers)")
    print("=" * 64)

    px, vol = panels(all_symbols(include_stocks=True))   # 188 names: 100 stocks + 88 ETFs, no leverage
    spy = load_close("SPY"); ro, _, _, _ = walk_forward(spy, None)
    risk_off = bool(ro.iloc[-1] > 0.5); asof = ro.index[-1].date()
    print(f"\n[1] STORM-DETECTOR ({asof}): {'RISK-OFF — go to cash, no bounce-buying' if risk_off else 'CALM — bounce candidates live'}")
    if risk_off:
        print("\n  >>> ACTION: storm regime -> hold CASH. Falling markets aren't safe to bounce-buy.")
        return

    # ffill so a name whose file ends a day early still gets scored on its latest available bar
    sc = reversal_score(px.ffill(), vol.ffill()).iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()
    ranked = sc[sc > 0].sort_values(ascending=False)
    # DIVERSITY CAP: top-10 shown, but picks capped at 2/sector so it can't be all-semis/all-oil
    picks = pd.Series({s: sc[s] for s in diversify(list(ranked.index), TOP_N, MAX_PER_SECTOR)})
    top10 = ranked.head(10)

    pxf = px.ffill()
    print(f"\n[2] TOP 10 BOUNCE CANDIDATES (of {len(sc)} names; picks diversified ≤{MAX_PER_SECTOR}/sector):")
    for i, (s, v) in enumerate(top10.items(), 1):
        d10 = (pxf[s].iloc[-1] / pxf[s].iloc[-11] - 1) * 100
        sec = SECTOR.get(s, "other")
        mark = " <- PICK" if s in picks.index else ""
        print(f"     {i:2d}. {s:5s} [{sec:12s}] 10d move {d10:+5.1f}%  score {v:.2f}{mark}")

    print(f"\n[3] FUNDAMENTALS (is the faller financially sound, or deteriorating?):")
    for s in picks.index:
        fv = fundamental_verdict(health_score(fetch_fundamentals(s))["score"])
        print(f"     {s:5s}  fundamentals {fv}")

    cur = load_json(HOLDINGS_FILE, {}).get("positions", {})
    news = {k.upper(): v.upper() for k, v in load_json(NEWS_FILE, {}).items() if not k.startswith("_")}

    # combine fundamentals + reversal-news into a final verdict + target weights (the ACTION)
    if news:
        print(f"\n[4] FINAL VERDICTS (fundamentals + falling-knife news check):")
        keep = []
        for s in picks.index:
            fv = fundamental_verdict(health_score(fetch_fundamentals(s))["score"])
            nv = news.get(s, "UNREAD")
            cv = "VETO" if nv == "VETO" or fv == "VETO" else ("CAUTION" if "CAUTION" in (fv, nv) else "BUY")
            print(f"     {s:5s}  fundamentals {fv:8s} news {nv:8s} -> {cv}")
            if cv != "VETO":
                keep.append((s, 0.5 if cv == "CAUTION" else 1.0))
        print("\n  >>> REVERSAL ACTION (equal-ish across non-vetoed bounce candidates):")
        if keep:
            tot = sum(w for _, w in keep)
            for s, w in keep:
                print(f"      BUY {s:5s}  {w/tot*100:4.0f}%  (bounce candidate, news-confirmed)")
            vetoed = [s for s in picks.index if s not in [k for k, _ in keep]]
            if vetoed:
                print(f"      SKIP (falling knife, news-vetoed): {', '.join(vetoed)}")
        else:
            print("      All bounce candidates were news-vetoed -> hold CASH.")
        return

    print(f"\n[4] CURRENT REVERSAL BOOK (from {HOLDINGS_FILE}):", cur or "(none)")
    print("\nNEWS_TO_READ (reversal framing):", ",".join(picks.index))
    print("\n>>> REVERSAL NEWS QUESTION (the falling-knife check) — for each pick ask:")
    print('    "This ETF/sector just dropped hard. Is that a TEMPORARY overreaction likely to')
    print('     bounce (CONFIRM), or is something genuinely BROKEN — a justified decline that')
    print('     keeps falling (VETO)? Heavy volume = capitulation (good); fresh bad fundamental')
    print('     news / structural break = falling knife (bad)."')
    print("    Then write", NEWS_FILE, 'as {"TICKER":"CONFIRM|CAUTION|VETO"} and re-run.')
    print("\n  ⚠ This is the AGGRESSIVE edge: ~16%/yr backtest but -46% worst drawdown (buys fallers).")
    print("  Independent from the momentum agent — treat as a separate strategy/book.")


if __name__ == "__main__":
    main()
