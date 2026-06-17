#!/usr/bin/env python3
"""the_bot.py — ONE MACHINE. The whole system tied together, built to eventually run as a bot.

Every run it does ALL of this:
  1. STORM-DETECTOR  — is it calm or risk-off? (the master switch)
  2. DISCOVER        — scan the universe, recommend NEW ETFs/stocks to buy (momentum)
  3. READ            — fundamentals (auto) + news verdicts (live web read, plugged in via JSON)
  4. MANAGE CURRENT  — read your holdings.json, say HOLD / ADD / TRIM / SELL each position
  5. ACTION REPORT   — one unified instruction sheet: target book + per-position orders

ARCHITECTURE (honest): this is the pure-Python STAGE A. It runs fully on its own and produces a
structured report + a NEWS-TO-READ list. Live news (STAGE B) needs a web search (the LLM/me, or an
API), so news verdicts are read from news_verdicts.json — fill it via a web read, or leave it and the
bot proceeds on price+fundamentals alone (marking news as 'unread'). This is exactly the seam where a
scheduled bot would call an LLM/news API. Everything else is automatable today.

Run:  python3 the_bot.py                 (uses holdings.json + news_verdicts.json if present)
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, LAG
from early_scanner import panels, score_leader, ETF_UNIVERSE
from fundamentals import fetch_fundamentals, health_score, fundamental_verdict

TOP_N = 5
LEVERAGED = {"SOXL", "TECL", "TQQQ", "QLD", "SSO", "UPRO", "SPXL", "FAS", "TNA", "UDOW", "ROM"}
LEV_CAP = 0.20
MAX_SINGLE = 0.35          # never put more than 35% of the book in one position (diversification floor)
TRIM_BAND = 0.05            # if a holding's target differs from current by >5% of book, act
HOLDINGS_FILE = "holdings.json"
NEWS_FILE = "news_verdicts.json"          # {"TICKER": "CONFIRM|CAUTION|VETO", ...} from a live web read


def load_json(path, default):
    if os.path.exists(path):
        try:
            d = json.load(open(path))
            return d
        except Exception:
            pass
    return default


def combined_verdict(ticker, fund_v, news_v):
    """Combine fundamentals + news into one verdict. The STRICTER of the two wins (safety-first):
    any VETO -> VETO; any CAUTION -> CAUTION; both CONFIRM/NEUTRAL -> CONFIRM."""
    order = {"VETO": 0, "CAUTION": 1, "NEUTRAL": 2, "CONFIRM": 3, "UNREAD": 2}
    worst = min(order.get(fund_v, 2), order.get(news_v, 2))
    return {0: "VETO", 1: "CAUTION", 2: "NEUTRAL", 3: "CONFIRM"}[worst]


def target_weights(picks, fund, news):
    """Build target book from momentum picks, combined verdict weighting, leverage cap."""
    rows = []
    for s, mom in picks.items():
        fv = fund.get(s, "NEUTRAL"); nv = news.get(s, "UNREAD")
        cv = combined_verdict(s, fv, nv)
        w = {"CONFIRM": 1.0, "NEUTRAL": 1.0, "CAUTION": 0.5, "VETO": 0.0}[cv]
        rows.append({"ticker": s, "mom": round(float(mom), 2), "fund": fv, "news": nv,
                     "verdict": cv, "w": w, "is_lev": s in LEVERAGED})
    df = pd.DataFrame(rows)
    kept = df[df["w"] > 0].copy()
    if len(kept) == 0:
        return df, kept
    kept["wt"] = kept["w"] / kept["w"].sum()
    lev_w = kept.loc[kept["is_lev"], "wt"].sum()
    if lev_w > LEV_CAP and (~kept["is_lev"]).any():
        kept.loc[kept["is_lev"], "wt"] *= LEV_CAP / lev_w
        excess = lev_w - LEV_CAP; nl = ~kept["is_lev"]
        kept.loc[nl, "wt"] += excess * (kept.loc[nl, "wt"] / kept.loc[nl, "wt"].sum())
    elif lev_w > LEV_CAP:
        kept["wt"] *= LEV_CAP / lev_w
    # single-position cap: never more than MAX_SINGLE in one name (excess -> cash, stay diversified)
    kept["wt"] = kept["wt"].clip(upper=MAX_SINGLE)
    return df, kept


def main():
    print("#" * 66)
    print("#   THE BOT — one machine: detect · discover · read · manage     #")
    print("#" * 66)

    holdings = load_json(HOLDINGS_FILE, {"cash": 1.0, "positions": {}})
    cur = holdings.get("positions", {})
    news = {k: v.upper() for k, v in load_json(NEWS_FILE, {}).items() if not k.startswith("_")}

    # 1. STORM-DETECTOR
    spy = load_close("SPY"); spy_ro, _, _, _ = walk_forward(spy, None)
    risk_off = spy_ro.iloc[-1] > 0.5
    asof = spy_ro.index[-1].date()
    print(f"\n[1] STORM-DETECTOR ({asof}): {'🛟 RISK-OFF — defensive' if risk_off else '📈 CALM — invest'}")

    # 2. DISCOVER (momentum scan)
    px, vol = panels(ETF_UNIVERSE)
    sc = score_leader(px).iloc[-1].dropna()
    picks = sc[sc > 0].nlargest(TOP_N)
    # always show the full top 10 ranking so you see what else is close / next in line
    top10 = sc.sort_values(ascending=False).head(10)
    print(f"\n[2a] TOP 10 by momentum (of {len(sc)} ETFs scanned; top {TOP_N} get picked):")
    for i, (s, v) in enumerate(top10.items(), 1):
        mark = "  ← PICKED" if s in picks.index else ("  (next in line)" if i <= TOP_N + 2 else "")
        lev = " [3x leveraged]" if s in LEVERAGED else ""
        print(f"     {i:2d}. {s:5s}  score {v:6.2f}{lev}{mark}")

    # 3. READ — fundamentals (auto) per pick + current holding
    universe_to_read = sorted(set(picks.index) | set(cur))
    fund = {}
    for s in universe_to_read:
        h = health_score(fetch_fundamentals(s))
        fund[s] = fundamental_verdict(h["score"])

    # decide target book
    if risk_off:
        target = {}    # all cash
        print("    Detector says storm -> TARGET BOOK = 100% CASH (no new buys).")
    else:
        df, kept = target_weights(picks, fund, news)
        target = dict(zip(kept["ticker"], kept["wt"])) if len(kept) else {}
        print(f"\n[2] DISCOVER — momentum recommends (calm):")
        print(df[["ticker", "mom", "fund", "news", "verdict"]].to_string(index=False))

    # 4. MANAGE CURRENT — compare holdings to target
    print(f"\n[3] MANAGE YOUR CURRENT POSITIONS (from {HOLDINGS_FILE}):")
    all_t = sorted(set(cur) | set(target))
    orders = []
    if not all_t:
        print("    (no current positions on file)")
    for t in all_t:
        c = cur.get(t, 0.0); tg = target.get(t, 0.0); d = tg - c
        if abs(d) <= TRIM_BAND:
            action = "HOLD" if c > 0 else "—"
        elif c == 0 and tg > 0:
            action = f"BUY new  -> {tg*100:.0f}%"
        elif tg == 0 and c > 0:
            action = "SELL all"
        elif d > 0:
            action = f"ADD     -> {tg*100:.0f}% (from {c*100:.0f}%)"
        else:
            action = f"TRIM    -> {tg*100:.0f}% (from {c*100:.0f}%)"
        why = []
        if t in target and target.get(t, 0) == 0 and t in (picks.index if not risk_off else []):
            why.append("vetoed")
        orders.append({"ticker": t, "current%": round(c*100, 1), "target%": round(tg*100, 1),
                       "action": action, "verdict": (news.get(t) or fund.get(t, "—"))})
    if orders:
        print(pd.DataFrame(orders).to_string(index=False))

    # 5. ACTION REPORT + the news-to-read list (Stage B seam)
    print("\n" + "=" * 66)
    print("  ➡️  THIS MONTH'S ORDERS:")
    print("=" * 66)
    if risk_off:
        print("   SELL everything to CASH/T-bills. Sit out until the detector turns calm.")
    elif not target:
        print("   Nothing passes -> hold CASH this month.")
    else:
        for o in orders:
            if o["action"] not in ("HOLD", "—"):
                print(f"   {o['action']:22s}  {o['ticker']}")
        held = [o['ticker'] for o in orders if o['action'] == 'HOLD']
        if held:
            print(f"   HOLD (already on target): {', '.join(held)}")
        tgt_cash = round(100 - sum(target.values())*100, 1)
        if tgt_cash > 0.5:
            print(f"   Keep {tgt_cash:.0f}% in CASH.")

    # news-to-read list: which picks still need a live news read (Stage B)
    unread = [s for s in (picks.index if not risk_off else []) if s not in news]
    if unread:
        print(f"\n  📰 NEWS TO READ (for a full verdict, do a live web read on): {', '.join(unread)}")
        print(f"     Then write {NEWS_FILE} as {{\"TICKER\":\"CONFIRM|CAUTION|VETO\"}} and re-run.")
    print("\n  (This is Stage A — fully automatable. Live news = Stage B, the LLM/news-API plug-in.)")


if __name__ == "__main__":
    main()
