#!/usr/bin/env python3
"""monthly_action.py — THE ONE-BUTTON MONTHLY TOOL (100%-satellite allocation).

Run this ~once a month. It chains the whole system and prints ONE clear instruction:
  1. STORM-DETECTOR (SJM regime on SPY): risk-off? -> GO TO CASH, stop.
  2. MOMENTUM SCAN: if calm, rank all ETFs, take the top-N strongest (the 100%-satellite engine).
  3. FUNDAMENTAL HEALTH: score each pick (P/E, margins, debt, analyst target) -> CONFIRM/CAUTION/VETO.
  4. FINAL ACTION: cash, or "buy these ETFs at these weights", with veto warnings.

This is the user's chosen target: 100% satellite = gated ETF momentum top5 monthly.
Lightweight, single-process. Run:  python3 monthly_action.py
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, LAG
from early_scanner import panels, score_leader, ETF_UNIVERSE
from fundamentals import fetch_fundamentals, health_score, fundamental_verdict

TOP_N = 5
LEVERAGED = {"SOXL", "TECL", "TQQQ", "QLD", "SSO", "UPRO", "SPXL", "FAS", "TNA", "UDOW", "ROM"}
LEV_CAP = 0.20    # leveraged ETFs combined may not exceed 20% of the book (wipeout-risk guardrail)


def main(do_fundamentals=True):
    print("=" * 64)
    print("  MONTHLY ACTION  —  100%-satellite (gated ETF momentum)")
    print("=" * 64)

    # --- 1. STORM-DETECTOR ---
    spy = load_close("SPY")
    spy_ro, _, _, _ = walk_forward(spy, None)
    risk_off = spy_ro.iloc[-1] > 0.5
    asof = spy_ro.index[-1].date()
    print(f"\n[1] STORM-DETECTOR (as of {asof}): "
          f"{'🛟 RISK-OFF' if risk_off else '📈 CALM'}")

    if risk_off:
        print("\n" + "=" * 64)
        print("  ➡️  ACTION THIS MONTH:  GO TO CASH / T-BILLS (BIL/SHY).")
        print("      The detector senses a storm. Sit out. Do NOT buy the momentum")
        print("      picks even if they look strong — that's the whole point of the gate.")
        print("=" * 64)
        return

    # --- 2. MOMENTUM SCAN (only runs when calm) ---
    px, vol = panels(ETF_UNIVERSE)
    sc = score_leader(px)
    last = sc.iloc[-1].dropna()
    picks = last[last > 0].nlargest(TOP_N)
    if len(picks) == 0:
        print("\n[2] MOMENTUM: nothing has positive momentum -> ACTION: stay in CASH this month.")
        return
    print(f"\n[2] MOMENTUM top {len(picks)} (calm -> these are this month's candidates):")
    for s, v in picks.items():
        print(f"      {s:5s}  momentum score {v:.2f}")

    # --- 3. FUNDAMENTAL HEALTH on each pick ---
    rows = []
    for s in picks.index:
        if do_fundamentals:
            h = health_score(fetch_fundamentals(s))
            fv = fundamental_verdict(h["score"])
            note = "; ".join(h["flags"]) if h.get("flags") else ("ETF/no-fundamentals" if h["score"] is None else "—")
        else:
            fv, note, h = "NEUTRAL", "(fundamentals skipped)", {"score": None}
        rows.append({"ticker": s, "mom_score": round(float(picks[s]), 2),
                     "fund_verdict": fv, "fund_score": h.get("score"), "flags": note})
    df = pd.DataFrame(rows)
    print(f"\n[3] FUNDAMENTAL HEALTH (current snapshot):")
    print(df.to_string(index=False))

    # --- 4. FINAL ACTION: weight the non-vetoed picks ---
    # VETO -> drop; CAUTION -> half weight; CONFIRM/NEUTRAL -> full weight
    wmap = {"CONFIRM": 1.0, "NEUTRAL": 1.0, "CAUTION": 0.5, "VETO": 0.0}
    df["w"] = df["fund_verdict"].map(wmap).fillna(0.5)
    kept = df[df["w"] > 0].copy()
    vetoed = df[df["w"] == 0]
    print("\n" + "=" * 64)
    print("  ➡️  ACTION THIS MONTH (100% satellite, equal-ish across survivors):")
    print("=" * 64)
    if len(kept) == 0:
        print("  Every pick was vetoed on fundamentals -> stay in CASH this month.")
    else:
        # base weights ~ proportional to verdict weight
        kept["wt"] = kept["w"] / kept["w"].sum()
        # LEVERAGE CAP: cap combined leveraged-ETF weight at LEV_CAP, redistribute the excess to the rest
        kept["is_lev"] = kept["ticker"].isin(LEVERAGED)
        lev_w = kept.loc[kept["is_lev"], "wt"].sum()
        capped = False
        if lev_w > LEV_CAP and (~kept["is_lev"]).any():
            capped = True
            scale = LEV_CAP / lev_w
            kept.loc[kept["is_lev"], "wt"] *= scale
            excess = lev_w - LEV_CAP
            non_lev = ~kept["is_lev"]
            kept.loc[non_lev, "wt"] += excess * (kept.loc[non_lev, "wt"] / kept.loc[non_lev, "wt"].sum())
        elif lev_w > LEV_CAP:
            # all picks are leveraged -> can't redistribute; hold the rest in cash
            capped = True
            kept["wt"] *= LEV_CAP / lev_w
        kept["final_weight%"] = (100 * kept["wt"]).round(1)
        if capped:
            print(f"  (⚙️ leverage cap applied: leveraged ETFs trimmed to ≤{int(LEV_CAP*100)}% combined)")
        for _, r in kept.iterrows():
            lev_tag = " [3x LEVERAGED — capped]" if r["is_lev"] else ""
            tag = "" if r["fund_verdict"] in ("CONFIRM", "NEUTRAL") else f"  ({r['fund_verdict'].lower()}: {r['flags']})"
            print(f"   BUY {r['ticker']:5s}  {r['final_weight%']:5.1f}% of the book{lev_tag}{tag}")
        cash_left = round(100 - kept["final_weight%"].sum(), 1)
        if cash_left > 0.5:
            print(f"   HOLD {cash_left:.1f}% in CASH (leverage cap left some book unallocated)")
    if len(vetoed):
        print("\n   ⚠️  VETOED (momentum liked them, fundamentals said NO — skip):")
        for _, r in vetoed.iterrows():
            print(f"       {r['ticker']:5s}  — {r['flags']}")
    print("\n  Reminder: this is the AGGRESSIVE 100%-satellite book (~11%/yr, but -27% worst-")
    print("  crash). Re-run next month. If the detector flips to RISK-OFF, sell to cash.")
    print("\n  NOTE: for a full news sanity-check on these picks, also run a live web read")
    print("  (news_check.py brief) — fundamentals catch the numbers, news catches the story.")


if __name__ == "__main__":
    main(do_fundamentals="--no-fund" not in sys.argv)
