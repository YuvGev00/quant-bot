#!/usr/bin/env python3
"""breakout_agent.py — the BREAKOUT SCOUT, Stage A (pure-python, runs in the cloud checkout).

⚠ HONESTY FIRST: this is SPECULATIVE discretionary research, NOT a backtested edge (cf. the
reversal edge, permutation p=0.000). It scans the ~100 large-cap stocks on disk and ranks the
beaten-down ones by how closely they match a hand-specified "blow-up fingerprint" — the classic
MU-2016 / INTC / DELL-before-the-run setup (deep drawdown → freefall stalls → volume dries up after
a capitulation washout → starting to turn). The fingerprint lives in breakout_pattern.json.

It is a RESEARCH LEAD GENERATOR, not a signal. Stage A is price/volume only; the fundamental floor
(analyst upside, rising EPS, profitability — what separates a turnaround from a value trap) is added
in Stage B (the cloud LLM reading live Bigdata.com tearsheets + news), written to breakout_ideas.json.

Outputs breakout_shortlist.json:
  - shortlist: ranked top-N candidates with per-feature scorecards
  - cohorts:   ALL beaten-down candidates grouped by sector, ranked (the same-situation peers the
               per-stock dossier pages compare against — 'why this one and not its peers')
  - news_to_read: suggested searches for the top names (Stage B reads these live)

Run: python3 breakout_agent.py   (or via: python3 cloud_bot_runner.py breakout)
"""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd

from early_scanner import load_ohlcv
from universe import STOCK_SECTOR

PATTERN_FILE = "breakout_pattern.json"
OUT_FILE = "breakout_shortlist.json"
TOP = 12


def load_pattern() -> dict:
    return json.load(open(PATTERN_FILE))


def _bump(x, lo, peak, hi):
    """Triangular preference: 0 outside [lo,hi], 1 at peak, linear in between."""
    if x is None or np.isnan(x): return 0.0
    if x <= lo or x >= hi: return 0.0
    return (x - lo) / (peak - lo) if x < peak else (hi - x) / (hi - peak)


def fingerprint(close: pd.Series, vol: pd.Series, pat: dict) -> dict | None:
    """Score one name against the blow-up fingerprint. Returns the 0-100 score, the raw feature
    values, and a per-feature 0-100 sub-score scorecard (transparent, no black box)."""
    f = pat["features"]
    c = close.dropna(); v = vol.reindex(c.index).ffill()
    if len(c) < pat["thresholds"]["min_history_days"]:
        return None
    px = float(c.iloc[-1])

    # --- raw feature values ---
    win = f["dd_from_high"]["win"]
    hi = float(c.tail(win).max()); lo = float(c.tail(win).min())
    dd = px / hi - 1.0                                   # drawdown from 1y high (negative)
    base_pos = (px - lo) / (hi - lo) if hi > lo else 0.5  # 0=at low, 1=at high

    ret = c.pct_change()
    r1m = px / float(c.iloc[-f["downtrend_stall"]["short"]]) - 1.0
    nlong = f["downtrend_stall"]["long"]
    r6m = px / float(c.iloc[-nlong]) - 1.0
    drift_per_short = r6m * (f["downtrend_stall"]["short"] / nlong)   # what 1m would be at the 6m pace
    stall = r1m - drift_per_short                         # >0 = decelerating / turning up

    rv21 = float(ret.tail(f["vol_compression"]["win"]).std())
    rv_ref = float(ret.tail(f["vol_compression"]["ref"]).rolling(f["vol_compression"]["win"]).std().median())
    vol_ratio = rv21 / rv_ref if rv_ref else 1.0          # <1 = compressed (coiling)

    v5 = v.rolling(5).mean()
    cap_win = f["capitulation_vol"]["win"]; cap_ref = f["capitulation_vol"]["ref"]
    vavg = float(v.tail(cap_ref).mean())
    cap_spike = float(v5.tail(cap_win).max()) / vavg if vavg else 1.0   # recent washout volume

    s, m = f["early_turn"]["short"], f["early_turn"]["med"]
    mom_s = px / float(c.iloc[-s]) - 1.0
    mom_m = px / float(c.iloc[-m]) - 1.0
    turn = mom_s - mom_m / (m / s)                        # short pace exceeding the trailing trend

    # --- per-feature sub-scores 0-100 ---
    sc = {}
    sc["dd_from_high"] = 100 * _bump(dd, f["dd_from_high"]["ideal_lo"], f["dd_from_high"]["peak"], f["dd_from_high"]["ideal_hi"])
    sc["base_position"] = 100 * max(0.0, min(1.0, (f["base_position"]["ideal_max"] - base_pos) / f["base_position"]["ideal_max"] + 0.2))
    sc["downtrend_stall"] = 100 * (1 / (1 + np.exp(-stall * 12)))          # logistic: turning up -> high
    sc["vol_compression"] = 100 * max(0.0, min(1.0, (1.25 - vol_ratio) / 0.75))   # <0.5 ->100, >1.25 ->0
    cap = (cap_spike - 1.0) / (f["capitulation_vol"]["min_spike"] - 1.0)
    sc["capitulation_vol"] = 100 * max(0.0, min(1.0, cap))
    sc["early_turn"] = 100 * (1 / (1 + np.exp(-turn * 14)))

    score = sum(sc[k] * f[k]["weight"] for k in f)        # weighted 0-100

    return {
        "score": round(score, 1),
        "price": round(px, 2),
        "dd_from_high_pct": round(dd * 100, 1),
        "base_pos_pct": round(base_pos * 100, 1),
        "hi_52w": round(hi, 2), "lo_52w": round(lo, 2),
        "stall": round(stall * 100, 1),
        "vol_ratio": round(vol_ratio, 2),
        "cap_spike": round(cap_spike, 2),
        "turn_pct": round(turn * 100, 1),
        "ret_1m_pct": round(r1m * 100, 1),
        "ret_6m_pct": round(r6m * 100, 1),
        "scorecard": {k: round(sc[k], 0) for k in sc},
    }


def screen() -> dict:
    pat = load_pattern()
    th = pat["thresholds"]
    rows = []
    for t in sorted(STOCK_SECTOR):
        d = load_ohlcv(t)
        if d is None: continue
        fp = fingerprint(d["close"], d["volume"], pat)
        if fp is None: continue
        # candidate = genuinely beaten down (in the fingerprint's drawdown band)
        if not (th["candidate_max_dd"] * 100 <= fp["dd_from_high_pct"] <= th["candidate_min_dd"] * 100):
            continue
        rows.append({"ticker": t, "sector": STOCK_SECTOR[t], **fp})

    rows.sort(key=lambda r: r["score"], reverse=True)

    # cohorts: ALL beaten-down candidates grouped by sector, ranked within sector (same-situation peers)
    cohorts: dict[str, list] = {}
    for r in rows:
        cohorts.setdefault(r["sector"], []).append({
            "ticker": r["ticker"], "score": r["score"], "price": r["price"],
            "dd_from_high_pct": r["dd_from_high_pct"], "base_pos_pct": r["base_pos_pct"],
            "ret_6m_pct": r["ret_6m_pct"],
        })
    for sec in cohorts:
        cohorts[sec].sort(key=lambda x: x["score"], reverse=True)

    shortlist = rows[:TOP]
    news_to_read = [{
        "ticker": r["ticker"], "sector": r["sector"],
        "query": f"recent news, earnings and analyst views on {r['ticker']} "
                 f"(down {abs(r['dd_from_high_pct']):.0f}% from its 1y high) — is the decline justified or is a turnaround underway?",
    } for r in shortlist[:6]]

    return {
        "_asof": str(pd.Timestamp.today().date()),
        "_pattern": pat.get("_name"),
        "_disclaimer": "SPECULATIVE discretionary research, NOT a backtested edge. Stage-A price/volume "
                       "fingerprint match only — ranks beaten-down names for human research. Live snapshot, "
                       "not point-in-time. Individual-stock survivorship applies.",
        "n_candidates": len(rows),
        "shortlist": shortlist,
        "cohorts": cohorts,
        "news_to_read": news_to_read,
    }


def main():
    print("=" * 74)
    print("  BREAKOUT SCOUT — Stage A — beaten-down names vs the blow-up fingerprint")
    print("  ⚠ SPECULATIVE research, NOT a backtested edge. Research leads, not signals.")
    print("=" * 74)
    res = screen()
    json.dump(res, open(OUT_FILE, "w"), indent=1)

    sl = res["shortlist"]
    print(f"\n[A] {res['n_candidates']} beaten-down candidates found (dd within the fingerprint band). "
          f"Ranked top {len(sl)}:\n")
    print(f"  {'#':>2} {'TICK':5} {'SECTOR':13} {'SCORE':>6} {'PRICE':>9} {'DD%':>7} {'52wPOS':>7} {'6m%':>7}")
    for i, r in enumerate(sl, 1):
        print(f"  {i:>2} {r['ticker']:5} {r['sector']:13} {r['score']:>6.1f} {r['price']:>9.2f} "
              f"{r['dd_from_high_pct']:>7.1f} {r['base_pos_pct']:>6.0f}% {r['ret_6m_pct']:>7.1f}")

    print("\n[A] COHORTS (same-sector beaten-down peers, ranked — the 'why this one not its peers' set):")
    for sec, peers in sorted(res["cohorts"].items(), key=lambda kv: -max(p["score"] for p in kv[1])):
        names = ", ".join(f"{p['ticker']}({p['score']:.0f})" for p in peers)
        print(f"   {sec:13}: {names}")

    print("\nNEWS_TO_READ (Stage B reads these live via Bigdata.com, then writes breakout_ideas.json):")
    for n in res["news_to_read"]:
        print(f"   - {n['ticker']}: {n['query']}")

    print(f"\nwrote {OUT_FILE}  ({res['n_candidates']} candidates, top {len(sl)} shortlisted)")
    print("NEXT: Stage B — research the top 6 on Bigdata.com, write breakout_ideas.json, build the site.")
    return res


if __name__ == "__main__":
    main()
