#!/usr/bin/env python3
"""breakout_discover.py — EARLY DISCOVERY mode for the Breakout Scout.

Finds UNDISCOVERED / pre-hype small & mid-cap stocks — cheap AND quietly improving, BEFORE the crowd
(the "MU early-2025 / NVDA-2023 / PLTR-at-$6" moment, caught EARLY, not after the run). This is the
SECOND mode of the Scout, complementing the large-cap "deeply-beaten fingerprint" screen (breakout_scout.py).

It is BIGDATA-NATIVE: the universe is LIVE-DISCOVERED via Bigdata.com (small-caps aren't in our local
parquet bars), so the discovery sweep runs in a Claude session / the cloud routine — NOT in pure python.
This module's job is to (1) score/format whatever the sweep returns into a ranked, tiered list and (2)
render it into the breakout_site/ research site. The sweep writes breakout_discover.json; this reads it.

EARLY-FIT criteria (what "before everyone else" means here):
  • UNDER-COVERED  : few analysts, small/mid-cap, not yet consensus
  • CHEAP          : reasonable valuation vs growth (room to re-rate)
  • IMPROVING      : rising estimates / revenue growth / positive earnings surprise (quietly getting better)
  • NOT-YET-RUN    : the anti-hype guard — exclude names that already exploded (handled in the sweep)
Tiers: EMERGING (~$1-20B mid-cap, steadier) vs SPECULATIVE_EARLY (~$0.25-2B micro/IPO, lottery-risk).

HONEST: forward conviction-research over a live-discovered universe — even LESS backtestable than the
large-cap mode (these names have thin/no clean history; small-caps carry liquidity/fraud/total-loss risk).

Run:  python3 breakout_discover.py            # score + summarize what's in breakout_discover.json
      (the live discovery sweep itself is run by the session/cloud agent — see DISCOVERY_QUERIES below)
"""
from __future__ import annotations
import os, json

DISCOVER = "breakout_discover.json"

# the sweep the session/cloud agent runs (kept here as the documented contract)
DISCOVERY_QUERIES = [
    "under-the-radar small-cap and mid-cap US stocks with low analyst coverage that are cheaply valued "
    "but showing improving revenue growth and rising earnings estimates",
    "small-cap US companies with growing contract backlogs or accelerating revenue that Wall Street has "
    "not yet noticed, trading cheaply",
    "recently IPO'd US companies that have based out after the post-IPO decline and are starting to show "
    "improving fundamentals and early institutional interest",
    "overlooked mid-cap stocks with insider buying and rising earnings estimates trading below fair value",
]

FIT_RANK = {"STRONG": 0, "MODERATE": 1, "WEAK": 2}
TIER_RANK = {"EMERGING": 0, "SPECULATIVE_EARLY": 1}


def early_score(d: dict) -> float:
    """A transparent 0-100 early-stage score from the four criteria (for ranking/display).
    Rewards: low coverage, cheap, improving (rising est + surprise + rev growth)."""
    s = 0.0
    na = d.get("n_analysts")
    if na is not None:
        s += 22 if na <= 3 else 14 if na <= 6 else 6 if na <= 12 else 0      # under-covered
    if d.get("cheap"):                       s += 22                          # cheap vs growth
    if d.get("eps_rising"):                  s += 20                          # estimates improving
    surp = d.get("earnings_surprise_pct")
    if surp is not None and surp > 5:        s += min(18, surp / 25)          # earnings momentum (capped)
    rg = d.get("rev_growth_pct")
    if rg is not None and rg > 8:            s += min(10, rg / 4)             # revenue growth
    bp = d.get("base_pos_pct")
    if bp is not None and bp < 45:           s += 8                           # based-out / not extended
    return round(min(s, 100.0), 1)


def load():
    if not os.path.exists(DISCOVER): return None
    try: return json.load(open(DISCOVER))
    except Exception: return None


def ranked_discoveries():
    d = load()
    if not d: return [], {}
    items = d.get("discoveries", [])
    for x in items:
        x["early_score"] = early_score(x)
    items.sort(key=lambda x: (FIT_RANK.get(x.get("early_fit"), 3),
                              TIER_RANK.get(x.get("tier"), 2),
                              -x.get("early_score", 0)))
    return items, d


def early_catch_check():
    """ANECDOTAL sanity check (NOT statistical): rewind known winners to an EARLY (pre-hype) window —
    well before their run — and ask whether the price/coverage criteria would have flagged them as
    'early' THEN. Uses our local bars for the price-based criteria (coverage/estimates need point-in-time
    Bigdata, noted as a limitation). Honest about being a handful of cherry-picked cases."""
    import pandas as pd
    try:
        from jump_model import load_close
    except Exception:
        return []
    # (ticker, early-window date BEFORE the hype, the eventual run for context)
    CASES = [("PLTR", "2023-01-31", "later 10x+"), ("SMCI", "2023-01-31", "later ~10x"),
             ("MU", "2024-09-30", "2025 AI-memory run"), ("NVDA", "2023-01-31", "AI explosion"),
             ("AVGO", "2023-01-31", "AI run")]
    out = []
    for tk, d0, note in CASES:
        s = load_close(tk)
        if s is None or s.empty: continue
        hist = s[s.index <= pd.Timestamp(d0)]
        if len(hist) < 60: continue
        px = hist.iloc[-1]; win = hist.iloc[-504:] if len(hist) >= 504 else hist
        lo, hi = win.min(), win.max()
        base_pos = (px - lo) / (hi - lo) if hi > lo else 0.5
        fwd = s[s.index > pd.Timestamp(d0)]
        run = (fwd.max() / px - 1) if not fwd.empty else None
        # 'early' price tell = based-out (not already extended): base_pos in a low-to-mid band, not at highs
        early_flag = base_pos < 0.55
        out.append({"ticker": tk, "asof": d0, "base_pos": round(base_pos, 2),
                    "fwd_max_run_pct": round(run * 100, 0) if run is not None else None,
                    "early_price_flag": early_flag, "note": note})
    return out


def main():
    items, d = ranked_discoveries()
    print("=" * 80)
    print("BREAKOUT SCOUT — EARLY DISCOVERY mode (pre-hype small/mid-caps, found before the crowd)")
    print("SPECULATIVE forward conviction-research over a LIVE-DISCOVERED universe — NOT a backtested edge.")
    print("Small-caps carry liquidity / fraud / total-loss risk. Research leads, you decide.")
    print("=" * 80)
    if not items:
        print(f"\nNo discoveries in {DISCOVER}. The live Bigdata sweep (session/cloud agent) must run first.")
        print("Sweep queries:")
        for q in DISCOVERY_QUERIES: print("  •", q)
        return
    emrg = [x for x in items if x.get("tier") == "EMERGING"]
    spec = [x for x in items if x.get("tier") == "SPECULATIVE_EARLY"]
    print(f"\nas of {d.get('_asof','')} — {len(items)} discoveries  "
          f"({len(emrg)} EMERGING · {len(spec)} SPECULATIVE_EARLY)\n")
    print(f"{'TICK':6s} {'TIER':17s} {'FIT':8s} {'SCORE':>5s} {'MC$B':>6s} {'AN':>3s} {'VALUATION':16s} thesis")
    for x in items:
        print(f"{x['ticker']:6s} {x.get('tier',''):17s} {x.get('early_fit',''):8s} "
              f"{x.get('early_score',0):>5.0f} {(x.get('market_cap_b') or 0):>6.2f} "
              f"{(x.get('n_analysts') or 0):>3d} {(x.get('valuation','') or '')[:16]:16s} "
              f"{(x.get('thesis','') or '')[:60]}")
    print("\nTop EARLY-FIT picks:", ", ".join(x["ticker"] for x in items if x.get("early_fit") == "STRONG") or "none rated STRONG")

    chk = early_catch_check()
    if chk:
        print("\n--- ANECDOTAL early-catch sanity check (NOT statistical — a few known cases) ---")
        print("Would the price-based 'early' tell (based-out, not-yet-extended) have flagged these BEFORE their run?")
        for c in chk:
            verdict = "FLAGGED early ✓" if c["early_price_flag"] else "missed (already extended) ✗"
            run = f"then ran +{c['fwd_max_run_pct']:.0f}%" if c["fwd_max_run_pct"] is not None else ""
            print(f"  {c['ticker']:5s} @ {c['asof']}: base_pos {c['base_pos']:.2f} -> {verdict}  {run}  ({c['note']})")
        print("CAVEAT: price tell only (coverage/estimates need point-in-time Bigdata); cherry-picked winners; "
              "no control group here => anecdotal, not a win-rate.")


if __name__ == "__main__":
    main()
