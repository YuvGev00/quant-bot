#!/usr/bin/env python3
"""cloud_bot_runner.py — the self-contained entry point a CLOUD scheduled agent runs.

It does the PURE-PYTHON part of the bot (Stage A) and prints a structured report the cloud
Claude agent then enriches with a live news read (Stage B, which Claude does natively) and emails
to the user. Designed to run in a fresh checkout with no local state.

Two modes:
  python3 cloud_bot_runner.py weekly   -> full deep cycle: refresh data, detect, discover top10,
                                          fundamentals, manage holdings, emit action sheet + news list
  python3 cloud_bot_runner.py hourly   -> lightweight emergency watch: detector + holdings price check;
                                          prints ALERT lines ONLY if action may be needed, else 'all quiet'

The cloud agent's PROMPT handles: refreshing prices (free_fetch), reading news on the flagged tickers,
optionally reading live IBKR positions, and emailing the result. This script is the deterministic core.
"""
from __future__ import annotations
import sys, json, os
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, LAG
from early_scanner import panels, score_leader
from universe import all_symbols, diversify, SECTOR
from fundamentals import fetch_fundamentals, health_score, fundamental_verdict
from bot_utils import ping_healthcheck, data_quality, worth_rotating

TOP_N = 5
LEVERAGED = {"SOXL", "TECL", "TQQQ", "QLD", "SSO", "UPRO", "SPXL", "FAS", "TNA", "UDOW", "ROM"}
ALERT_DROP = -0.08          # hourly: alert if a holding is down >8% over the last 5 sessions
HOLDINGS_FILE = "holdings.json"


def load_holdings():
    if os.path.exists(HOLDINGS_FILE):
        try: return json.load(open(HOLDINGS_FILE)).get("positions", {})
        except Exception: pass
    return {}


def detector():
    spy = load_close("SPY"); ro, _, _, _ = walk_forward(spy, None)
    return (ro.iloc[-1] > 0.5), ro.index[-1].date()


def weekly():
    print("=== WEEKLY DEEP CYCLE ===")
    px, _ = panels(all_symbols(include_stocks=True))
    ok, msg = data_quality(px)                 # FAIL LOUD before any decision
    print(f"DATA CHECK: {msg}")
    if not ok:
        print(">>> Cloud agent: data degraded -> email 'DATA DEGRADED, decisions suppressed', do NOT recommend trades.")
        ping_healthcheck("weekly", fail=True)
        return
    risk_off, asof = detector()
    print(f"STORM-DETECTOR ({asof}): {'RISK-OFF' if risk_off else 'CALM'}")
    sc = score_leader(px.ffill()).iloc[-1].dropna()
    sc = sc.drop(labels=[s for s in sc.index if s in LEVERAGED], errors="ignore")  # no leverage
    top10 = sc.sort_values(ascending=False).head(10)
    print("\nTOP 10 by momentum (scanned %d ETFs, leverage excluded):" % len(sc))
    for i, (s, v) in enumerate(top10.items(), 1):
        print(f"  {i:2d}. {s:5s} {v:6.2f}")
    picks = pd.Series({x: sc[x] for x in diversify(list(sc[sc>0].sort_values(ascending=False).index), TOP_N, 2)})
    print("\nCANDIDATE PICKS (top %d):" % TOP_N)
    for s in picks.index:
        fv = fundamental_verdict(health_score(fetch_fundamentals(s))["score"])
        print(f"  {s:5s}  momentum {picks[s]:.2f}  fundamentals {fv}")
    cur = load_holdings()
    print("\nCURRENT HOLDINGS (from holdings.json):", cur or "(none on file)")
    # tax-aware rotation check: is rotating into the new picks worth the IL tax + cost drag?
    tgt = {s: 1.0/len(picks) for s in picks.index}
    do_it, drag, why = worth_rotating(cur, tgt, expected_gross_gain=0.03)  # cloud agent refines gain est
    print(f"\nTAX-AWARE ROTATION CHECK: {why}")
    print("\nNEWS_TO_READ:", ",".join(picks.index))
    print("\n>>> Cloud agent: now (1) read live news/blogs on NEWS_TO_READ, (2) optionally read live")
    print(">>> IBKR positions, (3) decide BUY/SELL/HOLD per pick + per holding (leverage<=20%,")
    print(">>> single<=35%, news-VETO drops a name, and DON'T churn if the tax-aware check says HOLD),")
    print(">>> (4) email the user the final order sheet.")
    # regenerate the dashboard so it never goes stale (commit+push handled by the cloud agent)
    try:
        import dashboard; dashboard.main(); print("\n[dashboard.html regenerated — cloud agent should git add/commit/push it]")
    except Exception as e:
        print(f"\n[dashboard regen skipped: {e}]")
    ping_healthcheck("weekly")          # dead-man's-switch: success ping is the FINAL line


SPY_DAY_DROP = -0.025      # fast heads-up if SPY falls > 2.5% in a single day
VIX_SPIKE = 30.0           # fast heads-up if VIX jumps above 30 (fear gauge)


def hourly():
    print("=== HOURLY EMERGENCY WATCH ===")
    risk_off, asof = detector()
    alerts = []          # ACTION alerts (the disciplined daily signal)
    heads_up = []        # FAST info-only tripwires (NOT trade signals)

    # --- the slow, disciplined ACTION signal ---
    if risk_off:
        alerts.append(f"STORM-DETECTOR flipped RISK-OFF ({asof}) -> consider moving to cash.")
    cur = load_holdings()
    for sym in cur:
        c = load_close(sym)
        if c is None or len(c) < 6: continue
        drop5 = c.iloc[-1] / c.iloc[-6] - 1
        if drop5 <= ALERT_DROP:
            alerts.append(f"{sym} down {drop5*100:.1f}% over last 5 sessions -> review.")

    # --- the FAST heads-up tripwires (info only; money still moves on the daily gate) ---
    spy = load_close("SPY")
    if spy is not None and len(spy) > 2:
        day = spy.iloc[-1] / spy.iloc[-2] - 1
        if day <= SPY_DAY_DROP:
            heads_up.append(f"SPY down {day*100:.1f}% today — watch the close; the regime gate decides at EOD.")
    vix = load_close("VIX")
    if vix is not None and len(vix) and vix.iloc[-1] >= VIX_SPIKE:
        heads_up.append(f"VIX at {vix.iloc[-1]:.0f} (fear elevated) — heads-up, not a sell signal yet.")

    if heads_up:
        print("HEADS-UP (info only, NOT a trade signal):")
        for h in heads_up: print("  ~ " + h)

    if alerts:
        print("ALERT:")
        for a in alerts: print("  - " + a)
        print("\n>>> Cloud agent: email the user — ACTION alerts (disciplined signal).")
    if heads_up and not alerts:
        print("\n>>> Cloud agent: email a brief HEADS-UP (clearly labeled 'info, not a trade signal').")
    if not alerts and not heads_up:
        print("ALL QUIET — no action needed. (Cloud agent: stay silent / no notification.)")
    ping_healthcheck("hourly")          # dead-man's-switch: pings even on a quiet run = "alive & healthy"


DISCOVER_SWEEPS = [
    "under-the-radar small-cap and mid-cap US stocks with low analyst coverage that are cheaply valued but showing improving revenue growth and rising earnings estimates",
    "small-cap US companies with growing contract backlogs or accelerating revenue that Wall Street has not yet noticed, trading cheaply",
    "recently IPO'd US companies that have based out after the post-IPO decline and are starting to show improving fundamentals and early institutional interest",
    "overlooked mid-cap stocks with insider buying and rising earnings estimates trading below fair value",
]


def discover():
    """Print the EARLY-DISCOVERY brief + the exact Bigdata sweep queries.

    The deterministic core here is just the playbook; the live discovery sweep,
    tearsheet scoring, breakout_discover.json, site build and email are done by
    the cloud agent (it has the Bigdata.com + Gmail MCP tools)."""
    print("=== WEEKLY EARLY-DISCOVERY SCOUT ===")
    print("GOAL: discover pre-hype, under-the-radar small & mid-cap US stocks that are")
    print("      CHEAP and QUIETLY IMPROVING, caught EARLY (the MU-early-2025 / NVDA-2023 /")
    print("      PLTR-at-$6 moment) — NOT after they ran. Speculative, NOT a backtested edge.\n")
    print("ANTI-HYPE GUARD: drop mega-caps, ETFs/funds, non-US names, and anything already")
    print("      up >150% over the trailing year (already ran = NOT early). Drop names whose")
    print("      'improving' story is not actually in the numbers.\n")
    print("BIGDATA SWEEP QUERIES (run each as a SEPARATE smart-mode search, max_chunks ~25):")
    for i, q in enumerate(DISCOVER_SWEEPS, 1):
        print(f"  ({chr(96+i)}) {q}")
    print("\nTHEN per extracted ticker: find_securities -> rp_entity_id, then")
    print("      bigdata_company_tearsheet (company_overview/analyst_estimates/analyst_ratings/")
    print("      key_metrics/latest_earnings). Score TIER (EMERGING vs SPECULATIVE_EARLY) +")
    print("      EARLY_FIT (STRONG/MODERATE/WEAK). Write breakout_discover.json, build")
    print("      breakout_site.py + dashboard.py, commit/push, and email the ranked report.")
    try:
        import breakout_discover as _bd
        doc = _bd.load()
        n = len(doc.get("discoveries", []))
        if n:
            print(f"\n[last run: {doc.get('_asof')} — {n} discoveries on file]")
            for d in _bd.ranked(doc):
                print(f"  {d['early_fit']:8s} {d['tier']:17s} {d['ticker']:6s} {d['name']}")
    except Exception as e:
        print(f"\n[no prior discovery file: {e}]")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "weekly"
    {"weekly": weekly, "hourly": hourly, "discover": discover}.get(mode, weekly)()
