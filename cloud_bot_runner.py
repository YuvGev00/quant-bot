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
from early_scanner import panels, score_leader, ETF_UNIVERSE
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
    px, _ = panels(ETF_UNIVERSE)
    ok, msg = data_quality(px)                 # FAIL LOUD before any decision
    print(f"DATA CHECK: {msg}")
    if not ok:
        print(">>> Cloud agent: data degraded -> email 'DATA DEGRADED, decisions suppressed', do NOT recommend trades.")
        ping_healthcheck("weekly", fail=True)
        return
    risk_off, asof = detector()
    print(f"STORM-DETECTOR ({asof}): {'RISK-OFF' if risk_off else 'CALM'}")
    sc = score_leader(px).iloc[-1].dropna()
    top10 = sc.sort_values(ascending=False).head(10)
    print("\nTOP 10 by momentum (scanned %d ETFs):" % len(sc))
    for i, (s, v) in enumerate(top10.items(), 1):
        lev = " [3x LEV]" if s in LEVERAGED else ""
        print(f"  {i:2d}. {s:5s} {v:6.2f}{lev}")
    picks = sc[sc > 0].nlargest(TOP_N)
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
    ping_healthcheck("weekly")          # dead-man's-switch: success ping is the FINAL line


def hourly():
    print("=== HOURLY EMERGENCY WATCH ===")
    risk_off, asof = detector()
    alerts = []
    if risk_off:
        alerts.append(f"STORM-DETECTOR flipped RISK-OFF ({asof}) -> consider moving to cash.")
    cur = load_holdings()
    for sym in cur:
        c = load_close(sym)
        if c is None or len(c) < 6: continue
        drop5 = c.iloc[-1] / c.iloc[-6] - 1
        if drop5 <= ALERT_DROP:
            alerts.append(f"{sym} down {drop5*100:.1f}% over last 5 sessions -> review.")
    if alerts:
        print("ALERT:")
        for a in alerts: print("  - " + a)
        print("\n>>> Cloud agent: notify the user (push + email) with these alerts.")
    else:
        print("ALL QUIET — no action needed. (Cloud agent: stay silent / no notification.)")
    ping_healthcheck("hourly")          # dead-man's-switch: pings even on a quiet run = "alive & healthy"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "weekly"
    (weekly if mode == "weekly" else hourly)()
