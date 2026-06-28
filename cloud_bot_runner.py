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


def breakout():
    """BREAKOUT SCOUT (Phase-1 Stage A) — the pure-Python half of the weekly blow-up hunt.

    Runs the learned blow-up fingerprint (breakout_pattern.json) over every BEATEN-DOWN name on
    disk and emits a ranked shortlist + a per-sector COHORT block (the 'same-situation peers' each
    per-stock page compares against). The cloud Claude agent then does Stage B: live tearsheets +
    news per top name, writes breakout_ideas.json, and builds the multi-page research site.
    HONEST: structural price/volume screen only, survivorship-biased, NOT a backtested edge."""
    from early_scanner import panels
    from universe import all_symbols, SECTOR, STOCK_SECTOR
    import breakout_screen as bs

    print("=== BREAKOUT SCOUT :: STAGE A (learned blow-up fingerprint) ===")
    px, vol = panels(all_symbols(include_stocks=True))
    # A weekly STRUCTURAL turnaround screen uses 1y windows — a couple weeks of staleness is
    # immaterial (a name 40% off its high doesn't change category in 11 days). So we tolerate more
    # staleness here than the intraday momentum bot, but still fail loud on real coverage gaps.
    ok, dmsg = data_quality(px, max_stale_days=20)
    print(f"DATA CHECK: {dmsg}")
    if not ok:
        print(">>> Cloud agent: data degraded -> email 'DATA DEGRADED, scout suppressed', recommend nothing.")
        ping_healthcheck("weekly", fail=True)
        return

    pat = bs.load_or_learn(px, vol, refit=True)   # always re-fit on the freshest panel
    print(f"FINGERPRINT: {pat['_method']}")
    print(f"  centroid: dd {pat['centroid']['dd_from_high']*100:.0f}%  base {pat['centroid']['base_pos']*100:.0f}%  "
          f"accel {pat['centroid']['accel']:+.2f}  >ma50 {pat['centroid']['above_ma50']*100:+.0f}%  "
          f"vol-surge {pat['centroid']['vol_surge']:.2f}x")

    df = bs.screen(px, vol, pat)
    if df.empty:
        print(">>> No beaten-down candidates this cycle (market near highs). Cloud agent: email 'no setups'.")
        ping_healthcheck("weekly")
        return
    df["sector"] = df["ticker"].map(lambda t: SECTOR.get(t, "other"))
    df["kind"] = df["ticker"].map(lambda t: "stock" if t in STOCK_SECTOR else "etf")

    asof = str(px.index[-1].date())
    n = len(df)
    print(f"\n{n} BEATEN-DOWN CANDIDATES (>= {int(-pat.get('beaten_down_dd',-0.2)*100)}% off 1y high), "
          f"ranked by fingerprint match:")
    print(f"  {'#':>2} {'TICK':<6}{'MATCH':>6} {'PRICE':>9} {'DD%':>7} {'BASE%':>7} {'>MA50%':>7} {'VOLx':>6}  SECTOR")
    top12 = df.head(12)
    for i, r in top12.iterrows():
        print(f"  {i+1:>2} {r['ticker']:<6}{r['score']:>6.1f} {r['price']:>9.2f} {r['dd_from_high_pct']:>7.1f} "
              f"{r['base_pos_pct']:>7.1f} {r['above_ma50_pct']:>7.1f} {r['vol_surge']:>6.2f}  {r['sector']}")

    # COHORTS: every beaten-down candidate grouped by sector, ranked within the group. These are the
    # 'why this name and not its peers' comparison sets the per-stock pages render.
    cohorts = {}
    for sec, grp in df.groupby("sector"):
        cohorts[sec] = [
            {"ticker": rr["ticker"], "score": round(rr["score"], 1), "price": round(rr["price"], 2),
             "dd_from_high_pct": round(rr["dd_from_high_pct"], 1), "base_pos_pct": round(rr["base_pos_pct"], 1),
             "above_ma50_pct": round(rr["above_ma50_pct"], 1), "ret_3m_pct": round(rr["ret_3m_pct"], 1),
             "vol_surge": round(rr["vol_surge"], 2), "kind": rr["kind"]}
            for _, rr in grp.sort_values("score", ascending=False).iterrows()
        ]

    shortlist = [
        {"rank": i + 1, "ticker": r["ticker"], "sector": r["sector"], "kind": r["kind"],
         "score": round(r["score"], 1), "price": round(r["price"], 2),
         "dd_from_high_pct": round(r["dd_from_high_pct"], 1), "base_pos_pct": round(r["base_pos_pct"], 1),
         "ret_1m_pct": round(r["ret_1m_pct"], 1), "ret_3m_pct": round(r["ret_3m_pct"], 1),
         "above_ma50_pct": round(r["above_ma50_pct"], 1), "above_ma200_pct": round(r["above_ma200_pct"], 1),
         "ma50_slope_pct": round(r["ma50_slope_pct"], 1), "vol_surge": round(r["vol_surge"], 2),
         "base_age": round(r["base_age"], 2), "breakdown": r["breakdown"]}
        for i, r in df.iterrows()
    ]
    out = {
        "_asof": asof,
        "_generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "_stage": "A (pure-Python structural screen) — Stage B live research done by the cloud agent",
        "_disclaimer": "SPECULATIVE discretionary research, NOT a backtested edge. Structural "
                       "price/volume fingerprint on the bot's on-disk bars. Survivorship-biased "
                       "(today's survivors only). A research LEAD generator — verify every name live.",
        "pattern": {k: pat[k] for k in ("_method", "_caveat", "features", "weights", "centroid", "scale", "beaten_down_dd")},
        "shortlist": shortlist,
        "cohorts": cohorts,
        "news_to_read": list(top12["ticker"]),
    }
    json.dump(out, open("breakout_shortlist.json", "w"), indent=2)
    print(f"\n[wrote breakout_shortlist.json — {n} candidates, {len(cohorts)} sector cohorts]")
    print("\nNEWS_TO_READ:", ",".join(top12["ticker"]))
    print("\n>>> Cloud agent: STAGE B — for the TOP 6 names pull Bigdata.com tearsheets "
          "(overview/estimates/ratings/earnings/key_metrics) + recent news; compare 2-3 competitors;")
    print(">>> write a thesis/key-risk/verdict (CONFIRM/CAUTION/VETO) each; a VETO on a structurally")
    print(">>> broken name is a VALID, valuable output. Then write breakout_ideas.json, build the site,")
    print(">>> and email the ranked report. NOT a backtested edge — research leads, manual trades only.")
    ping_healthcheck("weekly")


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


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "weekly"
    {"weekly": weekly, "hourly": hourly, "breakout": breakout}.get(mode, weekly)()
