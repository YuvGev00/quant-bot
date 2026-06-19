#!/usr/bin/env python3
"""eps_revision.py — EPS estimate-revision momentum, FORWARD-collection harness.

THE EDGE (the one new signal rated 'strong-replicated' in the hunt): stocks whose forward
consensus EPS estimate is being revised UP tend to outperform; revised DOWN, underperform.
Price-independent — orthogonal to everything else in the bot (storm-gate, reversal, momentum).

WHY FORWARD-ONLY (honest): RavenPack/Bigdata.com gives TODAY's consensus estimate, not last
month's. There is NO revision HISTORY to backtest. So we BUILD the point-in-time panel ourselves:
snapshot NTM-EPS each week → next week's delta IS the revision signal. After ~12-24 weekly
snapshots we can permutation-test whether up-revisers outperformed, net of IL tax. Until then,
it is an UNVALIDATED live tilt — treat as such.

This module does NOT call the network itself (Bigdata is an MCP tool the agent/LLM calls). It:
  1. defines the universe + the NTM-EPS extraction from a tearsheet's analyst_estimates
  2. appends a dated snapshot row to eps_snapshots.csv  (the PIT panel, grown over time)
  3. computes the revision signal once >=2 snapshots exist, and (later) the forward eval

USAGE (two-stage, like the news layer):
  STAGE B (LLM/agent, weekly): for each ticker, call bigdata_company_tearsheet(..., ['analyst_estimates']),
    extract NTM-EPS via ntm_eps_from_estimates(), and write a snapshot via record_snapshot().
  STAGE A (this script): `python3 eps_revision.py signal`  -> rank names by latest revision
                          `python3 eps_revision.py eval`   -> once enough history, forward-test it
"""
from __future__ import annotations
import sys, os, csv, json
import pandas as pd, numpy as np

SNAP = "eps_snapshots.csv"          # the point-in-time panel we grow: date,ticker,ntm_eps,n_analysts
# tens of names (RavenPack is rate-limited) — large, liquid, well-covered. Edit freely.
UNIVERSE = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","ADBE","CRM","ORCL","AMD","NFLX",
            "JPM","BAC","V","MA","UNH","LLY","JNJ","PG","KO","WMT","COST","HD","XOM","CVX","CAT"]


def ntm_eps_from_estimates(forward_estimates: list) -> tuple[float, int]:
    """Next-twelve-months consensus EPS = sum of the next 4 quarterly eps.average (after today).
    Returns (ntm_eps, min_num_analysts). Robust to missing fields."""
    rows = sorted(forward_estimates, key=lambda r: r.get("period",""))
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    fut = [r for r in rows if r.get("period","") >= today][:4]
    if len(fut) < 4:
        fut = rows[:4]
    eps = [r.get("eps",{}).get("average") for r in fut]
    nan = [r.get("eps",{}).get("num_analysts") or 0 for r in fut]
    if any(e is None for e in eps): return (float("nan"), 0)
    return (float(sum(eps)), int(min(nan) if nan else 0))


def record_snapshot(date: str, ticker: str, ntm_eps: float, n_analysts: int):
    """Append one dated PIT row. Idempotent-ish: one row per (date,ticker)."""
    head = not os.path.exists(SNAP)
    existing = set()
    if not head:
        df = pd.read_csv(SNAP)
        existing = set(zip(df["date"].astype(str), df["ticker"]))
    if (date, ticker) in existing: return False
    with open(SNAP, "a", newline="") as f:
        w = csv.writer(f)
        if head: w.writerow(["date","ticker","ntm_eps","n_analysts"])
        w.writerow([date, ticker, f"{ntm_eps:.4f}", n_analysts])
    return True


def load_panel() -> pd.DataFrame:
    if not os.path.exists(SNAP): return pd.DataFrame()
    df = pd.read_csv(SNAP); df["date"] = pd.to_datetime(df["date"]); return df.sort_values("date")


def revision_signal() -> pd.DataFrame:
    """Latest revision per ticker = % change in NTM-EPS from its previous snapshot. >0 = upgrade."""
    df = load_panel()
    if df.empty: return df
    out = []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < 2: continue
        prev, last = g.iloc[-2], g.iloc[-1]
        if prev["ntm_eps"] and not np.isnan(prev["ntm_eps"]):
            rev = last["ntm_eps"]/prev["ntm_eps"] - 1
            out.append({"ticker": t, "ntm_eps": last["ntm_eps"], "prev": prev["ntm_eps"],
                        "revision%": round(rev*100, 2), "n_analysts": int(last["n_analysts"]),
                        "from": prev["date"].date(), "to": last["date"].date()})
    return pd.DataFrame(out).sort_values("revision%", ascending=False) if out else pd.DataFrame()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    df = load_panel()
    if mode == "status":
        if df.empty:
            print("No snapshots yet. The weekly cloud agent must collect them (Stage B).")
            print(f"Universe: {len(UNIVERSE)} names. Snapshots accumulate in {SNAP}.")
        else:
            snaps = df["date"].nunique()
            print(f"PIT panel: {len(df)} rows, {df['ticker'].nunique()} tickers, {snaps} weekly snapshots "
                  f"({df['date'].min().date()} -> {df['date'].max().date()})")
            print(f"Need >=12 snapshots before a forward permutation-test is meaningful. Have {snaps}.")
    elif mode == "signal":
        sig = revision_signal()
        if sig.empty:
            print("Need >=2 snapshots per name to compute a revision. Keep collecting weekly.")
        else:
            print("EPS-REVISION signal (rising NTM-EPS = analysts upgrading = the edge):")
            print(sig.to_string(index=False))
            print("\nTILT: overweight the top up-revisers, avoid/skip the down-revisers.")
            print("⚠ UNVALIDATED until the forward eval has >=12 snapshots. Live tilt, not proven.")
    elif mode == "eval":
        print("Forward eval (revision -> next-period return) needs >=12 snapshots + price join.")
        print("TODO once panel is deep enough: join each snapshot's revision to forward return from")
        print("bars_<ticker>_1d.parquet, permutation-test (shuffle revision->return), after IL tax.")


if __name__ == "__main__":
    main()
