#!/usr/bin/env python3
"""bot_utils.py — operational hardening borrowed from the trading-bot-landscape research:
  1. ping_healthcheck()  — dead-man's-switch (only model that catches a SILENT run that never fires)
  2. data_quality()      — fail-loud staleness + coverage asserts (suppress decisions on bad data)
  3. tax_turnover_penalty() / worth_rotating() — make IL tax + cost first-class in the rotation decision

Env vars (set in the cloud routine, NOT hardcoded):
  HC_WEEKLY_URL, HC_HOURLY_URL  — Healthchecks.io ping URLs (https://healthchecks.io, free tier)
"""
from __future__ import annotations
import os, sys
import pandas as pd

COST_BPS_RT = 3.0           # round-trip trading cost (bps of notional), Interactive-Israel-ish
TAX_RATE = 0.25             # IL CGT on realized gains (use 0.47 for marginal)


# ---------- 1. dead-man's-switch ----------
def ping_healthcheck(which="weekly", fail=False, start=False):
    """Ping a Healthchecks.io check. Call as the FINAL line after asserts pass (success),
    or with fail=True from an except block. No-op if the URL env var isn't set (local runs)."""
    url = os.environ.get("HC_HOURLY_URL" if which == "hourly" else "HC_WEEKLY_URL")
    if not url:
        return False
    if start: url = url.rstrip("/") + "/start"
    elif fail: url = url.rstrip("/") + "/fail"
    try:
        import urllib.request
        urllib.request.urlopen(url, timeout=10)
        return True
    except Exception:
        return False


# ---------- 2. fail-loud data quality ----------
def data_quality(prices: pd.DataFrame, min_coverage=0.80, max_stale_days=5):
    """Return (ok, message). prices = DataFrame (cols=tickers). FAIL LOUD: a silent stale/empty
    read feeding a confident 'rotate into X' is the worst failure mode for an emailed decision."""
    if prices is None or prices.empty:
        return False, "DATA DEGRADED: empty price frame"
    last = prices.dropna(how="all").index.max()
    if last is None:
        return False, "DATA DEGRADED: no dated rows"
    stale = (pd.Timestamp.today().normalize() - pd.Timestamp(last).normalize()).days
    # coverage = fraction of tickers that have a RECENT bar (within max_stale_days of the newest),
    # not just a value on the single literal last timestamp (files can end on slightly different days)
    recent = prices.loc[prices.index >= pd.Timestamp(last) - pd.Timedelta(days=max_stale_days + 2)]
    cov = recent.notna().any().mean() if len(recent) else 0.0
    if cov < min_coverage:
        return False, f"DATA DEGRADED: only {cov*100:.0f}% of tickers have data (<{min_coverage*100:.0f}%)"
    if stale > max_stale_days:
        return False, f"DATA DEGRADED: latest bar is {stale}d old (>{max_stale_days}d) — feed may be stale/429'd"
    return True, f"data OK ({cov*100:.0f}% coverage, latest bar {stale}d old)"


# ---------- 3. tax-aware rotation ----------
def tax_turnover_penalty(current: dict, target: dict, tax_rate=TAX_RATE):
    """Estimate the after-tax DRAG of rotating current->target weights: each sold fraction realizes
    gains taxed at tax_rate, plus round-trip cost on the turnover. Returns drag as a fraction of book.
    (Approximation: assumes sold positions are in gain; conservative for a momentum book.)"""
    keys = set(current) | set(target)
    turnover = sum(abs(target.get(k, 0.0) - current.get(k, 0.0)) for k in keys) / 2.0  # one-way
    sold = sum(max(current.get(k, 0.0) - target.get(k, 0.0), 0.0) for k in keys)
    tax_drag = sold * tax_rate * 0.10        # assume ~10% embedded gain on sold lots (tunable)
    cost_drag = turnover * 2 * (COST_BPS_RT / 1e4)
    return tax_drag + cost_drag


def worth_rotating(current: dict, target: dict, expected_gross_gain: float, tax_rate=TAX_RATE):
    """Should we actually rotate? Only if the expected gross improvement beats the after-tax+cost
    drag of doing it. Stops the bot churning for tiny momentum reshuffles that die after 25-47% tax.
    Returns (do_it: bool, drag, why)."""
    drag = tax_turnover_penalty(current, target, tax_rate)
    do_it = expected_gross_gain > drag * 1.5      # require a 1.5x margin over friction
    why = (f"expected gain {expected_gross_gain*100:.2f}% vs rotation drag {drag*100:.2f}% "
           f"(tax+cost) -> {'ROTATE' if do_it else 'HOLD (not worth the tax hit)'}")
    return do_it, drag, why


if __name__ == "__main__":
    # demo
    cur = {"EWY": 0.20, "SOXL": 0.20, "USO": 0.20, "TECL": 0.20}
    tgt = {"EWY": 0.35}
    print("data_quality demo:", data_quality(pd.DataFrame()))
    do, drag, why = worth_rotating(cur, tgt, expected_gross_gain=0.03)
    print("worth_rotating demo:", why)
    print("healthcheck (no env set):", ping_healthcheck("weekly"))
