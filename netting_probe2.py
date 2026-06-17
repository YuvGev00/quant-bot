#!/usr/bin/env python3
"""netting_probe2.py — CLEAN isolation of the tax-netting effect ONLY.
The proposal's claim is purely about WHEN tax is applied (per-sleeve vs on the netted book),
NOT about rebalancing. So we must compare on IDENTICAL pre-tax paths.

Method:
  Build the equal-weight daily blend ONCE -> this is the pre-tax book, fixed.
  (B) after_tax(blend) once  -> netting allowed (loss in one sleeve nets gain in other same year).
  (A) decompose the SAME blend into the per-sleeve contributions, tax each sleeve's annual P&L
      ALONE (no cross-offset), recombine. Same pre-tax dollars, only the tax timing differs.
The gap (B)-(A) is then PURE netting, no rebalancing/vol-drag confound.
"""
from __future__ import annotations
import numpy as np, pandas as pd

from jump_model import load_close as jm_load, walk_forward
from momentum_engine import price_panel, run_momentum, after_tax, ETF_ONLY
from core_portfolio import panel, yearly_rebal_weights, portfolio_ret, inv_vol_weights, CORE_ASSETS


def annual_pnl(daily, w):
    """Per-year dollar P&L of a sleeve weighted w within a fixed-notional book (1.0 total).
    Returns dict year->pnl_dollars assuming the sleeve's slice rides at constant 1/N notional
    rebased each Jan (i.e. tax lots realized yearly)."""
    out = {}
    for y in sorted(set(daily.index.year)):
        yv = daily[daily.index.year == y].dropna()
        out[y] = w * (((1+yv).prod()) - 1)   # dollar gain on w notional that year
    return out


def isolated_after_tax_wealth(sleeves_w, rate):
    """(A): each sleeve's ANNUAL P&L taxed alone (carryforward per-sleeve), summed.
    sleeves_w: list of (daily_ret, weight)."""
    years = sorted(set(sleeves_w[0][0].index.year))
    carry = [0.0]*len(sleeves_w)
    wealth = 1.0
    for y in years:
        net_after = 0.0
        for i, (d, w) in enumerate(sleeves_w):
            yv = d[d.index.year == y].dropna()
            g = w * (((1+yv).prod()) - 1)
            if g >= 0:
                u = min(g, carry[i]); carry[i] -= u; tx = g - u
            else:
                carry[i] += -g; tx = 0.0
            net_after += g - rate*max(0.0, tx)
        wealth += net_after
    return wealth


def netted_after_tax_wealth(sleeves_w, rate):
    """(B): blend the SAME annual P&Ls, net them, tax the netted book."""
    years = sorted(set(sleeves_w[0][0].index.year))
    carry = 0.0; wealth = 1.0
    for y in years:
        g = 0.0
        for d, w in sleeves_w:
            yv = d[d.index.year == y].dropna()
            g += w * (((1+yv).prod()) - 1)
        if g >= 0:
            u = min(g, carry); carry -= u; tx = g - u
        else:
            carry += -g; tx = 0.0
        wealth += g - rate*max(0.0, tx)
    return wealth


def main():
    spy = jm_load("SPY"); spy_ro, _, _, _ = walk_forward(spy, None)
    cpx = panel(list(CORE_ASSETS)).dropna()
    ivw = inv_vol_weights(cpx, list(CORE_ASSETS))
    core = portfolio_ret(cpx, yearly_rebal_weights(cpx, ivw), riskoff=spy_ro,
                         gate_assets=["SPY","EFA","EEM","VNQ"])
    mpx = price_panel(ETF_ONLY); mpx = mpx[mpx.index >= pd.Timestamp("2006-01-01")]
    sat, _ = run_momentum(mpx, top_n=5, rebal="ME", lev=1.0, riskoff=spy_ro)

    idx = core.dropna().index.intersection(sat.dropna().index)
    core, sat = core.reindex(idx).fillna(0.0), sat.reindex(idx).fillna(0.0)
    nyears = len(set(idx.year))

    print("=== CLEAN netting isolation (same pre-tax annual P&L, only tax-timing differs) ===")
    print("REAL pair: 50% core (risk-spread+SJM) / 50% sat (gated momentum)\n")
    for rate in (0.25, 0.47):
        sw = [(core, 0.5), (sat, 0.5)]
        wA = isolated_after_tax_wealth(sw, rate)
        wB = netted_after_tax_wealth(sw, rate)
        gapA = wA**(1/ (len(idx)/252)) - 1
        gapB = wB**(1/ (len(idx)/252)) - 1
        print(f"rate={rate:.0%}: isolated wealth={wA:.4f}  netted wealth={wB:.4f}  "
              f"=> netting CAGR gain ~ {(gapB-gapA)*1e4:+.2f} bps/yr "
              f"(abs terminal gain {(wB-wA)*100:+.3f}% of start capital over {nyears}y)")

    # The ONLY years netting helps: a sleeve must be NEGATIVE while the book is POSITIVE.
    print("\nYears where a sleeve had a realized LOSS (netting bites only then):")
    for y in sorted(set(idx.year)):
        cv = ((1+core[core.index.year==y]).prod()-1)*0.5
        sv = ((1+sat[sat.index.year==y]).prod()-1)*0.5
        if cv < 0 or sv < 0:
            print(f"  {y}: core_contrib={cv*100:+.2f}%  sat_contrib={sv*100:+.2f}%  "
                  f"{'<-- loss shelters gain' if cv*sv<0 else '<-- BOTH down, no offset, just deferral'}")


if __name__ == "__main__":
    main()
