#!/usr/bin/env python3
"""core_satellite.py — the full "mix of both" the user asked for:
  CORE   (set-and-mostly-forget): diversified risk-spread ETFs + storm-detector, yearly rebalance.
  SATELLITE (actively traded):    gated ETF momentum (the offense), monthly rebalance.
Blend them at a few core/satellite splits and compare to core-only and satellite-only, after IL tax.
Goal: keep most of the core's smoothness while adding some of the satellite's growth — and SHOW the
user which money is 'calm' vs 'active' so the structure is explicit.
"""
from __future__ import annotations
import numpy as np, pandas as pd

from jump_model import load_close as jm_load, walk_forward, LAG
from momentum_engine import price_panel, run_momentum, stats, after_tax, RF, COST_BPS_RT, ETF_ONLY
from core_portfolio import panel, yearly_rebal_weights, portfolio_ret, inv_vol_weights, CORE_ASSETS


def main():
    spy = jm_load("SPY"); spy_ro, _, _, _ = walk_forward(spy, None)

    # ---- CORE: diversified risk-spread + storm-detector on equity sleeves (the winner from core_portfolio) ----
    cpx = panel(list(CORE_ASSETS)).dropna()
    ivw = inv_vol_weights(cpx, list(CORE_ASSETS))
    core = portfolio_ret(cpx, yearly_rebal_weights(cpx, ivw), riskoff=spy_ro, gate_assets=["SPY","EFA","EEM","VNQ"])

    # ---- SATELLITE: gated ETF momentum top5 monthly (the trustworthy offense) ----
    mpx = price_panel(ETF_ONLY); mpx = mpx[mpx.index >= pd.Timestamp("2006-01-01")]
    sat, _ = run_momentum(mpx, top_n=5, rebal="ME", lev=1.0, riskoff=spy_ro)

    # align to common window
    idx = core.index.intersection(sat.index)
    core, sat = core.loc[idx], sat.loc[idx]

    print(f"Core+Satellite blend, {idx.min().date()} -> {idx.max().date()}, after Israeli tax @25%\n")
    bh = after_tax(jm_load("SPY").reindex(idx).pct_change(fill_method=None), 0.25)

    blends = {
        "100% core (set & forget)":      (1.00, 0.00),
        "85% core / 15% satellite":      (0.85, 0.15),
        "75% core / 25% satellite":      (0.75, 0.25),
        "60% core / 40% satellite":      (0.60, 0.40),
        "30% core / 70% satellite":      (0.30, 0.70),
        "20% core / 80% satellite":      (0.20, 0.80),
        "10% core / 90% satellite":      (0.10, 0.90),
        "100% satellite (all active)":   (0.00, 1.00),
    }
    rows = []
    store = {}
    for name,(cw, sw) in blends.items():
        # blend daily returns (rebalanced continuously is an approximation; fine at these weights)
        r = cw*core + sw*sat
        rat = after_tax(r, 0.25)
        rows.append({"blend": name, **stats(rat)})
        store[name] = rat
    rows.append({"blend": "buy-hold SPY (ref)", **stats(bh)})
    print("=== Core + Satellite blends (after-tax@25%) ===")
    print(pd.DataFrame(rows).set_index("blend").to_string(float_format=lambda x: f"{x:.2f}"))

    # crisis stress
    print("\n=== Crisis worst-drop (%) by blend ===")
    crises = {"GFC 2008":("2007-10-01","2009-03-31"), "COVID 2020":("2020-02-15","2020-04-30"), "Bear 2022":("2022-01-01","2022-10-31")}
    crows = []
    for name, r in store.items():
        row = {"blend": name}
        for cn,(a,b) in crises.items():
            w = r.loc[(r.index>=a)&(r.index<=b)]
            row[cn] = ((1+w).cumprod()/(1+w).cumprod().cummax()-1).min()*100 if len(w)>20 else np.nan
        crows.append(row)
    print(pd.DataFrame(crows).set_index("blend").to_string(float_format=lambda x: f"{x:.1f}"))

    print("\n=== Read (plain words) ===")
    print("- CORE = diversified risk-spread + storm-detector, rebalanced yearly (calm, low-tax, set & forget).")
    print("- SATELLITE = gated ETF momentum top5 monthly (the active growth engine).")
    print("- Pick the blend by how much active risk you want: more satellite = more growth + bigger swings.")
    print("- The sweet spot is usually 75/25 or 60/40 — most of the core's smoothness, a real growth kicker.")


if __name__ == "__main__":
    main()
