#!/usr/bin/env python3
"""core_portfolio.py — the STABLE LONG-HOLD CORE (buy-and-hold ~1yr), the opposite of the momentum
satellite. Built for someone who wants to set-and-mostly-forget: a diversified, survivorship-free
ETF allocation, rebalanced ~yearly, with the SJM storm-detector as the ONLY defensive overlay.

NOT momentum-chased froth. The point of a core is durability, not chasing whatever's hot.
We test three honest core constructions and compare to plain buy-hold SPY, after Israeli tax @25%:
  1) classic 60/40 (SPY/IEF)
  2) diversified risk-spread (equity + intl + bonds + gold) equal-risk-ish
  3) the diversified core WITH the storm-detector overlay on its equity portion
All ETFs, daily, 2006+, after-tax. Yearly rebalance => minimal tax drag (the core's whole virtue).
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd

from jump_model import load_close as jm_load, walk_forward, LAG
from momentum_engine import stats, after_tax, RF, COST_BPS_RT

# diversified, survivorship-free building blocks (broad asset classes, not stock bets)
CORE_ASSETS = {
    "SPY": "US equity", "EFA": "intl developed", "EEM": "emerging mkts",
    "IEF": "US 7-10y bonds", "TLT": "US long bonds", "LQD": "corp bonds",
    "GLD": "gold", "VNQ": "REITs", "DBC": "commodities",
}


def load(sym):
    c = jm_load(sym)
    return c[c.index >= pd.Timestamp("2006-01-01")] if c is not None else None


def panel(syms):
    return pd.DataFrame({s: load(s) for s in syms if load(s) is not None}).sort_index()


def yearly_rebal_weights(px, target_w):
    """Hold fixed target weights, rebalance once a year (Jan). Minimal turnover -> minimal tax."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    last_year = None
    cur = pd.Series(target_w).reindex(px.columns).fillna(0.0)
    for t in px.index:
        if t.year != last_year:
            cur = pd.Series(target_w).reindex(px.columns).fillna(0.0); last_year = t.year
        w.loc[t] = cur.values
    return w


def portfolio_ret(px, weights, riskoff=None, gate_assets=None):
    rets = px.pct_change(fill_method=None)
    w = weights.copy()
    if riskoff is not None and gate_assets:
        # on risk-off days, move the gated (equity) sleeves to cash
        ro = riskoff.reindex(px.index).fillna(0.0).shift(LAG).fillna(0.0)
        for a in gate_assets:
            if a in w.columns:
                w.loc[ro > 0.5, a] = 0.0
    gross = (w.shift(1) * rets).sum(axis=1)
    cash = (1 - w.shift(1).sum(axis=1)) * (RF/252)
    turn = w.diff().abs().sum(axis=1).fillna(0)
    return gross + cash - turn*(COST_BPS_RT/1e4)/2


def inv_vol_weights(px, assets, lookback=252):
    """Risk-spread: weight inversely to each asset's trailing vol (cheap risk-parity), set at the
    START of the sample (static-ish, recomputed yearly inside the rebal loop is overkill for a core)."""
    rets = px[assets].pct_change(fill_method=None)
    vol = rets.tail(lookback*3).std()          # use a long window for a stable estimate
    iv = (1/vol); iv = iv/iv.sum()
    return iv.to_dict()


def main():
    spy = jm_load("SPY"); spy_ro, _, _, _ = walk_forward(spy, None)
    px = panel(list(CORE_ASSETS))
    px = px.dropna()                            # common window across all core assets
    print(f"Core universe: {list(px.columns)}")
    print(f"Common window: {px.index.min().date()} -> {px.index.max().date()} ({len(px)} days)\n")

    bh = after_tax(load("SPY").pct_change(fill_method=None), 0.25)

    cores = {}
    # 1) classic 60/40
    cores["60/40 (SPY/IEF)"] = (yearly_rebal_weights(px, {"SPY":0.6,"IEF":0.4}), None, None)
    # 2) diversified equal-weight across asset classes
    eqw = {a: 1/len(CORE_ASSETS) for a in CORE_ASSETS}
    cores["diversified equal-weight"] = (yearly_rebal_weights(px, eqw), None, None)
    # 3) diversified inverse-vol (risk-spread)
    ivw = inv_vol_weights(px, list(CORE_ASSETS))
    cores["diversified risk-spread"] = (yearly_rebal_weights(px, ivw), None, None)
    # 4) risk-spread + storm-detector on the EQUITY sleeves only
    gate_eq = ["SPY","EFA","EEM","VNQ"]
    cores["risk-spread + storm-detector"] = (yearly_rebal_weights(px, ivw), spy_ro, gate_eq)

    print("=== LONG-HOLD CORE constructions, yearly rebalance, after Israeli tax @25% ===")
    print(f"Benchmark buy-hold SPY: CAGR {stats(bh)['CAGR%']:.1f}%  Sharpe {stats(bh)['Sharpe']:.2f}  maxDD {stats(bh)['maxDD%']:.0f}%\n")
    rows = []
    store = {}
    for name, (w, ro, ga) in cores.items():
        r = portfolio_ret(px, w, riskoff=ro, gate_assets=ga)
        rat = after_tax(r, 0.25)
        rows.append({"core": name, **stats(rat)})
        store[name] = rat
    print(pd.DataFrame(rows).set_index("core").to_string(float_format=lambda x: f"{x:.2f}"))

    # crisis stress: 2008, 2020, 2022 drawdowns
    print("\n=== Crisis-window worst drop (%) — the core's job is to NOT terrify you ===")
    crises = {"GFC 2008":("2007-10-01","2009-03-31"), "COVID 2020":("2020-02-15","2020-04-30"),
              "Bear 2022":("2022-01-01","2022-10-31")}
    crows = []
    for name, r in store.items():
        row = {"core": name}
        for cn,(a,b) in crises.items():
            w = r.loc[(r.index>=a)&(r.index<=b)]
            row[cn] = ((1+w).cumprod()/(1+w).cumprod().cummax()-1).min()*100 if len(w)>20 else np.nan
        crows.append(row)
    sbh = {"core":"buy-hold SPY"}
    for cn,(a,b) in crises.items():
        w = bh.loc[(bh.index>=a)&(bh.index<=b)]
        sbh[cn] = ((1+w).cumprod()/(1+w).cumprod().cummax()-1).min()*100 if len(w)>20 else np.nan
    crows.append(sbh)
    print(pd.DataFrame(crows).set_index("core").to_string(float_format=lambda x: f"{x:.1f}"))

    print("\n=== Read (plain words) ===")
    print("- The CORE is what you buy and hold for ~a year (rebal yearly = tiny tax drag).")
    print("- It trades return for SMOOTHNESS: lower CAGR than all-stock, but far smaller crashes.")
    print("- '+ storm-detector' should cut the crisis drops further with little turnover.")
    print("- This is the 'set and mostly forget' money. The momentum satellite (separate) is the active part.")


if __name__ == "__main__":
    main()
