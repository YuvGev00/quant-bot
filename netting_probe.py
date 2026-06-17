#!/usr/bin/env python3
"""netting_probe.py — adversarial test of the 'portfolio-level annual tax-netting' proposal.
Compares (A) sum of per-sleeve after-tax terminal wealth vs (B) after-tax of the equal-weight
BLENDED book. The gap = the cross-sleeve loss-offset benefit the proposal claims.

Uses the two sleeves already on disk that have honest after-tax paths:
  - CORE: diversified risk-spread + SJM storm-detector, yearly rebal (calm, mostly-up).
  - SAT : gated ETF momentum top5 monthly (the active sleeve).
This is the MOST FAVORABLE real pair available without re-running RSI2 (which only deepens
the point: if the gap is tiny on a 2-sleeve real blend, adding a 3rd won't rescue it).
Also runs a SYNTHETIC perfectly-anti-correlated pair to show the THEORETICAL ceiling.
"""
from __future__ import annotations
import numpy as np, pandas as pd

from jump_model import load_close as jm_load, walk_forward
from momentum_engine import price_panel, run_momentum, stats, after_tax, ETF_ONLY
from core_portfolio import panel, yearly_rebal_weights, portfolio_ret, inv_vol_weights, CORE_ASSETS


def terminal(r):
    return (1 + r.dropna()).prod()


def cagr(r):
    r = r.dropna(); return terminal(r) ** (252/len(r)) - 1


def netting_gap(sleeves, rate):
    """sleeves: dict name->daily ret Series on a COMMON index, equal notional.
    (A) tax each sleeve alone, sum terminal wealth (each starts at 1/N).
    (B) blend equal-weight daily, tax the blended book once.
    Returns (cagrA, cagrB, gap_bps)."""
    names = list(sleeves); N = len(names)
    idx = None
    for s in sleeves.values():
        idx = s.dropna().index if idx is None else idx.intersection(s.dropna().index)
    aligned = {k: v.reindex(idx).fillna(0.0) for k, v in sleeves.items()}
    # (A) each sleeve taxed alone; portfolio = equal-notional sum, rebal-free (let them run)
    wealthA = sum((1.0/N) * (1 + after_tax(aligned[k], rate)).cumprod() for k in names)
    # (B) equal-weight daily blend, taxed once
    blend = sum((1.0/N) * aligned[k] for k in names)
    wealthB = (1 + after_tax(blend, rate)).cumprod()
    cA = wealthA.iloc[-1] ** (252/len(idx)) - 1
    cB = wealthB.iloc[-1] ** (252/len(idx)) - 1
    return cA, cB, (cB - cA)*1e4, blend, aligned


def per_year_signs(aligned):
    rows = []
    for y in sorted(set(next(iter(aligned.values())).index.year)):
        row = {"year": y}
        for k, v in aligned.items():
            yv = v[v.index.year == y]
            row[k] = ((1+yv).prod()-1)*100
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def main():
    spy = jm_load("SPY"); spy_ro, _, _, _ = walk_forward(spy, None)

    cpx = panel(list(CORE_ASSETS)).dropna()
    ivw = inv_vol_weights(cpx, list(CORE_ASSETS))
    core = portfolio_ret(cpx, yearly_rebal_weights(cpx, ivw), riskoff=spy_ro,
                         gate_assets=["SPY","EFA","EEM","VNQ"])
    mpx = price_panel(ETF_ONLY); mpx = mpx[mpx.index >= pd.Timestamp("2006-01-01")]
    sat, _ = run_momentum(mpx, top_n=5, rebal="ME", lev=1.0, riskoff=spy_ro)

    print("=== REAL 2-sleeve pair: CORE (risk-spread+SJM) vs SAT (gated momentum) ===")
    for rate in (0.25, 0.47):
        cA, cB, gap, blend, aligned = netting_gap({"core": core, "sat": sat}, rate)
        print(f"\nrate={rate:.0%}:  (A) sum-of-isolated CAGR={cA*100:.3f}%   "
              f"(B) blended-book CAGR={cB*100:.3f}%   NETTING GAP={gap:+.1f} bps/yr")
        if rate == 0.25:
            yt = per_year_signs(aligned)
            yt["opp_sign"] = np.sign(yt["core"]) != np.sign(yt["sat"])
            print(yt.to_string(float_format=lambda x: f"{x:+.1f}"))
            opp_years = yt.index[yt["opp_sign"]].tolist()
            print(f"opposite-sign years (where netting CAN help): {opp_years}")

    # how often is a sleeve actually NEGATIVE on the year? (netting only bites then)
    yt = per_year_signs({"core": core, "sat": sat})
    core_neg = (yt["core"] < 0).sum(); sat_neg = (yt["sat"] < 0).sum()
    print(f"\ncore negative years: {core_neg}/{len(yt)}   sat negative years: {sat_neg}/{len(yt)}")

    # synthetic THEORETICAL CEILING: two sleeves with deliberately opposite annual P&L
    print("\n=== SYNTHETIC ceiling: perfectly anti-correlated ANNUAL sleeves ===")
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2006-01-01", "2026-06-16")
    yrs = sorted(set(idx.year))
    a = pd.Series(0.0, index=idx); b = pd.Series(0.0, index=idx)
    for i, y in enumerate(yrs):
        mask = idx.year == y; n = mask.sum()
        # alternate big win/big loss between sleeves each year
        ann_a = 0.30 if i % 2 == 0 else -0.20
        ann_b = -0.20 if i % 2 == 0 else 0.30
        a.loc[mask] = (1+ann_a)**(1/n) - 1
        b.loc[mask] = (1+ann_b)**(1/n) - 1
    cA, cB, gap, _, _ = netting_gap({"a": a, "b": b}, 0.25)
    print(f"rate=25%:  (A)={cA*100:.3f}%  (B)={cB*100:.3f}%  GAP={gap:+.1f} bps/yr "
          f"(this is a CONTRIVED upper bound)")


if __name__ == "__main__":
    main()
