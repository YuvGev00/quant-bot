#!/usr/bin/env python3
"""harden_meanrev.py — stress-test the one signal that passed (10d cross-sectional mean-reversion,
buy recent losers). Three honest gauntlets before we'd ever trust it:
  (1) OUT-OF-SAMPLE: split pre-2017 (where we 'found' it) vs 2017-2026 (unseen). Does it hold on
      data it wasn't chosen on? Re-run the permutation test on EACH half.
  (2) COST/SLIPPAGE STRESS: re-run at 3/5/10/20/40 bps round-trip — where does the edge die?
      (Mean-reversion trades more than momentum, so it's cost-sensitive; this is the real risk.)
  (3) PER-DECADE STABILITY: Sharpe in each sub-period — is it consistent or one lucky era?
After IL tax @25%, gated by the causal storm-detector.
"""
from __future__ import annotations
import numpy as np, pandas as pd, math

from jump_model import load_close, walk_forward, hysteresis, after_tax, stats, LAG
from early_scanner import panels, ETF_UNIVERSE

RNG = np.random.default_rng(2024)
N_PERM = 150
TOP_N = 5


def meanrev_score(px):
    return -(px / px.shift(10) - 1)        # prefer the biggest 10d losers


def backtest(px, spy_in, cost_bps=3.0, perm_map=None):
    score = meanrev_score(px); rets = px.pct_change(fill_method=None); cols = list(px.columns)
    w = pd.DataFrame(0.0, index=px.index, columns=cols); last=None; cur=None
    for t in px.index:
        per = t.to_period("M")
        if per != last:
            s = score.loc[t].dropna()
            if len(s):
                picks = list(s.nlargest(TOP_N).index)
                if perm_map is not None:
                    picks = [cols[perm_map[cols.index(p)]] for p in picks]
                cur = pd.Series(1.0/len(picks), index=picks)
            else: cur=None
            last=per
        if cur is not None: w.loc[t,cur.index]=cur.values
    port=(w.shift(1)*rets).sum(axis=1); inv=w.shift(1).sum(axis=1)
    gi=spy_in.reindex(port.index).fillna(1.0); port=port*gi
    turn=w.diff().abs().sum(axis=1).fillna(0)
    return port - turn*(cost_bps/1e4)/2 + (1-inv*gi)*(0.025/252)


def sh(r): return stats(after_tax(r.dropna(),0.25))["Sharpe"]
def perm_p(px, spy_in, real_sh, cost=3.0):
    nc=px.shape[1]
    null=np.array([sh(backtest(px,spy_in,cost,RNG.permutation(nc))) for _ in range(N_PERM)])
    return (null>=real_sh).mean()


def main():
    px,_=panels(ETF_UNIVERSE); px=px[px.index>=pd.Timestamp("2007-01-01")]
    spy=load_close("SPY"); ro,_,_,_=walk_forward(spy,None)
    spy_in=hysteresis((1-ro).shift(LAG).fillna(1.0),3,3)

    full=backtest(px,spy_in); fs=stats(after_tax(full,0.25))
    print(f"FULL-sample mean-reversion: CAGR {fs['CAGR%']:.1f}%  Sharpe {fs['Sharpe']:.2f}  maxDD {fs['maxDD%']:.0f}%")

    # (1) OOS split
    print("\n=== (1) OUT-OF-SAMPLE split (found pre-2017 / tested 2017-2026 unseen) ===")
    for lbl, lo, hi in [("IN-SAMPLE 2007-2016","2007-01-01","2016-12-31"),
                         ("OUT-OF-SAMPLE 2017-2026","2017-01-01","2026-12-31")]:
        sub=px[(px.index>=lo)&(px.index<=hi)]
        si=spy_in.reindex(sub.index).fillna(1.0)
        r=backtest(sub,si); st=stats(after_tax(r,0.25)); p=perm_p(sub,si,st["Sharpe"])
        verdict = "HOLDS (p<0.05)" if p<0.05 else ("weak (p<0.20)" if p<0.20 else "FAILS OOS")
        print(f"  {lbl:26s} CAGR {st['CAGR%']:5.1f}%  Sharpe {st['Sharpe']:.2f}  maxDD {st['maxDD%']:5.0f}%  p={p:.3f}  -> {verdict}")

    # (2) cost stress
    print("\n=== (2) COST/SLIPPAGE stress (where does the edge die?) ===")
    for c in (3,5,10,20,40):
        r=backtest(px,spy_in,cost_bps=c); st=stats(after_tax(r,0.25))
        print(f"  {c:2d} bps/RT:  CAGR {st['CAGR%']:5.1f}%  Sharpe {st['Sharpe']:.2f}")
    print("  (mean-reversion trades a lot -> watch how fast Sharpe decays with cost)")

    # (3) per-period stability
    print("\n=== (3) PER-PERIOD stability (consistent edge or one lucky era?) ===")
    for lo,hi in [("2007","2011"),("2012","2016"),("2017","2021"),("2022","2026")]:
        sub=px[(px.index>=f"{lo}-01-01")&(px.index<=f"{hi}-12-31")]
        r=backtest(sub,spy_in.reindex(sub.index).fillna(1.0)); st=stats(after_tax(r,0.25))
        print(f"  {lo}-{hi}:  CAGR {st['CAGR%']:5.1f}%  Sharpe {st['Sharpe']:5.2f}  maxDD {st['maxDD%']:5.0f}%")

    print("\n=== VERDICT ===")
    print("- Trust it ONLY if it HOLDS out-of-sample AND the edge survives realistic (>=10bps) costs")
    print("  AND it's positive across most sub-periods. Otherwise it's an in-sample/era artifact.")


if __name__ == "__main__":
    main()
