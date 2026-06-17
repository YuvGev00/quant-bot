#!/usr/bin/env python3
"""hunt_edges2.py — round 2 of the honest edge hunt, with VOLUME-aware signals (the daily-data
spirit of order-flow: did the move come on heavy volume = capitulation/conviction?). Plus seasonality,
gap-fade, dispersion. Same permutation gauntlet (p<0.05 = real), after IL tax, gated.
"""
from __future__ import annotations
import numpy as np, pandas as pd, math

from jump_model import load_close, walk_forward, hysteresis, after_tax, stats, LAG
from early_scanner import panels, ETF_UNIVERSE

RNG = np.random.default_rng(99)
N_PERM = 150
TOP_N = 5


def build_signals(px, vol):
    r = px.pct_change(fill_method=None)
    relvol = vol / vol.rolling(63).mean()                 # today's volume vs its norm
    sig = {}
    # plain 10d reversal (the known survivor — as control)
    sig["reversal-10d (control)"] = -(px/px.shift(10)-1)
    # VOLUME-CONFIRMED reversal: biggest losers THAT FELL ON HEAVY VOLUME (capitulation -> bounce)
    drop = -(px/px.shift(10)-1)
    sig["vol-confirmed reversal"] = drop * relvol.clip(0.5, 4.0)
    # dollar-volume-weighted reversal (favor liquid capitulations)
    sig["liquidity-weighted reversal"] = drop * np.log1p(vol.rolling(10).mean())
    # gap-fade: prefer names with a big DOWN gap (overnight overreaction)
    gap = px / px.shift(1) - 1
    sig["gap-fade (5d)"] = -gap.rolling(5).sum()
    # range-position: prefer names near the LOW of their 20d range (oversold)
    lo20 = px.rolling(20).min(); hi20 = px.rolling(20).max()
    sig["20d-range-low (oversold)"] = -((px - lo20) / (hi20 - lo20).replace(0, np.nan))
    # short reversal x low recent volatility (calm pullbacks bounce cleaner than volatile crashes)
    rv = r.rolling(20).std()
    sig["calm-pullback reversal"] = drop / (rv * math.sqrt(252)).replace(0, np.nan)
    # 1-day reversal (very short — strongest reversal horizon, but cost-sensitive)
    sig["reversal-1d"] = -r
    # 21d reversal (longer; tests horizon sensitivity)
    sig["reversal-21d"] = -(px/px.shift(21)-1)
    return sig


def backtest(px, score, spy_in, perm_map=None, cost_bps=5.0):
    rets = px.pct_change(fill_method=None); cols = list(px.columns)
    w = pd.DataFrame(0.0, index=px.index, columns=cols); last=None; cur=None
    for t in px.index:
        per = t.to_period("M")
        if per != last:
            s = score.loc[t].replace([np.inf,-np.inf],np.nan).dropna()
            if len(s):
                picks=list(s.nlargest(TOP_N).index)
                if perm_map is not None: picks=[cols[perm_map[cols.index(p)]] for p in picks]
                cur=pd.Series(1.0/len(picks),index=picks)
            else: cur=None
            last=per
        if cur is not None: w.loc[t,cur.index]=cur.values
    port=(w.shift(1)*rets).sum(axis=1); inv=w.shift(1).sum(axis=1)
    gi=spy_in.reindex(port.index).fillna(1.0); port=port*gi
    turn=w.diff().abs().sum(axis=1).fillna(0)
    return port - turn*(cost_bps/1e4)/2 + (1-inv*gi)*(0.025/252)


def sh(r): return stats(after_tax(r.dropna(),0.25))["Sharpe"]


def main():
    px,vol=panels(ETF_UNIVERSE); px=px[px.index>=pd.Timestamp("2007-01-01")]; vol=vol.reindex(px.index)
    spy=load_close("SPY"); ro,_,_,_=walk_forward(spy,None)
    spy_in=hysteresis((1-ro).shift(LAG).fillna(1.0),3,3)
    sigs=build_signals(px,vol); nc=px.shape[1]

    print(f"Round-2 hunt: {len(sigs)} volume/daily signals, permutation p<0.05, after tax, 5bps cost.\n")
    print(f"{'signal':30s} {'CAGR%':>6} {'Sharpe':>7} {'maxDD%':>7} {'p-value':>8}  verdict")
    print("-"*82)
    res=[]
    for name,score in sigs.items():
        real=backtest(px,score,spy_in); s=sh(real); st=stats(after_tax(real,0.25))
        null=np.array([sh(backtest(px,score,spy_in,RNG.permutation(nc))) for _ in range(N_PERM)])
        p=(null>=s).mean()
        v="** REAL **" if p<0.05 else ("marginal" if p<0.20 else "luck/fail")
        res.append((name,st["CAGR%"],s,st["maxDD%"],p,v))
        print(f"{name:30s} {st['CAGR%']:6.1f} {s:7.2f} {st['maxDD%']:7.0f} {p:8.3f}  {v}")
    print("-"*82)
    surv=[r[0] for r in res if r[4]<0.05]; marg=[r[0] for r in res if 0.05<=r[4]<0.20]
    print(f"\nSURVIVORS (p<0.05): {surv or 'NONE'}")
    print(f"MARGINAL (p<0.20): {marg or 'none'}")
    print("\nNote: 5bps cost here (vs 3 in round 1) is a stricter, more realistic bar for reversal signals.")


if __name__ == "__main__":
    main()
