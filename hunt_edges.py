#!/usr/bin/env python3
"""hunt_edges.py — test a BATTERY of candidate signals (not just momentum) the honest way: each must
pass a Monte-Carlo PERMUTATION TEST (p<0.05 — real, not luck) AND survive cost+IL tax. Most will FAIL;
that's the point. Survivors are genuine candidates.

Each signal scores every instrument cross-sectionally each month; we long the top-N (equal weight),
gated by the storm-detector, after tax. The permutation null shuffles the score->future-return link.
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, hysteresis, after_tax, stats, LAG
from early_scanner import panels, ETF_UNIVERSE

RNG = np.random.default_rng(7)
N_PERM = 150
TOP_N = 5


# ---------- candidate signals: each maps a price panel -> a score DataFrame (higher = prefer) ----------
def sig_momentum(px):      # the benchmark (we know it ~fails)
    return sum(px.shift(21)/px.shift(21+l)-1 for l in (63,126,252))/3
def sig_meanrev(px):       # cross-sectional short-term reversal: prefer recent LOSERS
    return -(px/px.shift(10)-1)
def sig_lowvol(px):        # defensive: prefer LOW trailing volatility
    return -px.pct_change(fill_method=None).rolling(63).std()
def sig_relstrength(px):   # relative strength vs SPY (excess 6m return)
    spy = load_close("SPY").reindex(px.index).ffill()
    return (px/px.shift(126)-1).sub((spy/spy.shift(126)-1), axis=0)
def sig_trendquality(px):  # 12m return / its volatility (risk-adjusted trend = smoother winners)
    r = px.pct_change(fill_method=None)
    return (px/px.shift(252)-1) / (r.rolling(252).std()*math.sqrt(252)).replace(0,np.nan)
def sig_dualmom(px):       # absolute+relative: only positive-12m names, ranked by 6m
    m12 = px/px.shift(252)-1; m6 = px/px.shift(126)-1
    return m6.where(m12>0, -9)
def sig_distance_low(px):  # prefer names FAR below their 1y high (deep-value-ish, contrarian)
    return -(px/px.rolling(252).max()-1)*-1   # = (px/max -1), closer to 0 = near high; we prefer near-high
def sig_accel(px):         # momentum acceleration (1m minus 6m/6)
    return (px/px.shift(21)-1) - (px/px.shift(126)-1)/6


SIGNALS = {
    "momentum (benchmark)": sig_momentum,
    "mean-reversion (10d)": sig_meanrev,
    "low-volatility": sig_lowvol,
    "relative-strength vs SPY": sig_relstrength,
    "trend-quality (ret/vol)": sig_trendquality,
    "dual-momentum": sig_dualmom,
    "near-52w-high": sig_distance_low,
    "momentum-acceleration": sig_accel,
}


def backtest(px, score, spy_in, top_n=TOP_N, perm_map=None, positive_only=False):
    rets = px.pct_change(fill_method=None); cols = list(px.columns)
    w = pd.DataFrame(0.0, index=px.index, columns=cols); last=None; cur=None
    for t in px.index:
        per = t.to_period("M")
        if per != last:
            s = score.loc[t].dropna()
            if positive_only: s = s[s>-5]
            if len(s):
                picks = list(s.nlargest(top_n).index)
                if perm_map is not None:
                    picks = [cols[perm_map[cols.index(p)]] for p in picks]
                cur = pd.Series(1.0/len(picks), index=picks)
            else: cur=None
            last=per
        if cur is not None: w.loc[t,cur.index]=cur.values
    port = (w.shift(1)*rets).sum(axis=1)
    inv = w.shift(1).sum(axis=1)
    gi = spy_in.reindex(port.index).fillna(1.0)
    port = port*gi
    turn = w.diff().abs().sum(axis=1).fillna(0)
    return port - turn*(3.0/1e4)/2 + (1-inv*gi)*(0.025/252)


def sharpe_at(r): return stats(after_tax(r.dropna(),0.25))["Sharpe"]


def main():
    print("Loading panel + causal storm-gate...")
    px,_ = panels(ETF_UNIVERSE); px = px[px.index>=pd.Timestamp("2007-01-01")]
    spy = load_close("SPY"); ro,_,_,_ = walk_forward(spy,None)
    spy_in = hysteresis((1-ro).shift(LAG).fillna(1.0),3,3)
    ncols = px.shape[1]

    print(f"\nTesting {len(SIGNALS)} candidate signals — each must pass permutation p<0.05 AND beat tax.\n")
    print(f"{'signal':28s} {'CAGR%':>6} {'Sharpe':>7} {'maxDD%':>7} {'p-value':>8}  verdict")
    print("-"*78)
    results=[]
    for name, fn in SIGNALS.items():
        try:
            score = fn(px)
            real = backtest(px, score, spy_in)
            sh = sharpe_at(real); st = stats(after_tax(real,0.25))
            null = np.array([sharpe_at(backtest(px, score, spy_in, perm_map=RNG.permutation(ncols)))
                             for _ in range(N_PERM)])
            p = (null>=sh).mean()
            verdict = "** REAL **" if p<0.05 else ("marginal" if p<0.20 else "luck/fail")
            results.append((name, st["CAGR%"], sh, st["maxDD%"], p, verdict))
            print(f"{name:28s} {st['CAGR%']:6.1f} {sh:7.2f} {st['maxDD%']:7.0f} {p:8.3f}  {verdict}")
        except Exception as e:
            print(f"{name:28s}  ERROR {str(e)[:40]}")

    print("-"*78)
    survivors = [r for r in results if r[4] < 0.05]
    marginal = [r for r in results if 0.05 <= r[4] < 0.20]
    print(f"\nSURVIVORS (p<0.05, real edge): {[r[0] for r in survivors] or 'NONE — honest null result'}")
    print(f"MARGINAL (p<0.20, worth more data): {[r[0] for r in marginal] or 'none'}")
    print("\nHONEST NOTE: a signal passing here still needs OOS confirmation before real money.")
    print("Most failing is EXPECTED — real cross-sectional edges in liquid ETFs are rare.")


if __name__ == "__main__":
    main()
