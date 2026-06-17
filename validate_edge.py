#!/usr/bin/env python3
"""validate_edge.py — is the 100%-satellite edge REAL, or luck/overfit? Two honest tests:

(1) MONTE-CARLO PERMUTATION (Neurotrader-style): the momentum strategy picks the top-N ETFs each
    month by trailing return. NULL hypothesis = "momentum has no predictive power; the same picks
    would do this well by chance." We test it by RANDOMLY PERMUTING which ETF each momentum-rank
    maps to (break the rank->future-return link) many times, building a null distribution of Sharpe.
    p-value = fraction of random runs that beat the real strategy. p<0.05 => the edge is real.

(2) WALK-FORWARD PARAMETER ROBUSTNESS: does the edge survive perturbing the knobs (lookback, top-N)?
    A real edge is a broad PLATEAU; an overfit one is a lonely spike. We report the Sharpe surface.

After Israeli tax @25%, gated by the (causal, hysteresis) storm-detector. Honest verdict at the end.
"""
from __future__ import annotations
import sys, math
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, gate_returns, hysteresis, after_tax, stats, LAG
from early_scanner import panels, ETF_UNIVERSE

RNG = np.random.default_rng(12345)
N_PERM = 200          # permutations (each is a full backtest; keep modest to stay laptop-cool)


def momentum_picks_returns(px, spy_in, top_n=5, lb=(63, 126, 252), perm_map=None):
    """Monthly: rank by blended trailing momentum, hold top_n equal-weight, gated by spy_in.
    If perm_map is given (array reindexing columns), apply it to BREAK the rank->return link (null)."""
    rets = px.pct_change(fill_method=None)
    cols = list(px.columns)
    # momentum score
    sc = sum(px.shift(21) / px.shift(21 + l) - 1 for l in lb) / len(lb)
    w = pd.DataFrame(0.0, index=px.index, columns=cols)
    last = None; cur = None
    for t in px.index:
        per = t.to_period("M")
        if per != last:
            s = sc.loc[t].dropna(); s = s[s > 0]
            if len(s):
                picks = list(s.nlargest(top_n).index)
                if perm_map is not None:
                    # null: keep the SAME momentum ranking, but assign each rank a RANDOM etf's
                    # future returns (shuffle the column->returns mapping) -> destroys predictive link
                    picks = [cols[perm_map[cols.index(p)]] for p in picks]
                cur = pd.Series(1.0 / len(picks), index=picks)
            else:
                cur = None
            last = per
        if cur is not None:
            w.loc[t, cur.index] = cur.values
    port = (w.shift(1) * rets).sum(axis=1)
    invested = w.shift(1).sum(axis=1)
    # gate: when storm, go flat
    gi = spy_in.reindex(port.index).fillna(1.0)
    port = port * gi
    turn = w.diff().abs().sum(axis=1).fillna(0)
    net = port - turn * (3.0 / 1e4) / 2 + (1 - invested * gi) * (0.025 / 252)
    return net


def sharpe_at(r):
    rt = after_tax(r.dropna(), 0.25)
    return stats(rt)["Sharpe"]


def main():
    print("Loading ETF panel + storm-detector (causal)...")
    px, _ = panels(ETF_UNIVERSE)
    px = px[px.index >= pd.Timestamp("2007-01-01")]
    spy = load_close("SPY"); ro, _, _, _ = walk_forward(spy, None)
    spy_in = hysteresis((1 - ro).shift(LAG).fillna(1.0), 3, 3)

    # real strategy
    real = momentum_picks_returns(px, spy_in)
    real_sh = sharpe_at(real)
    real_st = stats(after_tax(real, 0.25))
    print(f"\nREAL 100%-satellite (gated momentum top5): CAGR {real_st['CAGR%']:.1f}%  "
          f"Sharpe {real_sh:.2f}  maxDD {real_st['maxDD%']:.0f}%")

    # ---- (1) permutation test ----
    print(f"\n=== (1) Monte-Carlo permutation test ({N_PERM} runs) ===")
    print("Null: momentum rank has no link to future returns. Each run shuffles that link.")
    ncols = px.shape[1]
    null_sh = []
    for i in range(N_PERM):
        pm = RNG.permutation(ncols)
        null_sh.append(sharpe_at(momentum_picks_returns(px, spy_in, perm_map=pm)))
        if (i + 1) % 50 == 0: print(f"  ...{i+1}/{N_PERM}")
    null_sh = np.array(null_sh)
    pval = (null_sh >= real_sh).mean()
    print(f"\nReal Sharpe {real_sh:.2f}  vs  null mean {null_sh.mean():.2f} "
          f"(95th pctile {np.percentile(null_sh,95):.2f}, max {null_sh.max():.2f})")
    print(f">>> p-value = {pval:.3f}  ({(null_sh>=real_sh).sum()}/{N_PERM} random runs beat the real edge)")
    verdict = ("REAL — momentum edge unlikely to be luck (p<0.05)" if pval < 0.05 else
               "MARGINAL — beats most random but not decisively (0.05<=p<0.20)" if pval < 0.20 else
               "WEAK/LUCK — random permutations match it; edge is questionable (p>=0.20)")
    print(f">>> VERDICT: {verdict}")

    # ---- (2) parameter robustness ----
    print("\n=== (2) Parameter robustness (Sharpe surface; a real edge is a PLATEAU not a spike) ===")
    rows = []
    for top_n in (3, 5, 8):
        for lb in [(42, 84, 168), (63, 126, 252), (84, 168, 336)]:
            sh = sharpe_at(momentum_picks_returns(px, spy_in, top_n=top_n, lb=lb))
            rows.append({"top_n": top_n, "lookback": f"{lb[0]}/{lb[1]}/{lb[2]}", "Sharpe": sh})
    surf = pd.DataFrame(rows)
    print(surf.pivot(index="top_n", columns="lookback", values="Sharpe").to_string(float_format=lambda x: f"{x:.2f}"))
    spread = surf["Sharpe"].max() - surf["Sharpe"].min()
    print(f"\nSharpe range across params: {surf['Sharpe'].min():.2f} – {surf['Sharpe'].max():.2f}  (spread {spread:.2f})")
    print("PLATEAU (small spread, all positive) = robust edge. SPIKE (one high, rest low/neg) = overfit.")

    print("\n=== HONEST BOTTOM LINE ===")
    print(f"- Permutation: {verdict}")
    print(f"- Robustness: {'broad plateau, robust' if spread < 0.30 and surf['Sharpe'].min() > 0.2 else 'uneven — sensitive to params, be cautious'}")
    print("- Only chase higher % AFTER this says the base edge is real.")


if __name__ == "__main__":
    main()
