# Plain-picture comparison: the SJM "market storm-detector" gate vs just buy-and-hold SPY.
# Produces a 2-panel PNG: (top) growth of $10k, (bottom) when the detector was in stocks vs cash,
# shaded over the SPY price. Also prints the CURRENT live signal (in stocks or in cash today).
from __future__ import annotations
import math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from jump_model import load_close, walk_forward, gate_returns, LAG

COST_BPS_RT = 3.0


def main():
    spy = load_close("SPY")
    riskoff, ret, rf, lams = walk_forward(spy, None)
    in_mkt = (1 - riskoff)                      # 1 = in stocks, 0 = in cash
    in_mkt_lag = in_mkt.shift(LAG).fillna(1.0)
    sjm = gate_returns(in_mkt_lag, ret.loc[in_mkt_lag.index], rf.loc[in_mkt_lag.index])
    idx = sjm.index
    bh = ret.loc[idx]                           # buy & hold

    eq_sjm = 10000 * (1 + sjm).cumprod()
    eq_bh = 10000 * (1 + bh).cumprod()
    px = spy.reindex(idx)

    # ---- the picture ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[2, 1], sharex=True)
    fig.suptitle("Market 'storm-detector' (SJM) vs just Buy & Hold — SPY, $10,000 start",
                 fontsize=15, fontweight="bold")

    ax1.plot(eq_bh.index, eq_bh.values, lw=1.6, color="#888", label="Buy & Hold (always in stocks)")
    ax1.plot(eq_sjm.index, eq_sjm.values, lw=1.8, color="#1565c0", label="Storm-detector (in/out of stocks)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Account value ($, log scale)")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, which="both", alpha=0.25)
    # annotate the end values
    ax1.annotate(f"${eq_sjm.iloc[-1]:,.0f}", (eq_sjm.index[-1], eq_sjm.iloc[-1]),
                 color="#1565c0", fontweight="bold", fontsize=11, xytext=(8, 0), textcoords="offset points", va="center")
    ax1.annotate(f"${eq_bh.iloc[-1]:,.0f}", (eq_bh.index[-1], eq_bh.iloc[-1]),
                 color="#666", fontsize=11, xytext=(8, -12), textcoords="offset points", va="center")

    # bottom: SPY price with red shading where the detector pulled OUT to cash
    ax2.plot(px.index, px.values, lw=1.0, color="#333", label="SPY price")
    out = in_mkt_lag.reindex(idx).fillna(1.0) < 0.5
    # shade contiguous out-of-market spans
    ymin, ymax = px.min() * 0.9, px.max() * 1.05
    in_span = False
    for i in range(len(idx)):
        if out.iloc[i] and not in_span:
            start = idx[i]; in_span = True
        elif not out.iloc[i] and in_span:
            ax2.axvspan(start, idx[i], color="#e53935", alpha=0.20)
            in_span = False
    if in_span:
        ax2.axvspan(start, idx[-1], color="#e53935", alpha=0.20)
    ax2.set_ylabel("SPY price")
    ax2.set_xlabel("Year")
    ax2.set_ylim(ymin, ymax)
    ax2.grid(True, alpha=0.25)
    ax2.text(0.012, 0.92, "Red shading = detector moved you to CASH (sensed a storm)",
             transform=ax2.transAxes, fontsize=10, color="#b71c1c", va="top",
             bbox=dict(boxstyle="round", fc="white", ec="#e53935", alpha=0.8))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out_path = "sjm_vs_buyhold.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved {out_path}")

    # ---- numbers + the live signal ----
    def cagr(e): return (e.iloc[-1] / e.iloc[0]) ** (252 / len(e)) - 1
    def maxdd(e): return (e / e.cummax() - 1).min()
    print(f"\nOver {idx.min().date()} -> {idx.max().date()} ({len(idx)/252:.0f} years), $10,000 became:")
    print(f"  Storm-detector : ${eq_sjm.iloc[-1]:,.0f}   ({cagr(eq_sjm)*100:.1f}%/yr, worst drop {maxdd(eq_sjm)*100:.0f}%)")
    print(f"  Buy & Hold     : ${eq_bh.iloc[-1]:,.0f}   ({cagr(eq_bh)*100:.1f}%/yr, worst drop {maxdd(eq_bh)*100:.0f}%)")
    print("  (these are pre-tax USD; the storm-detector's edge is mainly the much smaller worst drop)")

    today_state = "IN STOCKS  📈" if in_mkt.iloc[-1] > 0.5 else "IN CASH  🛟"
    last_flip = in_mkt[in_mkt.diff().abs() > 0]
    since = last_flip.index[-1].date() if len(last_flip) else idx[0].date()
    print(f"\n>>> TODAY'S SIGNAL ({idx[-1].date()}): the detector says you should be {today_state}")
    print(f"    (last changed its mind on {since})")
    n_flips = int(in_mkt.diff().abs().sum())
    print(f"    Over the whole period it changed its mind only {n_flips} times "
          f"(~{n_flips/(len(idx)/252):.1f}x/year) — so you'd rarely trade.")


if __name__ == "__main__":
    main()
