# Part C: how much leverage can you SAFELY take? Sweep leverage on the trustworthy gated ETF-momentum
# strategy and find the highest multiplier that keeps the after-tax worst-crash above a ceiling
# (default -35%). Shows the full return-vs-drawdown curve so the tradeoff is explicit and honest.
#
# IMPORTANT REALITY CHECK printed at the end: a backtest applies leverage smoothly day-by-day. Real
# leverage can GAP through your stop in an overnight crash and get you margin-called at the bottom —
# so treat the "safe" leverage here as an UPPER bound, not a recommendation.
from __future__ import annotations
import math
import numpy as np, pandas as pd

from jump_model import load_close as jm_load, walk_forward
from momentum_engine import price_panel, run_momentum, stats, after_tax, ETF_ONLY

DD_CEILING = -35.0          # don't accept a strategy whose after-tax worst crash is worse than this


def main():
    print("Computing SJM storm-regime + sweeping leverage on gated ETF-momentum...")
    spy = jm_load("SPY"); spy_ro, _, _, _ = walk_forward(spy, None)
    px = price_panel(ETF_ONLY); px = px[px.index >= pd.Timestamp("2006-01-01")]

    spy_px = jm_load("SPY"); spy_px = spy_px[spy_px.index >= px.index.min()]
    bh = after_tax(spy_px.pct_change(fill_method=None), 0.25)
    print(f"\nBuy-hold SPY benchmark (after-tax@25%): "
          f"CAGR {stats(bh)['CAGR%']:.1f}%  Sharpe {stats(bh)['Sharpe']:.2f}  maxDD {stats(bh)['maxDD%']:.0f}%")
    print(f"DRAWDOWN CEILING you set: worst after-tax crash must stay better than {DD_CEILING:.0f}%\n")

    # Use the trustworthy top-5 ETF momentum, monthly, GATED. Sweep leverage finely.
    print("=== Leverage sweep: ETF-momentum top5 monthly, GATED (storm-detector on), after-tax@25% ===")
    rows = []
    for lev in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        net, _ = run_momentum(px, top_n=5, rebal="ME", lev=lev, riskoff=spy_ro)
        s = stats(after_tax(net, 0.25))
        rows.append({"leverage": f"{lev:.2f}x", "CAGR%": s["CAGR%"], "Sharpe": s["Sharpe"],
                     "maxDD%": s["maxDD%"], "vol%": s["vol%"],
                     "under_ceiling": "OK" if s["maxDD%"] >= DD_CEILING else "TOO RISKY"})
    df = pd.DataFrame(rows).set_index("leverage")
    print(df.to_string(float_format=lambda x: f"{x:.1f}" if isinstance(x, float) else x))

    # the answer
    ok = df[df["under_ceiling"] == "OK"]
    if len(ok):
        best_lev = ok["CAGR%"].idxmax()
        b = ok.loc[best_lev]
        print(f"\n>>> MAX SAFE LEVERAGE under a {DD_CEILING:.0f}% crash ceiling: {best_lev}")
        print(f"    -> after-tax {b['CAGR%']:.1f}%/yr, Sharpe {b['Sharpe']:.2f}, worst crash {b['maxDD%']:.0f}%")
        print(f"    vs buy-hold SPY {stats(bh)['CAGR%']:.1f}%/yr / {stats(bh)['maxDD%']:.0f}% crash")
    else:
        print(f"\n>>> Even 1x exceeds the {DD_CEILING:.0f}% ceiling — no leverage is 'safe' by this rule.")

    print("\n=== Honest reality check (read this before levering up) ===")
    print("- A backtest levers smoothly day-by-day. REAL 2x leverage can gap through the detector's exit")
    print("  in an overnight/weekend crash and get you margin-called at the bottom — a risk this number")
    print("  does NOT capture. Treat 'max safe leverage' as an optimistic UPPER bound.")
    print("- Leverage multiplies BOTH directions. The detector reduces but does not eliminate crash risk.")
    print("- A sane real-world choice is usually 1x-1.5x, not the max the backtest 'allows'.")


if __name__ == "__main__":
    main()
