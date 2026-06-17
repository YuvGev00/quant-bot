# Does the SJM SPY-regime gate improve the EXISTING RSI2 basket (not just buy-hold)?
# Idea: the Statistical Jump Model already pays its way as an SPY buy-hold gate. Here we use the
# SAME SPY-derived risk-off regime to GATE the 180-instrument RSI2 basket: in the high-vol regime,
# take NO new RSI2 entries (and/or hold flat). If regime-gating an already-good edge lifts after-tax
# Sharpe / cuts drawdown without much turnover, that's free improvement on a strategy you already own.
#
# Reuses: jump_model.walk_forward (SJM regime on SPY), meta_label.extract_trades + the REAL
# event-driven basket_fixed_fraction (honest Sharpe). All after Israeli CGT.
from __future__ import annotations
import glob, os, math
import numpy as np, pandas as pd

from jump_model import load_close as jm_load, walk_forward, LAG
from meta_label import (extract_trades, load_close as ml_load, vix_series, stats, after_tax,
                        basket_fixed_fraction, EXCLUDE, FEATURES)

N_SLOTS = 10


def spy_riskoff_regime():
    """SJM risk-off (1=high-vol) daily series on SPY, lagged by LAG (tradeable). 1 means 'gate off'."""
    spy = jm_load("SPY")
    riskoff, ret, rf, lams = walk_forward(spy, None)
    return riskoff.shift(LAG).fillna(0.0)        # default to risk-ON before first signal


def main():
    vix = vix_series()
    print("Computing SJM SPY regime + extracting RSI2 trades across the basket...")
    riskoff = spy_riskoff_regime()
    print(f"SPY regime OOS {riskoff.index.min().date()} -> {riskoff.index.max().date()}  "
          f"risk-off {riskoff.mean()*100:.1f}% of days")

    all_trades = []; price_rets = {}
    for f in sorted(glob.glob("bars_*_1d.parquet")):
        sym = os.path.basename(f)[len("bars_"):-len("_1d.parquet")]
        if sym in EXCLUDE or sym.endswith("_5m") or sym.endswith("_30m"): continue
        try:
            c = ml_load(f); all_trades += extract_trades(c, sym, vix); price_rets[sym] = c.pct_change()
        except Exception as e:
            print(f"  skip {sym}: {e}")
    td = pd.DataFrame(all_trades).dropna(subset=FEATURES).sort_values("entry_ts").reset_index(drop=True)

    # restrict to the regime OOS window so the comparison is apples-to-apples
    lo, hi = riskoff.index.min(), riskoff.index.max()
    td = td[(td["entry_ts"] >= lo) & (td["entry_ts"] <= hi)].reset_index(drop=True)
    # gate: drop any trade whose ENTRY day is in the SPY risk-off regime
    ro_on_entry = riskoff.reindex(td["entry_ts"]).fillna(0.0).values
    td["regime_off"] = ro_on_entry > 0.5
    print(f"{len(td)} basket trades in window; {td['regime_off'].mean()*100:.1f}% entered during risk-off "
          f"(these are the ones the gate removes)\n")

    ungated = basket_fixed_fraction(td, price_rets, n_slots=N_SLOTS)
    gated   = basket_fixed_fraction(td, price_rets, (~td["regime_off"]).values, n_slots=N_SLOTS)

    # also: regime-gate that FORCES cash in risk-off (even mid-trade) — stricter version
    # build by zeroing position returns on risk-off days at the portfolio level
    gated_hard = gated.copy()
    ro = riskoff.reindex(gated_hard.index).fillna(0.0)
    gated_hard = gated_hard.where(ro < 0.5, 0.025/252)   # risk-off days -> cash (RF)

    fx = jm_load("USDILSX")
    def block(name, r):
        rows = [{"k": name, **stats(r)},
                {"k": "  +after-tax @25%", **stats(after_tax(r, 0.25))},
                {"k": "  +after-tax @47%", **stats(after_tax(r, 0.47))}]
        if fx is not None:
            f = fx.reindex(r.index, method="ffill"); ils = (1+r)*(1+f.pct_change().fillna(0))-1
            rows.append({"k": "  ILS after-tax @25%", **stats(after_tax(ils, 0.25))})
        return rows

    print(f"=== SJM-gated vs ungated RSI2 basket ({N_SLOTS} slots, event-driven REAL Sharpe) ===")
    rows = []
    rows += block("UNGATED basket", ungated)
    rows += block("GATED (skip risk-off entries)", gated)
    rows += block("GATED-HARD (flat on risk-off days)", gated_hard)
    print(pd.DataFrame(rows).set_index("k").to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n=== Read ===")
    print("- If GATED beats UNGATED on after-tax Sharpe AND maxDD, the SPY regime adds value to your")
    print("  existing basket for free (it only REMOVES risk-off-entered trades -> lower turnover, tax-favorable).")
    print("- GATED-HARD also dodges mid-trade crashes but realizes more (forces cash); compare maxDD vs Sharpe.")
    print("- If GATED ~= UNGATED, the basket's own 200d-uptrend filter already captures most of the regime info.")


if __name__ == "__main__":
    main()
