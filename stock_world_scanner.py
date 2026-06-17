# Stock + WORLD-ETF scanner: combine 100 S&P/US large-caps with 22 country/region ETFs in ONE
# universe, run the same 3 signals (breakout+vol / accel / leader-momentum), gate with the SJM
# storm-detector, after Israeli tax. Compare to the ETF-only scanner to see if adding individual
# stocks + world exposure genuinely helps.
#
# *** SURVIVORSHIP-BIAS WARNING (flagged per user request) ***
# The 100 S&P names are TODAY'S survivors, chosen with hindsight. Momentum-into-known-survivors
# is partly double-counting hindsight. The WORLD ETFs are survivorship-free (countries don't
# delist), so the honest read is: trust the WORLD-ETF contribution, DISCOUNT the US-stock CAGR.
from __future__ import annotations
import os, math
import numpy as np, pandas as pd

from jump_model import load_close as jm_load, walk_forward, LAG
from momentum_engine import stats, after_tax, COST_BPS_RT, RF
from early_scanner import score_breakout, score_accel, score_leader, run_scan, load_ohlcv, panels

SP_STOCKS = """AAPL MSFT NVDA AVGO ORCL CRM ADBE AMD INTC CSCO QCOM TXN IBM NOW INTU
META NFLX DIS CMCSA TMUS T VZ AMZN TSLA HD NKE SBUX MCD LOW BKNG GM TGT
COST WMT PG KO PEP MO PM MDLZ GIS CL KMB UNH JNJ LLY ABBV MRK PFE TMO ABT DHR BMY AMGN GILD CVS
JPM BAC WFC GS MS C BLK SCHW SPGI AXP COF USB PNC BA HON UNP GE RTX LMT MMM CAT DE UPS FDX
XOM CVX COP SLB EOG OXY LIN APD SHW NEM FCX NEE DUK SO D AEP AMT PLD EQIX O""".split()
WORLD_ETFS = ["EWJ","EWY","EWZ","FXI","INDA","EFA","EEM","VGK","VWO","EWG","EWU","EWH","EWA",
              "EWC","EWT","EWS","EWW","EWP","EWI","EWL","EWN","EWQ"]
COMBINED = SP_STOCKS + WORLD_ETFS


def main():
    print(f"Loading combined universe: {len(SP_STOCKS)} S&P stocks + {len(WORLD_ETFS)} world ETFs "
          f"= {len(COMBINED)} instruments...")
    px, vol = panels(COMBINED)
    px = px[px.index >= pd.Timestamp("2007-01-01")]; vol = vol.reindex(px.index)
    print(f"panel: {px.shape[1]} with usable history, {px.index.min().date()} -> {px.index.max().date()}")
    spy = jm_load("SPY"); spy_ro, _, _, _ = walk_forward(spy, None)

    sigs = {"BREAKOUT+vol": score_breakout(px, vol),
            "ACCEL-momentum": score_accel(px),
            "leader-momentum": score_leader(px)}

    spy_px = jm_load("SPY"); spy_px = spy_px[spy_px.index >= px.index.min()]
    bh = after_tax(spy_px.pct_change(fill_method=None), 0.25)
    print(f"\n=== STOCK + WORLD-ETF scan, 2007-2026, after Israeli tax @25% ===")
    print(f"Benchmark buy-hold SPY: CAGR {stats(bh)['CAGR%']:.1f}%  Sharpe {stats(bh)['Sharpe']:.2f}  maxDD {stats(bh)['maxDD%']:.0f}%\n")

    rows = []; store = {}
    for name, sc in sigs.items():
        for top_n in (5, 8, 12):
            base, _ = run_scan(px, sc, top_n=top_n, riskoff=None)
            gated, w = run_scan(px, sc, top_n=top_n, riskoff=spy_ro)
            sb, sg = stats(after_tax(base,0.25)), stats(after_tax(gated,0.25))
            rows.append({"signal": name, "topN": top_n,
                         "BASE_CAGR": sb["CAGR%"], "BASE_DD": sb["maxDD%"],
                         "GATED_CAGR": sg["CAGR%"], "GATED_Shrp": sg["Sharpe"], "GATED_DD": sg["maxDD%"]})
            store[(name, top_n)] = (gated, w)
    df = pd.DataFrame(rows).sort_values("GATED_CAGR", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # how much of the winner's picks were STOCKS vs WORLD-ETFs? (to judge survivorship exposure)
    best = df.iloc[0]; bg, bw = store[(best["signal"], best["topN"])]
    sp_set = set(SP_STOCKS)
    # average weight in stocks vs world-etfs over time
    stock_cols = [c for c in bw.columns if c in sp_set]
    etf_cols = [c for c in bw.columns if c in set(WORLD_ETFS)]
    w_inv = bw.sum(axis=1).replace(0, np.nan)
    stock_share = (bw[stock_cols].sum(axis=1) / w_inv).mean() * 100
    etf_share = (bw[etf_cols].sum(axis=1) / w_inv).mean() * 100
    r = after_tax(bg, 0.25); yr = r.groupby(r.index.year).apply(lambda s: ((1+s).prod()-1)*100)
    print(f"\n=== BEST: {best['signal']} top{int(best['topN'])} (gated) ===")
    print(f"  after-tax {stats(r)['CAGR%']:.1f}%/yr, Sharpe {stats(r)['Sharpe']:.2f}, worst crash {stats(r)['maxDD%']:.0f}%")
    print(f"  picks were ~{stock_share:.0f}% individual US stocks, ~{etf_share:.0f}% world ETFs (rest cash)")
    print("  year by year: " + "  ".join(f"{y}:{v:+.0f}%" for y,v in yr.items()))

    # ---- LIVE SCAN ----
    print("\n=== 🔎 FLAGGING NOW (combined universe) — what each scanner would buy this month ===")
    last_t = px.index[-1]
    for name, sc in sigs.items():
        latest = sc.loc[last_t].dropna(); latest = latest[latest > 0].nlargest(10)
        tag = lambda s: f"{s}*" if s in sp_set else s     # * = individual stock (survivorship-risky)
        picks = ", ".join(f"{tag(s)}({v:.2f})" for s,v in latest.items()) if len(latest) else "(cash)"
        print(f"  {name:16s}: {picks}")
    print("  (* = individual US stock; un-starred = world/country ETF)")
    ro_now = spy_ro.iloc[-1] if len(spy_ro) else 0
    print(f"  Storm-detector: {'🛟 RISK-OFF' if ro_now>0.5 else '📈 calm — picks are live'}  ({last_t.date()})")

    print("\n=== ⚠️ SURVIVORSHIP-BIAS FLAG (read this) ===")
    print(f"- The winner leaned ~{stock_share:.0f}% on individual US stocks chosen with HINDSIGHT (today's survivors).")
    print("  That inflates the CAGR. The WORLD-ETF portion is survivorship-free and trustworthy.")
    print("- HONEST READ: compare this to the ETF-only scanner (~15%/yr). If stocks only add a few % of")
    print("  CAGR here, that gap is mostly hindsight, not a real edge you could have captured live.")


if __name__ == "__main__":
    main()
