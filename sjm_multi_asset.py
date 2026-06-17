# Test the SJM "storm-detector" across MANY instruments + asset classes, two ways:
#   (1) OWN-signal   : each asset gated by ITS OWN storm regime.
#   (2) MARKET-signal: every asset gated by SPY's market-wide storm regime (one weather map for all).
# Then build a diversified EQUAL-WEIGHT gated portfolio and compare to its buy-hold.
# Everything after Israeli CGT @25%. Goal: learn WHERE the detector helps most.
from __future__ import annotations
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

from jump_model import load_close, walk_forward, gate_returns, LAG

# Curated set with deep history + clear category labels.
UNIVERSE = {
    "stock-tech":   ["AAPL", "MSFT", "NVDA", "INTC", "CSCO", "ORCL", "IBM"],
    "stock-other":  ["JPM", "JNJ", "PG", "KO", "XOM", "DIS", "GE", "WMT", "MCD", "CAT"],
    "index-ETF":    ["SPY", "QQQ", "IWM"],
    "sector-ETF":   ["XLK", "XLF", "XLE", "XLV", "XLP", "XLU"],
    "bond-ETF":     ["TLT", "IEF", "LQD", "HYG", "AGG"],
    "commodity":    ["GLD", "SLV", "USO", "DBC"],
}


def metrics(r, ppy=252):
    r = r.dropna(); eq = (1 + r).cumprod()
    if len(r) < 60: return None
    return {"CAGR%": (eq.iloc[-1] ** (ppy / len(r)) - 1) * 100,
            "Sharpe": (r.mean() * ppy) / (r.std() * math.sqrt(ppy)) if r.std() > 0 else 0,
            "maxDD%": (eq / eq.cummax() - 1).min() * 100}


def after_tax(r, rate):
    r = r.dropna(); eq, carry, out, idx = 1.0, 0.0, [], []
    for y in sorted(set(r.index.year)):
        yr = r[r.index.year == y]; start = eq; eqs, m = [], eq
        for v in yr: m *= (1 + v); eqs.append(m)
        g = eqs[-1] - start
        if g >= 0: u = min(g, carry); carry -= u; tx = g - u
        else: carry += -g; tx = 0.0
        eqs[-1] -= rate * max(0, tx); prev = start
        for ev in eqs: out.append(ev / prev - 1); prev = ev
        idx.extend(list(yr.index)); eq = eqs[-1]
    return pd.Series(out, index=pd.DatetimeIndex(idx))


PROGRESS_FILE = "sjm_multi_progress.txt"
HIST_START = "2000-01-01"   # cap each asset's history here: keeps ~26yrs (plenty for the long-window
                            # test, and the warm-up + 13yr walk-forward) while making 64-yr stocks run
                            # at SPY-like speed. Pre-2000 single-stock data adds slow compute, not insight.


def _log_progress(sym, ok):
    """Append one line per finished instrument so you can watch live progress from another shell."""
    try:
        with open(PROGRESS_FILE, "a") as f:
            f.write(f"done\t{sym}\t{'ok' if ok else 'skip'}\n")
    except Exception:
        pass


def _own_regime_worker(sym):
    """Runs in a separate process: the EXPENSIVE per-asset walk-forward. Returns picklable dict
    of plain arrays (not Series, to keep pickling light). Computes the asset's OWN-signal gate.
    Logs a progress line the moment it finishes."""
    close = load_close(sym)
    if close is None:
        _log_progress(sym, False); return (sym, None)
    close = close[close.index >= pd.Timestamp(HIST_START)]    # cap history for tractable, comparable runtime
    riskoff, ret, rf, _ = walk_forward(close, None)
    if len(riskoff) < 250:
        _log_progress(sym, False); return (sym, None)
    own_in = (1 - riskoff).shift(LAG).fillna(1.0)
    g_own = gate_returns(own_in, ret.loc[own_in.index], rf.loc[own_in.index])
    bh = ret.loc[own_in.index]
    _log_progress(sym, True)
    return (sym, {"index": g_own.index, "bh": bh.values, "own": g_own.values,
                  "ret": ret.loc[own_in.index].values, "rf": rf.loc[own_in.index].values})


def main():
    import os
    all_syms = [(cat, s) for cat, syms in UNIVERSE.items() for s in syms]
    # fresh progress file ("done <sym>" per finished instrument). Check live with:
    #   wc -l sjm_multi_progress.txt   (e.g. "12" means 12 of N done)
    open(PROGRESS_FILE, "w").write(f"total\t{len(all_syms) + 1}\tinstruments (incl SPY market regime)\n")

    # SPY market regime, computed once (used to gate every asset with the market-wide weather)
    spy = load_close("SPY")
    spy_riskoff, _, spy_rf, _ = walk_forward(spy, None)
    spy_in_full = (1 - spy_riskoff)
    _log_progress("SPY-market-regime", True)

    # Gentle by default: use SJM_WORKERS env var, else just 3 cores (keeps the laptop cool).
    n_workers = int(os.environ.get("SJM_WORKERS", "3"))
    print(f"Testing the storm-detector on {len(all_syms)} instruments across {n_workers} CPU cores "
          f"(own-signal AND market-signal)...")
    print("All numbers AFTER Israeli tax @25%. 'beats?' = gated Sharpe > buy-hold Sharpe.\n")

    # PARALLEL: run the expensive per-asset walk-forwards across cores.
    # CACHE: dump raw worker output so the cheap portfolio analysis can rerun without recomputing.
    import pickle
    CACHE = "sjm_multi_cache.pkl"
    cat_of = {s: cat for cat, s in all_syms}
    if os.environ.get("SJM_ANALYZE_ONLY") and os.path.exists(CACHE):
        print(f"(analyze-only: reloading {CACHE}, skipping the heavy model fits)")
        worker_out = pickle.load(open(CACHE, "rb"))
    else:
        worker_out = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for sym, data in ex.map(_own_regime_worker, [s for _, s in all_syms]):
                worker_out[sym] = data
        pickle.dump(worker_out, open(CACHE, "wb"))

    per_asset = {}; rows = []
    for cat, sym in all_syms:
        data = worker_out.get(sym)
        if data is None:
            print(f"  skip {sym} (no/short data)"); continue
        idx = data["index"]
        bh = pd.Series(data["bh"], index=idx)
        g_own = pd.Series(data["own"], index=idx)
        # market gate applied here (cheap): gate this asset by SPY's regime
        mkt_in = (spy_in_full.reindex(idx).fillna(1.0)).shift(LAG).fillna(1.0)
        g_mkt = gate_returns(mkt_in, pd.Series(data["ret"], index=idx), pd.Series(data["rf"], index=idx))
        per_asset[sym] = {"cat": cat, "bh": bh, "own": g_own, "mkt": g_mkt}
        mb, mo = metrics(after_tax(bh, 0.25)), metrics(after_tax(g_own, 0.25))
        mm = metrics(after_tax(g_mkt, 0.25)) if g_mkt is not None else None
        rows.append({
            "cat": cat, "sym": sym,
            "BH_Shrp": mb["Sharpe"], "BH_DD": mb["maxDD%"],
            "OWN_Shrp": mo["Sharpe"], "OWN_DD": mo["maxDD%"], "own_beats": "Y" if mo["Sharpe"] > mb["Sharpe"] else "-",
            "MKT_Shrp": (mm["Sharpe"] if mm else np.nan), "MKT_DD": (mm["maxDD%"] if mm else np.nan),
            "mkt_beats": ("Y" if (mm and mm["Sharpe"] > mb["Sharpe"]) else "-"),
        })

    df = pd.DataFrame(rows)
    print("=== Per-instrument: Buy&Hold vs OWN-signal gate vs MARKET(SPY)-signal gate (after-tax@25%) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # category summary
    print("\n=== How often does the detector BEAT buy-hold, by category? ===")
    for cat in UNIVERSE:
        sub = df[df["cat"] == cat]
        if len(sub) == 0: continue
        print(f"  {cat:13s}: own-signal beats {sub['own_beats'].eq('Y').sum()}/{len(sub)},  "
              f"market-signal beats {sub['mkt_beats'].eq('Y').sum()}/{len(sub)}  "
              f"(avg OWN Sharpe {sub['OWN_Shrp'].mean():.2f} vs BH {sub['BH_Shrp'].mean():.2f})")

    # ---- diversified equal-weight portfolios on a common window ----
    print("\n=== Diversified EQUAL-WEIGHT portfolio (all instruments), common window ===")
    # align everything to the intersection of dates
    bh_df = pd.DataFrame({s: d["bh"] for s, d in per_asset.items()})
    own_df = pd.DataFrame({s: d["own"] for s, d in per_asset.items()})
    mkt_df = pd.DataFrame({s: d["mkt"] for s, d in per_asset.items() if d["mkt"] is not None})
    common = bh_df.dropna().index
    if len(common) > 250:
        port = {
            "Buy&Hold equal-weight":      bh_df.loc[common].mean(axis=1),
            "OWN-signal gated EW":        own_df.loc[common].mean(axis=1),
            "MARKET-signal gated EW":     mkt_df.loc[common].mean(axis=1),
        }
        prows = []
        for name, r in port.items():
            prows.append({"portfolio": name, **metrics(after_tax(r, 0.25))})
        print(f"(common window {common.min().date()} -> {common.max().date()}, {len(per_asset)} instruments)")
        print(pd.DataFrame(prows).set_index("portfolio").to_string(float_format=lambda x: f"{x:.2f}"))
    else:
        print("  (not enough overlapping history for a clean equal-weight portfolio on the full set)")

    # ---- LONG-WINDOW pressure test: use only assets whose gated history reaches back furthest ----
    print("\n=== LONG-WINDOW pressure test (assets with the longest gated history) ===")
    # how early does each gated series start? (cap at HIST_START=2000 + ~13yr warm-up => ~2013-14)
    starts = {s: d["own"].index.min() for s, d in per_asset.items()}
    # keep assets whose gated OOS starts by 2015 -> ~11+ yrs incl. BOTH calm years AND 2020/2022 crashes
    deep = [s for s, st in starts.items() if st <= pd.Timestamp("2015-01-01")]
    print(f"long-history set ({len(deep)} assets, gated OOS starting <=2015): {', '.join(sorted(deep))}")
    if len(deep) >= 5:
        bhd = pd.DataFrame({s: per_asset[s]["bh"] for s in deep})
        ownd = pd.DataFrame({s: per_asset[s]["own"] for s in deep})
        mktd = pd.DataFrame({s: per_asset[s]["mkt"] for s in deep})
        cl = bhd.dropna().index
        if len(cl) > 250:
            portL = {"Buy&Hold EW (long)": bhd.loc[cl].mean(axis=1),
                     "OWN-signal gated EW (long)": ownd.loc[cl].mean(axis=1),
                     "MARKET-signal gated EW (long)": mktd.loc[cl].mean(axis=1)}
            prows = [{"portfolio": n, **metrics(after_tax(r, 0.25))} for n, r in portL.items()]
            print(f"(LONG common window {cl.min().date()} -> {cl.max().date()}, {len(deep)} deep assets)")
            print(pd.DataFrame(prows).set_index("portfolio").to_string(float_format=lambda x: f"{x:.2f}"))

            # per-year: where does the gate's edge come from? (crash years vs calm years)
            print("\n=== Per-year return: is the gate's edge just crash-year protection? (after-tax@25%) ===")
            bh_y = after_tax(portL["Buy&Hold EW (long)"], 0.25)
            ow_y = after_tax(portL["OWN-signal gated EW (long)"], 0.25)
            yr = pd.DataFrame({"BuyHold%": bh_y.groupby(bh_y.index.year).apply(lambda s: ((1+s).prod()-1)*100),
                               "Gated%":   ow_y.groupby(ow_y.index.year).apply(lambda s: ((1+s).prod()-1)*100)})
            yr["edge"] = yr["Gated%"] - yr["BuyHold%"]
            print(yr.to_string(float_format=lambda x: f"{x:.1f}"))
            print(f"\nGate beat buy-hold in {(yr['edge']>0).sum()}/{len(yr)} years.  "
                  f"Avg edge in DOWN years for buy-hold: {yr.loc[yr['BuyHold%']<0,'edge'].mean():.1f}%  "
                  f"vs UP years: {yr.loc[yr['BuyHold%']>=0,'edge'].mean():.1f}%")
            print("(If the edge is ALL in down years and negative in up years, it's pure crash-insurance,")
            print(" not free alpha — still valuable, but know what you're buying.)")
    else:
        print("  (not enough deep-history assets)")

    print("\n=== Read (plain words) ===")
    print("- 'OWN-signal' = each thing watches its own storms. 'MARKET-signal' = everything follows SPY's weather.")
    print("- A category where the detector usually BEATS buy-hold = a good place to use it.")
    print("- The LONG-WINDOW portfolio + per-year table tell us if the 1.38 Sharpe was just the 2020-2022 crashes.")


if __name__ == "__main__":
    main()
