# OFFENSE: momentum-rotation growth engine + the SJM storm-detector as a safety net.
# Goal: MAX growth in good years (ride the strongest movers), with the detector cutting to cash
# before crashes. Tests universes (ETF-only / stocks+sectors / +leveraged), trading paces
# (monthly/quarterly), leverage multipliers, and WITH vs WITHOUT the storm-detector overlay.
# Everything after Israeli CGT (25%/47%) + per-trade cost. Honest year-by-year + maxDD shown.
#
# Momentum = the real, repeatable edge: each rebalance, rank the universe by a blend of trailing
# 3/6/12-month returns, hold the top-N equal-weight. Winners tend to keep winning for a while.
from __future__ import annotations
import glob, os, math
import numpy as np, pandas as pd

from jump_model import load_close as jm_load, walk_forward, LAG

COST_BPS_RT = 3.0; RF = 0.025; BORROW = 0.04   # annual borrow cost for simulated leverage

# Universes to test
ETF_ONLY = ["SPY","QQQ","IWM","EFA","EEM","XLK","XLF","XLE","XLV","XLP","XLU","XLY","XLI","XLB",
            "TLT","IEF","LQD","HYG","GLD","SLV","DBC","VNQ","SMH","SOXX","IGV","XBI","KRE"]
STOCKS_SECTORS = ETF_ONLY + ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","AMD","INTC","CSCO",
                             "META","NFLX","AMZN","TSLA","JPM","JNJ","LLY","UNH","XOM","CAT","GE"]
LEVERAGED = ETF_ONLY + ["TQQQ","QLD","SSO","SOXL","TECL"]


def load_close(sym):
    p = f"bars_{sym}_1d.parquet"
    if not os.path.exists(p): return None
    d = pd.read_parquet(p)[["ts","close"]].copy()
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)
    return d.sort_values("ts").set_index("ts")["close"]


def price_panel(syms):
    s = {}
    for sym in syms:
        c = load_close(sym)
        if c is not None: s[sym] = c
    return pd.DataFrame(s).sort_index()


def momentum_score(px):
    """Blend of trailing 3/6/12-month total returns (skip the most recent 1m to avoid short-term
    reversal), the classic momentum construction. Higher = stronger recent winner."""
    r1m = px.shift(21)
    m3  = px.shift(21) / px.shift(21+63) - 1
    m6  = px.shift(21) / px.shift(21+126) - 1
    m12 = px.shift(21) / px.shift(21+252) - 1
    return (m3 + m6 + m12) / 3.0


def stats(r, ppy=252):
    r = r.dropna();
    if len(r) < 60: return {"CAGR%": np.nan, "Sharpe": np.nan, "maxDD%": np.nan, "vol%": np.nan}
    eq = (1 + r).cumprod()
    return {"CAGR%": (eq.iloc[-1] ** (ppy/len(r)) - 1)*100, "vol%": r.std()*math.sqrt(ppy)*100,
            "Sharpe": (r.mean()*ppy)/(r.std()*math.sqrt(ppy)) if r.std()>0 else 0,
            "maxDD%": (eq/eq.cummax()-1).min()*100}


def after_tax(r, rate):
    r = r.dropna(); eq, carry, out, idx = 1.0, 0.0, [], []
    for y in sorted(set(r.index.year)):
        yr = r[r.index.year==y]; start=eq; eqs,m=[],eq
        for v in yr: m*=(1+v); eqs.append(m)
        g=eqs[-1]-start
        if g>=0: u=min(g,carry); carry-=u; tx=g-u
        else: carry+=-g; tx=0.0
        eqs[-1]-=rate*max(0,tx); prev=start
        for ev in eqs: out.append(ev/prev-1); prev=ev
        idx.extend(list(yr.index)); eq=eqs[-1]
    return pd.Series(out, index=pd.DatetimeIndex(idx))


def run_momentum(px, top_n=5, rebal="M", lev=1.0, riskoff=None):
    """Hold top-N momentum names, rebalance monthly('M')/quarterly('Q'). Optional leverage `lev`
    (with borrow cost on the >1 part) and optional SJM `riskoff` overlay (force cash when risk-off).
    Returns net daily return series."""
    rets = px.pct_change(fill_method=None)
    score = momentum_score(px)
    freq = rebal[0]                  # 'M' or 'Q' for to_period (rebal may be 'ME'/'QE')
    # build daily target weights (equal-weight top-N, chosen at each rebalance from data known then)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur = None
    last_rb_period = None
    for t in px.index:
        per = t.to_period(freq)
        if per != last_rb_period:
            sc = score.loc[t].dropna()
            # only names with enough history and positive momentum (don't buy losers)
            sc = sc[sc > 0]
            if len(sc) > 0:
                pick = sc.nlargest(top_n).index
                cur = pd.Series(1.0/len(pick), index=pick)
            else:
                cur = None      # nothing qualifies -> cash
            last_rb_period = per
        if cur is not None:
            weights.loc[t, cur.index] = cur.values
    # portfolio daily return
    port = (weights.shift(1) * rets).sum(axis=1)        # trade at next close (no look-ahead)
    invested = weights.shift(1).sum(axis=1)             # 0..1 (1 = fully in top-N)
    cash = (1 - invested) * (RF/252)
    gross = port + cash
    # leverage: scale the invested return, charge borrow on the levered-up part
    if lev != 1.0:
        gross = lev * port + cash - (lev - 1.0) * invested * (BORROW/252)
    # storm-detector overlay: on risk-off days, force flat (cash/RF)
    if riskoff is not None:
        ro = riskoff.reindex(gross.index).fillna(0.0).shift(LAG).fillna(0.0)
        gross = gross.where(ro < 0.5, RF/252)
    # turnover cost (on rebalance weight changes)
    turn = weights.diff().abs().sum(axis=1).fillna(0)
    net = gross - turn * (COST_BPS_RT/1e4)/2
    return net, invested


def main():
    print("Computing SJM SPY storm-regime (the safety net)...")
    spy = jm_load("SPY")
    spy_riskoff, _, _, _ = walk_forward(spy, None)

    universes = {"ETF-only": ETF_ONLY, "stocks+sectors": STOCKS_SECTORS, "+leveraged ETFs": LEVERAGED}
    configs = []
    for uname, syms in universes.items():
        px = price_panel(syms)
        px = px[px.index >= pd.Timestamp("2006-01-01")]   # common-ish window incl. leveraged inception
        for rebal, rname in [("ME","monthly"), ("QE","quarterly")]:
            for top_n in (3, 5):
                for lev in (1.0, 2.0):
                    base, inv = run_momentum(px, top_n=top_n, rebal=rebal, lev=lev, riskoff=None)
                    gated, _ = run_momentum(px, top_n=top_n, rebal=rebal, lev=lev, riskoff=spy_riskoff)
                    configs.append({
                        "universe": uname, "pace": rname, "topN": top_n, "lev": lev,
                        "base": base, "gated": gated})

    # buy-hold SPY benchmark on the same window
    spy_px = jm_load("SPY"); spy_px = spy_px[spy_px.index >= pd.Timestamp("2006-01-01")]
    bh = spy_px.pct_change(fill_method=None)

    print("\n=== GROWTH-engine sweep: momentum rotation, after Israeli tax @25% ===")
    print("'base' = momentum only (offense). 'gated' = momentum + storm-detector (offense+defense).")
    print("Benchmark buy-hold SPY @25%:", {k: round(v,2) for k,v in stats(after_tax(bh,0.25)).items()}, "\n")
    rows = []
    for c in configs:
        sb = stats(after_tax(c["base"], 0.25)); sg = stats(after_tax(c["gated"], 0.25))
        rows.append({"universe": c["universe"], "pace": c["pace"], "topN": c["topN"], "lev": c["lev"],
                     "BASE_CAGR": sb["CAGR%"], "BASE_Shrp": sb["Sharpe"], "BASE_DD": sb["maxDD%"],
                     "GATED_CAGR": sg["CAGR%"], "GATED_Shrp": sg["Sharpe"], "GATED_DD": sg["maxDD%"]})
    df = pd.DataFrame(rows)
    # rank by GATED CAGR (the "most profit with safety net" objective)
    df = df.sort_values("GATED_CAGR", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n=== TOP 3 by after-tax CAGR (most profit), with their year-by-year ===")
    top3 = df.head(3)
    for _, row in top3.iterrows():
        match = next(c for c in configs if c["universe"]==row["universe"] and c["pace"]==row["pace"]
                     and c["topN"]==row["topN"] and c["lev"]==row["lev"])
        for which in ["base", "gated"]:
            r = after_tax(match[which], 0.25)
            yr = r.groupby(r.index.year).apply(lambda s: ((1+s).prod()-1)*100)
            tag = f"{row['universe']} {row['pace']} top{int(row['topN'])} lev{row['lev']:.0f}x [{which}]"
            print(f"\n{tag}: {stats(r)['CAGR%']:.1f}%/yr  Sharpe {stats(r)['Sharpe']:.2f}  maxDD {stats(r)['maxDD%']:.0f}%")
            print("  " + "  ".join(f"{y}:{v:+.0f}%" for y, v in yr.items()))

    print("\n=== Read (plain words) ===")
    print("- BASE = pure offense (more profit, bigger crashes). GATED = offense + storm-detector safety net.")
    print("- Higher GATED_CAGR with a tolerable GATED_DD = the sweet spot you asked for.")
    print("- Leverage (lev 2x) boosts good years but watch the maxDD; the gate is what keeps 2x survivable.")


if __name__ == "__main__":
    main()
