# "CATCH IT EARLY" scanner: scan ALL ETFs monthly, flag EARLY risers (before they're leaders),
# two ways, and backtest both honestly with the storm-detector overlay. Survivorship-free
# (ETFs don't disappear the way single stocks do). After Israeli tax. Also prints what's flagging
# RIGHT NOW (the live scan you can act on).
#
# The two early signals (the honest version of "find the next pop"):
#  (1) BREAKOUT + VOLUME : price breaks to a new N-day high AND recent volume >> its average.
#      Classic "something is starting here" — you enter as the move begins, not after it's mature.
#  (2) ACCEL MOMENTUM    : short-term momentum is turning UP fast and now exceeds long-term momentum
#      from a low/negative base (the 2nd-derivative kick) — catches the early part of a trend.
# Compared head-to-head vs plain leader-momentum and buy-hold SPY, gated by the SJM detector.
from __future__ import annotations
import glob, os, math
import numpy as np, pandas as pd

from jump_model import load_close as jm_load, walk_forward, LAG
from momentum_engine import stats, after_tax, COST_BPS_RT, RF

ETF_UNIVERSE = ['AGG','BNO','DBC','DBMF','DVY','EEM','EFA','EMB','EWA','EWC','EWG','EWH','EWI','EWJ',
 'EWL','EWN','EWP','EWQ','EWS','EWT','EWU','EWW','EWY','EWZ','FXI','GDX','GDXJ','GLD','HYG','IBB',
 'IEF','IGV','INDA','ITB','IWM','IYR','IYT','IYZ','KBE','KIE','KMLM','KRE','LQD','MTUM','OIH','QLD',
 'QQQ','QUAL','SDY','SIL','SLV','SMH','SOXL','SOXX','SPY','SSO','TECL','TIP','TLT','USMV','USO','VGK',
 'VIG','VLUE','VNQ','VWO','VYM','XBI','XHB','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY','XME','XRT']


def load_ohlcv(sym):
    p = f"bars_{sym}_1d.parquet"
    if not os.path.exists(p): return None
    d = pd.read_parquet(p)[["ts","close","volume"]].copy()
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)
    return d.sort_values("ts").set_index("ts")


def panels(syms):
    closes, vols = {}, {}
    for s in syms:
        d = load_ohlcv(s)
        if d is not None and len(d) > 300:
            closes[s] = d["close"]; vols[s] = d["volume"]
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(vols).sort_index()


# ---------- the two early-riser scores (higher = stronger early-flag) ----------
def score_breakout(px, vol, hi_win=126, vol_win=63):
    """Breakout strength: how far price is above its trailing hi_win high (new-high pressure),
    times a volume surge factor (recent vol / its longer average). Positive only on genuine breakouts."""
    roll_hi = px.rolling(hi_win).max()
    breakout = px / roll_hi - 1.0                       # ~0 at a new high, negative below
    near_high = (breakout > -0.02).astype(float)        # within 2% of the high = "breaking out"
    vsurge = vol.rolling(10).mean() / vol.rolling(vol_win).mean()   # >1 = volume picking up
    # score: reward being at new highs WITH volume confirmation
    raw = near_high * (vsurge.clip(0.5, 3.0))
    # add a touch of recent momentum so ties break toward the stronger mover
    mom = px / px.shift(63) - 1.0
    return raw * (1 + mom.clip(-0.5, 1.0))


def score_accel(px):
    """Accelerating momentum: short-term (1m) minus medium-term (6m) momentum — positive when the
    recent move is faster than the trailing trend (the early kick), from a not-already-extended base."""
    m1 = px / px.shift(21) - 1.0
    m3 = px / px.shift(63) - 1.0
    m6 = px / px.shift(126) - 1.0
    accel = (m1 - m6 / 6.0) + (m3 - m6 / 2.0)           # recent pace exceeding the long trend
    not_extended = (m6 < 0.30).astype(float) * 0.5 + 0.5  # mild penalty if already up a lot in 6m
    return accel * not_extended


def score_leader(px):
    """Plain leader-momentum (the benchmark to beat): trailing 3/6/12m blend."""
    m3 = px.shift(21)/px.shift(21+63)-1; m6 = px.shift(21)/px.shift(21+126)-1; m12 = px.shift(21)/px.shift(21+252)-1
    return (m3 + m6 + m12) / 3.0


def run_scan(px, score, top_n=5, lev=1.0, riskoff=None, min_score=0.0):
    """Monthly: hold top_n by `score` (must exceed min_score), equal-weight, optional leverage + gate."""
    rets = px.pct_change(fill_method=None)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cur, last = None, None
    for t in px.index:
        per = t.to_period("M")
        if per != last:
            sc = score.loc[t].dropna()
            sc = sc[sc > min_score]
            cur = pd.Series(1.0/min(top_n, len(sc)), index=sc.nlargest(top_n).index) if len(sc) else None
            last = per
        if cur is not None:
            weights.loc[t, cur.index] = cur.values
    port = (weights.shift(1) * rets).sum(axis=1)
    invested = weights.shift(1).sum(axis=1)
    cash = (1 - invested) * (RF/252)
    gross = port + cash if lev == 1.0 else lev*port + cash - (lev-1)*invested*(0.04/252)
    if riskoff is not None:
        ro = riskoff.reindex(gross.index).fillna(0.0).shift(LAG).fillna(0.0)
        gross = gross.where(ro < 0.5, RF/252)
    turn = weights.diff().abs().sum(axis=1).fillna(0)
    net = gross - turn*(COST_BPS_RT/1e4)/2
    return net, weights


def main():
    print("Loading 84-ETF universe + computing SJM storm-regime...")
    px, vol = panels(ETF_UNIVERSE)
    px = px[px.index >= pd.Timestamp("2007-01-01")]; vol = vol.reindex(px.index)
    spy = jm_load("SPY"); spy_ro, _, _, _ = walk_forward(spy, None)

    s_break = score_breakout(px, vol)
    s_accel = score_accel(px)
    s_lead  = score_leader(px)

    spy_px = jm_load("SPY"); spy_px = spy_px[spy_px.index >= px.index.min()]
    bh = spy_px.pct_change(fill_method=None)

    print(f"\n=== EARLY-RISER scan on {px.shape[1]} ETFs, 2007-2026, after Israeli tax @25% ===")
    print("Goal: catch risers EARLY (not just hold the established leaders).")
    print(f"Benchmark buy-hold SPY: {', '.join(f'{k} {v:.1f}' for k,v in stats(after_tax(bh,0.25)).items())}\n")

    sigs = {"BREAKOUT+vol": s_break, "ACCEL-momentum": s_accel, "leader-momentum (benchmark)": s_lead}
    rows = []
    series_store = {}
    for name, sc in sigs.items():
        for top_n in (5, 8):
            base, _ = run_scan(px, sc, top_n=top_n, riskoff=None)
            gated, w = run_scan(px, sc, top_n=top_n, riskoff=spy_ro)
            sb, sg = stats(after_tax(base,0.25)), stats(after_tax(gated,0.25))
            rows.append({"signal": name, "topN": top_n,
                         "BASE_CAGR": sb["CAGR%"], "BASE_Shrp": sb["Sharpe"], "BASE_DD": sb["maxDD%"],
                         "GATED_CAGR": sg["CAGR%"], "GATED_Shrp": sg["Sharpe"], "GATED_DD": sg["maxDD%"]})
            series_store[(name, top_n)] = (gated, w)
    df = pd.DataFrame(rows).sort_values("GATED_CAGR", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # best gated config: year-by-year
    best = df.iloc[0]
    bg, bw = series_store[(best["signal"], best["topN"])]
    r = after_tax(bg, 0.25); yr = r.groupby(r.index.year).apply(lambda s: ((1+s).prod()-1)*100)
    print(f"\n=== BEST: {best['signal']} top{int(best['topN'])} (gated) — year by year (after-tax@25%) ===")
    print("  " + "  ".join(f"{y}:{v:+.0f}%" for y,v in yr.items()))

    # ---- THE LIVE SCAN: what is flagging RIGHT NOW? ----
    print("\n=== 🔎 FLAGGING NOW (latest data) — what each scanner would buy this month ===")
    last_t = px.index[-1]
    for name, sc in sigs.items():
        latest = sc.loc[last_t].dropna()
        latest = latest[latest > 0].nlargest(8)
        picks = ", ".join(f"{s}({v:.2f})" for s, v in latest.items()) if len(latest) else "(nothing qualifies — sit in cash)"
        print(f"  {name:28s}: {picks}")
    ro_now = spy_ro.iloc[-1] if len(spy_ro) else 0
    print(f"\n  Storm-detector says: {'🛟 RISK-OFF (gate to cash, ignore the picks)' if ro_now>0.5 else '📈 calm — the picks above are live'}  (as of {last_t.date()})")

    print("\n=== Read (plain words) ===")
    print("- BREAKOUT+vol and ACCEL try to catch risers EARLY; leader-momentum just holds what's already up.")
    print("- If an early signal beats leader-momentum on gated CAGR/Sharpe, catching-early genuinely adds value.")
    print("- 'FLAGGING NOW' is your actual monthly shopping list — but only act on it if the detector says calm.")


if __name__ == "__main__":
    main()
