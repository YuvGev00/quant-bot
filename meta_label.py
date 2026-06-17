# Experiment 3 — Meta-labeling on the RSI2 basket (Lopez de Prado, AFML Ch.3).
# The PRIMARY model (RSI(2)<10 + 200d-uptrend) decides DIRECTION. A SECONDARY ML model
# learns P(this specific trade ends net-profitable) from signal-time features, and is used
# ONLY to VETO weak trades. It never invents trades -> it can only REDUCE turnover, which is
# tax-favorable for an Israeli taxpayer (fewer realizations -> more deferral).
#
# Discipline (the whole point — meta-labeling overfits trivially if you cheat):
#  * Pool trades across the WHOLE ~180-instrument basket so there are enough labels.
#  * Features are known AT ENTRY only (no look-ahead).
#  * PURGED + EMBARGOED walk-forward: train on trades that CLOSED before the test block starts,
#    minus an embargo, so overlapping-horizon leakage can't inflate OOS.
#  * Judge on AFTER-TAX CAGR *and* Sharpe — a precision filter that lifts Sharpe but cuts the
#    right-tail winners can LOWER after-tax CAGR. We check the kept-vs-skipped return dists.
#  * CONTROLS: (a) leaky random-CV to measure how much leakage would inflate it; (b) plain
#    LogisticRegression — if it matches the gradient-boost, the fancy model adds nothing.
from __future__ import annotations
import glob, math, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = "."; COST_BPS_RT = 3.0; RF = 0.025; MAXHOLD = 10
EXCLUDE = {"USDILSX", "VIX", "IRX", "BIL", "DBMF", "KMLM"}     # not tradeable equities
EMBARGO_DAYS = 10                                              # >= MAXHOLD so test trades can't leak into train


def rsi(s, n):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100 / (1 + up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean())


def load_close(path):
    d = pd.read_parquet(path)[["ts", "close"]].copy()
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)
    return d.sort_values("ts").set_index("ts")["close"]


def vix_series():
    p = f"{DATA}/bars_VIX_1d.parquet"
    return load_close(p) if os.path.exists(p) else None


def extract_trades(close, sym, vix=None):
    """Replay the RSI2 strategy on one instrument. For each TRADE, record entry-time features and
    the trade's net return (after cost). Returns a list of dicts (one per trade)."""
    c = close
    r2 = rsi(c, 2); sma5 = c.rolling(5).mean(); sma10 = c.rolling(10).mean(); sma200 = c.rolling(200).mean()
    ret = c.pct_change()
    vol20 = ret.rolling(20).std() * math.sqrt(252)
    dd20 = c / c.rolling(20).max() - 1.0          # drawdown from 20d high
    dist200 = c / sma200 - 1.0                    # % above 200d MA
    dist5 = c / sma5 - 1.0
    mom20 = c / c.shift(20) - 1.0                 # 20d momentum
    vixs = vix.reindex(c.index, method="ffill") if vix is not None else None
    vix_chg = vixs.pct_change(5) if vixs is not None else None

    entry = ((r2 < 10) & (c > sma200)).values
    exit_ = (c > sma5).values
    n = len(c); idx = c.index
    trades = []; holding = False; et = None; epx = None
    cv = c.values
    for t in range(1, n):
        if not holding and entry[t] and not np.isnan(sma200.values[t]):
            # ENTER at close t (features known now)
            holding = True; et = t; epx = cv[t]
            feat = {
                "sym": sym, "entry_i": t, "entry_ts": idx[t],
                "rsi2": r2.values[t], "dist5": dist5.values[t], "dist200": dist200.values[t],
                "vol20": vol20.values[t], "dd20": dd20.values[t], "mom20": mom20.values[t],
                "dow": idx[t].dayofweek,
                "vix": (vixs.values[t] if vixs is not None else np.nan),
                "vix_chg5": (vix_chg.values[t] if vix_chg is not None else np.nan),
            }
        elif holding and (exit_[t] or (t - et) >= MAXHOLD):
            # EXIT at close t
            gross = cv[t] / epx - 1.0
            net = gross - COST_BPS_RT / 1e4              # one round trip
            feat["exit_i"] = t; feat["exit_ts"] = idx[t]; feat["hold_days"] = t - et
            feat["ret_net"] = net; feat["win"] = int(net > 0)
            trades.append(feat); holding = False
    return trades


FEATURES = ["rsi2", "dist5", "dist200", "vol20", "dd20", "mom20", "dow", "vix", "vix_chg5"]


def stats(r, ppy=252):
    r = r.dropna(); eq = (1 + r).cumprod()
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


def basket_daily_from_trades(trades_df, kept_mask=None):
    """Turn a set of trades into an equal-weight BASKET daily-return series. Each day's return =
    mean over instruments currently in a trade of that instrument's per-day trade return
    (approximated as the trade's net return spread evenly across its holding days), else RF/252.
    This is a returns-level approximation consistent with swing_basket.py's equal-weight framing."""
    df = trades_df if kept_mask is None else trades_df[kept_mask]
    if len(df) == 0:
        return pd.Series(dtype=float)
    # spread each trade's net return across its holding days as a daily geometric rate
    all_days = {}
    for _, tr in df.iterrows():
        days = pd.bdate_range(tr["entry_ts"], tr["exit_ts"])
        if len(days) == 0: continue
        daily = (1 + tr["ret_net"]) ** (1 / len(days)) - 1
        for d in days:
            all_days.setdefault(d, []).append(daily)
    if not all_days:
        return pd.Series(dtype=float)
    s = pd.Series({d: np.mean(v) for d, v in all_days.items()}).sort_index()
    # fill non-trading-position days with RF (idle cash), across the full span
    full = pd.bdate_range(s.index.min(), s.index.max())
    s = s.reindex(full).fillna(RF / 252)
    return s


def basket_fixed_fraction(trades_df, price_rets, kept_mask=None, n_slots=10):
    """REALISTIC event-driven basket: capital split into n_slots fixed slots; each open position
    holds 1/n_slots of capital and earns its ACTUAL daily price return that day (from price_rets),
    unused slots earn RF/252. Slot-constrained: if more positions want in than free slots, the
    excess is skipped (FIFO by entry). This preserves true daily vol — no smearing of trade-level
    scalars across calendar days — so the Sharpe is REAL.
      price_rets: dict[sym] -> daily pct_change Series (datetime index, tz-naive).
    Returns the portfolio daily-return Series."""
    df = trades_df if kept_mask is None else trades_df[kept_mask]
    if len(df) == 0:
        return pd.Series(dtype=float)
    df = df.sort_values("entry_ts")
    # map each trade to the set of trading days it is HELD (day after entry .. exit inclusive):
    # entry at close t means we hold the price move from t -> exit over days (t, exit].
    holds_by_day = {}     # day -> list of syms held that day
    # use each symbol's own index to get the exact trading days between entry and exit
    for _, tr in df.iterrows():
        sym = tr["sym"]; pr = price_rets.get(sym)
        if pr is None: continue
        # trading days strictly after entry_ts up to and including exit_ts
        mask = (pr.index > tr["entry_ts"]) & (pr.index <= tr["exit_ts"])
        for d in pr.index[mask]:
            holds_by_day.setdefault(d, []).append(sym)
    if not holds_by_day:
        return pd.Series(dtype=float)
    all_days = pd.DatetimeIndex(sorted(holds_by_day)).sort_values()
    full = pd.bdate_range(all_days.min(), all_days.max())
    out = {}
    for d in full:
        syms = holds_by_day.get(d, [])[:n_slots]      # slot cap (FIFO by entry order)
        k = len(syms)
        pos_ret = 0.0
        for sym in syms:
            r = price_rets[sym].get(d, 0.0)
            pos_ret += (r if pd.notna(r) else 0.0)
        # k slots earn their price return; (n_slots-k) slots earn cash; minus per-position RT cost
        # is already inside ret_net at the trade level, so charge cost on entry days only:
        out[d] = (pos_ret + (n_slots - k) * (RF / 252)) / n_slots
    s = pd.Series(out).sort_index()
    # subtract round-trip cost on each trade's entry day (1 RT per trade, spread is negligible daily)
    cost_by_day = {}
    for _, tr in df.iterrows():
        d0 = tr["entry_ts"]
        cost_by_day[d0] = cost_by_day.get(d0, 0.0) + (COST_BPS_RT / 1e4) / n_slots
    for d, cst in cost_by_day.items():
        if d in s.index:
            s.loc[d] -= cst
    return s


def main():
    vix = vix_series()
    print("Extracting RSI2 trades across the basket...")
    all_trades = []; price_rets = {}
    for f in sorted(glob.glob(f"{DATA}/bars_*_1d.parquet")):
        sym = os.path.basename(f)[len("bars_"):-len("_1d.parquet")]
        if sym in EXCLUDE or sym.endswith("_5m") or sym.endswith("_30m"): continue
        try:
            c = load_close(f)
            all_trades += extract_trades(c, sym, vix)
            price_rets[sym] = c.pct_change()             # for the realistic event-driven basket
        except Exception as e:
            print(f"  skip {sym}: {e}")
    td = pd.DataFrame(all_trades).dropna(subset=FEATURES).sort_values("exit_ts").reset_index(drop=True)
    print(f"Pooled {len(td)} trades across {td['sym'].nunique()} instruments, "
          f"{td['entry_ts'].min().date()} -> {td['exit_ts'].max().date()}")
    print(f"Base win-rate {td['win'].mean()*100:.1f}%   mean net ret/trade {td['ret_net'].mean()*100:.3f}%\n")

    # ---- PURGED + EMBARGOED walk-forward: train on trades CLOSED before each test year (minus embargo) ----
    td["test_year"] = td["entry_ts"].dt.year
    years = sorted(td["test_year"].unique())
    test_years = [y for y in years if y >= 2015]      # train pre-2015, then expand
    for model_name, make_model, scale in [
        ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                        max_iter=300, l2_regularization=1.0,
                                                                        min_samples_leaf=50, random_state=0), False),
        ("LogisticRegression",  lambda: LogisticRegression(max_iter=1000, C=0.5), True),
    ]:
        pwin = pd.Series(index=td.index, dtype=float)
        for ty in test_years:
            test_mask = td["test_year"] == ty
            # purge: train only on trades that CLOSED before the embargo cutoff (start of test year - embargo)
            cutoff = pd.Timestamp(year=ty, month=1, day=1) - pd.Timedelta(days=EMBARGO_DAYS)
            train_mask = td["exit_ts"] < cutoff
            if train_mask.sum() < 300 or test_mask.sum() == 0:
                continue
            Xtr, ytr = td.loc[train_mask, FEATURES].values, td.loc[train_mask, "win"].values
            Xte = td.loc[test_mask, FEATURES].values
            if scale:
                sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            m = make_model().fit(Xtr, ytr)
            pwin.loc[test_mask] = m.predict_proba(Xte)[:, 1]
        td[f"pwin_{model_name}"] = pwin

    # ---- LEAKY CONTROL: random K-fold (no purge) — measures how much leakage would inflate ----
    from sklearn.model_selection import cross_val_predict, KFold
    sub = td.dropna(subset=["pwin_HistGradientBoosting"]).copy()
    leaky = cross_val_predict(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                             l2_regularization=1.0, min_samples_leaf=50, random_state=0),
                              sub[FEATURES].values, sub["win"].values,
                              cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    sub["pwin_LEAKY"] = leaky

    # ---- Evaluate: unfiltered vs filtered baskets, after tax ----
    oos = td.dropna(subset=["pwin_HistGradientBoosting"]).copy()
    print(f"OOS trades (2015+, purged-embargoed): {len(oos)}  base win {oos['win'].mean()*100:.1f}%\n")

    def filt_results(scores, label, thr_grid=(0.45, 0.50, 0.55, 0.60)):
        rows = []
        unf = basket_daily_from_trades(oos)
        rows.append({"variant": f"[{label}] UNFILTERED", "trades": len(oos),
                     **stats(after_tax(unf, 0.25)), "Sharpe_pre": stats(unf)["Sharpe"]})
        for thr in thr_grid:
            keep = scores >= thr
            kept = oos[keep]
            if len(kept) < 50: continue
            r = basket_daily_from_trades(oos, keep.values)
            rows.append({"variant": f"  P(win)>={thr}", "trades": int(keep.sum()),
                         **stats(after_tax(r, 0.25)), "Sharpe_pre": stats(r)["Sharpe"]})
        return rows

    print("=== Meta-label filter on the basket — after-tax@25% (Sharpe_pre = pre-tax Sharpe) ===")
    rows = []
    rows += filt_results(oos["pwin_HistGradientBoosting"], "GBM")
    rows += filt_results(oos["pwin_LogisticRegression"], "Logit")
    rr = pd.DataFrame(rows).set_index("variant")
    print(rr.to_string(float_format=lambda x: f"{x:.2f}"))

    # leaky comparison at one threshold
    print("\n=== LEAKAGE CONTROL (random-CV, NOT tradeable — shows inflation if you cheat) ===")
    lk = []
    for thr in (0.50, 0.55, 0.60):
        keep = sub["pwin_LEAKY"] >= thr
        if keep.sum() < 50: continue
        r = basket_daily_from_trades(sub, keep.values)
        lk.append({"variant": f"LEAKY P>={thr}", "trades": int(keep.sum()), **stats(after_tax(r, 0.25))})
    print(pd.DataFrame(lk).set_index("variant").to_string(float_format=lambda x: f"{x:.2f}"))

    # right-tail check: are we throwing away big winners?
    print("\n=== Right-tail check (GBM @ best threshold): kept vs skipped trade returns ===")
    thr = 0.55; keep = oos["pwin_HistGradientBoosting"] >= thr
    kd, sk = oos[keep]["ret_net"], oos[~keep]["ret_net"]
    print(f"  kept:    n={len(kd):5d} mean={kd.mean()*100:.3f}% win={ (kd>0).mean()*100:.1f}% "
          f"p95={kd.quantile(0.95)*100:.2f}% max={kd.max()*100:.2f}%")
    print(f"  skipped: n={len(sk):5d} mean={sk.mean()*100:.3f}% win={ (sk>0).mean()*100:.1f}% "
          f"p95={sk.quantile(0.95)*100:.2f}% max={sk.max()*100:.2f}%")
    print("  If skipped p95/max >> kept, the filter is culling right-tail winners (bad for after-tax CAGR).")

    # ---- REALISTIC fixed-fraction basket (de-inflated Sharpe) ----
    print("\n=== FIXED-FRACTION basket (10 slots, 1/10 capital each) — REAL Sharpe, not vol-diversified ===")
    print("    (equal-weight-of-active above inflates Sharpe by averaging away daily vol; this is the honest view)")
    ff_rows = []
    for n_slots in (5, 10, 20):
        unf = basket_fixed_fraction(oos, price_rets, n_slots=n_slots)
        s_pre, s_at = stats(unf), stats(after_tax(unf, 0.25))
        ff_rows.append({"variant": f"UNFILTERED ({n_slots} slots)", "Sharpe_pre": s_pre["Sharpe"],
                        "CAGR%_at25": s_at["CAGR%"], "Sharpe_at25": s_at["Sharpe"], "maxDD%_at25": s_at["maxDD%"]})
    # filtered vs unfiltered at 10 slots, GBM best threshold
    for thr in (0.50, 0.55, 0.60):
        keep = oos["pwin_HistGradientBoosting"] >= thr
        if keep.sum() < 50: continue
        r = basket_fixed_fraction(oos, price_rets, keep.values, n_slots=10)
        s_pre, s_at = stats(r), stats(after_tax(r, 0.25))
        ff_rows.append({"variant": f"GBM P>={thr} (10 slots)", "Sharpe_pre": s_pre["Sharpe"],
                        "CAGR%_at25": s_at["CAGR%"], "Sharpe_at25": s_at["Sharpe"], "maxDD%_at25": s_at["maxDD%"]})
    print(pd.DataFrame(ff_rows).set_index("variant").to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n=== Read ===")
    print("- Filter PASSES only if a threshold beats UNFILTERED on BOTH after-tax Sharpe AND CAGR.")
    print("- If GBM ~= Logit, the model adds nothing — use the simpler one.")
    print("- If purged-embargoed gains << LEAKY gains, most apparent edge was leakage. Trust the purged row.")
    print("- FIXED-FRACTION Sharpe is the REAL number; the equal-weight-of-active Sharpe above is inflated.")


if __name__ == "__main__":
    main()
