# Experiment 2 — Statistical Jump Model (SJM) regime gate on SPY (Shu, Yu & Mulvey 2024).
# A k-means-style regime classifier with an explicit JUMP PENALTY lambda on every state
# transition, so the fitted regime path is sticky (few switches/yr -> few taxable round-trips).
# That stickiness is the whole point for an Israeli taxpayer: regime timers normally die on
# the 25-47% CGT-no-deferral wall; SJM is built to switch ~2-3x/yr.
#
# Algorithm (coordinate descent), per Shu-Yu-Mulvey:
#   given features X (T x p, standardized) and K states with centroids mu_k:
#   (1) ASSIGN: choose state path s minimizing sum_t ||x_t - mu_{s_t}||^2 + lambda * sum_t 1[s_t != s_{t-1}]
#       -> exact DP (Viterbi-like) over the loss matrix.
#   (2) UPDATE: mu_k = mean of x_t assigned to k.
#   iterate to convergence. lambda=0 recovers k-means; large lambda -> one regime.
#
# We do this WALK-FORWARD with NO look-ahead:
#   - standardize features on the training window only;
#   - fit centroids on training;
#   - pick lambda on a validation window by NET-OF-COST strategy Sharpe (penalizes churn);
#   - assign the out-of-sample period online; trade T+1.
# Then run the full cost + 25/47% CGT + USD/ILS gauntlet and print the MANDATED baselines:
#   buy-hold SPY, a plain VIX-percentile gate, and buy-hold DBMF.
from __future__ import annotations
import math
import numpy as np, pandas as pd

DATA = "."; COST_BPS_RT = 3.0; LAG = 1            # T+1 execution (set 2 for the paper's T+2)
RF_FALLBACK = 0.025                                # used before BIL exists (2007) / if BIL missing
TRAIN_YEARS = 10; VALID_YEARS = 3                  # walk-forward windows
K_STATES = 3
LAMBDAS = [25.0, 50.0, 100.0, 200.0]               # jump-penalty grid; floored >0 so we never collapse
                                                   # to pure k-means (lam=0), which churns and re-creates
                                                   # the very tax-wall problem SJM exists to avoid.
VALID_TAX = 0.25                                   # select lambda on AFTER-TAX validation Sharpe (the real
                                                   # objective) so the selector cannot reward taxable churn.


# ---------- data ----------
def load_close(sym):
    p = f"{DATA}/bars_{sym}_1d.parquet"
    try:
        d = pd.read_parquet(p)[["ts", "close"]].copy()
    except FileNotFoundError:
        return None
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)
    return d.sort_values("ts").set_index("ts")["close"]


def tbill_daily(index):
    """Daily risk-free return aligned to `index`. Use IRX (13wk T-bill yield, %) if present, else flat."""
    irx = load_close("IRX")
    if irx is not None:
        y = (irx.reindex(index, method="ffill") / 100.0).fillna(RF_FALLBACK)
        return y / 252.0
    return pd.Series(RF_FALLBACK / 252.0, index=index)


# ---------- features (all causal, paper's family) ----------
def make_features(ret_excess):
    """EWM downside deviation (hl=10) and EWM Sortino-like ratios (hl=20,60) on excess returns."""
    dn = ret_excess.clip(upper=0.0)
    dd10 = (dn.pow(2).ewm(halflife=10, adjust=False).mean()).pow(0.5)
    feats = {}
    for hl in (20, 60):
        mu = ret_excess.ewm(halflife=hl, adjust=False).mean()
        dd = (dn.pow(2).ewm(halflife=hl, adjust=False).mean()).pow(0.5)
        feats[f"sortino{hl}"] = mu / dd.replace(0, np.nan)
    feats["dd10"] = dd10
    X = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
    return X


# ---------- Statistical Jump Model core ----------
def _assign_path(D, lam):
    """Exact DP: minimize sum_t D[t, s_t] + lam * 1[s_t != s_{t-1}].  D = (T x K) distance-to-centroid.
    Vectorized forward pass: transition cost from prev-state i to state k is lam*(i!=k), so the
    min over i of (cost[t-1,i] + lam*(i!=k)) = min( min_i cost[t-1,i] + lam,  cost[t-1,k] ),
    i.e. either jump from the globally-cheapest other state (+lam) or stay in k (+0)."""
    T, K = D.shape
    cost = np.empty((T, K)); back = np.zeros((T, K), dtype=int)
    cost[0] = D[0]
    for t in range(1, T):
        prev = cost[t - 1]
        gmin = prev.min(); garg = int(prev.argmin())
        # candidate "jump from cheapest other state": if k IS the global argmin, the cheapest
        # OTHER state is the 2nd-smallest; handle by comparing stay vs (gmin+lam) per k.
        jump_from = np.full(K, gmin + lam)          # cost of arriving at k via a jump
        jump_src = np.full(K, garg, dtype=int)
        # if k == garg, a "jump" must originate elsewhere -> use 2nd smallest
        if K > 1:
            tmp = prev.copy(); tmp[garg] = np.inf
            gmin2 = tmp.min(); garg2 = int(tmp.argmin())
            jump_from[garg] = gmin2 + lam; jump_src[garg] = garg2
        stay = prev                                  # cost of staying in k (no penalty)
        use_stay = stay <= jump_from
        cost[t] = D[t] + np.where(use_stay, stay, jump_from)
        back[t] = np.where(use_stay, np.arange(K), jump_src)
    s = np.empty(T, dtype=int); s[-1] = int(np.argmin(cost[-1]))
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s


def fit_jump(X, K, lam, n_init=5, max_iter=30, seed=0):
    """Fit centroids by jump-penalized coordinate descent. Returns (centroids, labels, inertia)."""
    Xv = X.values; T, p = Xv.shape
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        mu = Xv[rng.choice(T, K, replace=False)].copy()
        prev = None
        for _ in range(max_iter):
            D = ((Xv[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)   # T x K
            s = _assign_path(D, lam)
            new = mu.copy()
            for k in range(K):
                m = s == k
                if m.any():
                    new[k] = Xv[m].mean(axis=0)
            if prev is not None and np.allclose(new, mu):
                mu = new; break
            mu = new; prev = s
        D = ((Xv[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
        s = _assign_path(D, lam)
        inertia = D[np.arange(T), s].sum() + lam * (np.diff(s) != 0).sum()
        if best is None or inertia < best[2]:
            best = (mu, s, inertia)
    return best


def assign_online(X, mu, lam):
    """Assign labels for X given fixed centroids (still DP so the jump penalty applies OOS)."""
    Xv = X.values
    D = ((Xv[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
    return _assign_path(D, lam)


def rank_states_by_vol(mu, feat_cols):
    """Order states from calm->stormy by the downside-deviation centroid (dd10). Highest = risk-off."""
    dd_idx = list(feat_cols).index("dd10")
    return np.argsort(mu[:, dd_idx])     # ascending: calm first, stormy last


# ---------- strategy evaluation ----------
def gate_returns(in_mkt, asset_ret, rf_daily):
    """Long asset when in_mkt else earn T-bill. Net of cost on switches. in_mkt is a 0/1 Series."""
    pos = in_mkt.astype(float)
    turn = pos.diff().abs().fillna(0)
    return pos * asset_ret + (1 - pos) * rf_daily - turn * (COST_BPS_RT / 1e4) / 2


def stats(r, ppy=252):
    r = r.dropna(); eq = (1 + r).cumprod()
    return {"CAGR%": (eq.iloc[-1] ** (ppy / len(r)) - 1) * 100, "vol%": r.std() * math.sqrt(ppy) * 100,
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


def to_ils(usd, fx):
    if fx is None: return None
    f = fx.reindex(usd.index, method="ffill")
    return (1 + usd) * (1 + f.pct_change().fillna(0)) - 1


# ---------- walk-forward driver ----------
def walk_forward(close, fx=None):
    ret = close.pct_change().fillna(0)
    rf = tbill_daily(close.index)
    excess = ret - rf
    X_all = make_features(excess)
    valid = X_all.dropna().index
    X_all = X_all.loc[valid]; ret = ret.loc[valid]; rf = rf.loc[valid]

    years = sorted(set(X_all.index.year))
    regime = pd.Series(index=X_all.index, dtype=float)
    chosen_lams = {}
    start_i = TRAIN_YEARS + VALID_YEARS
    for yi in range(start_i, len(years)):
        test_y = years[yi]
        tr = X_all[(X_all.index.year >= years[yi - start_i]) & (X_all.index.year < years[yi - VALID_YEARS])]
        va = X_all[(X_all.index.year >= years[yi - VALID_YEARS]) & (X_all.index.year < test_y)]
        te = X_all[X_all.index.year == test_y]
        if len(tr) < 250 or len(va) < 60 or len(te) < 5:
            continue
        mean, std = tr.mean(), tr.std().replace(0, 1.0)
        trS, vaS, teS = (tr - mean) / std, (va - mean) / std, (te - mean) / std

        best_lam, best_sh = None, -1e9
        for lam in LAMBDAS:
            mu, _, _ = fit_jump(trS, K_STATES, lam)
            order = rank_states_by_vol(mu, X_all.columns)
            riskoff = order[-1]
            s_va = assign_online(vaS, mu, lam)
            in_mkt = pd.Series((s_va != riskoff).astype(float), index=va.index)
            r = gate_returns(in_mkt, ret.loc[va.index], rf.loc[va.index])
            r_at = after_tax(r, VALID_TAX)          # judge lambda on AFTER-TAX Sharpe, not gross
            sh = (r_at.mean() * 252) / (r_at.std() * math.sqrt(252)) if r_at.std() > 0 else -1e9
            if sh > best_sh:
                best_sh, best_lam = sh, lam
        # refit on train+valid with the chosen lambda, assign the test year online
        trva = X_all[(X_all.index.year >= years[yi - start_i]) & (X_all.index.year < test_y)]
        mean, std = trva.mean(), trva.std().replace(0, 1.0)
        mu, _, _ = fit_jump((trva - mean) / std, K_STATES, best_lam)
        order = rank_states_by_vol(mu, X_all.columns); riskoff = order[-1]
        s_te = assign_online((te - mean) / std, mu, best_lam)
        regime.loc[te.index] = (s_te == riskoff).astype(float)   # 1 = risk-off (high vol)
        chosen_lams[test_y] = best_lam

    return regime.dropna(), ret, rf, chosen_lams


def ma_gate(close, idx, rf, n=200):
    """Plain 200-day moving-average gate: in market when close > SMA(n), else T-bill. The cheap
    benchmark SJM must justify itself against — if a free SMA crossover ties it, SJM isn't worth it."""
    sma = close.rolling(n).mean()
    in_mkt = (close > sma).astype(float).reindex(idx).fillna(0.0).shift(LAG).fillna(1.0)
    return gate_returns(in_mkt, close.pct_change().reindex(idx).fillna(0), rf.loc[idx])


def evaluate(sym, fx=None, vix=None, dbmf_ret=None):
    """Run the SJM gate + baselines for one asset. Returns (verdict_dict, printed already)."""
    close = load_close(sym)
    if close is None:
        print(f"[{sym}] no data, skipping."); return None
    riskoff, ret, rf, lams = walk_forward(close, fx)
    if len(riskoff) < 250:
        print(f"[{sym}] too little OOS, skipping."); return None
    in_mkt = (1 - riskoff)
    in_mkt_lagged = in_mkt.shift(LAG).fillna(1.0)
    sjm = gate_returns(in_mkt_lagged, ret.loc[in_mkt_lagged.index], rf.loc[in_mkt_lagged.index])
    idx = sjm.index

    switches = int(in_mkt.diff().abs().sum()); yrs = (idx.max() - idx.min()).days / 365.25
    print(f"\n################  {sym}  ################")
    print(f"SJM gate OOS {idx.min().date()} -> {idx.max().date()}  ({len(idx)} days)  "
          f"switches={switches} ({switches/yrs:.2f}/yr)  "
          + ("OK low-turnover" if switches/yrs <= 3.5 else "** WARN >3.5/yr (tax-wall erodes)"))

    bh = ret.loc[idx]
    series = {f"SJM gate": sjm, f"buy-hold {sym}": bh, "200d-MA gate": ma_gate(close, idx, rf)}
    if vix is not None:
        v = vix.reindex(idx, method="ffill"); thr = v.rolling(252, min_periods=60).quantile(0.80)
        vix_in = (v <= thr).astype(float).shift(LAG).fillna(1.0)
        series["VIX-80pct gate"] = gate_returns(vix_in, bh, rf.loc[idx])
    if dbmf_ret is not None:
        series["buy-hold DBMF"] = dbmf_ret.reindex(idx).fillna(0)

    # build the USD after-tax-25% comparison table (the decision view)
    rows = []
    for name, r in series.items():
        s25 = stats(after_tax(r, 0.25)); s = stats(r)
        rows.append({"strat": name, "CAGR%": s["CAGR%"], "Sharpe": s["Sharpe"], "maxDD%": s["maxDD%"],
                     "CAGR%_at25": s25["CAGR%"], "Sharpe_at25": s25["Sharpe"], "maxDD%_at25": s25["maxDD%"]})
    tbl = pd.DataFrame(rows).set_index("strat")
    print(tbl.to_string(float_format=lambda x: f"{x:.2f}"))

    # verdict: does SJM beat EVERY baseline on after-tax-25% Sharpe?
    sjm_sh = tbl.loc["SJM gate", "Sharpe_at25"]
    beat = {b: sjm_sh > tbl.loc[b, "Sharpe_at25"] for b in series if b != "SJM gate"}
    verdict = "PASS — beats all baselines" if all(beat.values()) else \
              f"PARTIAL — loses to: {[b for b,v in beat.items() if not v]}"
    print(f"VERDICT [{sym}] after-tax@25% Sharpe: {verdict}")
    return {"sym": sym, "sjm_sharpe_at25": sjm_sh, "switches_per_yr": switches/yrs,
            "beats": beat, "verdict": verdict}


def main():
    fx = load_close("USDILSX"); vix = load_close("VIX")
    dbmf = load_close("DBMF"); dbmf_ret = dbmf.pct_change() if dbmf is not None else None
    print("SJM regime-gate external-validity test — does it generalize beyond SPY?")
    print("Baselines per asset: buy-hold, 200d-MA gate, VIX-80pct gate, buy-hold DBMF.")
    print("Cols: raw 3 = pre-tax; *_at25 = after Israeli CGT 25% (loss-carryforward).\n")
    # Widened, all DEEP-history (EWG/EWA/EWC/EWQ refetched to full 1996+ via free_fetch).
    # US: SPY/QQQ/IWM ; Europe: EWG/EWQ/EWU ; Asia-Pac: EWJ/EWH/EWA ; Americas: EWC/EWZ ; broad: EFA/EEM.
    results = []
    for sym in ["SPY", "QQQ", "IWM", "EWG", "EWQ", "EWU", "EWJ", "EWH", "EWA", "EWC", "EWZ", "EFA", "EEM"]:
        r = evaluate(sym, fx=fx, vix=vix, dbmf_ret=dbmf_ret)
        if r: results.append(r)

    print("\n================  CROSS-ASSET SUMMARY  ================")
    print("If SJM passes on SPY but FAILS on EWG/EWJ, the SPY result is likely overfit.")
    for r in results:
        print(f"  {r['sym']}: SJM after-tax Sharpe {r['sjm_sharpe_at25']:.2f}  "
              f"({r['switches_per_yr']:.1f} sw/yr)  ->  {r['verdict']}")


if __name__ == "__main__":
    main()
