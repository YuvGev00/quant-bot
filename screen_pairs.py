from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from cost_model import IBKRCosts, Leg, TradePlan, round_trip_cost

# Economically-motivated candidates only (a reason to co-move), NOT all combinations.
CANDIDATE_PAIRS = [
    ("GLD", "GDX"), ("GDX", "GDXJ"), ("SLV", "SIL"),        # precious metals / miners
    ("XLE", "XOP"), ("XOM", "CVX"), ("USO", "BNO"),          # energy
    ("KO", "PEP"), ("HD", "LOW"), ("V", "MA"), ("MCD", "YUM"),  # consumer peers
    ("UPS", "FDX"), ("JPM", "WFC"), ("XLF", "KBE"),          # logistics / financials
    ("XLK", "VGT"), ("SPY", "IVV"), ("EWA", "EWC"),          # near-duplicate / linked economies
    ("XLP", "XLU"), ("IWM", "IJR"), ("GOOGL", "GOOG"),       # defensives / small cap / share classes
    ("T", "VZ"), ("GS", "MS"), ("C", "BAC"), ("CAT", "DE"),  # telecom / banks / industrials
    ("DUK", "SO"), ("PFE", "MRK"), ("WMT", "TGT"),           # utilities / pharma / retail
    ("CL", "PG"), ("AXP", "COF"), ("EWG", "EWQ"),            # staples / cards / Europe
]

LOOKBACK = 60
Z_ENTRY, Z_EXIT, Z_STOP = 2.0, 0.5, 3.5
LEG_NOTIONAL = 25_000
HALF_SPREAD_CENTS = 1.0
ANNUAL_BORROW = 0.005
COSTS = IBKRCosts.interactive_israel()

# survivor thresholds
ADF_MAX = 0.10            # require residual stationarity in BOTH halves (loose p, strict by demanding both)
HL_MIN, HL_MAX = 2.0, 60.0
BETA_DRIFT_MAX = 0.50    # |beta_oos - beta_is| / |beta_is|


def load_symbol(sym: str) -> pd.DataFrame | None:
    path = f"bars_{sym}_1d.parquet"
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)[["ts", "close"]].rename(columns={"close": sym})
    return d


def beta_resid(la: np.ndarray, lb: np.ndarray):
    x = sm.add_constant(lb)
    fit = sm.OLS(la, x).fit()
    beta = fit.params[1]
    resid = la - (fit.params[0] + beta * lb)
    return beta, resid


def halflife(resid: np.ndarray) -> float:
    s = pd.Series(resid)
    ds = s.diff().dropna()
    lag = s.shift(1).dropna().iloc[: len(ds)]
    k = -sm.OLS(ds.values, sm.add_constant(lag.values)).fit().params[1]
    return np.log(2) / k if k > 0 else np.inf


def coint_metrics(df: pd.DataFrame, a: str, b: str) -> dict:
    la, lb = np.log(df[a].values), np.log(df[b].values)
    h = len(df) // 2
    beta_is, res_is = beta_resid(la[:h], lb[:h])
    beta_oos, _ = beta_resid(la[h:], lb[h:])
    res_oos = la[h:] - (np.mean(la[:h] - beta_is * lb[:h]) + beta_is * lb[h:])  # apply IS fit OOS
    try:
        adf_is = adfuller(res_is, maxlag=1, autolag=None)[1]
        adf_oos = adfuller(res_oos, maxlag=1, autolag=None)[1]
    except Exception:
        adf_is = adf_oos = 1.0
    drift = abs(beta_oos - beta_is) / abs(beta_is) if beta_is != 0 else np.inf
    hl = halflife(res_is)
    survivor = (adf_is < ADF_MAX and adf_oos < ADF_MAX
                and HL_MIN <= hl <= HL_MAX and drift < BETA_DRIFT_MAX)
    return dict(beta_is=beta_is, adf_is=adf_is, adf_oos=adf_oos,
                beta_drift=drift, half_life=hl, survivor=survivor)


def rolling_backtest(df: pd.DataFrame, a: str, b: str):
    n = len(df)
    la, lb = np.log(df[a].values), np.log(df[b].values)
    pa, pb = df[a].values, df[b].values
    z = np.full(n, np.nan)
    for t in range(LOOKBACK, n):
        xb = sm.add_constant(lb[t - LOOKBACK:t])
        beta = sm.OLS(la[t - LOOKBACK:t], xb).fit().params[1]
        sw = la[t - LOOKBACK:t] - beta * lb[t - LOOKBACK:t]
        st = la[t] - beta * lb[t]
        sd = sw.std()
        z[t] = (st - sw.mean()) / sd if sd > 0 else 0.0

    pos, entry = 0, None
    pnl, cost = np.zeros(n), np.zeros(n)
    trades = []
    for t in range(LOOKBACK, n - 1):
        if pos != 0:
            rA = pa[t + 1] / pa[t] - 1
            rB = pb[t + 1] / pb[t] - 1
            pnl[t + 1] = pos * (rA - rB) * LEG_NOTIONAL
            cost[t + 1] += LEG_NOTIONAL * ANNUAL_BORROW / 360.0
        zt = z[t]
        if np.isnan(zt):
            continue
        new = pos
        if pos == 0:
            new = -1 if zt > Z_ENTRY else (+1 if zt < -Z_ENTRY else 0)
        elif abs(zt) < Z_EXIT or abs(zt) > Z_STOP:
            new = 0
        if new != pos:
            plan = TradePlan(Leg(a, int(LEG_NOTIONAL / pa[t]), pa[t], HALF_SPREAD_CENTS),
                             Leg(b, int(LEG_NOTIONAL / pb[t]), pb[t], HALF_SPREAD_CENTS),
                             entry_passive=(True, False), exit_passive=(False, False))
            rt = round_trip_cost(plan, COSTS)
            if pos == 0:
                cost[t] += rt["entry_cost"]; entry = t
            elif new == 0:
                cost[t] += rt["exit_cost"]; trades.append(t - entry)
            pos = new
    cap = 2 * LEG_NOTIONAL
    gross, net = pnl / cap, (pnl - cost) / cap
    return gross, net, trades


def sharpe(r):
    return np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 0 else 0.0


def maxdd(r):
    eq = np.cumprod(1 + r)
    return (eq / np.maximum.accumulate(eq) - 1).min()


def main():
    rows = []
    for a, b in CANDIDATE_PAIRS:
        da, db = load_symbol(a), load_symbol(b)
        if da is None or db is None:
            continue
        df = da.merge(db, on="ts").dropna().sort_values("ts").reset_index(drop=True)
        if len(df) < 2 * LOOKBACK + 20:
            continue
        m = coint_metrics(df, a, b)
        gross, net, trades = rolling_backtest(df, a, b)
        h = len(net) // 2
        m.update(pair=f"{a}/{b}", n=len(df), trades=len(trades),
                 sharpe_net_full=sharpe(net), sharpe_net_oos=sharpe(net[h:]),
                 ret_net_oos=np.mean(net[h:]) * 252, dd_net_oos=maxdd(net[h:]))
        rows.append(m)

    if not rows:
        print("No candidate pairs found. Pull the universe with free_fetch.py first.")
        return
    res = pd.DataFrame(rows)

    print("=== ALL CANDIDATES (cointegration screen) ===")
    cols = ["pair", "adf_is", "adf_oos", "beta_drift", "half_life", "survivor"]
    print(res[cols].sort_values("survivor", ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    surv = res[res["survivor"]].sort_values("sharpe_net_oos", ascending=False)
    print("\n=== SURVIVORS (passed both-halves cointegration), ranked by OOS net Sharpe ===")
    if surv.empty:
        print("None. Every candidate failed the stability screen on this data window.")
    else:
        sc = ["pair", "trades", "sharpe_net_full", "sharpe_net_oos", "ret_net_oos", "dd_net_oos"]
        print(surv[sc].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\nScreened {len(res)} pairs. Caveat: even with both-halves + OOS filtering, surviving")
    print("a multi-pair scan invites selection bias. Survivors are CANDIDATES for forward")
    print("paper-trading, not validated strategies. Re-confirm on fresh out-of-sample data.")


if __name__ == "__main__":
    main()
