from __future__ import annotations
import math
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

DATA = "."
TENOR_DAYS = 21
VRP_MULT = 1.20
IV_FLOOR = 0.08
RF_FLAT = 0.025
DIV_Y = 0.017
COST_PER_ROLL = 0.0010
HEDGE_CARRY = -0.010     # net annual carry from FX-hedging = (ILS short rate - USD short rate).
                         # Negative today because ILS rates sit slightly below USD. Sensitivity shown below.


def norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


@dataclass
class Variant:
    name: str
    otm: float
    spread_width: float


def load_aux(name):
    p = f"{DATA}/bars_{name}_1d.parquet"
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)[["ts", "close"]].copy()
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)
    return d.sort_values("ts").set_index("ts")["close"]


def load_spy():
    d = pd.read_parquet(f"{DATA}/bars_SPY_1d.parquet")[["ts", "close"]].copy()
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)
    d = d.sort_values("ts").reset_index(drop=True)
    d["ret"] = d["close"].pct_change()
    d["rv"] = d["ret"].rolling(21).std() * math.sqrt(252)
    return d


def aligned(series, ts):
    return None if series is None else series.reindex(ts, method="ffill").reset_index(drop=True)


def backtest(df, v, iv=None, rate=None):
    px = df["close"].values; rv = df["rv"].values; n = len(df); T = TENOR_DAYS / 252
    rets, dates, t = [], [], 60
    while t + TENOR_DAYS < n:
        S0 = px[t]
        r = float(rate.iloc[t]) if rate is not None and not np.isnan(rate.iloc[t]) else RF_FLAT
        sig = (float(iv.iloc[t]) if iv is not None and not np.isnan(iv.iloc[t])
               else (max(rv[t] * VRP_MULT, IV_FLOOR) if not np.isnan(rv[t]) else IV_FLOOR))
        K = S0 * (1 - v.otm)
        prem = bs_put(S0, K, r, DIV_Y, sig, T)
        if v.spread_width > 0:
            K2 = S0 * (1 - v.otm - v.spread_width); prem -= bs_put(S0, K2, r, DIV_Y, sig, T)
        S1 = px[t + TENOR_DAYS]
        loss = max(0.0, K - S1)
        if v.spread_width > 0:
            loss -= max(0.0, K2 - S1)
        rets.append((prem - loss) / S0 + r * T - COST_PER_ROLL)
        dates.append(df["ts"].iloc[t + TENOR_DAYS]); t += TENOR_DAYS
    return pd.Series(rets, index=pd.DatetimeIndex(dates))


def stats(r, ppy=12):
    r = r.dropna(); eq = (1 + r).cumprod()
    return {"CAGR%": (eq.iloc[-1] ** (ppy / len(r)) - 1) * 100,
            "vol%": r.std() * math.sqrt(ppy) * 100,
            "Sharpe": (r.mean() * ppy - RF_FLAT) / (r.std() * math.sqrt(ppy)) if r.std() > 0 else 0,  # EXCESS of rf
            "maxDD%": (eq / eq.cummax() - 1).min() * 100}


def to_ils(usd, fx_px):
    if fx_px is None:
        return None
    fx = fx_px.reindex(usd.index, method="ffill")
    return (1 + usd) * (1 + fx.pct_change().fillna(0)) - 1


def to_ils_hedged(usd, carry_annual, ppy=12):
    # Full FX hedge: shekel value tracks the USD return (FX vol removed); rolling the hedge
    # earns/pays the rate differential (ILS - USD) ~ covered interest parity forward points.
    return usd + carry_annual / ppy


def after_tax(rets, rate):
    rets = rets.dropna(); eq, carry = 1.0, 0.0; out_r, out_i = [], []
    for y in sorted(set(rets.index.year)):
        yr = rets[rets.index.year == y]; start = eq; eqs, m = [], eq
        for x in yr:
            m *= (1 + x); eqs.append(m)
        gain = eqs[-1] - start
        if gain >= 0:
            used = min(gain, carry); carry -= used; taxable = gain - used
        else:
            carry += -gain; taxable = 0.0
        eqs[-1] -= rate * max(0.0, taxable)
        prev = start
        for e in eqs:
            out_r.append(e / prev - 1); prev = e
        out_i.extend(list(yr.index)); eq = eqs[-1]
    return pd.Series(out_r, index=pd.DatetimeIndex(out_i))


if __name__ == "__main__":
    df = load_spy(); ts = df["ts"]
    iv = aligned(load_aux("VIX"), ts); iv = iv / 100.0 if iv is not None else None
    irx = aligned(load_aux("IRX"), ts); rate = irx / 100.0 if irx is not None else None
    fx = load_aux("USDILSX")
    print(f"SPY {df['ts'].min().date()} -> {df['ts'].max().date()}  "
          f"IV={'real VIX' if iv is not None else 'RV proxy'}  "
          f"rate={'real 13wk T-bill' if rate is not None else f'flat {RF_FLAT:.1%}'}  "
          f"FX={'USD/ILS' if fx is not None else 'none'}  hedge_carry={HEDGE_CARRY:+.1%}\n")

    v = Variant("ATM put-spread (5% wide)", 0.00, 0.05)
    usd = backtest(df, v, iv=iv, rate=rate)

    print(f"=== {v.name}: unhedged vs FX-HEDGED, layered to after-tax shekels ===")
    layers = [("USD pre-tax", usd)]
    ils = to_ils(usd, fx)
    if ils is not None:
        h = to_ils_hedged(usd, HEDGE_CARRY)
        layers += [
            ("ILS unhedged  pre-tax", ils),
            ("ILS unhedged  after-tax @25%", after_tax(ils, 0.25)),
            (f"ILS HEDGED    pre-tax  (= in a tax-deferred wrapper)", h),
            ("ILS HEDGED    after-tax @25%", after_tax(h, 0.25)),
        ]
    print(pd.DataFrame([{"layer": n, **stats(r)} for n, r in layers]
                       ).set_index("layer").to_string(float_format=lambda x: f"{x:.2f}"))

    if fx is not None:
        print("\n=== hedged pre-tax sensitivity to the ILS-USD rate gap (the hedge carry) ===")
        rows = [{"hedge_carry": f"{c:+.1%}", **stats(to_ils_hedged(usd, c))}
                for c in (-0.02, -0.01, 0.0, 0.01)]
        print(pd.DataFrame(rows).set_index("hedge_carry").to_string(float_format=lambda x: f"{x:.2f}"))
        print("\nHedging removes the FX vol -> drawdown & Sharpe snap back to the USD profile;")
        print("only the LEVEL shifts with the rate gap. Wrapper deletes the tax rows.")
