from __future__ import annotations
import math
import pandas as pd
import numpy as np

DATA = "."

def stats(r, ppy=252):
    r = r.dropna(); eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (ppy / len(r)) - 1
    vol = r.std() * math.sqrt(ppy)
    sharpe = (r.mean() * ppy) / vol if vol > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR%": cagr*100, "vol%": vol*100, "Sharpe": sharpe, "maxDD%": dd*100, "total_x": eq.iloc[-1]}

d = pd.read_parquet(f"{DATA}/bars_SPY_1d.parquet")[["ts","open","close"]].copy()
d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)
d = d.sort_values("ts").reset_index(drop=True)

d["overnight"] = d["open"] / d["close"].shift(1) - 1     # prior close -> today open
d["intraday"]  = d["close"] / d["open"] - 1              # today open -> today close
d["buyhold"]   = d["close"].pct_change()                 # = (1+overnight)(1+intraday)-1

print(f"SPY {d['ts'].min().date()} -> {d['ts'].max().date()}  ({len(d)} days)\n")
print("=== Return decomposition: where does SPY's return actually happen? ===")
rows = {"Overnight (close->open)": d["overnight"], "Intraday (open->close)": d["intraday"],
        "Buy & hold (close->close)": d["buyhold"]}
print(pd.DataFrame({k: stats(v) for k,v in rows.items()}).T.to_string(float_format=lambda x: f"{x:.2f}"))

print("\n=== 'Buy the close, sell the open' STRATEGY, after costs + Israeli tax ===")
on = d["overnight"].dropna()
def after_cost_tax(r, cost_bps_rt, tax):
    net = r - cost_bps_rt/1e4                       # one round-trip (buy MOC + sell MOO) per day
    # annual tax on net gains, loss carryforward
    s = pd.Series(net.values, index=d["ts"].iloc[1:1+len(net)].values)
    eq, carry, out, idx = 1.0, 0.0, [], []
    for y in sorted(set(pd.DatetimeIndex(s.index).year)):
        yr = s[pd.DatetimeIndex(s.index).year == y]; start=eq; eqs,m=[],eq
        for x in yr: m*=(1+x); eqs.append(m)
        gain=eqs[-1]-start
        if gain>=0: used=min(gain,carry); carry-=used; taxable=gain-used
        else: carry+=-gain; taxable=0.0
        eqs[-1]-=tax*max(0,taxable); prev=start
        for e in eqs: out.append(e/prev-1); prev=e
        idx.extend(list(yr.index)); eq=eqs[-1]
    return pd.Series(out, index=pd.DatetimeIndex(idx))

configs = [("gross (0 cost, 0 tax)", 0.0, 0.0),
           ("IBKR-ish 1bp/RT, 47% tax", 1.0, 0.47),
           ("3bp/RT, 47% tax", 3.0, 0.47),
           ("Interactive-Israel-ish 5bp/RT, 47% tax", 5.0, 0.47)]
print(pd.DataFrame({n: stats(after_cost_tax(on, c, t)) for n,c,t in configs}).T.to_string(float_format=lambda x: f"{x:.2f}"))
print("\n252 round-trips/yr: even ~3bp/trip = ~7.5%/yr drag before tax. That's the wall.")
