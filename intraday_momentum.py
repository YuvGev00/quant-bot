# Market Intraday Momentum (Gao, Han, Li & Zhou). Signal: sign of the first 30-min return
# predicts the last 30-min return. Trade ONLY the last 30 min: position = sign(first-30m).
# Needs 30-min bars: bars_SPY_30m.parquet (ts, open, high, low, close). Pull via IBKR (ib_fetch).
from __future__ import annotations
import math, os
import pandas as pd, numpy as np

DATA = "."; SYM = "SPY"; COST_BPS_RT = 3.0; TAX = 0.47

def stats(r, ppy=252):
    r = r.dropna(); eq=(1+r).cumprod()
    return {"CAGR%":(eq.iloc[-1]**(ppy/len(r))-1)*100,"vol%":r.std()*math.sqrt(ppy)*100,
            "Sharpe":(r.mean()*ppy)/(r.std()*math.sqrt(ppy)) if r.std()>0 else 0,
            "maxDD%":(eq/eq.cummax()-1).min()*100,"win%":(r>0).mean()*100}

def after_tax(r, tax):
    eq,carry,out,idx=1.0,0.0,[],[]
    for y in sorted(set(r.index.year)):
        yr=r[r.index.year==y]; start=eq; eqs,m=[],eq
        for x in yr: m*=(1+x); eqs.append(m)
        g=eqs[-1]-start
        if g>=0: u=min(g,carry); carry-=u; tx=g-u
        else: carry+=-g; tx=0.0
        eqs[-1]-=tax*max(0,tx); prev=start
        for e in eqs: out.append(e/prev-1); prev=e
        idx.extend(list(yr.index)); eq=eqs[-1]
    return pd.Series(out,index=pd.DatetimeIndex(idx))

p=f"{DATA}/bars_{SYM}_30m.parquet"
if not os.path.exists(p):
    print(f"Need {p} (30-min bars). Pull via IBKR with ib_fetch.py, then re-run."); raise SystemExit
d=pd.read_parquet(p)[["ts","open","high","low","close"]].copy()
d["ts"]=pd.to_datetime(d["ts"]).dt.tz_localize(None); d["day"]=d["ts"].dt.date
rows=[]
for day,g in d.groupby("day"):
    g=g.sort_values("ts")
    if len(g)<3: continue
    first=g.iloc[0]; last=g.iloc[-1]
    r_first=first["close"]/first["open"]-1
    r_last=last["close"]/last["open"]-1
    sig=np.sign(r_first)
    rows.append((pd.Timestamp(day), sig*r_last))
s=pd.Series(dict(rows)).sort_index()
gross=s; net=s-COST_BPS_RT/1e4; net_tax=after_tax(net,TAX)
print(f"Intraday Momentum {SYM}  {s.index.min().date()}->{s.index.max().date()}  ({len(s)} days)\n")
out={"gross":gross,f"net {COST_BPS_RT:.0f}bp/RT":net,f"net +{int(TAX*100)}% tax":net_tax}
print(pd.DataFrame({k:stats(v) for k,v in out.items()}).T.to_string(float_format=lambda x:f"{x:.2f}"))
