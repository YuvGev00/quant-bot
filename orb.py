# 5-minute Opening Range Breakout (Zarattini-Barbon-Aziz), single-instrument version.
# Opening range = first 5-min bar. If it closed up, go long on a break above its high (stop=low);
# if down, short on break below its low (stop=high). Exit at EOD. Needs 5-min bars:
# bars_SPY_5m.parquet. The full "Stocks in Play" version scans a universe by relative volume.
from __future__ import annotations
import math, os
import pandas as pd, numpy as np

DATA="."; SYM="SPY"; COST_BPS_RT=3.0; TAX=0.47

def stats(r, ppy=252):
    r=r.dropna(); eq=(1+r).cumprod()
    return {"CAGR%":(eq.iloc[-1]**(ppy/len(r))-1)*100,"vol%":r.std()*math.sqrt(ppy)*100,
            "Sharpe":(r.mean()*ppy)/(r.std()*math.sqrt(ppy)) if r.std()>0 else 0,
            "maxDD%":(eq/eq.cummax()-1).min()*100,"win%":(r>0).mean()*100,"trades":int((r!=0).sum())}

def after_tax(r,tax):
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

p=f"{DATA}/bars_{SYM}_5m.parquet"
if not os.path.exists(p):
    print(f"Need {p} (5-min bars). Pull via IBKR with ib_fetch.py, then re-run."); raise SystemExit
d=pd.read_parquet(p)[["ts","open","high","low","close"]].copy()
d["ts"]=pd.to_datetime(d["ts"]).dt.tz_localize(None); d["day"]=d["ts"].dt.date
rows=[]
for day,g in d.groupby("day"):
    g=g.sort_values("ts").reset_index(drop=True)
    if len(g)<4: continue
    orb=g.iloc[0]; rest=g.iloc[1:]
    up = orb["close"]>orb["open"]
    entry = orb["high"] if up else orb["low"]
    stop  = orb["low"]  if up else orb["high"]
    ret=0.0; in_pos=False
    for _,b in rest.iterrows():
        if not in_pos:
            if (up and b["high"]>=entry) or ((not up) and b["low"]<=entry):
                in_pos=True; px=entry
        if in_pos:
            if up and b["low"]<=stop:  ret=(stop/px-1); break
            if (not up) and b["high"]>=stop: ret=(px/stop-1); break
    else:
        if in_pos:
            last=rest.iloc[-1]["close"]; ret=(last/px-1) if up else (px/last-1)
    rows.append((pd.Timestamp(day), ret))
s=pd.Series(dict(rows)).sort_index()
gross=s; net=s-np.where(s!=0,COST_BPS_RT/1e4,0); net=pd.Series(net,index=s.index)
net_tax=after_tax(net,TAX)
print(f"ORB 5m {SYM}  {s.index.min().date()}->{s.index.max().date()}  ({len(s)} days)\n")
out={"gross":gross,f"net {COST_BPS_RT:.0f}bp/RT":net,f"net +{int(TAX*100)}% tax":net_tax}
print(pd.DataFrame({k:stats(v) for k,v in out.items()}).T.to_string(float_format=lambda x:f"{x:.2f}"))
