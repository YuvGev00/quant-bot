# Swing strategies on DAILY bars (holds of ~1-10 days). Long-only on SPY.
# Needs only bars_SPY_1d.parquet (you already have 1993-2026). Costs + Israeli tax applied,
# but turnover is low so costs barely bite. Benchmarked vs buy & hold.
from __future__ import annotations
import math
import numpy as np, pandas as pd

DATA="."; COST_BPS_RT=3.0; RF=0.025  # idle cash earns T-bills (historical avg ~2.5%; today ~4.5%)

def rsi(s, n):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.ewm(alpha=1/n,adjust=False).mean()/dn.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+rs)

def run(df, entry, exit_, maxhold):
    n=len(df); pos=np.zeros(n); holding=False; et=None
    e=entry.values; x=exit_.values
    for t in range(1,n):
        pos[t]=1.0 if holding else 0.0
        if not holding and e[t]:
            holding=True; et=t
        elif holding and (x[t] or (t-et)>=maxhold):
            holding=False
    pos=pd.Series(pos,index=df.index)
    ret=df["close"].pct_change().fillna(0)
    gross=pos*ret + (1-pos)*(RF/252)        # idle capital earns the T-bill rate
    turn=pos.diff().abs().fillna(0)
    net=gross-turn*(COST_BPS_RT/1e4)/2
    trades=int((pos.diff()==1).sum())
    return net, pos, trades

def after_tax(r, rate):
    r=r.dropna(); eq,carry,out,idx=1.0,0.0,[],[]
    for y in sorted(set(r.index.year)):
        yr=r[r.index.year==y]; start=eq; eqs,m=[],eq
        for v in yr: m*=(1+v); eqs.append(m)
        g=eqs[-1]-start
        if g>=0: u=min(g,carry); carry-=u; tx=g-u
        else: carry+=-g; tx=0.0
        eqs[-1]-=rate*max(0,tx); prev=start
        for ev in eqs: out.append(ev/prev-1); prev=ev
        idx.extend(list(yr.index)); eq=eqs[-1]
    return pd.Series(out,index=pd.DatetimeIndex(idx))

def stats(r, pos=None, trades=None, ppy=252):
    r=r.dropna(); eq=(1+r).cumprod()
    d={"CAGR%":(eq.iloc[-1]**(ppy/len(r))-1)*100,"vol%":r.std()*math.sqrt(ppy)*100,
       "Sharpe":(r.mean()*ppy)/(r.std()*math.sqrt(ppy)) if r.std()>0 else 0,
       "maxDD%":(eq/eq.cummax()-1).min()*100}
    if pos is not None: d["inMkt%"]=pos.mean()*100
    if trades is not None: d["trd/yr"]=trades/(len(r)/ppy)
    return d

df=pd.read_parquet(f"{DATA}/bars_SPY_1d.parquet")[["ts","close"]].copy()
df["ts"]=pd.to_datetime(df["ts"]).dt.tz_localize(None); df=df.sort_values("ts").set_index("ts")
c=df["close"]
r2=rsi(c,2); sma5=c.rolling(5).mean(); sma10=c.rolling(10).mean(); sma200=c.rolling(200).mean()
hi20=c.rolling(20).max().shift(1); lo10=c.rolling(10).min().shift(1)
down=(c.diff()<0); down3=down&down.shift(1)&down.shift(2)

print(f"SPY {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} days)  cost={COST_BPS_RT}bp/RT  idle-cash rf={RF:.1%}\n")

strats={
 "RSI(2)<10 MR, exit>SMA5":        (run(df, r2<10, c>sma5, 10)),
 "RSI(2)<10 MR + 200d uptrend":    (run(df, (r2<10)&(c>sma200), c>sma5, 10)),
 "3 down days, exit on up day":    (run(df, down3, c>c.shift(1), 8)),
 "dip < SMA10, exit > SMA10":      (run(df, c<sma10, c>sma10, 15)),
 "Donchian 20-hi breakout, 10d":   (run(df, c>hi20, c<lo10, 10)),
}
rows=[]
for name,(net,pos,trd) in strats.items():
    rows.append({"strategy":name, **stats(net,pos,trd)})
    at25=after_tax(net,0.25); at47=after_tax(net,0.47)
    rows.append({"strategy":"   + after-tax @25% / @47%",
                 "CAGR%":stats(at25)["CAGR%"], "vol%":stats(at47)["CAGR%"],
                 "Sharpe":stats(at25)["Sharpe"], "maxDD%":stats(at25)["maxDD%"]})
bh=c.pct_change().fillna(0)
rows.append({"strategy":"buy & hold SPY", **stats(bh)})
out=pd.DataFrame(rows).set_index("strategy")
print(out.to_string(float_format=lambda x: f"{x:.2f}" if pd.notna(x) else ""))
print("\n(' + after-tax' row: CAGR% col = @25% net CAGR ; vol% col here reused = @47% net CAGR)")
