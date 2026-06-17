# Multi-instrument swing: RSI(2)<10 + 200d-uptrend across every bars_<SYM>_1d.parquet.
# Honesty fixes: (1) Sharpe is EXCESS-of-rf (kills the low-vol cash-like illusion, e.g. SHY).
# (2) Baskets split ETF-only (survivorship-FREE = trustworthy) vs all vs stock-only.
from __future__ import annotations
import glob, math, os
import numpy as np, pandas as pd

DATA="."; COST_BPS_RT=3.0; RF=0.025; MAXHOLD=10
EXCLUDE={"USDILSX","VIX","IRX","BIL","DBMF","KMLM","SHY"}   # non-tradeable / cash-like (SHY ~= cash)

# Survivorship-free instruments (sector/industry/country/factor/bond/commodity/index ETFs).
# Everything NOT in here is treated as an individual stock (survivorship-biased).
ETF={"SPY","IVV","QQQ","IWM","IJR",
 "XLK","XLF","XLV","XLE","XLP","XLY","XLI","XLU","XLB","VGT",
 "SMH","SOXX","IGV","XBI","IBB","KRE","ITB","XHB","XRT","OIH","XME","KIE","IYR","IYT","IYZ","KBE","VNQ",
 "EFA","EEM","VGK","VWO","EWJ","EWU","EWZ","EWW","EWT","EWY","EWH","INDA","FXI","EWP","EWI","EWL","EWN","EWS","EWA","EWC","EWG","EWQ",
 "MTUM","QUAL","VLUE","USMV","VIG","VYM","SDY","DVY",
 "TLT","IEF","LQD","HYG","AGG","TIP","EMB",
 "GLD","SLV","GDX","GDXJ","SIL","DBC","USO","BNO","XOP"}

def rsi(s,n):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    return 100-100/(1+up.ewm(alpha=1/n,adjust=False).mean()/dn.ewm(alpha=1/n,adjust=False).mean())

def strat_returns(c):
    r2=rsi(c,2); sma5=c.rolling(5).mean(); sma200=c.rolling(200).mean()
    entry=((r2<10)&(c>sma200)).values; exit_=(c>sma5).values
    n=len(c); pos=np.zeros(n); holding=False; et=None
    for t in range(1,n):
        pos[t]=1.0 if holding else 0.0
        if not holding and entry[t]: holding=True; et=t
        elif holding and (exit_[t] or (t-et)>=MAXHOLD): holding=False
    pos=pd.Series(pos,index=c.index); ret=c.pct_change().fillna(0); turn=pos.diff().abs().fillna(0)
    return pos*ret + (1-pos)*(RF/252) - turn*(COST_BPS_RT/1e4)/2

def stats(r,ppy=252):
    r=r.dropna(); eq=(1+r).cumprod(); vol=r.std()*math.sqrt(ppy)
    return {"CAGR%":(eq.iloc[-1]**(ppy/len(r))-1)*100,
            "Sharpe":(r.mean()*ppy-RF)/vol if vol>0 else 0,        # EXCESS of rf
            "maxDD%":(eq/eq.cummax()-1).min()*100,"yrs":len(r)/ppy}

def after_tax(r,rate):
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

def load(p):
    d=pd.read_parquet(p)[["ts","close"]].copy(); d["ts"]=pd.to_datetime(d["ts"]).dt.tz_localize(None)
    return d.sort_values("ts").set_index("ts")["close"]

syms={}
for f in sorted(glob.glob(f"{DATA}/bars_*_1d.parquet")):
    sym=os.path.basename(f)[len("bars_"):-len("_1d.parquet")]
    if sym in EXCLUDE: continue
    try: syms[sym]=strat_returns(load(f))
    except Exception as e: print(f"  skip {sym}: {e}")

netf=sum(1 for s in syms if s in ETF); nstk=len(syms)-netf
print(f"RSI(2)<10 + 200d uptrend | {len(syms)} instruments ({netf} ETF, {nstk} stock) | cost={COST_BPS_RT}bp rf={RF:.1%} | Sharpe=EXCESS\n")
rows=[{"instrument":s,"kind":"ETF" if s in ETF else "stk", **stats(r)} for s,r in syms.items()]
df=pd.DataFrame(rows).sort_values("Sharpe",ascending=False).set_index("instrument")
print("=== per-instrument (excess Sharpe) ===")
print(df.to_string(float_format=lambda x:f"{x:.2f}"))

def basket(keys,label):
    if not keys: return {"basket":label,"n":0,"CAGR%":float("nan"),"Sharpe":float("nan"),"maxDD%":float("nan"),"yrs":float("nan"),"Sharpe@25%":float("nan"),"CAGR%@25%":float("nan")}
    b=pd.DataFrame({k:syms[k] for k in keys}).mean(axis=1).dropna()
    pre=stats(b); at=stats(after_tax(b,0.25))
    return {"basket":label,"n":len(keys),**{f"{k}":pre[k] for k in pre},
            "Sharpe@25%":at["Sharpe"],"CAGR%@25%":at["CAGR%"]}

etf=[s for s in syms if s in ETF]; stk=[s for s in syms if s not in ETF]
print("\n=== baskets (equal-weight) ===")
bk=pd.DataFrame([basket(etf,"ETF-ONLY (survivorship-free = TRUST)"),
                 basket(list(syms),"ALL (etf+stock)"),
                 basket(stk,"STOCK-ONLY (survivor-biased)")]).set_index("basket")
print(bk.to_string(float_format=lambda x:f"{x:.2f}"))
print("\nETF-ONLY is the honest headline: survivorship-free, excess-of-cash Sharpe, full history.")
