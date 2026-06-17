#!/usr/bin/env python3
"""universe.py — the EXPANDED candidate universe + a sector map + a diversity-capped picker.

Why this exists: with only ~80 ETFs (a dozen of them volatile enough to ever rank), the scanners
kept surfacing the SAME names. Fix = bigger universe (100 stocks + 80 ETFs) AND a per-sector cap so
a single hot theme (semis, oil) can't take all the slots. NOT hardcoded picks — this just widens the
pool and forces spread. Leverage excluded.
"""
from __future__ import annotations
import os, glob

LEVERAGED = {"SOXL","TECL","TQQQ","QLD","SSO","UPRO","SPXL","FAS","TNA","UDOW","ROM"}

# sector/theme map — used to cap picks per theme so output diversifies
SECTOR = {
    # --- ETFs ---
    "SPY":"US-broad","QQQ":"US-broad","IWM":"US-broad","IVV":"US-broad","DIA":"US-broad",
    "XLK":"tech","SMH":"semis","SOXX":"semis","IGV":"software","VGT":"tech",
    "XLF":"financials","KBE":"financials","KRE":"financials","KIE":"financials",
    "XLE":"energy","XOP":"energy","OIH":"energy","USO":"oil","BNO":"oil","DBC":"commodities","DBMF":"mgd-futures","KMLM":"mgd-futures",
    "XLV":"health","IBB":"biotech","XBI":"biotech","XLP":"staples","XLY":"discretionary","XRT":"retail","ITB":"homebuild","XHB":"homebuild",
    "XLI":"industrials","IYT":"transport","XLU":"utilities","XLB":"materials","XME":"metals","GDX":"goldminers","GDXJ":"goldminers","SIL":"silverminers",
    "GLD":"gold","SLV":"silver","IYR":"reits","VNQ":"reits","IYZ":"telecom",
    "TLT":"bonds","IEF":"bonds","SHY":"bonds","AGG":"bonds","LQD":"bonds","HYG":"bonds","TIP":"bonds","EMB":"bonds","BIL":"bonds",
    "EFA":"intl-dev","VGK":"europe","EWG":"europe","EWQ":"europe","EWU":"europe","EWI":"europe","EWP":"europe","EWL":"europe","EWN":"europe",
    "EEM":"em","VWO":"em","FXI":"china","INDA":"india","EWZ":"latam","EWW":"latam","EWJ":"japan","EWH":"asia","EWS":"asia","EWT":"taiwan","EWY":"korea","EWA":"apac","EWC":"canada",
    "MTUM":"factor","QUAL":"factor","VLUE":"factor","USMV":"factor","VIG":"dividend","VYM":"dividend","SDY":"dividend","DVY":"dividend",
}
# individual stocks -> sector (the ~100 large caps on disk)
STOCK_SECTOR = {
    **{t:"tech" for t in ["AAPL","MSFT","ORCL","CRM","ADBE","CSCO","IBM","NOW","INTU","AMD","NVDA","AVGO","QCOM","TXN","INTC"]},
    **{t:"comm" for t in ["META","NFLX","DIS","CMCSA","TMUS","T","VZ"]},
    **{t:"discretionary" for t in ["AMZN","TSLA","HD","NKE","SBUX","MCD","LOW","BKNG","GM","TGT"]},
    **{t:"staples" for t in ["COST","WMT","PG","KO","PEP","MO","PM","MDLZ","GIS","CL","KMB"]},
    **{t:"health" for t in ["UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","ABT","DHR","BMY","AMGN","GILD","CVS"]},
    **{t:"financials" for t in ["JPM","BAC","WFC","GS","MS","C","BLK","SCHW","SPGI","AXP","COF","USB","PNC"]},
    **{t:"industrials" for t in ["BA","HON","UNP","GE","RTX","LMT","MMM","CAT","DE","UPS","FDX"]},
    **{t:"energy" for t in ["XOM","CVX","COP","SLB","EOG","OXY"]},
    **{t:"materials" for t in ["LIN","APD","SHW","NEM","FCX"]},
    **{t:"utilities" for t in ["NEE","DUK","SO","D","AEP"]},
    **{t:"reits" for t in ["AMT","PLD","EQIX","O"]},
}
SECTOR.update(STOCK_SECTOR)


def all_symbols(include_stocks=True):
    """Every tradeable symbol on disk (leverage + non-equity excluded), optionally with stocks."""
    syms = []
    for f in glob.glob("bars_*_1d.parquet"):
        s = os.path.basename(f)[len("bars_"):-len("_1d.parquet")]
        if s in LEVERAGED: continue
        if s in {"VIX","IRX","USDILSX"}: continue          # not tradeable
        if not include_stocks and s in STOCK_SECTOR: continue
        syms.append(s)
    return sorted(set(syms))


def diversify(ranked_syms, top_n=5, max_per_sector=2):
    """From a ranked list (best first), pick top_n but allow at most max_per_sector from one theme.
    This is what stops '5 semis / 4 oil' — it forces spread across sectors."""
    picks, counts = [], {}
    for s in ranked_syms:
        sec = SECTOR.get(s, "other")
        if counts.get(sec, 0) >= max_per_sector:
            continue
        picks.append(s); counts[sec] = counts.get(sec, 0) + 1
        if len(picks) >= top_n:
            break
    return picks


if __name__ == "__main__":
    u = all_symbols()
    print(f"universe: {len(u)} symbols ({len([s for s in u if s in STOCK_SECTOR])} stocks, "
          f"{len([s for s in u if s not in STOCK_SECTOR])} ETFs), leverage excluded")
    print("sectors covered:", len(set(SECTOR.values())))
    demo = ["SOXX","SMH","XLK","NVDA","AMD","USO","BNO","XLE","JPM","GLD"]
    print("diversify demo (max 2/sector):", diversify(demo, 5, 2))
