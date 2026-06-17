# Broadens the swing universe. Run LOCALLY after free_fetch.py, then re-run swing_basket.py
# (it auto-discovers every bars_*_1d.parquet). Daily, dividend-adjusted, max history.
from free_fetch import fetch, save

# 77 large-cap stocks, spread across all 11 sectors (NOT tech-only) to avoid a single-bet basket.
# HONEST CAVEAT: these are today's survivors -> survivorship-biased. They broaden the test but
# also flatter it; anchor trust on the ETFs below, which are survivorship-free.
STOCKS = [
    "AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","CSCO","INTC","AMD","QCOM","TXN","IBM","NOW","INTU",  # tech
    "META","NFLX","DIS","CMCSA","TMUS",                                                                    # comm
    "AMZN","TSLA","NKE","SBUX","BKNG","GM",                                                                # cons disc
    "COST","PM","MO","MDLZ","GIS","KMB",                                                                   # staples
    "UNH","JNJ","LLY","ABBV","TMO","ABT","DHR","BMY","AMGN","GILD","CVS",                                  # health
    "BLK","SCHW","SPGI","CB","PNC","USB",                                                                  # financials
    "BA","HON","UNP","GE","RTX","LMT","MMM",                                                               # industrials
    "COP","SLB","EOG","OXY",                                                                               # energy
    "LIN","APD","SHW","NEM","FCX",                                                                         # materials
    "NEE","D","AEP",                                                                                       # utilities
    "AMT","PLD","EQIX","O",                                                                                # REITs
]

# 45 survivorship-free ETFs: industries, countries, factors, bonds. These STRENGTHEN the test.
ETFS = [
    "SMH","SOXX","IGV","XBI","IBB","KRE","ITB","XHB","XRT","OIH","XME","KIE","IYR","IYT","IYZ",            # industries
    "EWJ","EWU","EWZ","EWW","EWT","EWY","EWH","INDA","FXI","EWP","EWI","EWL","EWN","EWS","VGK","VWO",      # countries
    "MTUM","QUAL","VLUE","USMV","VIG","VYM","SDY","DVY",                                                   # factors
    "LQD","HYG","AGG","SHY","TIP","EMB",                                                                   # bonds
]

UNIVERSE = STOCKS + ETFS
ok=fail=0
for sym in UNIVERSE:
    name=sym.replace("=","").replace("^","").replace("-","").replace(".","")
    try:
        df=fetch(sym, period="max", interval="1d")
        if df.empty: print(f"  FAIL {sym}: empty"); fail+=1; continue
        save(df, f"bars_{name}_1d.parquet"); ok+=1
    except Exception as e:
        print(f"  FAIL {sym}: {e}"); fail+=1
print(f"\nDone. {ok} pulled, {fail} failed. Now re-run: python3 swing_basket.py")
