from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd

DATA = "."
CORE = {"SPY": 0.60, "IEF": 0.40}      # passive-core proxy (balanced). Edit to taste.
SLEEVE_W = 0.20                          # how much of the book the overlay occupies
SLEEVES = ["SPY", "EFA", "EEM", "VNQ", "GLD", "DBC", "TLT", "IEF"]
RF_ANNUAL = 0.03

CRISES = {
    "GFC 2008":  ("2007-10-01", "2009-03-31"),
    "COVID 2020": ("2020-02-19", "2020-04-30"),
    "Bear 2022": ("2022-01-03", "2022-10-31"),
}


def load_prices() -> pd.DataFrame:
    series = {}
    for path in glob.glob(os.path.join(DATA, "bars_*_1d.parquet")):
        key = os.path.basename(path)[len("bars_"):-len("_1d.parquet")]
        d = pd.read_parquet(path)[["ts", "close"]].copy()
        d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None).dt.normalize()
        series[key] = d.set_index("ts")["close"]
    px = pd.DataFrame(series).sort_index()
    return px


def metrics(ret: pd.Series) -> dict:
    r = ret.dropna()
    if len(r) < 60:
        return {}
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - RF_ANNUAL) / vol if vol > 0 else 0
    downside = r[r < 0].std() * np.sqrt(252)
    sortino = (r.mean() * 252 - RF_ANNUAL) / downside if downside > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    return {"start": r.index[0].date(), "CAGR%": cagr * 100, "vol%": vol * 100,
            "Sharpe": sharpe, "Sortino": sortino, "maxDD%": dd * 100,
            "Calmar": calmar}


def combined_returns(px: pd.DataFrame, overlay: pd.Series) -> pd.Series:
    rets = px.pct_change(fill_method=None)
    core = sum(w * rets[s] for s, w in CORE.items() if s in rets)
    return (1 - SLEEVE_W) * core + SLEEVE_W * overlay


def main():
    px = load_prices()
    print(f"loaded {px.shape[1]} series, {px.index.min().date()} -> {px.index.max().date()}\n")
    rets = px.pct_change(fill_method=None)

    core_ret = sum(w * rets[s] for s, w in CORE.items() if s in rets)
    basket = rets[[s for s in SLEEVES if s in rets]].mean(axis=1)

    overlays = {"core only (no sleeve)": core_ret}
    for name, ov in [("+ BIL (cash)", rets.get("BIL")),
                     ("+ GLD (gold)", rets.get("GLD")),
                     ("+ static basket", basket),
                     ("+ DBMF", rets.get("DBMF")),
                     ("+ KMLM", rets.get("KMLM"))]:
        if ov is not None:
            overlays[name] = combined_returns(px, ov)

    print("=== USD terms: passive core (60/40) blended 80/20 with each overlay ===")
    rows = []
    for name, r in overlays.items():
        m = metrics(r)
        if m:
            ov_key = name.split()[1] if "+" in name else None
            corr = (rets.get(ov_key).corr(core_ret) if ov_key in rets else np.nan) if ov_key else np.nan
            m["overlay~core_corr"] = corr
            rows.append({"portfolio": name, **m})
    df = pd.DataFrame(rows).set_index("portfolio")
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n=== Crisis-window max drawdown (%) — blank = overlay had no data then ===")
    crow = []
    for name, r in overlays.items():
        row = {"portfolio": name}
        for cname, (a, b) in CRISES.items():
            w = r.loc[(r.index >= a) & (r.index <= b)].dropna()
            if len(w) > 20:
                eq = (1 + w).cumprod()
                row[cname] = (eq / eq.cummax() - 1).min() * 100
            else:
                row[cname] = np.nan
        crow.append(row)
    print(pd.DataFrame(crow).set_index("portfolio").to_string(float_format=lambda x: f"{x:.1f}"))

    fx_keys = [c for c in px.columns if c.startswith("USDILS")]
    if fx_keys:
        print("\n=== ILS terms: same portfolios converted to shekels (Codex's FX point) ===")
        fx = px[fx_keys[0]].reindex(px.index).ffill()
        fxret = fx.pct_change()
        irows = []
        for name, r in overlays.items():
            ils = (1 + r) * (1 + fxret) - 1          # USD return compounded with USD/ILS move
            m = metrics(ils)
            if m:
                irows.append({"portfolio": name, "Sharpe_ILS": m["Sharpe"], "maxDD%_ILS": m["maxDD%"]})
        print(pd.DataFrame(irows).set_index("portfolio").to_string(float_format=lambda x: f"{x:.2f}"))
        print("Note: '+ BIL (cash)' in ILS is NOT risk-free — it carries USD/ILS risk.")

    print("\n=== Read ===")
    print("If '+ GLD', '+ static basket', or '+ DBMF/KMLM' already give most of the")
    print("crisis-drawdown reduction and a better combined Sharpe than 'core only',")
    print("then a custom monthly momentum strategy must beat THOSE, after tax, to justify building.")


if __name__ == "__main__":
    main()
