#!/usr/bin/env python3
"""value_agent.py — the THIRD agent: cheap stocks with potential, thesis-backed.

⚠ HONESTY FIRST: unlike the reversal agent (a statistically-VALIDATED edge, permutation p<0.01),
this is DISCRETIONARY VALUE RESEARCH — an idea generator, NOT a backtested edge. It scores the
~100 stocks on disk for "cheap + quality" from current fundamentals and gives a rough fair-value
estimate. Treat its output as SPECULATIVE THESIS IDEAS to investigate, not signals to trade blindly.

Two stages:
  STAGE A (this script, local): screen our ~100 stocks → cheapness+quality score + rough fair value.
  STAGE B (cloud/LLM): hunt RECENT IPOs + cheap small-caps off-disk, read filings, write a thesis +
    valuation per name. Needs web search → runs in the cloud agent, writes value_thesis.json.

Run: python3 value_agent.py
"""
from __future__ import annotations
import json, os
import numpy as np, pandas as pd

from fundamentals import fetch_fundamentals
from universe import STOCK_SECTOR, SECTOR

TOP = 12
MAX_PRICE = 100.0          # affordability filter: only show stocks trading <= $100/share
THESIS_FILE = "value_thesis.json"          # Stage-B web thesis verdicts (cloud agent fills this)


def value_score(f: dict) -> dict:
    """Transparent 'cheap WITH quality' score 0-100 + a rough fair-value estimate. Cheap-but-broken
    (value trap) scores LOW; cheap-and-profitable-and-growing scores HIGH."""
    fpe = f.get("forwardPE"); pm = f.get("profitMargins"); roe = f.get("returnOnEquity")
    rg = f.get("revenueGrowth"); peg = f.get("pegRatio"); tgt = f.get("targetMeanPrice")
    px = f.get("currentPrice"); dy = f.get("dividendYield")
    if px is None or (fpe is None and pm is None):
        return {"score": None}
    score, flags, thesis = 0, [], []

    # CHEAP (the value half)
    if fpe is not None and fpe > 0:
        if fpe < 10: score += 26; thesis.append(f"very cheap (fwd P/E {fpe:.0f})")
        elif fpe < 15: score += 18; thesis.append(f"cheap (fwd P/E {fpe:.0f})")
        elif fpe < 22: score += 8
        elif fpe > 40: score -= 12; flags.append(f"expensive (P/E {fpe:.0f})")
    if peg is not None and 0 < peg < 1: score += 10; thesis.append(f"cheap vs growth (PEG {peg:.2f})")
    if dy and dy > 3: score += 6; thesis.append(f"pays {dy:.1f}% dividend")

    # QUALITY (so we don't buy value traps)
    if pm is not None:
        if pm < 0: score -= 30; flags.append(f"UNPROFITABLE ({pm*100:.0f}% margin) — likely a TRAP")
        elif pm > 0.20: score += 18; thesis.append(f"high margins ({pm*100:.0f}%)")
        elif pm > 0.08: score += 8
    if roe is not None:
        if roe > 0.20: score += 14; thesis.append(f"strong ROE ({roe*100:.0f}%)")
        elif roe < 0: score -= 10; flags.append("negative ROE")
    if rg is not None:
        if rg > 0.10: score += 10; thesis.append(f"growing ({rg*100:.0f}%/yr)")
        elif rg < 0: score -= 8; flags.append("shrinking revenue")

    # analyst upside (a sanity tilt, not the thesis)
    upside = (tgt/px - 1) if (tgt and px) else None
    if upside is not None:
        if upside > 0.20: score += 10; thesis.append(f"analysts see +{upside*100:.0f}%")
        elif upside < -0.05: score -= 10; flags.append(f"analyst target {upside*100:.0f}% below price")

    # rough fair value: blend a normalized P/E (15x) reprice and the analyst target
    fv = None
    if fpe and px and fpe > 0:
        repriced = px * (15.0 / fpe)            # what it'd be at a 'fair' 15x
        fv = np.mean([v for v in [repriced, tgt] if v])
    score = max(0, min(100, score))
    return {"score": score, "fwd_PE": (round(fpe,1) if fpe else None),
            "margin%": (round(pm*100,1) if pm is not None else None),
            "ROE%": (round(roe*100,0) if roe is not None else None),
            "upside%": (round(upside*100,0) if upside is not None else None),
            "fair_value": (round(fv,2) if fv else None), "price": (round(px,2) if px else None),
            "fv_gap%": (round((fv/px-1)*100,0) if (fv and px) else None),
            "thesis": "; ".join(thesis) or "—", "flags": "; ".join(flags) or "—"}


def main():
    print("="*70)
    print("  VALUE AGENT  —  cheap stocks w/ potential (SPECULATIVE thesis ideas)")
    print("  ⚠ NOT a backtested edge — discretionary research. Investigate before buying.")
    print("="*70)

    rows = []; over_price = 0
    for t in sorted(STOCK_SECTOR):
        vs = value_score(fetch_fundamentals(t))
        if vs.get("score") is None: continue
        if vs.get("price") and vs["price"] > MAX_PRICE:    # affordability filter (<= $100/share)
            over_price += 1; continue
        rows.append({"ticker": t, "sector": SECTOR.get(t,"?"), **vs})
    df = pd.DataFrame(rows).sort_values("score", ascending=False)

    print(f"\n[A] DATA SCREEN — cheap+quality AND <= ${MAX_PRICE:.0f}/share "
          f"(filtered out {over_price} names priced over ${MAX_PRICE:.0f}); top {TOP} of {len(df)}:")
    show = df.head(TOP)[["ticker","sector","score","fwd_PE","margin%","ROE%","upside%","fv_gap%","thesis"]]
    print(show.to_string(index=False, max_colwidth=42))

    # snapshot for the dashboard (so it doesn't re-fetch fundamentals on every render)
    ideas = [{"ticker": r["ticker"], "sector": r["sector"], "score": int(r["score"]),
              "fv_gap%": r.get("fv_gap%"), "thesis": r["thesis"]} for _, r in df.head(8).iterrows()]
    json.dump({"_asof": str(pd.Timestamp.today().date()), "ideas": ideas}, open("value_ideas.json","w"), indent=1)

    print("\n[A] VALUE TRAPS to AVOID (cheap but broken — low score):")
    traps = df[df["flags"].str.contains("TRAP|UNPROFIT|shrink|negative", na=False)].head(6)
    for _, r in traps.iterrows():
        print(f"   {r['ticker']:5s} score {r['score']:.0f} — {r['flags']}")

    # Stage-B web thesis (cloud agent fills value_thesis.json with IPOs / small-caps + thesis)
    if os.path.exists(THESIS_FILE):
        try:
            tw = json.load(open(THESIS_FILE))
            print("\n[B] WEB THESIS IDEAS (recent IPOs / small-caps, from the cloud agent's read):")
            for name, info in tw.items():
                if name.startswith("_"): continue
                print(f"   {name}: {info.get('verdict','?')} — {info.get('thesis','')[:80]}")
        except Exception: pass
    else:
        print(f"\n[B] WEB LAYER (IPOs / off-disk cheap small-caps): run in the CLOUD agent — it web-searches")
        print(f"    for recent IPOs + cheap names, reads filings, writes thesis+valuation to {THESIS_FILE}.")
        print(f"    (Local python can't browse, so this stage is cloud-only.)")

    print("\n=== READ ===")
    print("- HIGH score = cheap AND profitable/growing (not a value trap). Each line is a STARTING thesis.")
    print("- fv_gap% = rough fair-value vs price (positive = possibly undervalued). Crude, verify yourself.")
    print("- This is IDEAS, not the validated reversal edge. Do your own diligence before any buy.")


if __name__ == "__main__":
    main()
