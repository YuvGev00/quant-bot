#!/usr/bin/env python3
"""breakout_scout.py — PHASE 1: SCREEN today's universe with the LEARNED blow-up fingerprint.

Loads breakout_pattern.json (the signature Phase 0 learned from labeled winners-vs-controls) and scores
every single stock in universe.STOCK_SECTOR by how much TODAY's setup resembles what the winners looked
like at their pre-explosion bottoms: deeply beaten down + near lows + volatile/contested + (Stage-B) the
news capitulation arc. Tags each EARLY (beaten base just starting to kick) vs CONFIRMED (already breaking
out on volume). Writes breakout_shortlist.json + a NEWS_TO_READ list for the Stage-B live read.

Stage A here uses ONLY on-disk bars + the yfinance snapshot, so the news/trajectory factors (capitulation,
negative-headline count, earnings-surprise) are left PENDING — filled live in Stage B from Bigdata.com.
Scores are therefore explicitly PROVISIONAL.

HONEST: SPECULATIVE discretionary research, NOT a backtested edge (cf reversal p=0.000). Individual-stock
survivorship caveat. "Low share price != cheap." News-veto on froth/falling-knives happens in Stage B.

Run:  python3 breakout_scout.py
"""
from __future__ import annotations
import os, json, math
import numpy as np, pandas as pd

from universe import STOCK_SECTOR, SECTOR, diversify
from early_scanner import panels, score_breakout, score_accel, load_ohlcv
from fundamentals import fetch_fundamentals

PATTERN_FILE = "breakout_pattern.json"
OUT = "breakout_shortlist.json"

# a-priori fallback signature if Phase 0 hasn't run (winner-direction effects, hand-set)
DEFAULT_SIG = {
    "vol_ann":      {"available": True, "winner_mean": 0.50, "control_mean": 0.27, "effect": 1.59},
    "dd_from_high": {"available": True, "winner_mean": -0.43, "control_mean": -0.15, "effect": -1.25},
    "base_pos":     {"available": True, "winner_mean": 0.32, "control_mean": 0.64, "effect": -1.08},
}


def load_signature():
    if os.path.exists(PATTERN_FILE):
        try:
            p = json.load(open(PATTERN_FILE))
            return p.get("signature", DEFAULT_SIG), p.get("hit_rate")
        except Exception:
            pass
    return DEFAULT_SIG, None


def price_base(close):
    """Point-in-time price features mirroring breakout_backtest.price_features (consistency is key)."""
    s = close.dropna()
    if len(s) < 60: return None
    px = s.iloc[-1]
    win = s.iloc[-504:] if len(s) >= 504 else s
    lo, hi = win.min(), win.max()
    base_pos = (px - lo) / (hi - lo) if hi > lo else 0.5
    dd = px / hi - 1
    ret_12m = px / s.iloc[-252] - 1 if len(s) >= 252 else 0.0
    vol_ann = s.pct_change().iloc[-63:].std() * math.sqrt(252) if len(s) >= 64 else None
    return {"price": float(px), "base_pos": float(base_pos), "dd_from_high": float(dd),
            "ret_12m": float(ret_12m), "vol_ann": float(vol_ann) if vol_ann is not None else None}


def score_features(feats, sig):
    """Apply the learned signature -> single resemblance score (same math as backtest.score_with_signature)."""
    tot, used = 0.0, 0
    for f, s in sig.items():
        if not s.get("available") or feats.get(f) is None: continue
        anchor = (s["winner_mean"] + s["control_mean"]) / 2
        spread = abs(s["winner_mean"] - s["control_mean"]) or 1e-9
        z = (feats[f] - anchor) / spread
        tot += z * s["effect"]; used += 1
    return (tot / used) if used else None


def stage(brk, accel, base_pos):
    """Stage by where the name is in the blow-up arc:
       CONFIRMED = already breaking out on volume from a base (the move has started, lower-risk).
       EARLY     = deeply beaten / near its lows (the pre-explosion bottom — catch-it-before-the-move).
                   An accel kick is a PLUS (early turn underway) but NOT required: the truest catch
                   (MU at $50, PLTR at $6) had no kick yet — it was just hated and beaten.
       Names mid-range with no breakout and no beaten base aren't candidates."""
    if brk is not None and brk > 0.05 and (base_pos is None or base_pos < 0.95):
        return "CONFIRMED"
    if base_pos is not None and base_pos < 0.35:          # deeply beaten = the winner-bottom zone
        return "EARLY"
    if base_pos is not None and base_pos < 0.50 and accel is not None and accel > 0:
        return "EARLY"                                     # mid-low base WITH a fresh upward kick
    return None


def fingerprint(ticker, sig, brk_latest, accel_latest):
    close = load_ohlcv(ticker)
    if close is None or "close" not in close: return None
    pf = price_base(close["close"])
    if pf is None: return None
    st = stage(brk_latest, accel_latest, pf["base_pos"])
    if st is None: return None                       # not a candidate at all
    resemblance = score_features(pf, sig)
    fund = fetch_fundamentals(ticker) or {}
    fpe = fund.get("forwardPE"); pm = fund.get("profitMargins"); roe = fund.get("returnOnEquity")
    tgt = fund.get("targetMeanPrice"); px = fund.get("currentPrice") or pf["price"]
    rkey = (fund.get("recommendationKey") or "").replace("_", " ")
    nan = fund.get("numberOfAnalystOpinions")
    upside = (tgt / px - 1) if (tgt and px) else None

    thesis = []
    thesis.append(f"{pf['base_pos']*100:.0f}% up its 2yr range" if pf["base_pos"] < 0.45
                  else "breaking to new highs")
    thesis.append(f"{pf['dd_from_high']*100:.0f}% off its high" if pf["dd_from_high"] < -0.15 else "near highs")
    if upside is not None and upside > 0.15: thesis.append(f"analysts see +{upside*100:.0f}%")
    flags = []
    if px and px > 300: flags.append("high share price (not 'cheap' by price)")
    if fpe and fpe > 60: flags.append(f"rich fwd P/E {fpe:.0f}")

    return {"ticker": ticker, "sector": SECTOR.get(ticker, "other"), "stage": st,
            "resemblance": round(resemblance, 3) if resemblance is not None else None,
            "price": round(px, 2) if px else None,
            "base_pos%": round(pf["base_pos"] * 100, 1), "dd%": round(pf["dd_from_high"] * 100, 1),
            "ret_12m%": round(pf["ret_12m"] * 100, 1),
            "vol_ann%": round(pf["vol_ann"] * 100, 1) if pf["vol_ann"] is not None else None,
            "fwd_PE": round(fpe, 1) if fpe else None,
            "margin%": round(pm * 100, 1) if pm is not None else None,
            "roe%": round(roe * 100, 1) if roe is not None else None,
            "analyst": rkey or "—", "n_analysts": int(nan) if nan else 0,
            "upside%": round(upside * 100, 1) if upside is not None else None,
            "thesis": "; ".join(thesis), "flags": flags,
            # trajectory/news factors filled in Stage B:
            "pending_live": ["est_revision_up", "earnings_surprise", "news_capitulation_arc", "competitor_edge"]}


def scan_all():
    """Fingerprint the WHOLE stock universe once; return all candidates ranked by resemblance (desc)."""
    syms = list(STOCK_SECTOR)
    px, vol = panels(syms)
    brk = score_breakout(px, vol).iloc[-1]
    acc = score_accel(px).iloc[-1]
    cands = [fp for s in syms if (fp := fingerprint(s, SIG, brk.get(s), acc.get(s)))]
    cands.sort(key=lambda c: (c["resemblance"] if c["resemblance"] is not None else -9), reverse=True)
    return cands


def sector_cohorts(cands=None):
    """Group ALL beaten-down candidates by sector, ranked within sector — NO diversify cap.
    This is the 'other stocks in the same situation' set each stock page compares against."""
    if cands is None: cands = scan_all()
    cohorts = {}
    for c in cands:                       # cands already resemblance-sorted, so each sector list stays ranked
        cohorts.setdefault(c["sector"], []).append(c)
    return cohorts


def stage_a_shortlist(top=12, cands=None):
    """The diversified top-N picks that go to Stage-B research (<=3 per sector)."""
    if cands is None: cands = scan_all()
    keep = set(diversify([c["ticker"] for c in cands], top_n=top, max_per_sector=3))
    return [c for c in cands if c["ticker"] in keep][:top]


SIG, HIT = load_signature()


def main():
    print("=" * 78)
    print("BREAKOUT SCOUT — Phase 1 screen (LEARNED blow-up fingerprint applied to today's universe)")
    print("SPECULATIVE discretionary research — NOT a backtested edge. Individual-stock survivorship applies.")
    if HIT is not None:
        print(f"Learned signature winners-vs-controls hit-rate: {HIT*100:.0f}% (from breakout_backtest.py)")
    print("=" * 78)
    cands = scan_all()                       # one scan -> both shortlist and cohorts
    short = stage_a_shortlist(cands=cands)
    cohorts = sector_cohorts(cands)
    if not short:
        print("\nNo candidates today (nothing matches the beaten-base or breakout setup)."); return
    asof = pd.Timestamp.now().strftime("%Y-%m-%d")
    print(f"\n{'#':>2} {'TICK':6s} {'STAGE':9s} {'RESEM':>6s} {'price':>8s} {'base%':>6s} {'dd%':>6s} "
          f"{'vol%':>6s} {'analyst':12s} thesis")
    for i, c in enumerate(short, 1):
        print(f"{i:>2} {c['ticker']:6s} {c['stage']:9s} "
              f"{(c['resemblance'] if c['resemblance'] is not None else 0):>6.2f} "
              f"{(c['price'] or 0):>8.2f} {c['base_pos%']:>6.1f} {c['dd%']:>6.1f} "
              f"{(c['vol_ann%'] or 0):>6.1f} {c['analyst'][:12]:12s} {c['thesis']}")
    news = [c["ticker"] for c in short]
    print(f"\nNEWS_TO_READ (Stage B — pull Bigdata tearsheets + point-in-time-style news read): {','.join(news)}")
    print("Factors awaiting the Stage-B live read on each:", short[0]["pending_live"])
    out = {"_asof": asof,
           "_disclaimer": "SPECULATIVE discretionary research, NOT a backtested edge (cf reversal p=0.000). "
                          "Stage-A scores are PROVISIONAL — news/earnings/estimate-trajectory factors are "
                          "filled live in Stage B. Individual-stock survivorship applies. Low price != cheap.",
           "learned_hit_rate": HIT, "candidates": short, "cohorts": cohorts}
    json.dump(out, open(OUT, "w"), indent=1)
    ncoh = sum(len(v) for v in cohorts.values())
    print(f"\nwrote {OUT} ({len(short)} picks + {ncoh} candidates across {len(cohorts)} sector cohorts)")


if __name__ == "__main__":
    main()
