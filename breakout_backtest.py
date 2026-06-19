#!/usr/bin/env python3
"""breakout_backtest.py — PHASE 0: LEARN the blow-up fingerprint from labeled history (the user's angle).

Instead of hand-guessing what a pre-explosion stock looks like, learn it from stocks we KNOW blew up:
  1. WINNERS = labeled blow-ups, each with its pre-explosion (trough) date.
  2. CONTROLS = same-era names that did NOT blow up (the discriminator guard vs survivorship storytelling).
  3. rewind_features(ticker, asof) -> a POINT-IN-TIME feature vector (NO future leak): everything is
     computed using only data up to `asof`.
  4. mine_signature(winners, controls) -> which features actually DISCRIMINATE winners from controls,
     with simple explainable weights + a winners-vs-controls hit-rate. Writes breakout_pattern.json.

PRICE/FUNDAMENTAL features are computed here offline from the parquet bars (fully point-in-time).
NEWS/ANALYST features come from a cache (breakout_news_cache.json) that the Claude session fills by
calling bigdata_search with filters.timestamp.end = asof (verified leak-free this session). If the
cache is missing a (ticker, asof) the news features are left as None and the miner ignores them.

HONEST LIMITS (printed loudly): small labeled N; point-in-time news clean only back to ~2022; this learns
a DISCRETIONARY research signature, NOT a permutation-tested statistical edge.

Run:  python3 breakout_backtest.py            # mine + report using whatever is cached
      python3 breakout_backtest.py --emit-news-plan   # print the bigdata_search calls the session should run
"""
from __future__ import annotations
import os, json, argparse, math
import numpy as np, pandas as pd
from jump_model import load_close

DATA = "."
CACHE = "breakout_news_cache.json"
OUT = "breakout_pattern.json"

# --- labeled set: (ticker, pre-explosion trough date) ; date = roughly BEFORE the run began ---
WINNERS = {
    "MU":   "2022-11-30",   # ~$50 trough before the AI-memory run to >$1000
    "INTC": "2025-08-29",   # foundry/turnaround base before the run
    "DELL": "2023-05-31",   # pre AI-server run
    "NVDA": "2022-10-31",   # post-2022 bottom before the AI explosion
    "AVGO": "2022-10-31",   # same AI-cycle base
    "SMCI": "2022-12-30",   # before the AI-server parabola
    "PLTR": "2022-12-30",   # ~$6 base before the multi-bagger
}
# controls: same era, liquid, did NOT blow up from these dates (stayed flat / fell / mild)
CONTROLS = {
    "INTC_2021": ("INTC", "2021-01-29"),   # INTC was dead money 2021-2024
    "T":         ("T",    "2022-11-30"),
    "VZ":        ("VZ",   "2022-11-30"),
    "PFE":       ("PFE",  "2022-12-30"),
    "KO":        ("KO",   "2022-11-30"),
    "WMT":       ("WMT",  "2022-11-30"),
    "CVX":       ("CVX",  "2023-05-31"),
    "IBM":       ("IBM",  "2022-10-31"),
}
FWD_WINDOW = 378  # ~18 months of trading days to measure the "did it blow up" label (sanity check only)


def _asof_close(sym, asof):
    s = load_close(sym)
    if s is None or s.empty: return None
    s = s[s.index <= pd.Timestamp(asof)]
    return s if len(s) > 60 else None


def price_features(sym, asof):
    """Point-in-time price/technical features as of `asof`."""
    s = _asof_close(sym, asof)
    if s is None: return None
    px = s.iloc[-1]
    win = s.iloc[-504:] if len(s) >= 504 else s
    lo, hi = win.min(), win.max()
    base_pos = (px - lo) / (hi - lo) if hi > lo else 0.5     # 0=at lows (beaten), 1=at highs
    dd_from_high = px / hi - 1                                # how far below its own high
    ret_3m = px / s.iloc[-63] - 1 if len(s) >= 63 else 0.0
    ret_12m = px / s.iloc[-252] - 1 if len(s) >= 252 else 0.0
    vol_3m = s.pct_change().iloc[-63:].std() * math.sqrt(252) if len(s) >= 64 else None
    return {"base_pos": float(base_pos), "dd_from_high": float(dd_from_high),
            "ret_3m": float(ret_3m), "ret_12m": float(ret_12m),
            "vol_ann": float(vol_3m) if vol_3m is not None else None}


def forward_label(sym, asof, thresh=1.5, window=FWD_WINDOW):
    """Did it actually blow up after asof? (sanity check on our labels, NOT a feature.)"""
    s = load_close(sym)
    if s is None: return None
    fut = s[s.index > pd.Timestamp(asof)].iloc[:window]
    base = _asof_close(sym, asof)
    if base is None or fut.empty: return None
    run = fut.max() / base.iloc[-1] - 1
    return {"fwd_max_run": float(run), "blew_up": bool(run >= thresh)}


def load_news_cache():
    if os.path.exists(CACHE):
        try: return json.load(open(CACHE))
        except Exception: return {}
    return {}


def news_features(key):
    """Point-in-time news/analyst features from the session-filled cache (None if absent)."""
    c = load_news_cache().get(key)
    if not c: return {"news_neg": None, "news_capitulation": None, "analyst_split": None,
                      "cheap_mentions": None}
    # cache stores counts the session extracted from bigdata_search (timestamp.end=asof)
    return {"news_neg": c.get("neg_headlines"),
            "news_capitulation": c.get("capitulation"),     # downgrades/layoffs/loss = trough signature
            "analyst_split": c.get("analyst_split"),          # mix of buy+hold/sell = contested = pre-run
            "cheap_mentions": c.get("cheap_mentions")}        # "deeply discounted"/"cheap" mentions


def rewind_features(sym, asof, key):
    pf = price_features(sym, asof)
    if pf is None: return None
    feats = dict(pf); feats.update(news_features(key))
    feats["_label"] = forward_label(sym, asof)
    return feats


def mine_signature(winners, controls):
    """Mean-difference discrimination per feature; sign + magnitude = weight. Explainable on purpose."""
    feat_names = ["base_pos", "dd_from_high", "ret_12m", "vol_ann",
                  "news_neg", "news_capitulation", "analyst_split", "cheap_mentions"]
    sig = {}
    for f in feat_names:
        w = [d[f] for d in winners if d.get(f) is not None]
        c = [d[f] for d in controls if d.get(f) is not None]
        if len(w) < 2 or len(c) < 2:
            sig[f] = {"available": False}; continue
        wm, cm = float(np.mean(w)), float(np.mean(c))
        pooled = float(np.std(w + c)) or 1e-9
        effect = (wm - cm) / pooled            # standardized mean diff (Cohen's d-ish)
        sig[f] = {"available": True, "winner_mean": round(wm, 3), "control_mean": round(cm, 3),
                  "effect": round(effect, 3), "direction": "higher-in-winners" if effect > 0 else "lower-in-winners"}
    return sig


def score_with_signature(feats, sig):
    """Apply learned standardized weights to a feature vector -> a single discriminant score."""
    tot, used = 0.0, 0
    for f, s in sig.items():
        if not s.get("available") or feats.get(f) is None: continue
        # z the feature toward the winner direction using winner/control means as anchors
        anchor = (s["winner_mean"] + s["control_mean"]) / 2
        spread = abs(s["winner_mean"] - s["control_mean"]) or 1e-9
        z = (feats[f] - anchor) / spread
        tot += z * s["effect"]; used += 1
    return tot / used if used else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-news-plan", action="store_true")
    args = ap.parse_args()

    keys_w = {t: (t, d) for t, d in WINNERS.items()}
    if args.emit_news_plan:
        print("# bigdata_search calls the session should run (timestamp.end = asof, max_chunks ~30):")
        for label, (sym, d) in {**keys_w, **CONTROLS}.items():
            print(f'  {label}: entity={sym} end={d}T23:59:59Z  query="news, analyst ratings and outlook for {sym} around {d}"')
        print(f"\n# write results into {CACHE} as: "
              '{"<label>": {"neg_headlines":int,"capitulation":int,"analyst_split":int,"cheap_mentions":int}}')
        return

    print("=" * 78)
    print("PHASE 0 — LEARN the blow-up fingerprint from labeled winners vs controls")
    print("SPECULATIVE discretionary research — NOT a permutation-tested edge. Small N. News PIT to ~2022.")
    print("=" * 78)

    wfeats, cfeats = [], []
    print("\nWINNERS (rewound to pre-explosion date):")
    for sym, d in WINNERS.items():
        f = rewind_features(sym, d, sym)
        if f is None: print(f"  {sym:6s} {d}  -- no price history, skipped"); continue
        wfeats.append(f); lab = f["_label"]
        run = f"{lab['fwd_max_run']*100:+.0f}% / {'BLEW UP' if lab['blew_up'] else 'no'}" if lab else "n/a"
        news = "news✓" if f["news_neg"] is not None else "news–pending"
        print(f"  {sym:6s} {d}  base_pos={f['base_pos']:.2f} dd={f['dd_from_high']*100:+.0f}% "
              f"12m={f['ret_12m']*100:+.0f}%  fwd:{run}  {news}")

    print("\nCONTROLS (same era, did NOT blow up):")
    for label, (sym, d) in CONTROLS.items():
        f = rewind_features(sym, d, label)
        if f is None: print(f"  {label:10s} {sym} {d}  -- no history, skipped"); continue
        cfeats.append(f); lab = f["_label"]
        run = f"{lab['fwd_max_run']*100:+.0f}%" if lab else "n/a"
        print(f"  {label:10s} {sym:5s} {d}  base_pos={f['base_pos']:.2f} dd={f['dd_from_high']*100:+.0f}% "
              f"12m={f['ret_12m']*100:+.0f}%  fwd:{run}")

    sig = mine_signature(wfeats, cfeats)
    print("\n--- LEARNED DISCRIMINATING SIGNATURE (winners vs controls) ---")
    print(f"{'feature':18s} {'winner':>9s} {'control':>9s} {'effect':>8s}  interpretation")
    ranked = sorted([(f, s) for f, s in sig.items() if s.get("available")],
                    key=lambda x: -abs(x[1]["effect"]))
    for f, s in ranked:
        print(f"{f:18s} {s['winner_mean']:>9.2f} {s['control_mean']:>9.2f} {s['effect']:>8.2f}  {s['direction']}")
    pending = [f for f, s in sig.items() if not s.get("available")]
    if pending: print(f"  (pending news cache — run --emit-news-plan then fill {CACHE}): {pending}")

    # winners-vs-controls hit-rate: does the learned score separate them?
    ws = [score_with_signature(f, sig) for f in wfeats]; ws = [x for x in ws if x is not None]
    cs = [score_with_signature(f, sig) for f in cfeats]; cs = [x for x in cs if x is not None]
    if ws and cs:
        thr = (np.median(ws) + np.median(cs)) / 2
        hit = (np.mean([w > thr for w in ws]) + np.mean([c <= thr for c in cs])) / 2
        print(f"\nWinners-vs-controls separation: winner score median={np.median(ws):+.2f}, "
              f"control median={np.median(cs):+.2f}, balanced hit-rate={hit*100:.0f}%")
    else:
        hit = None; print("\n(can't compute hit-rate yet — need news cache or more features)")

    out = {"_disclaimer": "SPECULATIVE learned signature, NOT a permutation-tested edge. Small N; news PIT to ~2022.",
           "winners": list(WINNERS), "controls": list(CONTROLS),
           "signature": sig, "hit_rate": (round(float(hit), 3) if hit is not None else None)}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
