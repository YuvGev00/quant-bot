#!/usr/bin/env python3
"""breakout_screen.py — the Phase-1 Stage-A "blow-up fingerprint" screen.

GOAL: find BEATEN-DOWN stocks that look like they could blow up (the MU / Intel / Dell
-before-they-ran archetype): deeply off their highs, basing near the lows, but showing the
EARLY signs of a turn — relative momentum improving, reclaiming a moving average, volume
picking up. This is NOT momentum-chasing (those names are already up); it is the opposite —
catch the inflection while the name is still hated.

HONEST METHODOLOGY (no overclaiming):
- The "fingerprint" is LEARNED from the on-disk daily history: we scan every symbol's past,
  find episodes where a name was >35% off its 1y high and then ran +40%+ over the next 6
  months, and average the feature vector AT THE TRIGGER. That centroid (+ spread) is the
  fingerprint. A current candidate is scored by weighted Gaussian similarity to it.
- SURVIVORSHIP CAVEAT: the universe on disk is today's survivors. Names that were beaten down
  and then DELISTED are absent, so the learned pattern is optimistic. This is a research LEAD
  generator, NOT a backtested edge. Treat every name as a starting thesis to investigate.
- Fundamentals are NOT point-in-time here — Stage A is pure price/volume structure. The live
  fundamental + news read (Stage B) is what actually confirms or vetoes a name.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

PATTERN_FILE = "breakout_pattern.json"

# --- the feature set: every value is computable from daily close+volume, point-in-time-safe
#     (only trailing windows are used). Order matters for the stored centroid/scale arrays.
FEATURES = [
    "dd_from_high",   # drawdown from trailing 252d high (negative; deeply negative = beaten down)
    "base_pos",       # position in the 52w range (0=at the low, 1=at the high)
    "ret_1m",         # 21d return (recent move)
    "ret_3m",         # 63d return (the trailing-quarter move)
    "accel",          # ret_1m - ret_3m  (recent pace exceeding the trailing trend = turning up)
    "above_ma50",     # px/ma50 - 1   (reclaiming the 50d = early-turn confirmation)
    "above_ma200",    # px/ma200 - 1  (still below the 200d for a true beaten-down name)
    "ma50_slope",     # 21d slope of the 50d MA (flattening / curling up out of the base)
    "vol_surge",      # mean(vol,20)/mean(vol,63)  (>1 = accumulation / interest returning)
    "base_age",       # days since the 63d low / 63  (>0 = has stopped making fresh lows)
]

# weights: how much each feature matters to "this looks like a pre-run base". Beaten-down-ness
# and the early-turn signals carry the most weight; volume and base-age are confirmation.
WEIGHTS = {
    "dd_from_high": 1.6, "base_pos": 1.3, "ret_1m": 0.8, "ret_3m": 0.7, "accel": 1.2,
    "above_ma50": 1.3, "above_ma200": 0.6, "ma50_slope": 1.1, "vol_surge": 0.9, "base_age": 0.7,
}

MIN_BARS = 300
BEATEN_DOWN_DD = -0.20   # a name must be at least 20% off its 1y high to count as "beaten down"


def _feat_at(close: np.ndarray, vol: np.ndarray, i: int) -> dict | None:
    """Feature vector computed using ONLY data up to and including index i (point-in-time safe)."""
    if i < 252:
        return None
    c = close[: i + 1]
    v = vol[: i + 1]
    px = c[-1]
    win = c[-252:]
    hi, lo = float(np.nanmax(win)), float(np.nanmin(win))
    if not np.isfinite(px) or hi <= 0 or hi == lo:
        return None
    ma50 = float(np.nanmean(c[-50:]))
    ma50_prev = float(np.nanmean(c[-71:-21])) if len(c) >= 71 else ma50
    ma200 = float(np.nanmean(c[-200:]))
    v20 = float(np.nanmean(v[-20:])); v63 = float(np.nanmean(v[-63:]))
    low63_idx = int(np.nanargmin(c[-63:]))
    base_age = (62 - low63_idx) / 63.0          # 0 if the low is today, ~1 if the low was 63d ago
    return {
        "dd_from_high": px / hi - 1.0,
        "base_pos": (px - lo) / (hi - lo),
        "ret_1m": px / c[-22] - 1.0 if len(c) >= 22 else 0.0,
        "ret_3m": px / c[-64] - 1.0 if len(c) >= 64 else 0.0,
        "accel": (px / c[-22] - 1.0) - (px / c[-64] - 1.0) if len(c) >= 64 else 0.0,
        "above_ma50": px / ma50 - 1.0 if ma50 > 0 else 0.0,
        "above_ma200": px / ma200 - 1.0 if ma200 > 0 else 0.0,
        "ma50_slope": (ma50 - ma50_prev) / ma50_prev if ma50_prev > 0 else 0.0,
        "vol_surge": v20 / v63 if v63 > 0 else 1.0,
        "base_age": base_age,
    }


def learn_fingerprint(px: pd.DataFrame, vol: pd.DataFrame,
                      fwd_days: int = 126, run_thresh: float = 0.40,
                      trigger_dd: float = -0.35) -> dict:
    """LEARN the blow-up fingerprint from history. An 'episode' = a bar where the name was at least
    `trigger_dd` off its 1y high AND went on to run +`run_thresh` over the next `fwd_days`. We
    average the feature vector at those trigger bars -> the centroid; the spread -> the scale.
    Sampled every ~15 sessions and de-duplicated so one big run is not counted many times."""
    vecs = []
    episodes = 0
    for sym in px.columns:
        s = px[sym].dropna()
        c = s.values.astype(float)
        v = vol[sym].reindex(s.index).ffill().values.astype(float)
        n = len(c)
        last_used = -999
        for i in range(252, n - fwd_days, 15):
            if not np.isfinite(c[i]) or i - last_used < 63:
                continue
            f = _feat_at(c, v, i)
            if f is None or f["dd_from_high"] > trigger_dd:
                continue
            fwd = c[i + fwd_days] / c[i] - 1.0
            if np.isfinite(fwd) and fwd >= run_thresh:
                vecs.append([f[k] for k in FEATURES])
                last_used = i
                episodes += 1
    if len(vecs) < 8:
        # Fallback: a hand-specified archetype centroid (still honest — documented below).
        centroid = {"dd_from_high": -0.45, "base_pos": 0.18, "ret_1m": 0.04, "ret_3m": -0.02,
                    "accel": 0.06, "above_ma50": 0.03, "above_ma200": -0.18, "ma50_slope": 0.01,
                    "vol_surge": 1.15, "base_age": 0.45}
        scale = {k: max(abs(centroid[k]) * 0.6, 0.05) for k in FEATURES}
        method = "archetype-specified (too few historical episodes found to fit)"
    else:
        arr = np.array(vecs)
        centroid = {k: float(np.nanmean(arr[:, j])) for j, k in enumerate(FEATURES)}
        scale = {k: float(max(np.nanstd(arr[:, j]), 0.03)) for j, k in enumerate(FEATURES)}
        method = (f"learned from {episodes} historical blow-up episodes "
                  f"(>{int(-trigger_dd*100)}% off 1y high, then +{int(run_thresh*100)}% in "
                  f"{fwd_days} sessions) across {len(px.columns)} on-disk names")
    return {
        "_what": "Blow-up fingerprint: the price/volume shape of a beaten-down name at the bar it "
                 "began a major run. A current candidate is scored by weighted similarity to this.",
        "_method": method,
        "_caveat": "SURVIVORSHIP-BIASED: on-disk universe = today's survivors; beaten-down names that "
                   "delisted are absent, so the pattern is optimistic. A research LEAD generator, NOT "
                   "a backtested edge.",
        "_asof": str(px.index[-1].date()),
        "features": FEATURES,
        "weights": WEIGHTS,
        "centroid": centroid,
        "scale": scale,
        "beaten_down_dd": BEATEN_DOWN_DD,
    }


def load_or_learn(px: pd.DataFrame, vol: pd.DataFrame, refit: bool = False) -> dict:
    if not refit and os.path.exists(PATTERN_FILE):
        try:
            pat = json.load(open(PATTERN_FILE))
            if "centroid" in pat and "scale" in pat:
                return pat
        except Exception:
            pass
    pat = learn_fingerprint(px, vol)
    json.dump(pat, open(PATTERN_FILE, "w"), indent=2)
    return pat


def score_one(feat: dict, pat: dict) -> tuple[float, dict]:
    """Weighted Gaussian similarity of a feature vector to the fingerprint -> 0..100, plus a
    per-feature match breakdown (0..1 each) used by the site scorecard."""
    cen, scl, w = pat["centroid"], pat["scale"], pat["weights"]
    total_w = sum(w[k] for k in FEATURES)
    sim_sum = 0.0
    breakdown = {}
    for k in FEATURES:
        z = (feat[k] - cen[k]) / (scl[k] if scl[k] else 1.0)
        m = float(np.exp(-0.5 * z * z))          # 1 at the centroid, ->0 far away
        breakdown[k] = {"value": feat[k], "target": cen[k], "match": m}
        sim_sum += w[k] * m
    return 100.0 * sim_sum / total_w, breakdown


def screen(px: pd.DataFrame, vol: pd.DataFrame, pat: dict) -> pd.DataFrame:
    """Score every beaten-down name in the panel against the fingerprint. Returns a ranked frame."""
    rows = []
    for sym in px.columns:
        s = px[sym].dropna()
        if len(s) < MIN_BARS:
            continue
        c = s.values.astype(float)
        v = vol[sym].reindex(s.index).ffill().values.astype(float)
        f = _feat_at(c, v, len(c) - 1)
        if f is None:
            continue
        if f["dd_from_high"] > pat.get("beaten_down_dd", BEATEN_DOWN_DD):
            continue   # not beaten down enough — not a turnaround candidate
        sc, bd = score_one(f, pat)
        rows.append({
            "ticker": sym, "score": sc, "price": float(c[-1]),
            "dd_from_high_pct": f["dd_from_high"] * 100.0,
            "base_pos_pct": f["base_pos"] * 100.0,
            "ret_1m_pct": f["ret_1m"] * 100.0, "ret_3m_pct": f["ret_3m"] * 100.0,
            "accel": f["accel"], "above_ma50_pct": f["above_ma50"] * 100.0,
            "above_ma200_pct": f["above_ma200"] * 100.0, "vol_surge": f["vol_surge"],
            "ma50_slope_pct": f["ma50_slope"] * 100.0, "base_age": f["base_age"],
            "features": f, "breakdown": bd,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    from early_scanner import panels
    from universe import all_symbols
    px, vol = panels(all_symbols(include_stocks=True))
    pat = load_or_learn(px, vol, refit=True)
    print("METHOD:", pat["_method"])
    print("CENTROID:", {k: round(v, 3) for k, v in pat["centroid"].items()})
    df = screen(px, vol, pat)
    print(f"\n{len(df)} beaten-down candidates; top 12:")
    print(df[["ticker", "score", "price", "dd_from_high_pct", "base_pos_pct", "above_ma50_pct"]].head(12).to_string(index=False))
