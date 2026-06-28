#!/usr/bin/env python3
"""breakout_discover.py — schema + helpers for the WEEKLY EARLY-DISCOVERY scout.

This is the SPECULATIVE, forward-looking discovery mode of the bot. Unlike the
large-cap momentum/value engines elsewhere in the repo, this mode does NOT run a
backtested edge. It is a live, qualitative conviction-research sweep over a
freshly DISCOVERED universe of small & mid-cap US names that are CHEAP and
QUIETLY IMPROVING — caught EARLY, before the crowd/hype (the 'MU early-2025 /
NVDA-2023 / PLTR-at-$6' moment, NOT after they ran).

The weekly cloud agent fills breakout_discover.json (repo root) using the schema
below, then breakout_site.py renders it. Only REAL, sourced numbers go in;
anything not actually retrieved stays null. A thin list of real early names beats
a padded one.

Discovery is sourced from Bigdata.com live search sweeps + company tearsheets.

ANTI-HYPE GUARD (enforced by the agent, documented here):
  - DROP mega-caps, ETFs/funds, non-US names.
  - DROP anything already up >150% over the trailing year (already ran = NOT early).
  - DROP names whose 'improving' story is not actually showing in the numbers.

TIERS:
  EMERGING          ~$1-20B mid-cap, steadier
  SPECULATIVE_EARLY ~$0.25-2B micro/IPO, lottery-risk / total-loss possible

EARLY_FIT verdict per name: STRONG / MODERATE / WEAK.
"""
from __future__ import annotations
import json, os

DISCOVER_FILE = "breakout_discover.json"

TIERS = ("EMERGING", "SPECULATIVE_EARLY")
EARLY_FITS = ("STRONG", "MODERATE", "WEAK")

# ---- schema (documentation + light validation) -----------------------------
# Top level:
#   _asof:       ISO date string of the run
#   _source:     provenance string
#   _disclaimer: speculative / live-discovered / small-cap-risk wording
#   discoveries: list[ DISCOVERY ]
#
# DISCOVERY fields (report ONLY numbers actually retrieved; else null):
DISCOVERY_FIELDS = {
    "ticker": str,            # US listing symbol
    "name": str,              # company name
    "tier": str,              # one of TIERS
    "early_fit": str,         # one of EARLY_FITS
    "market_cap_b": float,    # market cap in $B
    "price": float,           # last price
    "base_pos_pct": float,    # position in 52w range, 0-100 (low = beaten down)
    "n_analysts": int,        # number of covering analysts (LOW = under-the-radar)
    "coverage": str,          # qualitative: 'low' / 'moderate' / 'high'
    "ratings": dict,          # {buy, hold, sell}
    "valuation": str,         # short valuation note (fwd P/E or P/S vs growth)
    "cheap": bool,            # is it genuinely cheap vs its growth?
    "rev_growth_pct": float,  # revenue growth %, if obtained
    "eps_rising": bool,       # forward EPS estimates rising?
    "earnings_surprise_pct": float,  # latest earnings EPS surprise %
    "margin_pct": float,      # margin / ROE if obtained
    "thesis": str,            # 2-3 sentence early-mover thesis
    "key_risk": str,          # the dominant risk
    "early_signals": list,    # list[str] of concrete early signals
    "news": list,             # list[{headline, source, date, tone}]
}

REQUIRED = ("ticker", "name", "tier", "early_fit", "thesis", "key_risk")


def validate(doc: dict) -> list:
    """Return a list of human-readable validation problems (empty == OK)."""
    problems = []
    if "discoveries" not in doc or not isinstance(doc["discoveries"], list):
        return ["missing 'discoveries' list"]
    for i, d in enumerate(doc["discoveries"]):
        tag = d.get("ticker", f"#{i}")
        for r in REQUIRED:
            if not d.get(r):
                problems.append(f"{tag}: missing required field '{r}'")
        if d.get("tier") not in TIERS:
            problems.append(f"{tag}: bad tier {d.get('tier')!r}")
        if d.get("early_fit") not in EARLY_FITS:
            problems.append(f"{tag}: bad early_fit {d.get('early_fit')!r}")
    return problems


def load(path: str = DISCOVER_FILE) -> dict:
    if not os.path.exists(path):
        return {"_asof": None, "_source": None, "_disclaimer": None, "discoveries": []}
    with open(path) as f:
        return json.load(f)


# rank: STRONG first, then EMERGING before SPECULATIVE_EARLY, then bigger cap
_FIT_RANK = {"STRONG": 0, "MODERATE": 1, "WEAK": 2}
_TIER_RANK = {"EMERGING": 0, "SPECULATIVE_EARLY": 1}


def rank_key(d: dict):
    return (_FIT_RANK.get(d.get("early_fit"), 9),
            _TIER_RANK.get(d.get("tier"), 9),
            -(d.get("market_cap_b") or 0))


def ranked(doc: dict) -> list:
    return sorted(doc.get("discoveries", []), key=rank_key)


if __name__ == "__main__":
    doc = load()
    probs = validate(doc)
    print(f"{DISCOVER_FILE}: {len(doc.get('discoveries', []))} discoveries; "
          f"{'OK' if not probs else str(len(probs)) + ' problems'}")
    for p in probs:
        print("  -", p)
    for d in ranked(doc):
        print(f"  {d['early_fit']:8s} {d['tier']:17s} {d['ticker']:6s} "
              f"{d.get('market_cap_b','?')}B  {d['name']}")
