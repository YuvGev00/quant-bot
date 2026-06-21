#!/usr/bin/env python3
"""breakout_discover.py — schema + writer for the weekly EARLY DISCOVERY scout.

This is the SPECULATIVE forward-conviction discovery mode of the bot. Each week the
cloud agent runs a Bigdata.com live sweep over small/mid-cap US stocks, scores the
genuinely *early* ones (cheap AND quietly improving, caught BEFORE the hype), and
writes breakout_discover.json using the schema defined here. breakout_site.py renders
that JSON into the static research site.

It is NOT a backtested edge. Small-caps carry liquidity, fraud and total-loss risk.
Research leads, the human decides, manual trades only.

Schema (breakout_discover.json):
{
  "_asof": "YYYY-MM-DD",
  "_source": "Bigdata.com live discovery sweep (small/mid-cap, pre-hype)",
  "_disclaimer": "...",
  "discoveries": [ Discovery, ... ]
}

Discovery = {
  "ticker": str,
  "name": str,
  "tier": "EMERGING" | "SPECULATIVE_EARLY",   # ~$1-20B steadier vs ~$0.25-2B lottery
  "early_fit": "STRONG" | "MODERATE" | "WEAK",
  "market_cap_b": float | None,               # USD billions
  "price": float | None,
  "base_pos_pct": float | None,               # position in 52w range, 0-100
  "n_analysts": int | None,                   # LOW coverage = the under-the-radar signal
  "coverage": "Low" | "Moderate" | "High" | None,
  "ratings": {"buy": int, "hold": int, "sell": int} | None,
  "valuation": str | None,                    # human-readable fwd P/E or P/S vs growth
  "cheap": bool | None,
  "rev_growth_pct": float | None,
  "eps_rising": bool | None,                  # the "improving" signal
  "earnings_surprise_pct": float | None,
  "margin_pct": float | None,
  "thesis": str,                              # why it could be an early MU/NVDA-style mover
  "key_risk": str,
  "early_signals": [str, ...],
  "news": [ {"headline": str, "source": str, "date": "YYYY-MM-DD", "tone": str}, ... ]
}
"""
from __future__ import annotations
import json, os
from datetime import date

OUT = "breakout_discover.json"

DISCLAIMER = (
    "SPECULATIVE forward conviction-research over a live-discovered small/mid-cap "
    "universe. NOT a backtested edge; small-caps carry liquidity, fraud and total-loss "
    "risk. Research leads, you decide. Manual trades only."
)
SOURCE = "Bigdata.com live discovery sweep (small/mid-cap, pre-hype)"

_TIERS = {"EMERGING", "SPECULATIVE_EARLY"}
_FITS = {"STRONG", "MODERATE", "WEAK"}


def validate(disc: dict) -> list[str]:
    """Return a list of problems (empty list = OK). Keeps the JSON honest."""
    errs = []
    for d in disc.get("discoveries", []):
        t = d.get("ticker", "?")
        if d.get("tier") not in _TIERS:
            errs.append(f"{t}: bad tier {d.get('tier')!r}")
        if d.get("early_fit") not in _FITS:
            errs.append(f"{t}: bad early_fit {d.get('early_fit')!r}")
        if not d.get("thesis"):
            errs.append(f"{t}: missing thesis")
        if not d.get("key_risk"):
            errs.append(f"{t}: missing key_risk")
    return errs


def write(discoveries: list[dict], asof: str | None = None, path: str = OUT) -> str:
    payload = {
        "_asof": asof or date.today().isoformat(),
        "_source": SOURCE,
        "_disclaimer": DISCLAIMER,
        "discoveries": discoveries,
    }
    errs = validate(payload)
    if errs:
        raise ValueError("breakout_discover validation failed:\n  " + "\n  ".join(errs))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return os.path.abspath(path)


if __name__ == "__main__":
    if os.path.exists(OUT):
        data = json.load(open(OUT))
        errs = validate(data)
        print(f"{OUT}: {len(data.get('discoveries', []))} discoveries, asof {data.get('_asof')}")
        print("validation:", "OK" if not errs else errs)
    else:
        print(f"{OUT} not present — the cloud agent writes it from the weekly Bigdata sweep.")
