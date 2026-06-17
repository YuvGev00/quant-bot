#!/usr/bin/env python3
"""news_check.py — the NEWS-READING layer for the scanner.

The backtest scanner is PRICE-BLIND: it knows a ticker is rising, not WHY. This module turns the
price-only monthly pick-list into a list of (ticker, why-to-research) prompts, so a news read can
give each candidate a CONFIRM / CAUTION verdict before you buy. It does two things:

  1) build_research_brief(tickers): emit, per ticker, the exact web-search queries + the questions a
     human (or an LLM-with-web-search like Claude) should answer to sanity-check the price signal.
  2) score_candidates(verdicts): combine the price-momentum rank with a manual/LLM news verdict
     into a final BUY / WATCH / SKIP decision, so news can VETO a frothy momentum pick.

WHY it's structured this way (honest design): this script cannot itself fetch the live web in a
normal `python3` run — live news requires a web-search tool. So the workflow is:
  step A: run the price scanner (early_scanner.py / stock_world_scanner.py) -> monthly pick-list
  step B: run THIS to print the research brief for those picks
  step C: a web-search agent (Claude) answers the brief -> fills a verdicts dict
  step D: run score_candidates(verdicts) -> final decision list
This keeps the price math reproducible and the (slower, non-reproducible) news read as an explicit,
auditable overlay rather than hidden magic. LIMITATION: news reads are current-but-not-realtime;
this is a "why is it moving / is it hype" sanity check, NOT a speed/HFT edge.
"""
from __future__ import annotations
import json, sys

# what to ask about each flagged name — the price chart can't answer any of these
RESEARCH_QUESTIONS = [
    "WHY is it rising right now — the actual catalyst (earnings? AI demand? takeover? macro/commodity?)",
    "Is the move backed by fundamentals or is it hype / momentum-chasing / a short squeeze?",
    "RED FLAGS a chart hides: recent bad earnings, analyst target BELOW price, overvaluation/froth, "
    "sector at a historic extreme, geopolitical/commodity dependence, insider selling?",
    "Current analyst lean (bullish / bearish / mixed) and average price target vs today's price?",
]


def search_queries(ticker: str) -> list[str]:
    """The web searches to run for a ticker (year-stamped so results are current)."""
    return [f"{ticker} stock why rising 2026",
            f"{ticker} stock 2026 outlook analyst",
            f"{ticker} stock overvalued OR bubble OR downgrade 2026",
            f"{ticker} earnings 2026 risk"]


def build_research_brief(tickers: list[str]) -> str:
    """Print the per-ticker research brief a news-read should answer (step B)."""
    out = ["=" * 70, "NEWS-READING BRIEF — answer these before buying the price-flagged names",
           "(price momentum got them onto this list; news decides if they're REAL)", "=" * 70]
    for t in tickers:
        out.append(f"\n### {t}")
        out.append("  Run searches: " + " | ".join(f'\"{q}\"' for q in search_queries(t)))
        out.append("  Answer:")
        for i, q in enumerate(RESEARCH_QUESTIONS, 1):
            out.append(f"    {i}. {q}")
        out.append("  -> verdict: CONFIRM (real, buy) / CAUTION (mixed, half-size) / VETO (froth/red-flag, skip)")
    out.append("\nThen put answers into a verdicts dict and run score_candidates(). See __main__ demo.")
    return "\n".join(out)


# verdict -> position multiplier: news can shrink or veto a momentum pick (never enlarge it past 1)
VERDICT_WEIGHT = {"CONFIRM": 1.0, "CAUTION": 0.5, "VETO": 0.0}


def score_candidates(price_ranked: list[str], verdicts: dict[str, str]) -> list[dict]:
    """Combine price rank (order of price_ranked = strongest first) with news verdicts (step D).
    Returns a final action list. Equal price weight across non-vetoed names, scaled by verdict."""
    rows = []
    for i, t in enumerate(price_ranked):
        v = verdicts.get(t, "CAUTION").upper()        # default CAUTION if not researched
        w = VERDICT_WEIGHT.get(v, 0.5)
        action = {"CONFIRM": "BUY", "CAUTION": "BUY (half size)", "VETO": "SKIP"}[
            "CONFIRM" if w == 1.0 else ("VETO" if w == 0.0 else "CAUTION")]
        rows.append({"ticker": t, "price_rank": i + 1, "news_verdict": v,
                     "position_mult": w, "action": action})
    # renormalize the kept weights so the book is fully invested across survivors
    live = [r for r in rows if r["position_mult"] > 0]
    tot = sum(r["position_mult"] for r in live) or 1.0
    for r in rows:
        r["final_weight%"] = round(100 * r["position_mult"] / tot, 1) if r["position_mult"] > 0 else 0.0
    return rows


if __name__ == "__main__":
    # DEMO using the names the live web research actually flagged + the verdicts that research reached.
    # (This is the real June-2026 read from the parallel research agents.)
    price_ranked = ["INTC", "AMD", "EWY", "CAT", "CSCO", "TXN", "EWT", "SLB", "NEM", "AVGO"]

    if len(sys.argv) > 1 and sys.argv[1] == "brief":
        print(build_research_brief(price_ranked)); sys.exit()

    # Verdicts from the live research done 2026-06-17 (semis = late-stage froth; Intel worst):
    verdicts = {
        "INTC": "VETO",     # +250% YTD on HOPED-for foundry deals; mean target ~21% BELOW price; squeeze spent
        "AMD":  "CAUTION",  # real AI demand but ~53x fwd P/E, DCF ~21-27% overvalued
        "EWY":  "CAUTION",  # real earnings but 42% Samsung+SK Hynix, Buffett-indicator 256%, won fragile
        "CAT":  "CAUTION",  # data-center power real, but $2.6B tariff hit + mining -39%, late-cycle
        "CSCO": "CAUTION",  # real AI orders but re-rating "already absorbed"
        "TXN":  "CONFIRM",  # most fundamentally grounded; data-center +90%, capex cycle winding down
        "EWT":  "CAUTION",  # cheap-ish (P/E 21) but leveraged-TSMC bet under Taiwan-Strait tail risk
        "SLB":  "VETO",     # pure war-driven oil-spike premium; JPM baseline Brent ~$60 vs ~$95 now
        "NEM":  "CAUTION",  # great FCF but leveraged on gold which just corrected 16-25%; output falling
        "AVGO": "CAUTION",  # strong but dropped 15% on June-4 guide miss; expectations priced for perfection
    }
    print(build_research_brief(price_ranked))
    print("\n" + "=" * 70)
    print("FINAL DECISIONS (price momentum + live news verdict, June 2026 demo)")
    print("=" * 70)
    rows = score_candidates(price_ranked, verdicts)
    import pandas as pd
    df = pd.DataFrame(rows)[["ticker", "price_rank", "news_verdict", "action", "final_weight%"]]
    print(df.to_string(index=False))
    kept = [r["ticker"] for r in rows if r["position_mult"] > 0]
    vetoed = [r["ticker"] for r in rows if r["position_mult"] == 0]
    print(f"\nBUY/WATCH: {', '.join(kept)}")
    print(f"NEWS VETOED (price said buy, news said NO): {', '.join(vetoed)}")
    print("\nNote: news VETO'd INTC & SLB — both were strong price-momentum picks the chart loved,")
    print("but the news revealed froth (Intel) and a mean-reverting war premium (SLB). That veto is")
    print("the entire point of this layer: stop the price-blind scanner buying a rising-for-bad-reasons name.")
