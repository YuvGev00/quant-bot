I have everything I need to synthesize this. The 38 adversarially-verified candidates have already been probed against the live RavenPack/Bigdata.com MCP and the IL tax/permutation constraints. My job is to rank and write the report, not re-run the probes. Let me produce the tight markdown.

# New Trading Strategies on RavenPack/Bigdata.com — Ranked Shortlist for an IL Retail Quant

**Bottom line up front:** Of 38 verified candidates, **zero are build-and-test-now standalone alphas.** The data is real but the *backtestable history* mostly isn't (snapshot-only feeds), the strongest legs are short-side or small-cap (untradeable here), and the IL tax wall kills anything faster than monthly. What survives is a small set of **low-turnover overlays and forward-validated tilts** that ride your already-proven edges (storm-gate + volume-confirmed reversal). Rank below is by (edge-real × tax-survival × buildable).

---

## 1. Top new strategies to test

### #1 — Sentiment-stabilization filter on the reversal edge (falling-knife guard)
- **Mechanism:** Your validated 10d volume-confirmed reversal (p=0.000, Sharpe 0.70, but −46% maxDD) buys recent losers. Overlay: only take the long-the-loser leg when the name's media-sentiment slope has flattened/turned up over trailing 5–10d, and **skip losers whose sentiment is still deteriorating** or that carry a hard negative catalyst (8-K guidance cut, litigation). Separates overreaction-bounce from real-information-drift.
- **RavenPack data:** MEDIA SENTIMENT time-series (slope) + NLP SEARCH over 8-K/news for the binary catalyst flag.
- **Why NEW:** This is the first thing that *attacks the documented weakness* of our one real edge rather than proposing a fresh edge. The veto direction is backed by source literature (Heston-Sinha: negative news drifts/continues, so still-falling = keep-falling).
- **Verdict:** promising-needs-validation. Tax-friendly (a veto only *removes* trades, can extend holds → reduces turnover). Buildable LIVE; **not backtestable** (no historical sentiment panel).
- **First test:** Forward-only A/B. Each weekly reversal run, log two parallel paper books — (A) unfiltered candidates, (B) candidates that survive the sentiment-slope + catalyst veto — snapshotting sentiment at decision time (no leak). After ~20–30 weekly cohorts, compare forward win-rate and especially the **left tail / realized maxDD** of B vs A. Justified only if it cuts the tail without crushing median return. No permutation test possible until forward data exists.

### #2 — Macro news-sentiment/attention as a market-level vol/regime gate (incremental to SJM)
- **Mechanism:** Aggregate daily sentiment + attention on macro entities ("US economy", SPY). Falling sentiment + rising macro attention → forecast higher vol → dial down equity beta / tilt defensive. A *gate*, not a stock-picker.
- **RavenPack data:** MEDIA SENTIMENT + ATTENTION on a tiny macro entity set (low-name-count, fits rate limits). The cleaner **point-in-time** input is `bigdata_country_tearsheet(US)` consensus-vs-actual macro SURPRISE (NFP/CPI/FOMC/ISM/GDP).
- **Why NEW:** Three confirmed papers show macro news improves vol *forecasts*; we've never had a sentiment/macro-surprise gate. Forward macro-surprise data is genuinely point-in-time (unlike per-company sentiment snapshots).
- **Verdict:** marginal/promising. Tax-clean (monthly-ish flips, ~0.5–1.3 switches/yr like SJM). **Redundancy risk is the real problem** — attention *follows* VIX, so it may be a lagging twin of the price-vol signal SJM already captures.
- **First test:** Falsify the *incremental* claim. Build a monthly macro-surprise index from `country_tearsheet`; regress next-month realized SPY vol on (lagged RV/VIX) vs (+macro-surprise term), strictly OOS walk-forward, T+1 timestamps. Permutation-test the OOS incremental R² of the macro term (p<0.05). **Kill if it adds nothing beyond VIX/RV** — then it can't beat the SJM you already ship. Do NOT backtest the FinBERT sentiment version (snapshot-only, unbacktestable).

### #3 — Analyst estimate-revision BREADTH as a price-independent momentum confirmation
- **Mechanism:** Net-up-minus-down EPS-estimate revisions over trailing ~100d, used as a **tie-breaker/booster** on momentum picks ("confirmed momentum" = positive price momentum AND positive revision breadth). Rescues the momentum sleeve whose stock-picking failed permutation (p=0.34) by adding an orthogonal, price-independent signal.
- **RavenPack data:** ANALYST estimate revisions (diffusion of up vs down). Best-pedigreed new dimension (Arnott→PEAD lineage; Mill Street MAER IC=0.23, ~83% sign-persistence, genuine 2013–22 OOS).
- **Why NEW:** Price-independent fundamental momentum — something our price-only stack literally could not see. Strongest statistical footing of any candidate.
- **Verdict:** promising-needs-validation. Tax-friendly (slow, 100d window, deployed as re-rank adds ~zero turnover). **Buildable only partly** — `analyst_estimates` is a *current snapshot*, no revision history; breadth must be reconstructed from dated `bigdata_search` events (PT-biased, earnings-clustered) or self-collected forward.
- **First test:** Forward paper A/B — RAW momentum top-N vs CONFIRMED (momentum AND positive live breadth), monthly, net of 25% IL CGT. Snapshot breadth to a dated file each rebalance (build the PIT panel we lack). After ~24–36 snapshots, run the permutation rig (shuffle breadth→return) and require p<0.05 AND that breadth adds over what price already encodes. **Do NOT inherit the 15.6%/7.6% numbers** — those are global, gross, long-short decile spreads that collapse at our 30–80 name N.

### #4 — Downgrade / negative-revision AVOIDANCE veto (long-only exit/skip rule)
- **Mechanism:** Downgrade/negative-revision drift is materially stronger and longer than the upgrade drift (analyst optimism bias). Can't short → use as a VETO: exit/skip any held or candidate name with a fresh consensus cut or sharp FY1 EPS downward revision, even if price momentum still looks fine.
- **RavenPack data:** Consensus RATING CHANGES + EPS revision events (dated `bigdata_search` news stream with old→new values, publication-timestamped = causal, avoids restated-consensus look-ahead).
- **Why NEW:** Captures the *strong* (negative) leg of a well-replicated 30-yr asymmetry (Womack) in a long-only-compatible, tax-neutral form. More buildable historically than first assumed — the dated downgrade stream reaches back to ~2019.
- **Verdict:** promising-needs-validation. Tax-clean (a veto reduces, never adds, turnover). The honest payoff is **avoided-drawdown, not per-name alpha** — so it won't clear a per-name permutation IC test by construction; test it as a conditional contribution.
- **First test:** Veto event study with the right denominator. Pull dated downgrade/lowered-PT/negative-EPS events for ~40 IBI-tradable large-caps over 2019–25; measure forward 1/3/6mo returns of vetoed names vs same-name non-veto periods and matched controls (hit-rate of underperformance, avoided-drawdown magnitude). **Decisive cut:** condition on SJM gate state — if downgrades only matter when the gate is already OFF, it's redundant with existing defense. Ship only if it adds avoided-drawdown *conditional on the gate being ON*; else fold into `news_check.py`.

---

## 2. The single best bet

**Build #1 — the sentiment-stabilization falling-knife guard on the reversal sleeve.**

Why it wins:
- It is the **only candidate that improves an edge we have *already proven real*** (volume-confirmed reversal, p=0.000, survives OOS + costs to 40bps + every era). Everything else is a fresh, unproven, decayed, or short-side anomaly. The base rate for new commercial-sentiment alphas surviving our rig is dismal; the base rate for *trimming a known left tail* is far better.
- It directly targets the reversal sleeve's documented **−46% maxDD** — the single biggest risk in the live book — and the veto direction is the *well-supported* half of the sentiment literature (negative-news drift), not the speculative half.
- **Tax-perfect:** a veto only removes trades and can extend holds → it can only *reduce* realizations under the 25–47% no-deferral wall. Zero added turnover, zero new tax exposure.
- **Buildable LIVE today** on a tens-of-names weekly universe, well within rate limits, slotting into the existing `news_check.py`/`reversal_agent.py` plumbing.

The honest constraint: it **cannot be backtested or permutation-tested** (no historical sentiment panel), so it ships as a forward-logged left-tail trimmer with **no attached return number** — never as a standalone alpha. That's an acceptable trade because the downside is bounded (worst case: it does nothing or over-filters; it cannot blow up the sleeve).

---

## 3. Reject pile (one line each)

**Tax-killed (turnover/holding-period vs IL no-deferral CGT):**
- **Analyst-rating-change sentiment momentum** — 5.4-day hold; gross 3.3% already nets to ~zero (Barber 2001), dead under IL CGT.
- **Vanilla single-surprise PEAD** — drift exists only in untradeable microcaps (ex-microcap t=1.43); needs a short leg; dead in large-caps.
- **Pre-Earnings Announcement (PrEA) momentum** — days-to-2-week hold = 100% short-term gains every cycle; sub-1%/mo edge nets to ~zero.
- **Earnings-Announcement Premium (EAP)** — US premium *disappeared* post-2004 (8-K reform); monthly-rotation tax wall (frontrun precedent: 9.5%→2.3% CAGR).
- **Pre-/post-earnings sentiment-conditioned drift** — earnings-clustered realization; honest long-only net is sub-1%/yr noise.
- **Earnings-call Q&A transcript-tone drift** — vendor's own signal decays in ~8 days; tax-dead at real turnover.
- **Conditional PEAD (divergence-gated)** — ~4wk holds realize ST gains monthly; ~120bps gross has no headroom after IL CGT + costs.

**Secretly needs data we lack / not buildable on this feed:**
- **13F best-ideas cloning** — no 13F endpoint anywhere in the MCP; needs WhaleWisdom/Novus or an EDGAR parser.
- **Revision breadth diffusion index** — feed gives consensus mean + counts, NOT up/down revision counts; the diffusion numerator doesn't exist.
- **Classic hiring-RATE anomaly (BLB)** — feed exposes job *postings flow* (avg_active_jobs), not audited Compustat headcount; wrong variable, no PIT panel.
- **Job-postings hiring-acceleration** — snapshot-only (no historical postings panel), de-rated monthly slice fights the durable level factor.
- **ESG-improvement momentum** — only 6 *rewritten-vintage* annual snapshots; a change-of-level signal trades methodology churn, not company improvement.
- **Lazy Prices (10-K YoY similarity)** — `bigdata_search` returns relevance-ranked chunks, not deterministic full-text; can't compute stable cosine; alpha is on the short leg anyway.
- **Sell-side report TEXT alpha** — needs 1.2M full-text Investext reports + GPU LLaMA3 embeddings; feed gives scored sentiment, not raw text.

**Sentiment/alt-data decay or wrong-universe (edge weak/decayed for us):**
- **Novelty/similarity-gap sentiment** — RavenPack's own novelty score isn't exposed; proxy can't distinguish "no prior event" from "not retrieved"; edge lives in microcaps.
- **Abnormal-media-attention amplifier** — `media_attention` field returns empty even for Apple; edge is short-leg microcap.
- **Analyst-rating-revision sentiment** — the cited paper (Lv 2025) *refutes* this exact construction (revision alpha t=1.56, insignificant).
- **Sentiment momentum-then-reversal** — regime-unstable splice; redundant with reversal; ~3yr shallow history can't fit the month-7 inflection.
- **Media-sentiment extreme + attention reversal** — attention field empty; mechanism-splice (monthly reversal vs 2–5d continuation); short leg untradeable.
- **Pre-earnings revision tilt / revision+sentiment confirmation / EPS-revision momentum (standalone)** — all need a PIT revision panel the snapshot feed lacks; thin after universe + decay + tax.
- **Sector rotation by constituent sentiment** — needs PIT ETF constituent *weights* (absent from RavenPack); headline is mega-cap growth beta in disguise.
- **Sector-level revision breadth** — aggregating to ~11 ETFs destroys the dispersion that makes the spread; no evidence at sector level.
- **Sentiment-conditioned overpricing gate / price-vs-sentiment divergence / attention-spike sector overlay / low-attention underreaction / conference-presentation effect** — all reject as standalone: short-side or cross-sectional or microcap or snapshot-only; usable at most as live vetoes.
- **Sentiment-attention SURGE confirm on reversal exit** — sound idea but host mismatch (our reversal book is long-only, never fades winners) + attention metric is empty.

**Meta-constraint (not a strategy):**
- **Commercialized-signal decay / crowding anchor** — correct *prior*, not a tradeable sleeve; adopt it as a standing IR discount on all sentiment work.

---

## 4. Honest caveat — is commercial-feed sentiment alpha already gone in 2026?

**Mostly yes, for the fast/liquid/scalable slice — which is exactly the slice we can trade.** Three converging realities:

1. **Crowding is real and replicated.** McLean-Pontiff (2016): −26% OOS, −58% post-publication, with rising cross-predictor correlation = the literal crowding signature. RavenPack has been institutionally distributed 10+ years; news sentiment is the textbook "most-crowded, decays-first" signal. The AI-era acceleration numbers (42% convergence, 18mo half-life) are *simulation outputs*, not measured fact — don't quote them, but the direction is right.

2. **The surviving alpha lives where we can't go.** Every documented effect concentrates in small/illiquid/low-coverage names (limits-to-arbitrage is *why* it persists) and on the short leg (negative-news drift is the stronger half). A long-only, large-cap, manual IL retail account harvests the weakest, most-arbitraged slice by construction.

3. **The decisive practical kill is data, not just decay.** The MCP serves **live snapshots, not downloadable multi-year point-in-time panels.** `media_attention` returns empty even for Apple/NVIDIA; `analyst_estimates`/`analyst_ratings`/`sentiment_tearsheet` are as-of-today only. So most sentiment strategies **cannot be permutation-tested on our rig at all** — we'd be trusting a vendor signal blind. For a shop whose discipline already exposed momentum as p=0.34 noise, "I literally cannot validate it offline" is reason enough to refuse it as standalone alpha.

**What would tell us alpha still exists:** A forward-collected, point-in-time signal panel (built going forward, no backfill) that, after ~6–18 months, lets a **veto/gate/confirmation overlay beat its ungated baseline net of IL tax at p<0.05** on our *liquid* universe. A useful crowding tell: rolling correlation of any sentiment-sorted book vs generic momentum/our reversal book — rising = crowded → discount hard; low/falling = it carries something orthogonal → worth weight. Until a forward panel clears that bar, the correct posture is the one we already run: **commercial sentiment is veto-only, never sized as standalone alpha.** The genuinely new, defensible use of RavenPack is not a new return engine — it is sharper *defense and confirmation* layered on the storm-gate and reversal edges we already trust.