I'll write this report directly from the adversarially-verified research provided. The material is comprehensive and already verified by 3 skeptics each — no need to re-research. Let me synthesize it into the report.

# Online Trading-Bot Landscape — Final Report for a Deployed Equities/ETF Decision Engine

**Your context, restated so the recommendations stay honest:** You already run a working, deployed Python *decision* engine — Statistical-Jump-Model (SJM) regime "storm-detector" gate, momentum rotation over ~80 US ETFs, yfinance fundamentals screen, LLM news/sentiment read, position/leverage caps, holdings.json state — running as scheduled claude.ai cloud routines off a private repo. You **do not auto-trade**; you place orders **manually** via Interactive Israel / IBKR, as Israeli retail, after a **25–47% tax** drag. Stack is pandas/numpy/sklearn/statsmodels/yfinance.

That profile changes almost every verdict below. Most famous "trading bots" are crypto-first, auto-executing, and microstructure-obsessed — three things you don't need. Your edge lives in signal logic and tax-aware sizing, not execution. So the right question for every tool is **"does this improve the brain, the validation, or the data — without forcing a rewrite?"** Very few do. The ones that do, do so as *borrowable components or offline research benches*, not as frameworks to migrate onto.

---

## 1. Top tools worth adopting

Ranked by (usefulness to *your* bot) × (maintained). I've separated **adopt** from **borrow-the-idea**, because for your architecture most "adoptions" are surgical, not wholesale.

### Tier 1 — Highest value, clean fit

**jumpmodels (Statistical Jump Models, Python)** — https://github.com/Yizhan-Oliver-Shu/jump-models
*What:* sklearn-style (`.fit/.predict/.predict_proba`) implementation of discrete, continuous (CJM), and sparse (SJM) jump models, plus **`.predict_online` / `.predict_proba_online`** for streaming inference. Apache-2.0, deps = your exact stack (numpy/pandas/scipy/sklearn).
*Why it's #1:* This is the **canonical reference implementation of the exact method your storm-detector already uses** (Nystrup/Kolm/Mulvey/Shu lineage; the author is a co-author on the CJM paper). It is not net-new capability — it's a chance to *validate and upgrade* code you hand-rolled.
*How to use, concretely:*
- **Validate first.** Run your gate and `jumpmodels` on the same features/history and diff the regime labels. Free correctness check on a gate that's making real allocation decisions.
- **Fix look-ahead.** The single most important finding across all three skeptics: `predict_proba_online` is causal (each bar uses only prior data via a forward DP value matrix); plain `.predict` runs a full-window E-step and **leaks future info into past labels**. If your gate (or your validation of it) ever labels regimes with full-window `.predict`, that's an in-sample look-ahead bug. Use the `_online` methods for live and for any honest backtest.
- **Upgrade binary → soft.** Use CJM `predict_proba` as a *continuous* throttle on leverage/position caps (and add a hysteresis band, e.g. exit risk >0.7, re-enter <0.3) instead of a hard on/off flip. Fewer false flips = fewer taxable round-trips, which matters enormously at 25–47%.
- **Prune features.** `SparseJumpModel` does L1 feature selection over your candidate regime features.
*Caveats:* **Stale** — one release (v0.1.1, Oct 2024), ~13 commits, dormant ~17 months, single academic author. So **vendor/fork it** (pin `==0.1.1`, copy the ~6 modules into your private repo), don't take a live upstream dependency in a scheduled cloud bot. It is labels-only: no backtester, no costs, no fills — all of that stays in your code.

**skfolio** — https://github.com/skfolio/skfolio
*What:* portfolio optimization + leakage-aware cross-validation built *on top of scikit-learn* (estimators, Pipeline, GridSearchCV). BSD-3, genuinely active (v0.20.1, Apr 2026; ~weekly releases; backed by arXiv 2507.04176).
*Why:* It solves the two parts of your pipeline that are weakest, and it composes with your existing sklearn idiom with near-zero friction.
*How to use:*
- **Leakage-free validation (the high-value piece).** `from skfolio.model_selection import WalkForward` / `CombinatorialPurgedCV`. These implement `split()`, so they drop straight into `cross_val_predict`/`GridSearchCV` over **any** estimator — you do **not** have to adopt a single skfolio optimizer. Use them to re-tune your momentum lookback, rank cutoff, rebalance cadence, and SJM thresholds **without** the look-ahead that naive KFold/`TimeSeriesSplit`-without-gap introduces on overlapping return windows. This is the cleanest fix for the single most common silent failure in retail momentum bots.
- **Sizing (optional).** After your screen picks the basket, feed returns to `HierarchicalRiskParity` or `RiskBudgeting` for correlation-aware weights instead of equal-weight/ad-hoc caps. Start with HRP — no return forecasts, no covariance inversion, robust on ~80 noisy ETFs.
- **Regime-aware covariance / views.** It ships `RegimeAdjustedEWCovariance` (directly relevant to a regime bot) and `EntropyPooling`/`OpinionPooling` (a principled place to inject your LLM news read as "views" if you ever want sizing, not just direction).
*Caveats:* It is **NOT** a fill-realistic backtester — `transaction_costs`/`management_fees` are linear penalties on turnover, no slippage, **no tax**. Its Sharpe is a pre-tax, pre-execution upper bound. Pin the version (pre-1.0, API moves). Use it offline in the weekly deep cycle, never in the hourly watch.

### Tier 2 — Strong, narrowly scoped

**vectorbt (open-source, polakowo)** — https://github.com/polakowo/vectorbt
*What:* Numba/NumPy-vectorized backtester; sweeps thousands of parameter combos in seconds. Genuinely **alive** (v1.0.0, Apr 2026, optional Rust backend) — the "abandoned, go-pay-for-PRO" reputation is **outdated**.
*Why:* Your momentum sleeve has many tunable knobs and (per the architecture) no systematic sweep harness. vectorbt's one killer use is exactly that: pull your ~80-ETF panel, build entry/exit signal arrays, run `Portfolio.from_signals(close, entries, exits, fees=..., slippage=...)` across a grid of lookbacks × top-N × cadence × SJM threshold in one pass, and read the Sharpe/maxDD *surface* — looking for robust **plateaus**, not lucky spikes.
*How to use:* **Offline research bench only** — never wire it into the live cloud agents. Two highest-value experiments: (1) A/B the regime gate ON vs OFF over identical history to *quantify* what the storm-detector actually buys you in drawdown reduction; (2) walk-forward / out-of-sample to confirm your live params aren't curve-fit.
*Caveats:* It is also an **overfitting machine** — the speed that lets you sweep 10k combos is the same speed that lets you data-mine a spurious lookback; always pair with OOS/walk-forward. It models **no tax and no FX** — haircut returns by your 25–47% band + IBKR/ILS costs *before* ranking, or it'll favor high-turnover configs that die after tax. License is Apache-2.0 + Commons Clause ("fair-code") — fine for private personal research; only matters if you ever resell. Skip vectorbt**PRO** ($240/yr): the OSS edition covers sweeps + walk-forward + data download for a solo trader.

**bt (pmorissette)** — https://github.com/pmorissette/bt
*What:* portfolio-rebalancing backtester whose native paradigm (`SelectMomentum → WeighInvVol/WeighEqually → RunWeekly → Rebalance` composed as a tree) **is exactly your strategy shape.** Actively maintained (v1.2.0, Apr 2026), MIT.
*Why/how:* The best *architectural* match for your momentum-rotation sleeve, and `ffn` (its dependency) gives free tear-sheet stats (Sharpe/Sortino/maxDD/CAGR) on your realized equity curve — a low-effort, high-value add for your weekly email. Use it as an independent cross-check: re-express the rotation in bt and diff its picks/weights against your `holdings.json` logic to catch weighting or look-ahead bugs.
*Caveats:* **Default commission is literally `0.0`** and fills are at the decision-bar price with no slippage — set `commissions=...` and bid/offer or it flatters you. Models no tax. The SJM gate / LLM read / fundamentals must be precomputed and injected as a signal; they don't live in bt.

### Tier 3 — Data-layer hardening (do this regardless of everything else)

**Your single biggest operational risk is yfinance, and it's already in production.** The Feb-2025 Yahoo redesign brought new quotas + HTML schema changes; `YFRateLimitError` 429s now fire even at low volume, **especially from cloud IPs** like yours (issue #2411, closed "not planned"; reporter hit 429s across rotated IPs). yfinance is also survivorship-biased (delisted tickers vanish) and has no SLA. It's still actively maintained (v1.4.1, May 2026) and fine as *one* input — but harden how you consume it:

1. **Fail loud, never silent.** Wrap every `yf` call so a 429/empty/NaN frame raises a flagged state, not a zeroed signal. Add a staleness assert (last bar within N days) and a coverage assert (data for ≥K of ~80 ETFs). If either fails, the email says "DATA DEGRADED — decisions suppressed," not a clean recommendation. A silent stale-data read feeding the momentum rank or fundamentals screen is the worst-case failure for a confident emailed "rotate into X."
2. **Promote IBKR to your primary feed.** You already have the **IBKR MCP tools right in this environment** (`get_price_history`, `get_price_snapshot`, `search_contracts`, `get_account_positions`, `get_account_balances`). These give broker-grade OHLCV from the venue you actually trade — no Gateway to babysit, no rate-limit roulette — and cross-checking IBKR vs yfinance catches both a 429-stale read *and* a bad tick. Demote yfinance to fallback + the niche fundamentals fields IBKR doesn't expose. **Caveat:** IBKR bars are cached snapshots (guard freshness; `chart_end` can run a day ahead), and `get_company_themes` came back empty for ETF conids — it's single-name oriented, so it won't replace your ETF fundamentals screen.
3. **Reconcile holdings.json against reality.** Use `get_account_positions`/`get_account_trades` to diff your hand-maintained holdings file against the actual account each weekly cycle — kills the silent state-drift bug.

**Add a dead-man's-switch (Healthchecks.io)** — https://healthchecks.io
*What:* open-source, self-hostable cron monitor on the dead-man's-switch model — your job pings a URL on success; if the ping doesn't arrive within period+grace, it alerts (email/Telegram/etc.). Free tier (20 checks) is plenty.
*Why this is `adopt-strong`:* An **email-only scheduled bot has one blind spot: the run that silently never fires, or dies before the email step** — and "no email" is indistinguishable from "quiet market." For a regime gate you trust to flag storms before real money moves, a silent-death window during a selloff is exactly the scenario that costs you. A dead-man's-switch is the *only* model that catches absence-of-signal; try/except + email structurally cannot.
*How:* Separate checks per failure mode (`weekly-deep-cycle` with generous grace; `hourly-emergency-watch` with ~15-min grace; `data-feed-fresh` pinged only after data passes your sanity checks). **Make the success ping the FINAL line, after your own asserts pass** — that converts "process alive" into "healthy run completed." Use `/start` and `/fail` variants to distinguish "ran and failed" from "never ran." Route alerts to a *second* channel (Telegram or a different email) so a dead-bot alert doesn't depend on the path that just died. ~30 min, $0.

---

## 2. The framework landscape — should you migrate? (No.)

Honest one-line answer first: **keep your custom bot.** Every big framework below is either the wrong asset class, the wrong execution model (auto-trade vs. your manual flow), the wrong layer (execution microstructure you don't need), or a rewrite of a working system. Here's the map.

| Framework | What it's for | Asset class | Verdict for you |
|---|---|---|---|
| **NautilusTrader** (https://github.com/nautechsystems/nautilus_trader) | Rust-core, event-driven, nanosecond execution + backtest *parity* | Multi-asset, crypto-first | **Reference only.** Excellent, actively maintained (v1.228.0, Jun 2026). But its entire crown jewel is the *execution/microstructure layer* (fills, latency, queue position) — which a manual, weekly, liquid-ETF decision engine **never touches**. Overkill; would be a rewrite for value you can't use. Read its FillModel docs as a friction checklist, nothing more. |
| **QuantConnect LEAN** (https://github.com/QuantConnect/Lean) | C#-core (Python algos) backtest+live engine, clean corporate-action handling | Multi-asset, equities-strong | **Reference only.** Genuinely active (commits *today*; the frozen-2017-releases-tab is a red herring — it ships via master/Docker/CLI). But it's a `.NET 9 + Docker` monolith you write *inside*; adopting it = abandoning your pandas/sklearn stack. **Killer catch:** local US equity data isn't free (AlgoSeek/QC Dataset Market, paid) — the headline "clean splits/dividends since 1998" sits behind a paywall the moment you leave the cloud IDE. *Borrow:* its per-security cost-model template (FeeModel/SlippageModel) and corporate-action correctness as an audit checklist for your yfinance pipeline. |
| **vectorbt (OSS)** | Vectorized parameter sweeps | Asset-agnostic | **Worth trying** as an offline bench (see §1). The one big framework that genuinely fits one of your needs. |
| **backtrader** (https://github.com/mementum/backtrader) | Event-driven single-instrument backtester w/ realistic order sim | Asset-agnostic | **Avoid / reference only.** **Effectively abandoned** — last release/commit Apr 2023, Python classifiers cap at 3.7, GPL-3.0 (copyleft, bad for a private repo). Its strength (broker-style fill sim) is the thing you don't need; its event-driven model fights cross-sectional ranking. The maintained fork `backtrader_next` exists but is ~28 stars / bus-factor-1. |
| **zipline-reloaded** (https://github.com/stefan-jansen/zipline-reloaded) | Revival of Quantopian Zipline; Pipeline API for cross-sectional factors | Equities | **Reference only.** Alive but *low-tempo* (v3.1.1, compat-maintenance only). Heavyweight Cython + mandatory data-bundle ingestion — hostile to ephemeral cloud agents. *Borrow:* the Pipeline `rank().top(n)` cross-sectional pattern and `pyfolio-reloaded`/`alphalens-reloaded` for tear-sheets and factor IC analysis (validate whether your momentum factor actually predicts forward returns). |
| **Freqtrade** (https://github.com/freqtrade/freqtrade) | The flagship OSS bot | **Crypto only** (CCXT) | **Reference only.** Best-in-class and very active (51k★), but crypto-only, auto-executing, GPL-3.0. Zero IBKR/equities path. *Borrow:* its honest backtest-realism docs (fees applied twice, zero slippage by default, lookahead-bias warnings) as a "ways your backtest lies" checklist; and the idea of optimizing against Sortino/Calmar (drawdown-aware) via Optuna rather than raw return. |
| **Hummingbot** (https://github.com/hummingbot/hummingbot) | Crypto market-making / arbitrage | **Crypto only** | **Avoid for this bot.** Wrong asset, wrong strategy (HFT two-sided quoting), wrong execution. Its own backtester is coarse (flat trade-cost %, no order book). Nothing to borrow but the V2 "Controllers" decision/execution separation pattern. |
| **Microsoft Qlib** (https://github.com/microsoft/qlib) | AI/ML alpha platform, factor zoo | Equities (China-first) | **Reference only.** Maintenance-mode (0.9.x since 2022). Built for ML-ranking *large single-stock universes*; an 80-ETF rotation is the wrong shape. China-centric cost defaults, broken/second-class US data path, no tax model. *Borrow:* read `qlib/backtest/exchange.py` once for its cost/fill math (per-side bps, `max(notional*ratio, min_cost)` floor, quadratic impact term) and port the *idea* into your pandas backtest. |
| **NautilusTrader / LEAN as "someday"** | — | — | Re-evaluate **only if your mandate changes** to true automated IBKR execution or intraday signals. Until then, they solve problems you deliberately don't have. |

**Bottom line:** there is no migration here that's positive-EV. Your custom bot is already on the right side of the design choices these frameworks would force on you (manual execution, tax-awareness, transparent rules, a regime gate more principled than anything in Freqtrade/Qlib). Use the frameworks as **idea sources and offline benches**, never as a destination.

**On broker connectivity specifically:** `ib_async` (https://github.com/ib-api-reloaded/ib_async, the maintained ib_insync successor, v2.1.0) is real and the right library *if you ever auto-trade or run a persistent VM*. But it needs a long-lived TWS/IB-Gateway session — architecturally hostile to ephemeral cloud routines, and redundant with the IBKR MCP tools you already have for read-only data/positions. Skip it; use the MCP layer. **Definitely avoid** `alpaca-py` (would mean a second broker you don't have), `ccxt` (crypto-only, wrong asset class entirely), `schwab-py`/`Tradier` (US-resident-gated, you can't even get an API token), and `robin_stocks`/`pyrh` (reverse-engineered Robinhood endpoints with on-disk credentials — an anti-pattern, and geo-blocked for you anyway).

---

## 3. Best people & content to follow

Organized by where your time actually pays off. (These are scouted, not verification-graded — but the signal/noise calls are consistent across the research.)

### Highest signal — methodology & rigor (spend the most time here)
- **Marcos López de Prado** — https://www.quantresearch.org/ — purged/embargoed CV, **Deflated Sharpe Ratio**, Probability of Backtest Overfitting (PBO). The canonical "why backtests lie" frame. Read "The 10 Reasons Most ML Funds Fail" (https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf). His tooling discipline is *exactly* the antidote to overfitting an 80-ETF rotation with swept parameters.
- **Robert Carver (qoppac + pysystemtrade)** — https://qoppac.blogspot.com — the most rigorous *open* practitioner on your exact problems: vol-targeted sizing, continuous forecast scaling, leverage caps. His `-20..+20` continuous-forecast abstraction is a clean upgrade path to turn your binary gate into a weighted tilt. (Note: pysystemtrade moved to `pst-group/pysystemtrade` org as of Jan 2026.)
- **Neurotrader888** — https://github.com/neurotrader888 — **Monte Carlo permutation tests** (`mcpt`) and meta-labeling for retail. The single most directly borrowable creator: run MCPT on your momentum backtest to check the edge is real, not data-mined; meta-label the SJM gate to filter false "storm" signals.
- **Kevin Davey** — https://www.youtube.com/channel/UCjTZtWVBchDTJuxy_7GjySQ — verified competition track record; walk-forward + overfitting-avoidance discipline. Concept > code, but high-value process.
- **The SJM source papers** — "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach" (https://arxiv.org/abs/2402.05272) and the 2024 dynamic-factor-allocation companion (arXiv:2410.14841). This is the peer-reviewed foundation *under your own gate* — it gives you the author-intended way to pick the jump penalty (time-series CV optimizing your strategy's downside metric, not log-likelihood) and shows regime signals applied to factor/sector rotation. *Honest caveat:* validated on broad indices, not an 80-ETF rotation universe — validate transfer before relying.

### High signal — idea pipeline & staying current
- **PyQuantNews (Jason Strimpel)** — https://www.pyquantnews.com — best single radar for *which Python quant libraries are maintained and worth adopting*.
- **QuantSeeker (Substack)** — https://www.quantseeker.com — ex-bank/hedge-fund quant; weekly vetted filter over systematic-investing research, repeatedly surfacing your themes (regime rotation, cross-asset momentum).
- **Quantocracy** — https://quantocracy.com — daily aggregator of the credible quant blogosphere; top-of-funnel for implementable ETF-rotation/regime posts.
- **Alpha Architect** — https://alphaarchitect.com/blog/ — peer-reviewed momentum research (authors of *Quantitative Momentum*). Reference for momentum design.
- **Corey Hoffstein / Flirting with Models** — https://www.flirtingwithmodels.com — his "rebalance timing luck" work argues for *tranching* your weekly rebalance instead of one snapshot date — directly applicable to your weekly deep cycle.
- **Ernie Chan** — https://epchan.blogspot.com — meta-labeling concept (train a classifier on P(signal works | regime, vol, breadth)) to turn your binary gate into a probabilistic sizer. (Treat his PredictNow product claims skeptically; the books/blog are solid.)
- **machine-learning-for-trading (Stefan Jansen / ML4T)** — https://github.com/stefan-jansen/machine-learning-for-trading — best free ML-for-trading curriculum; the alpha-factor chapter (information coefficient, factor decay) gives a disciplined way to weight momentum + fundamentals and to test whether the fundamentals screen *adds* alpha over momentum alone.

### Useful but noisy (search/lurk, don't follow a feed)
- **r/algotrading** — https://www.reddit.com/r/algotrading/ — high beginner noise, but the broker-API threads and the reflexive "show me out-of-sample / include costs" culture are genuinely useful. Best for vetting IBKR quirks and avoiding known overfitting traps.
- **Quant Stack Exchange** — https://quant.stackexchange.com — the rigorous counterweight: ask *narrow* questions about jump-model vs HMM estimation, online updating, and label-leakage.
- **Algovibes / Code Trading / Part Time Larry** — solid, code-honest YouTube for plumbing patterns (pandas signal construction, multi-asset backtesting.py scaffolding, alert/notification I/O). Reference for *plumbing*, not strategy. Note: Part Time Larry is **not** Hudson & Thames — those are unrelated.
- **HMM regime tutorials (QuantConnect/QuantInsti/QuantStart)** — https://www.quantconnect.com/docs/v2/writing-algorithms/machine-learning/popular-libraries/hmmlearn — useful as a *benchmark* to cross-check your jump-model regime calls against a 2–3 state Gaussian HMM. (Jump models generally beat HMMs by avoiding frequent false-alarm flips — confirm that's true on your data.)

---

## 4. Avoid — hype, scams, dead repos

Plainly flagged so you don't spend money or attention:

- **"AI MEV bots" / Telegram "your wallet is flagged" lures** (https://lukka.tech/investigation-mev-bot-powered-by-ai-scam/) — outright scams: victims deploy a contract or paste a key, funds swept. Not your asset class, but worth knowing the pattern. **Never paste a key into any "bot dashboard," never deploy code from influencer/Telegram sources.**
- **Telegram signal-sellers / pump-and-dump rings** (e.g. PumpCell, ~$800k/month) — fake P&L, fabricated win rates, fake "verification" bots that harvest exchange API keys. Red flags: guaranteed returns, urgency, anonymous admins, deleted criticism, suspiciously round member counts. **Hard rule: never connect any third-party bot or "signal verifier" to an IBKR/exchange key.**
- **SaaS grid/copy bots — 3Commas, Pionex, Bitsgap** — crypto-only, auto-executing, and not your problem. Reality vs. claim: grid bots backtest +38% and run -18% live; **Pionex** ate a multi-state US consent order (May 2025) for unlicensed money transmission + AMF France blacklisting. Their backtesters are toys (flat trade-cost %, no slippage). Irrelevant to a US-ETF/IBKR/manual workflow.
- **Moon Dev (AlgoTradeCamp)** — real coding content wrapped in a paid-camp + "get funded $1,000+" funnel; light on out-of-sample rigor. Skim for LLM-agent plumbing ideas; **ignore the monetization and all performance claims.**
- **mlfinlab (Hudson & Thames)** — the *techniques* (purged CV, Deflated Sharpe, meta-labeling) are gold and free in the de Prado book; the *package* is now **paywalled** (£100/mo, closed-source, `pip install mlfinlab` 404s) and the open repo is a stub. Read the methods, **don't pay for the package, don't trust the open fork `mlfinpy`** (alpha, ~20 months stale) for production.
- **"best-of-algorithmic-trading" SEO repos** (TitanFlow-Systems, PlaceNL2026, etc.) and "Top 10 AI crypto bot" Medium listicles — keyword-stuffed, auto-generated, inflated "310K stars." Ignore their rankings. The credible curated lists are **awesome-quant** (https://github.com/wilsonfreitas/awesome-quant) and **awesome-systematic-trading**.
- **FinGPT-Forecaster as a signal source** — weights frozen at Sept-2023, DOW30-only; "biased demo" by its own README. The *FinBERT* sentiment baseline and the structured-output prompt schema are borrowable; the forecaster is not.
- **"$0/month GitHub-Actions-as-bot-runner" Medium tutorials** — useful pattern validation, but they undersell cron jitter (GitHub cron can drift 30–60 min and auto-disables after 60 days of no commits) and handle secrets loosely. "Bulletproof" is overselling. Read for the pattern, defend against the gaps (→ dead-man's-switch).
- **Backtrader as a new dependency** — dead since 2023, GPL-3.0, wrong shape. **Don't `pip install` it for a new project.**

**Base-rate honesty:** the widely-cited "bots lost 77× more per user / 80%+ of retail bot users lose money / ~95% of AI bots are repackaged RSI-MACD" stats are real-ish but mostly measure crypto wash-trading bots and naive indicator scripts — **not** a disciplined, regime-gated, tax-aware, human-in-the-loop ETF decision engine. They are an argument *for* your design, not a verdict on it. Don't internalize the pessimism as applying to your book — but do take the CFTC's blunt line as a guardrail: **"AI can't predict the future or sudden market changes."** Keep the LLM as a context/sentiment *input* under hard caps, never a predictive oracle.

---

## 5. What to borrow for YOUR bot specifically — prioritized

Six concrete moves, ordered by (impact × low effort):

**1. Add a dead-man's-switch (Healthchecks.io). [highest ROI, ~30 min, $0]**
Your email-only architecture is blind to the run that never fires or dies pre-email — precisely the failure that hurts during a real storm. Ping a free Healthchecks check as the *final* line of each cycle, after your data/regime/holdings asserts pass. Separate checks for weekly vs. hourly; route alerts to a second channel. This closes your single biggest operational blind spot. (https://healthchecks.io)

**2. Kill look-ahead in regime labeling + tuning. [correctness, high]**
Two parts: (a) audit your gate — if any regime labeling or *validation* uses full-window `.predict`-style logic, switch to the causal online method (`jumpmodels`' `predict_proba_online` is the reference for how to do this right). (b) Re-tune every swept parameter (momentum lookback, rank cutoff, SJM threshold) under **purged/embargoed walk-forward** via `skfolio.model_selection.WalkForward`/`CombinatorialPurgedCV` dropped into your existing sklearn `GridSearchCV` — no need to adopt any optimizer. This is the most likely place your bot is quietly fooling itself.

**3. Make tax + costs first-class in every backtest/ranking. [edge-defining, medium]**
None of vectorbt/bt/skfolio/Qlib models your 25–47% Israeli tax or IBKR/ILS friction — and that drag *dominates* a momentum rotation's net edge. Bolt an explicit per-rebalance cost + tax-on-realized-gains term onto your ranking *before* it greenlights a rotation, so the engine prefers low-turnover allocations. Cross-check against your existing IBKR→IL tax tool (the Moses floored-currency rule) so the two stay consistent. A pre-tax signal can rank trades that are net-negative after a 47% haircut.

**4. Turn the binary gate into a soft, validated throttle. [strategy upgrade, medium]**
Borrow CJM `predict_proba` (from vendored `jumpmodels`) + a hysteresis band to scale leverage/position caps continuously instead of flipping on/off — fewer whipsaw round-trips = less tax bleed. Optionally layer Chan/de Prado **meta-labeling**: a small sklearn classifier predicting P(momentum signal pays | regime, vol, breadth) as the position-size scaler. Validate the gate's *contribution* with a vectorbt A/B (gate ON vs OFF over identical history) and an `mcpt` permutation test (Neurotrader) so you know the edge is real, not data-mined.

**5. Add a parameter-sweep + tear-sheet research bench. [validation infra, medium]**
Stand up vectorbt (OSS) *offline* to sweep your momentum knobs across the ~80-ETF panel and look for robust plateaus; use `bt` + `ffn` (or `pyfolio-reloaded`) to generate Sharpe/Sortino/maxDD/"interesting-times" tear-sheets on your realized equity curve for the weekly email. Keep both out of the live cloud agents.

**6. Harden and diversify the data layer. [reliability, medium]**
Promote the IBKR MCP feed (`get_price_history`) to primary for the ~80 ETFs and cross-check against yfinance; demote yfinance to fallback + niche fundamentals. Make every external call (yfinance, LLM, IBKR) fail loud with hard timeouts + non-empty/staleness asserts, so a 429 or a hung call suppresses the email rather than producing a confident decision on stale data. Auto-reconcile `holdings.json` against `get_account_positions` each cycle.

**One LLM-specific guardrail threaded through #3/#4:** feed the news LLM **only** articles timestamped *before* the decision moment and make it cite dates — otherwise look-ahead/memorization (a documented failure mode of LLM-trading backtests) makes any backtest of the news layer overstate live edge (https://arxiv.org/html/2512.23847). Per the FinBen benchmark (https://arxiv.org/abs/2402.12659): trust the LLM for *parsing/summarizing/sentiment* (its strength), never for price targets or directional forecasts (its documented weakness) — keep direction coming from your SJM+momentum+fundamentals stack. Your current division of labor is already correct; just enforce the timestamp discipline.

---

**Closing honest take:** You're in the ~1-in-many camp that already built the disciplined thing most of this ecosystem only markets. Nothing here should replace your bot. The real wins are unglamorous: leakage-free validation, tax-aware ranking, a dead-man's-switch, a hardened data feed, and a soft regime throttle — borrowed as components from `jumpmodels`, `skfolio`, vectorbt, Healthchecks, and the de Prado/Carver/Neurotrader methodology canon, not adopted as frameworks.