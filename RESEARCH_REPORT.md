I'll write this report directly from the two inputs provided. No file reads or web fetches are needed — all findings, verdicts, and constraints are in the prompt. Let me synthesize.

# Frontier Trading-Edge & Learning-Landscape Report
**For: Israeli retail quant — daily bars on ~180 US instruments (1993–2026) + limited SPY 5m/30m, IBKR Interactive-Israel costs, 25%/47% CGT, USD/ILS friction.**
**Bar every idea must clear: beat GLD / managed-futures (DBMF/KMLM) / RSI2-basket *after* cost+tax+FX, ideally on daily/low-freq data.**

---

## The brutal one-paragraph summary

Almost every "exciting" 2024–2026 short-horizon equity edge in the candidate set is a high-turnover, cross-sectional, gross-of-cost, often-microcap or options-surface strategy that you either **cannot build** (no single-name IV, no TAQ, no point-in-time CRSP, no futures panel) or **cannot keep** (25–47% CGT realized continuously with no deferral, the exact wall that already killed your overnight effect, front-run-seasonality, and vol-managed work). Of ~50 candidate edges, **3 scored a verdict of `plausible-needs-testing` that is genuinely actionable for you** — and notably, **none of them is a return signal**. They are all *meta-layers* that wrap or gate the strategies you already own. The real frontier for your specific constraint set is **conditioning, sizing, and turnover-suppression**, not new alpha. The single best thing in the entire dossier is arguably the **Statistical Jump Model** (a low-turnover regime gate that, uniquely, was designed to minimize taxable round-trips) and the **meta-labeling** layer on your RSI2 basket.

---

## 1. Top findings — ranked by (novelty × survivability)

These are the only candidates that survived 3-skeptic adversarial review with a verdict better than "data/cost-prohibitive" *and* are testable on your data. Ranked by how much they could actually move your after-tax book.

### #1 — Statistical Jump Model (SJM) regime gate — *the one regime-timer built to dodge the tax wall*
- **Source:** Shu, Yu & Mulvey, arXiv [2402.05272](https://arxiv.org/abs/2402.05272)
- **Verdict:** `plausible-needs-testing` · novelty 3 · survives_costs **yes-on-daily-data** (the only regime overlay in the batch with this rating)
- **Mechanism:** An HMM-style regime classifier with an explicit **jump penalty λ** at every state transition, making the fitted state path far *stickier* than a raw HMM. Crucially, λ is tuned by walk-forward CV to **maximize net strategy Sharpe** — i.e. the model is penalized directly for generating taxable round-trips. De-risk (cut equity → T-bills) only in the high-vol state. Tested US/DE/JP 1990–2023 *with* costs and T+2 delay.
- **Why novel vs your stack:** Every prior regime/timing idea you've built or evaluated died on turnover→CGT (vol-managed ~7.5×/yr churn; front-run 0.61→0.23 Sharpe). SJM is the **first one whose entire design point is ~2–3 regime shifts/yr (44% turnover, ~0.4–0.7 round-trips/yr)** — roughly an order of magnitude below the 252-RT/yr wall. It is *not* continuous vol-targeting (which you killed) and *not* a CPD exit-gate (which you've already extracted). The transferable insight is "solve regime persistence as a **jump-penalized clustering** problem, tune the penalty on net P&L."
- **The honest caveat (all 3 skeptics agree):** Part of the vol/MDD reduction is mechanical beta-reduction (avg leverage ~75–84%); absolute Sharpe gain is modest (SPY 0.48→0.68); the HMM baseline is weak (no comparison to vol-targeting); single study, US-favorable window. **The real open question is whether it beats *buy-hold DBMF after tax*, not whether it survives costs** — it survives costs; it just may not beat what you already hold.
- **Concrete next test:** `pip install jumpmodels`. Fit a 3-state JM on SPY excess-return features (EWM downside-dev hl=10, EWM Sortino hl=20 & 60), rolling 6-mo Θ refit, monthly λ re-select on validation Sharpe. Bear state → SHY/BIL, else SPY. Push the realized ~2–3 shifts/yr through `cost_model.py` + 25%/47% IL CGT (no deferral) + USD/ILS at T+2. **Must beat: buy-hold SPY, the DBMF/KMLM overlay, AND a plain VIX-percentile gate on the same SPY** (net-of-tax Sharpe & MaxDD). **30-min pre-test:** count realized regime shifts/yr on YOUR SPY — if materially > ~3/yr, the tax-wall advantage erodes and it collapses toward the vol-managed failure. If it clears, re-run on EWG/EWJ as cheap external-validity checks against the paper's DAX/Nikkei.

### #2 — Meta-labeling on the RSI2 basket — *a turnover-REDUCING filter, working with your tax wall not against it*
- **Source:** López de Prado AFML Ch.3 / Hudson & Thames — [whatworksintrading.substack](https://whatworksintrading.substack.com/p/meta-labeling-the-technique-that)
- **Verdict:** `plausible-needs-testing` · novelty 4 (relative to your built stack) · survives_costs **unknown** (genuinely — depends on whether OOS precision lift is real)
- **Mechanism:** Keep the primary rule (RSI(2)<10 + 200d-uptrend) to decide *direction*; train a separate binary classifier on signal-time features to predict P(this specific trade wins) and use it **only to veto/skip trades** (and optionally size). Side and size decoupled; the ML never invents trades.
- **Why novel vs your stack:** grep of `~/Desktop/pair` confirmed **no ML/meta-label/p_win/kelly code anywhere** — this is a genuinely new architectural pattern for you. And it is the *first idea in the whole batch that attacks turnover/precision rather than chasing more signals*: it can only **remove** trades → fewer taxable realizations → more deferral → the exact asymmetry that makes your GLD/RSI2 baselines win works *for* you.
- **The honest caveats:** (1) The cited evidence is near-worthless — the EURUSD/PredictNow backtest is 1-min, gross, ~15mo OOS, the "3-4× Sharpe / >50%" claims are unsubstantiated marketing. The *technique* is sound; the *evidence* is not. (2) Real risk is overfitting: a single-name RSI2 yields too few labels; you **must pool all trades across the 180-instrument basket** to get enough labels, and use **purged + embargoed walk-forward CV** (AFML Ch.7). (3) The likely honest outcome is **after-tax CAGR drops while Sharpe/MaxDD improve** — a precision filter can cull the right-tail winners the rule's positive skew depends on. Judge on after-tax CAGR, *not* just Sharpe.
- **Concrete next test:** On the 180-instrument basket, log a feature vector at each entry (RSI(2) level, distance below SMA5/10, % above SMA200, 20d realized vol, ATR%, drawdown-from-20d-high, VIX level/change, sector-ETF 5d return, day-of-week) + label = did this trade exit net-profitable. Pool all ~thousands of trades, train a gradient-boosted classifier under strict purged+embargoed walk-forward (train pre-2015, test 2015–2026). Apply P(win)>threshold as a **binary skip/take filter** (do not chase continuous sizing first). Run filtered stream through `cost_model.py` + `after_tax(...,0.25)/(...,0.47)`. **Must beat:** unfiltered basket AND GLD/managed-futures on after-tax CAGR *and* after-tax Sharpe — and explicitly confirm the filter isn't just discarding right-tail winners (compare kept-vs-skipped per-trade return distributions). **Control:** re-run with leaky random-CV to quantify how much leakage would have inflated it; swap in plain LogisticRegression — if it matches GBM, the model adds nothing.

### #3 — CPD / BOCPD changepoint as a low-turnover EXIT gate — *the one transferable kernel from the CTA literature*
- **Source:** Wood/Roberts/Zohren "Slow Momentum with Fast Reversion" [2105.13727](https://arxiv.org/abs/2105.13727) + CTA-replication re-eval
- **Verdict:** `plausible-needs-testing` · novelty 3 · survives_costs **no** for the full LSTM/fast-reversion book, but the EXIT-gate kernel is low-turnover by construction
- **Mechanism:** Discard the neural net and the high-turnover fast-reversion leg entirely. Keep only **a Bayesian Online Changepoint Detector (Adams–MacKay 2007) used purely as an exit/risk-off gate** on a slow trend sleeve: stay in the trend, cut to flat for K days when a regime-break severity score fires.
- **Why novel vs your stack:** You have entry rules and crude exits, but **no explicit regime-break exit gate**. BOCPD fires only on rare detected breaks (~a handful of extra exits/yr), so it adds almost no taxable turnover — categorically different from the high-turnover ideas that died on your tax wall.
- **The honest caveat:** The full paper's 2.16 Sharpe is gross and survives only to ~2bps; the headline is concentrated in 2015–2020. The kernel's realistic win is **lower MaxDD, probably not higher net Sharpe** (the extra regime-exit turnover realizes CGT a plain trailing stop avoids).
- **Concrete next test (cheap 1-hr pre-test first):** Plot the BOCPD severity score νₜ on SPY/DBMF and check whether it **fires *before* known crisis drawdowns or merely lags them** — if it lags, it only adds taxable turnover for no protection; dead on arrival. If it leads: on DBMF + a simple in-house 200d/12-1 slow-trend sleeve, run gated vs ungated through `cost_model.py` + 25% CGT + USD/ILS. **Must beat:** the same trend sleeve with a plain trailing-stop/vol-target exit, AND buy-hold DBMF, on net-of-tax return/MaxDD.

### #4 — Vol-targeting, applied *selectively* (equity/credit sleeves only) — *a free turnover saving via a negative result*
- **Source:** QuantPedia ["An Introduction to Volatility Targeting"](https://quantpedia.com/an-introduction-to-volatility-targeting/)
- **Verdict:** Multiplier, not an edge — but the **negative result is the actionable part**.
- **Mechanism / why it matters for you:** Vol-targeting lifts Sharpe meaningfully **only for risk assets with vol clustering + a negative vol/return (leverage) relationship** (equities, credit). For bonds, FX, commodities, and managed futures, **the Sharpe gain is negligible** — so vol-targeting those sleeves just pays turnover (and CGT) for nothing. The money-saving takeaway: **do NOT vol-target your DBMF/KMLM/GLD/bond sleeves.** Only consider it on the RSI2/equity sleeve, and even there your own vol-managed repro (–0.025 OOS Sharpe vs buy-hold, gross) is a warning.
- **Concrete next test:** None needed as a return-hunt — you already proved single-series vol-scaling fails for you. Use it purely as a **rule: vol-target nothing, or at most the equity sleeve with a wide hysteresis band and annual rebalance.** This is a guardrail, not a project.

### #5 — Deflated Sharpe / PBO retro-audit — *the cheapest, highest-EV thing in the entire report*
- **Source:** López de Prado / Hudson & Thames; Bailey-LdP DSR & PBO/CSCV
- **Verdict:** Methodology, zero turnover, survives trivially. Novelty for you: you almost certainly *don't* formally do this across your search history.
- **Mechanism:** You have prototyped ≥8 strategies + adversarially evaluated ~50 findings. That is a massive multiple-testing surface. **Deflated Sharpe haircuts your best-looking Sharpe (the RSI2 basket) for the number of trials you ran.** If the RSI2-basket Sharpe is not distinguishable from the best-of-N-random expectation after deflation, even your surviving baseline is partly luck.
- **Concrete next test (an afternoon):** Take your full list of tried variants (RSI2 single + basket, 3-down-days, Donchian-20, ORB, intraday-momentum, overnight, pairs, put-writing) as N trials. Compute DSR for the RSI2 basket from: its Sharpe, the cross-trial Sharpe variance, sample length, skew, kurtosis. **This is a guardrail that protects every other decision in this report.** Run it first on the *known-dead* overnight effect to validate the guardrail against a true negative.

---

## 2. Frontier & emerging (2025–2026) — honestly flagged for hype

| Idea | Source | What's real | The hype / why it's not yours |
|---|---|---|---|
| **DRIF — daily-return "timing > magnitude" reversal** (Cakici, Fieberg, Neszveda, Bianchi, Zaremba) | [SSRN 6005614](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6005614) | Genuinely non-obvious insight: *which day* recent moves occurred dominates their *magnitude*; unifies short-term reversal / MAX / ivol; expanding-window OOS. novelty 3. | EW (2.26%) ≫ VW (1.57%) = **small-cap fingerprint**. It's a decile L/S over thousands of CRSP names — **your universe is ~120 mega-caps + ETFs, the LOW-MAX, low-reversal end.** Monthly full-turnover, short leg, "net" is a breakeven *argument* (36–42bps vs assumed 10–20bps institutional), not a net backtest. Dies on your tax wall. *But the kernel — "recency of the down-move" — is worth one cheap placebo test on your RSI2 entries.* |
| **DeltaLag — learned dynamic lead-lag NN** | [2511.00390](https://arxiv.org/abs/2511.00390) | Real methodological twist (per-pair, time-varying lag via sparse cross-attention). | ~25-33% AR / Sharpe 2.1-2.9 is **GROSS, no turnover reported, single seed, 2yr OOS**. It's a daily-rebalanced 10/10 decile L/S over 500–1140 names — needs point-in-time cross-section you don't have; the per-order $2.50 min alone swamps a 10bp/day edge. Same family as the attention-factors paper you already shelved. |
| **Salience / weekly-MAX reversal refinements** | [4649393](https://papers.ssrn.com/sol3/Delivery.cfm/4649393.pdf), [4622831](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4622831) | Theory-elegant conditional reversal. **Skeptic-3 literally reproduced MAX×reversal on your own data: spread ≈ 0.008%/wk, t=0.07 vs the paper's 1.66%/wk.** | Confirmed: the alpha lives in the microcap/lottery tail you cannot trade. Dead on your liquid universe before costs. |
| **Eksi-Roy RV–IV-spread / abnormal-turnover hygiene** | [5234112](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5234112) | The *portable kernel* (turnover shocks transiently inflate realized vol → noisier RV signals) is real and daily-computable. novelty 3. | The full L/S needs single-name IV you don't have. Only the **abnormal-turnover screen** transfers. |
| **0DTE attenuation / GEX / VIX1D / index-reconstitution / closing-auction imbalance / EOD reversal (Baltussen)** | [5039009](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5039009), [DSPX](https://resonanzcapital.com/insights/dispersion-trading-and-the-dspx-index), VIX1D [fut.70023](https://onlinelibrary.wiley.com/doi/full/10.1002/fut.70023) | Real microstructure phenomena. | **All require intraday/options/auction data you lack** and live at intraday frequency where your tax+commission wall is fatal. The EOD-reversal "FINDING" was also *materially mis-described* (it's same-day intraday, not overnight; mechanism is the one the authors *reject*). |
| **LLM-agent trading (TradingAgents, agentic surveys, EIA-speed)** | [2412.20138](https://arxiv.org/abs/2412.20138), [2605.19337](https://arxiv.org/abs/2605.19337) | The *audit survey* is honest and confirms your priors (2/19 cost models, 0/19 reproducible). | TradingAgents' headline = 3 stocks, Q1-2024, gross, **inside the LLMs' training window (look-ahead by memorization)**. EIA-speed edge is a latency race you structurally lose from Tel Aviv. Hype as tradable alpha. |
| **Crypto (spot-perp carry, perp factor zoo, Hyperliquid HLP, liquidation reversal, BTC seasonality)** | [2510.14435](https://arxiv.org/html/2510.14435v2), [HLP](https://0xian.substack.com/p/understanding-hyperliquids-hlp-vault) | Carry decay is real evidence; HLP is genuinely novel (novelty 4, no equity analogue). | **Carry Sharpe 6.45 full-sample → NEGATIVE 2025.** Everything else needs perp/funding/on-chain data you don't have, an offshore venue, and Israeli ordinary-income tax on funding. HLP is short-gamma with a governance-backstopped fat tail (JELLY ~27% DD). Not your rails. |

**Net frontier read:** The genuinely new academic work clusters around (a) cross-sectional reversal refinements that live in microcaps you can't trade, (b) options/intraday microstructure you can't observe, and (c) ML cross-sectional L/S that dies on turnover×tax. The *transferable* frontier ideas are all **measurement/construction hygiene** (DRIF's timing decomposition, Eksi-Roy's turnover screen, jump-penalized regime persistence), not signals.

---

## 3. Probably won't survive costs/tax — here's why (don't waste time)

Grouped by failure mode so you can pattern-match future ideas instantly.

**A. Data-prohibitive (you cannot even build the signal):**
- End-of-Day Reversal, Maxing-Out reversal, RV–IV spread, Intraday Option Reversals ([5081696](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5081696)), Option-skew/index-inclusion ([4192142](https://papers.ssrn.com/sol3/Delivery.cfm/4192142.pdf)), Option Volume Imbalance ([2201.09319](https://arxiv.org/abs/2201.09319)), Attention Factors, SNAP deep asset pricing ([2509.04812](https://arxiv.org/abs/2509.04812)), Expected-Returns-with-LLMs ([4416687](https://papers.ssrn.com/sol3/Delivery.cfm/4416687.pdf)), FinGPT sentiment, Interday CSM, DeltaLag, Anomaly-concentration/"overlap stocks" ([6005614](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6005614) is DRIF; overlap is a separate Sobotka paper), convertible-bond lead-lag, cross-chain MEV. **Common thread:** need TAQ / single-name IV / point-in-time CRSP / fundamentals / futures panel / options chains / on-chain data — none of which is in your daily-bar ETF/large-cap set.

**B. Cost+tax-prohibitive (real but high-turnover → 25-47% CGT with no deferral kills it):**
- Front-running seasonality (your own repro: CAGR 9.5%→2.3%, Sharpe 0.61→0.23 on 25% CGT; +1 offset failed a placebo sweep — [5119553](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5119553)), FOMC even-week timer ([2687614](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2687614), also decayed post-2016), DeePM/Deep-Momentum HFT-turnover trend ([0331391](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0331391); gross, dies by 5bps, authors say "economically infeasible"), 0DTE iron condors ([5285330](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5285330); 3,100-trade sample is a look-ahead artifact — daily SPX 0DTE didn't exist pre-2022), RL portfolio allocation ([2605.27848](https://arxiv.org/abs/2605.27848); degenerate 2-rule heuristic mislabeled "RL", Sharpe ties equal-weight).

**C. Already-built / decayed / wrong-direction (no genuine new twist for you):**
- Korea NXT overnight reversal ([6752783](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6752783); self-defeating — the paper *is* an obituary; ETF-level EWY shows no reversal). Fast-slow-arbitrage flows ([3675163](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3675163); the FINDING's "contrarian" sign is **backwards** — the paper documents continuation, plus non-causal full-sample band-pass look-ahead). Single-stock 0DTE gamma-trap ([Citadel notes](https://www.citadelsecurities.com/news-and-insights/march-macro-checklist/); instrument only live ~Jan 2026, ~95 days of history, no GEX data). SVOL/VRP ETF wrapper ([Simplify](https://www.simplify.us/blog/volatility-premium-harvesting-reimagined); monthly distribution destroys deferral — strictly *worse* than your own put-writing for an IL taxpayer; live CAGR ~7-8% < SPY with -33.5% DD). LETF vol-decay shorts (the decay you'd "harvest" cancels against borrow; net the expense ratio).

**D. Crowded / debunked / self-refuting:**
- Short-interest "crash-then-normalize" ([0378426625000561](https://www.sciencedirect.com/science/article/pii/S0378426625000561)): the "71%/12.7%" numbers are a **Medium tutorial confabulation** grafted onto an unrelated prevalence paper; evidence quality "none." Factor-crowding alpha-decay ([2512.11913](https://arxiv.org/abs/2512.11913)): **the author WITHDREW the paper** 2025-12-27 because Sections 5-7 don't support the claims. Anomalies-once-public ([Jacobs-Müller](https://ideas.repec.org/a/eee/jfinec/v135y2020i1p213-230.html)): real, but it's gross, the surviving edge lives exactly where frictions are worst, and "no decay outside US" is contradicted by its own text. Vol-managed portfolios ([JoF 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13395)): you already reproduced the OOS failure (-0.025 Sharpe vs buy-hold, gross); DeMiguel's rescue needs a 9-factor L/S desk + trade-netting you can't build.

---

## 4. Multipliers — techniques that improve ANY signal you already own

These don't generate alpha; they reshape the risk/turnover/tax profile of your existing book. Ranked by EV for your constraints.

1. **Deflated Sharpe / PBO / purged-embargoed CV** (López de Prado, Hudson & Thames). Zero turnover, protects every decision. *Do this first.* Source: [hudsonthames.org/research](https://hudsonthames.org/research/).
2. **Statistical Jump Model as a regime gate** (#1 above) — the rare regime overlay whose turnover is low enough to clear your tax wall.
3. **Meta-labeling as a binary trade filter** (#2 above) — turnover-*reducing*, works with the tax asymmetry.
4. **BOCPD changepoint as an exit gate** (#3 above) — for MaxDD reduction, not Sharpe.
5. **Selective vol-targeting** (#4): vol-target *only* the equity sleeve, never bonds/commodities/MF. The negative result saves turnover.
6. **Kelly / fractional-Kelly — STATIC only.** ([Kelly criterion](https://en.wikipedia.org/wiki/Kelly_criterion)) The *static* version (set half-Kelly relative weights across your sleeves once, annual rebalance, wide no-trade band) is tax-neutral and worth wiring in. The *dynamic shrinkage-Kelly* version resizes continuously → dies on your tax wall exactly like vol-managed. Pre-test: if each sleeve's mean t-stat < ~2, the Kelly fraction should collapse toward equal-weight (the overlay adds nothing).
7. **Carver "dynamic optimization" / buffered trading** ([systematicmoney.org](https://www.systematicmoney.org/advanced-futures), pysystemtrade) — integer-contract greedy optimizer + no-trade buffers to approximate a many-instrument risk-parity book at retail size and cut turnover. The genuinely under-used piece for your 180-name basket.
8. **Tranching / rebalance-timing-luck** ([Allocate Smartly](https://allocatesmartly.com/timing-luck-portfolio-tranching/)) — **use as a diagnostic, not a trade.** Measure how much of your reported Sharpe is just the rebalance day you picked. Do *not* tranche to N=8 (multiplies taxable realizations); at most N=2–3 on a low-frequency sleeve.
9. **EVaR / worst-window objective for sleeve weights** (DeePM kernel, [2601.05975](https://arxiv.org/abs/2601.05975)) — annual sleeve-weight optimization on a tail-aware objective; likely buys lower net-of-tax MaxDD, not higher Sharpe. Low priority.
10. **Closing-auction (MOC/LOC) routing for close-signalled daily rules** ([Goyal-Jegadeesh-Wu JFQA 2026](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/0F72910A79C5B42CF6E85F55164CE846)) — execution hygiene only. At retail size price *impact* is ~0; the real (tiny, ~1–3bp) benefit is spread avoidance via a single auction price. Validates your existing close-fill backtest assumption rather than improving it. Route your RSI2/Donchian exits as LOC, not market.

---

## 5. Learning landscape — where to actually spend time

Signal/noise ratings are for *your* profile (experienced daily-bar quant who already builds and backtests with a cost/tax model). A resource that's 5/5 for a beginner can be 2/5 for you.

### Tier 1 — Highest signal, directly relevant to your binding constraints
- **Ernie Chan — [epchan.blogspot.com](http://epchan.blogspot.com/) / predictnow.ai.** The clearest articulation of the "ML as overlay (meta-labeling, conditional parameter optimization / regime-conditioning), NOT as primary signal" thesis — exactly how you should apply ML to your RSI2/Donchian/pairs book. Lowest-hype source in the set. **S/N 5/5.** (Discount the predictnow product; the public blog is the value.)
- **Robert Carver — *Advanced Futures Trading Strategies* + [systematicmoney.org](https://www.systematicmoney.org/advanced-futures) + pysystemtrade.** Honest (full system ~0.5 Sharpe). The **dynamic-optimization / buffered-trading** chapter is the genuinely under-used piece for running a 180-instrument daily basket at retail size. **S/N 5/5.**
- **Hudson & Thames + PyQuant Newsletter — [hudsonthames.org](https://hudsonthames.org/research/), [pyquantnews.com](https://www.pyquantnews.com/).** Production-grade purged/embargoed + combinatorial CV, meta-labeling, deflated Sharpe. This is your **overfitting-defense toolkit** — methodology, no decay. (Discount the "ML alpha" framing.) **S/N 4.5/5.**
- **QuantSeeker (Substack) — [quantseeker.com](https://www.quantseeker.com/p/popular-investing-research-in-2025).** The single best continuous discovery feed for low-turnover, implementable, cost-aware academic ideas — it pre-filters for turnover/data-frequency, exactly your concerns. Most of section 1–2's leads came from here. Subscribe and treat as top-of-funnel. **S/N 4.5/5.**

### Tier 2 — High signal, portfolio-construction / research-process depth
- **Flirting with Models / Newfound Research (Corey Hoffstein) — [flirtingwithmodels.com](https://www.flirtingwithmodels.com/), [blog.thinknewfound.com](https://blog.thinknewfound.com/).** Best thinking on *combining* the strategies you already have (RSI2 = concave/short-vol; trend = convex/long-vol) into a payoff-shaped ensemble, plus rebalance-timing-luck and return-stacking. Zero hype, high density. **S/N 4.5/5.**
- **Robot Wealth / "Brave New Backtest" (Kris Longmore) — [robotwealth.com](https://robotwealth.com/blog/).** 2026 contrarian theme: LLMs make hypothesis-generation cheap → multiple-comparisons false discoveries explode → FDR control / deflated Sharpe is the binding constraint. Directly relevant to your 50-findings, 180-instrument search surface. **S/N 4/5.**
- **Concretum / Zarattini — [concretumgroup.com](https://concretumgroup.com/papers/).** Known-good methodology (their ORB work is in your stack). The **position-sizing comparison** (vol-targeting vs vol-parity vs pyramiding) holds the signal fixed and varies only sizing — clean attribution most retail content skips; use to pick the lowest-turnover sizing. **S/N 4/5** (the LETF/intraday papers themselves are not yours to trade).
- **Quantpedia Premium — [quantpedia.com](https://quantpedia.com/strategies/).** A faster idea-funnel into the published-anomaly landscape, with a portfolio-construction layer. But you're already a manual Quantpedia (you front-ran their seasonality paper and killed it). **Buy only if a free-tier funnel test yields ≥1 IL-tax-survivable survivor.** **S/N 3/5 for you specifically** (lower than for most because you already do the downstream net-of-tax validation it omits).

### Tier 3 — Aggregators & community (mine for leads/failure-data, not alpha)
- **Quantocracy — [quantocracy.com](https://quantocracy.com/).** The "Hacker News of quant"; one feed over hundreds of blogs. Durable infrastructure, not a strategy. **S/N 4/5 as a discovery layer.**
- **Quantitativo — [quantitativo.com](https://www.quantitativo.com/).** Fully-specified backtested daily rules; the **RSI2 *threshold-curve ensemble*** (Sharpe-weighted across RSI thresholds 5–30) is a mild, under-discussed robustness twist on your existing RSI2 basket worth one OOS test (verify it beats equal-weighting the variants — it often barely does). **S/N 3.5/5.**
- **r/algotrading + r/quant.** Anti-signal on posted strategies (anything public is crowded), but uniquely valuable for **failure data and broker/cost/tax friction reports** blogs omit. **S/N 2/5 strategies, 4/5 friction reality-checks.**
- **Top Traders Unplugged / SG Trend / TTU indices — [toptradersunplugged.com](https://www.toptradersunplugged.com/podcast-series/systematic-investor/).** Free CTA benchmarks to sanity-check whether your DBMF/KMLM overlay is tracking real trend beta. **S/N 3/5 (benchmark utility).**

### Free credentials / courses (you likely don't need them)
- **WorldQuant University MScFE (free, accredited) + BRAIN — [wqu.edu/mscfe](https://www.wqu.edu/mscfe).** Free is free; BRAIN's alpha-expression formalism teaches cost/turnover-aware cross-sectional signal generation. But BRAIN alphas are tuned to WQ's universe/costs and don't transfer to your 180-name book. **S/N 3/5 (BRAIN as a feature generator).**
- **CQF / QuantInsti EPAT / Coursera ML-for-Trading — [cqf.com](https://www.cqf.com/about-cqf/program-structure/program-overview), [quantinsti.com](https://www.quantinsti.com/).** Credential/structure plays, expensive, front-loaded with material you already know. EPAT's execution/broker-integration content is the only differentiator. **S/N 2/5 for you** — honestly, you don't need them.

### Treat skeptically
- **SpotGamma / SqueezeMetrics GEX/DIX — [spotgamma.substack](https://spotgamma.substack.com/p/the-new-volatility-regime).** The academic 0DTE-attenuation papers undercut the "gamma bomb" narrative; GEX is a noisy proxy and the actionable use is intraday (data you lack). Only the **free daily DIX/GEX series as a candidate daily regime feature** is worth probing. **S/N 2.5/5.**
- **Macro Hive Deep Dives, setup4alpha listicle, Quant Galore, The Quant Stack.** Useful context/curation; none surfaces a new tradable edge for your constraints. **S/N 2–3/5.**

---

## 6. What I'd test next in the `pair/` folder — prioritized specs

Ordered by EV given your data + cost/tax/FX model. Each notes the baseline it must beat. Pre-tests are explicitly cheap kill-switches so you waste minimal time.

### Experiment 1 (do first — protects everything else) — **Deflated-Sharpe retro-audit**
- **Spec:** Treat your full tried-strategy list as N trials. Compute DSR (Bailey–LdP) for the RSI2 basket from: observed Sharpe, cross-trial Sharpe variance, sample length, return skew & kurtosis. Validate the harness on the *known-dead* overnight effect first.
- **Cost/tax:** N/A (analytic).
- **Must beat:** Itself — if RSI2-basket DSR isn't distinguishable from best-of-N-random, downgrade your confidence in the baseline you benchmark everything against.
- **Effort:** ~1 afternoon. **Highest EV in the report** because it re-weights the credibility of your entire stack.

### Experiment 2 — **Statistical Jump Model regime gate on SPY**
- **Pre-test (30 min):** Fit 3-state JM, count realized regime shifts/yr on your SPY 1993–2026. If > ~3/yr → abort (tax-wall advantage gone).
- **Spec:** `jumpmodels` 3-state JM, features = EWM downside-dev hl=10 + EWM Sortino hl=20/60 on SPY excess-over-BIL returns; rolling 6-mo Θ refit, monthly λ re-select on validation Sharpe; bear state → BIL/SHY, else SPY; T+2 execution.
- **Cost/tax:** `cost_model.py` (Interactive Israel $0.01/sh, $2.50 min) + 25%/47% CGT no-deferral + USD/ILS.
- **Must beat:** buy-hold SPY **AND** the DBMF/KMLM overlay **AND** a plain VIX-percentile gate on the same SPY, on net-of-tax Sharpe + MaxDD. External validity: re-run on EWG, EWJ.
- **Effort:** ~1 day.

### Experiment 3 — **Meta-labeling filter on the 180-instrument RSI2 basket**
- **Pre-test (30 min):** Compute RSI2-basket trade win-rate conditioned on VIX-percentile buckets. If flat across buckets → the meta-model has nothing to learn; stop.
- **Spec:** Log signal-time features per entry (RSI2 level, dist below SMA5/10, %>SMA200, 20d vol, ATR%, 20d-drawdown, VIX level/change, sector-ETF 5d return, day-of-week); label = net-profitable exit. Pool all trades; gradient-boost under purged+embargoed walk-forward (train pre-2015 / test 2015–26). Binary P(win)>threshold skip/take filter (no continuous sizing yet).
- **Cost/tax:** full IL stack + `after_tax(0.25)`/`after_tax(0.47)`.
- **Must beat:** unfiltered RSI2 basket AND GLD/managed-futures, on **after-tax CAGR *and* Sharpe** (confirm the filter isn't culling right-tail winners; control with leaky-CV and a LogisticRegression baseline).
- **Effort:** ~2–3 days.

### Experiment 4 — **BOCPD exit-gate on a slow-trend/DBMF sleeve**
- **Pre-test (1 hr):** Plot BOCPD severity νₜ on SPY/DBMF; confirm it *leads* (not lags) 2008/2020/2022 drawdowns. If it lags → abort.
- **Spec:** On DBMF + an in-house 200d/12-1 slow-trend ETF sleeve, add a νₜ-threshold exit (flat for K days on fire).
- **Cost/tax:** full IL stack.
- **Must beat:** the same sleeve with a plain trailing-stop/vol-target exit, AND buy-hold DBMF, on net-of-tax return/MaxDD. (Realistic win = lower MaxDD, not higher Sharpe.)
- **Effort:** ~1 day.

### Experiment 5 (cheap placebo, only if curious) — **DRIF "timing > magnitude" kernel on your liquid universe**
- **Spec:** Single-name elastic-net (or OLS) mapping each instrument's last 21 daily returns → next-21-day return, expanding-window; monthly cross-sectional long-short quintile over the 180 names. **Placebo control:** compare chronological-component model vs magnitude-only vs **shuffled-day-order** — if shuffling doesn't degrade it, the "timing" mechanism doesn't replicate outside CRSP small-caps (expected).
- **Cost/tax:** full IL stack + 25%/47% CGT on monthly realizations vs tax-deferring GLD/MF/RSI2.
- **Must beat:** GLD / managed-futures / RSI2 basket net-of-tax. **Prior:** fails on the same tax wall as front-run-seasonality; run mainly to confirm the kernel is microcap-locked and close the question.
- **Effort:** ~1 day.

---

## Bottom line for your next quarter

Spend your time on **Experiments 1→3** in order. Experiment 1 (deflated Sharpe) is a free credibility audit of your whole stack. Experiment 2 (Statistical Jump Model) is the single most promising genuinely-new idea that respects your tax wall by design. Experiment 3 (meta-labeling) is the highest-novelty addition to your stack and structurally tax-favorable. Everything sexier in the 2024–2026 literature — the cross-sectional reversal refinements, the ML lead-lag nets, the options-microstructure alphas, the crypto carry — is either unbuildable on your data or dies on the 25–47% CGT-with-no-deferral wall you have already mapped four separate times. The frontier *for you* is conditioning, sizing, and turnover suppression, not new signals.