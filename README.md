# Quant Bot — storm-detector + momentum + news/fundamentals

A defensive-momentum trading **decision** engine for an Israeli retail investor (US ETFs via
Interactive Israel / IBKR). It decides what to buy/sell/hold; **you place the trades** (no auto-trading).

All results are **after Interactive-Israel costs + 25%/47% Israeli capital-gains tax + USD/ILS FX**.

## What it does
- **Storm-detector (Statistical Jump Model):** low-turnover regime gate — pulls to cash before crashes.
  Beats buy-hold after tax on 10/13 world indices; cuts worst drawdown roughly in half.
- **Momentum discovery:** scans ~80 liquid ETFs, ranks by 3/6/12-month trend, recommends the top names.
- **Fundamentals screen:** P/E, margins, debt, analyst-target-vs-price per pick (live snapshot).
- **News read:** live web read on each pick → CONFIRM / CAUTION / VETO (catches froth a chart can't see).
- **Position management:** reads `holdings.json`, says HOLD / ADD / TRIM / SELL each current position.
- **Guardrails:** leverage ETFs capped at 20% of book, single position capped at 35%.

## Core files
- `the_bot.py` — the unified local one-shot (detect → discover → read → manage → orders).
- `cloud_bot_runner.py` — entry point for the **cloud scheduled agent** (`weekly` / `hourly` modes).
- `jump_model.py` — the Statistical Jump Model storm-detector.
- `early_scanner.py` / `momentum_engine.py` — momentum discovery + backtests.
- `fundamentals.py` / `news_check.py` — the reading layers.
- `core_portfolio.py` / `core_satellite.py` — long-hold core + core/satellite blends.
- `holdings.json` — YOUR current positions (edit when you trade). `news_verdicts.json` — live verdicts.
- `bars_*_1d.parquet` — daily price data (seed; the cloud agent refreshes via `free_fetch.py`).

## Cloud bot (scheduled)
- **Weekly:** `python3 cloud_bot_runner.py weekly` → refresh data, full analysis, email the order sheet.
- **Hourly:** `python3 cloud_bot_runner.py hourly` → emergency watch; notify only if action may be needed.

Requires: `pip install pandas numpy scipy scikit-learn statsmodels yfinance pyarrow`.

## Honest limits
- Decisions only — does NOT place trades. Interactive Israel has no retail trading API; you click the orders.
- Signals are daily — re-checking intraday rarely changes them (the hourly job is a guard, not a re-analysis).
- Fundamentals/news are current snapshots, not point-in-time → live screening, not backtestable.
