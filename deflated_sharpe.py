# Experiment 1 — Deflated-Sharpe / PSR retro-audit (Bailey & Lopez de Prado).
# You have tried >=8 strategy families on SPY. That is a multiple-testing surface: the
# best-looking Sharpe is biased UP by the number of trials. The Deflated Sharpe Ratio (DSR)
# haircuts your best Sharpe by the expected maximum Sharpe under N independent NULL trials,
# correcting for sample length + return skew/kurtosis (fat tails inflate naive significance).
#
# DSR = PSR evaluated against the deflated benchmark SR0* = E[max Sharpe under the null].
# DSR is P(true Sharpe > SR0*). Rule of thumb: DSR > 0.95 => the edge survives the haircut.
#
# We also VALIDATE the harness on a known-dead strategy (the overnight effect you already
# killed): a correct harness must NOT certify it. Gut-check before trusting it on RSI2.
from __future__ import annotations
import math
import numpy as np, pandas as pd
from scipy.stats import norm

DATA = "."; COST_BPS_RT = 3.0; RF = 0.025; MAXHOLD = 10


# ---------- shared helpers (same idiom as swing_test.py) ----------
def load_ohlc(sym):
    d = pd.read_parquet(f"{DATA}/bars_{sym}_1d.parquet")[["ts", "open", "close"]].copy()
    d["ts"] = pd.to_datetime(d["ts"]).dt.tz_localize(None)
    return d.sort_values("ts").set_index("ts")


def rsi(s, n):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    return 100 - 100 / (1 + up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean())


def run_swing(c, entry, exit_, maxhold=MAXHOLD):
    """Long-only swing: enter on `entry`, exit on `exit_` or after maxhold days. Returns net daily returns."""
    n = len(c); pos = np.zeros(n); holding = False; et = None
    e = entry.values; x = exit_.values
    for t in range(1, n):
        pos[t] = 1.0 if holding else 0.0
        if not holding and e[t]:
            holding = True; et = t
        elif holding and (x[t] or (t - et) >= maxhold):
            holding = False
    pos = pd.Series(pos, index=c.index); ret = c.pct_change().fillna(0)
    turn = pos.diff().abs().fillna(0)
    return pos * ret + (1 - pos) * (RF / 252) - turn * (COST_BPS_RT / 1e4) / 2


def sharpe(r, ppy=252):
    r = r.dropna()
    return (r.mean() * ppy) / (r.std() * math.sqrt(ppy)) if r.std() > 0 else 0.0


# ---------- Deflated Sharpe machinery ----------
def psr(sr_hat, sr0, n, skew, kurt):
    """Probabilistic Sharpe Ratio: P(true SR > sr0) given estimation error, skew, kurtosis.
    sr_hat, sr0 in the SAME (per-period) units; n = number of observations.
    Variance of the Sharpe estimator (Mertens/Lo, used by Bailey-LdP):
        se^2 = (1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1)"""
    se = math.sqrt(max((1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2) / (n - 1), 1e-12))
    return norm.cdf((sr_hat - sr0) / se)


def expected_max_sharpe(sr_trials_std, n_trials):
    """E[max] of N iid standard-normal Sharpe draws (per-period units), scaled by the
    cross-trial Sharpe dispersion. Bailey-LdP closed form with Euler-Mascheroni gamma."""
    if n_trials < 2:
        return 0.0
    g = 0.5772156649  # Euler-Mascheroni
    e = math.e
    z = ((1 - g) * norm.ppf(1 - 1.0 / n_trials)
         + g * norm.ppf(1 - 1.0 / (n_trials * e)))
    return sr_trials_std * z


def deflated_sharpe(best_sr_per_period, all_sr_per_period, n_obs, skew, kurt):
    """DSR = PSR with the benchmark set to the expected-max-Sharpe under the null of N trials."""
    sr_std = float(np.std(all_sr_per_period, ddof=1)) if len(all_sr_per_period) > 1 else 0.0
    sr0 = expected_max_sharpe(sr_std, len(all_sr_per_period))
    dsr = psr(best_sr_per_period, sr0, n_obs, skew, kurt)
    return dsr, sr0, sr_std


# ---------- build the trial set on SPY ----------
def build_trials():
    spy = load_ohlc("SPY"); c = spy["close"]
    r2 = rsi(c, 2); sma5 = c.rolling(5).mean(); sma10 = c.rolling(10).mean(); sma200 = c.rolling(200).mean()
    hi20 = c.rolling(20).max().shift(1); lo10 = c.rolling(10).min().shift(1)
    down = (c.diff() < 0); down3 = down & down.shift(1) & down.shift(2)

    trials = {
        "RSI(2)<10 MR exit>SMA5":      run_swing(c, r2 < 10, c > sma5),
        "RSI(2)<10 + 200d uptrend":    run_swing(c, (r2 < 10) & (c > sma200), c > sma5),  # the headline
        "3 down days exit up-day":     run_swing(c, down3, c > c.shift(1), 8),
        "dip<SMA10 exit>SMA10":        run_swing(c, c < sma10, c > sma10, 15),
        "Donchian-20 breakout":        run_swing(c, c > hi20, c < lo10, 10),
    }
    # overnight effect (the KNOWN-DEAD control): buy close, sell next open, 252 round-trips/yr
    on = (spy["open"] / spy["close"].shift(1) - 1) - COST_BPS_RT / 1e4
    trials["[CONTROL] overnight effect"] = on.dropna()
    return trials, "RSI(2)<10 + 200d uptrend", "[CONTROL] overnight effect"


def main():
    trials, headline, control = build_trials()
    rows = []
    for name, r in trials.items():
        r = r.dropna()
        rows.append({"strategy": name, "Sharpe": sharpe(r), "n": len(r),
                     "skew": float(r.skew()), "kurt": float(r.kurtosis() + 3.0),  # pandas kurtosis is excess
                     "mean_d": r.mean(), "std_d": r.std()})
    df = pd.DataFrame(rows).set_index("strategy")
    n_trials = len(df)

    print("=== Trial set (every strategy family you have tried on SPY) ===")
    print(df[["Sharpe", "n", "skew", "kurt"]].to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nN trials = {n_trials}.  Cross-trial Sharpe dispersion (annualized) = "
          f"{df['Sharpe'].std():.3f}")

    # Per-PERIOD (daily) Sharpe for the DSR math (annualized = daily * sqrt(252))
    sqrt_ppy = math.sqrt(252)
    all_daily_sr = (df["Sharpe"] / sqrt_ppy).values

    print("\n=== Deflated Sharpe Ratio (DSR) — does the edge survive N-trial selection? ===")
    print("DSR = P(true Sharpe > expected-best-under-null).  Survives if DSR > 0.95.\n")
    out = []
    for name in [headline, control]:
        row = df.loc[name]
        best_daily = row["Sharpe"] / sqrt_ppy
        dsr, sr0_daily, sr_std_daily = deflated_sharpe(best_daily, all_daily_sr, int(row["n"]), row["skew"], row["kurt"])
        # also the naive PSR vs 0 (no multiple-testing haircut) for contrast
        psr0 = psr(best_daily, 0.0, int(row["n"]), row["skew"], row["kurt"])
        out.append({"strategy": name, "Sharpe_ann": row["Sharpe"],
                    "PSR_vs_0": psr0, "deflated_bench_ann": sr0_daily * sqrt_ppy,
                    "DSR": dsr, "survives_0.95": "YES" if dsr > 0.95 else "no"})
    res = pd.DataFrame(out).set_index("strategy")
    print(res.to_string(float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

    print("\n=== Read ===")
    print("- PSR_vs_0 ignores multiple testing; DSR haircuts for the", n_trials, "things you tried.")
    print("- The [CONTROL] overnight effect is KNOWN-DEAD after costs. If its DSR clears 0.95,")
    print("  the harness is broken. It should NOT certify it.")
    print("- If the RSI2 headline DSR < 0.95, your benchmark-of-benchmarks is partly luck:")
    print("  every other experiment that 'must beat RSI2' should be judged against a softer bar.")


if __name__ == "__main__":
    main()
