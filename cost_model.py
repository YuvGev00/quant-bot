from __future__ import annotations
from dataclasses import dataclass, field

# Canonical schema (Parquet, one file per symbol per day, UTC nanosecond timestamps):
#   quotes: ts, symbol, bid, ask, bid_size, ask_size
#   trades: ts, symbol, price, size, bid, ask        # bid/ask = prevailing NBBO for Lee-Ready signing
#   bars:   ts, symbol, open, high, low, close, volume, vwap, bar_seconds
# cost_model is data-agnostic: it takes scalar cost inputs, not the dataframes.

# NOTE: regulatory rates change. Verify SEC Section 31 + FINRA TAF before trusting absolute $.
SEC_FEE_PER_DOLLAR_SELL = 0.0000278     # ~ $27.80 per $1M of principal, sells only — VERIFY
FINRA_TAF_PER_SHARE_SELL = 0.000166     # capped per order, sells only — VERIFY
FINRA_TAF_MAX_PER_ORDER = 8.30          # VERIFY


@dataclass
class IBKRCosts:
    model: str = "tiered"                # "tiered" or "fixed"
    fixed_per_share: float = 0.005
    fixed_min: float = 1.00
    tiered_per_share: float = 0.0035
    tiered_min: float = 0.35
    taker_passthrough_per_share: float = 0.0003   # exchange/clearing when removing liquidity
    maker_rebate_per_share: float = 0.0020        # credit when adding liquidity (passive leg)
    max_pct_of_trade: float = 0.01

    def commission(self, shares: int, price: float, is_sell: bool, is_maker: bool) -> float:
        notional = shares * price
        if self.model == "fixed":
            comm = max(self.fixed_min, self.fixed_per_share * shares)
        else:
            base = max(self.tiered_min, self.tiered_per_share * shares)
            if is_maker:
                comm = base - self.maker_rebate_per_share * shares
            else:
                comm = base + self.taker_passthrough_per_share * shares
        comm = min(comm, self.max_pct_of_trade * notional)
        if is_sell:
            comm += SEC_FEE_PER_DOLLAR_SELL * notional
            comm += min(FINRA_TAF_PER_SHARE_SELL * shares, FINRA_TAF_MAX_PER_ORDER)
        return comm

    @classmethod
    def interactive_israel(cls) -> "IBKRCosts":
        # Interactive Israel: $0.01/share, $2.5 min, fixed (no maker rebate). VERIFY current tariff.
        return cls(model="fixed", fixed_per_share=0.01, fixed_min=2.5,
                   maker_rebate_per_share=0.0, taker_passthrough_per_share=0.0)


@dataclass
class Leg:
    symbol: str
    shares: int
    price: float
    half_spread_cents: float             # half the bid-ask, in cents/share


@dataclass
class TradePlan:
    leg_a: Leg
    leg_b: Leg
    # which legs are posted passive (maker, no spread crossed) vs active (taker, cross half-spread)
    entry_passive: tuple[bool, bool] = (True, False)   # (leg_a, leg_b)
    exit_passive: tuple[bool, bool] = (False, False)    # pessimistic default: both taker on exit
    short_leg: str = "a"                 # which leg is the short (borrow accrues here)
    annual_borrow_rate: float = 0.005    # 50 bps/yr default; small-cap shorts can be 10x+
    holding_days: float = 0.0            # 0 = intraday; >0 = swing


def _execution_cost(leg: Leg, costs: IBKRCosts, is_sell: bool, is_passive: bool) -> float:
    comm = costs.commission(leg.shares, leg.price, is_sell=is_sell, is_maker=is_passive)
    spread_cost = 0.0 if is_passive else (leg.half_spread_cents / 100.0) * leg.shares
    return comm + spread_cost


def round_trip_cost(plan: TradePlan, costs: IBKRCosts) -> dict:
    a, b = plan.leg_a, plan.leg_b
    short = a if plan.short_leg == "a" else b
    # Entry: open the position. Short leg is sold to open; long leg bought to open.
    a_short = plan.short_leg == "a"
    entry = (
        _execution_cost(a, costs, is_sell=a_short, is_passive=plan.entry_passive[0])
        + _execution_cost(b, costs, is_sell=not a_short, is_passive=plan.entry_passive[1])
    )
    # Exit: reverse both legs.
    exit_ = (
        _execution_cost(a, costs, is_sell=not a_short, is_passive=plan.exit_passive[0])
        + _execution_cost(b, costs, is_sell=a_short, is_passive=plan.exit_passive[1])
    )
    borrow = (short.shares * short.price) * plan.annual_borrow_rate * (plan.holding_days / 360.0)
    total = entry + exit_ + borrow
    gross_notional = a.shares * a.price + b.shares * b.price
    return {
        "entry_cost": entry,
        "exit_cost": exit_,
        "borrow_cost": borrow,
        "total_cost": total,
        "gross_notional": gross_notional,
        "breakeven_bps_of_notional": 1e4 * total / gross_notional,
        "breakeven_cents_per_share_combined": 100.0 * total / (a.shares + b.shares),
    }


def breakeven_report(plan: TradePlan, costs: IBKRCosts) -> str:
    r = round_trip_cost(plan, costs)
    lines = [
        f"  entry cost        : ${r['entry_cost']:.2f}",
        f"  exit cost         : ${r['exit_cost']:.2f}",
        f"  borrow cost       : ${r['borrow_cost']:.2f}",
        f"  TOTAL round trip  : ${r['total_cost']:.2f}",
        f"  gross notional    : ${r['gross_notional']:,.0f}",
        f"  >>> BREAKEVEN     : {r['breakeven_bps_of_notional']:.1f} bps of notional"
        f"  ({r['breakeven_cents_per_share_combined']:.2f} c/share combined)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    leg_a = Leg("AAA", shares=500, price=50.0, half_spread_cents=0.5)
    leg_b = Leg("BBB", shares=500, price=50.0, half_spread_cents=0.5)

    intraday = TradePlan(leg_a, leg_b, entry_passive=(True, False), exit_passive=(False, False))
    intraday_taker = TradePlan(leg_a, leg_b, entry_passive=(False, False), exit_passive=(False, False))
    swing = TradePlan(leg_a, leg_b, entry_passive=(True, False), exit_passive=(False, False),
                      holding_days=2.0, annual_borrow_rate=0.005)

    for label, costs in [("IBKR tiered (direct/global)", IBKRCosts(model="tiered")),
                         ("Interactive Israel ($0.01/sh, $2.5 min)", IBKRCosts.interactive_israel())]:
        print(f"\n===== {label} =====")
        print("intraday, passive entry leg A:")
        print(breakeven_report(intraday, costs))
        print("intraday, both legs taker:")
        print(breakeven_report(intraday_taker, costs))
        print("swing (2 days, 50bps borrow), passive entry leg A:")
        print(breakeven_report(swing, costs))
