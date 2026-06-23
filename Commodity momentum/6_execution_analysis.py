"""
6_execution_analysis.py - execution analysis

Covered here:
  Turnover - how much do we trade, and what does that cost us?
  Exposure - how much of the time are we actually in the market?
  Capacity - how big can this strategy scale before it breaks itself?

A strategy can look great on paper and still be un-investable if:
  - turnover is so high that costs eat all the edge
  - exposure patterns are erratic / inconsistent with the stated thesis
  - it only works at $50k AUM and falls apart at $5M 
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config


# Section 1: Turnover  

def compute_turnover(weights: pd.DataFrame) -> dict:
    """
    Turnover = how much of the portfolio gets traded on each rebalance.

    Daily turnover[t] = sum of |weight_change| across all instruments
    This is the same quantity used to compute transaction costs
    in 3_backtest_core.py

    We report it at three levels:
    - Per-rebalance turnover (average size of each trade event)
    - Annualised turnover (total trading activity per year)
    - Cost-equivalent (how much of the year's return turnover alone consumes)

    High turnover requires more sophisticated execution infrastructure, tighter broker relationships,
    and can flag regulatory/reporting overhead.

    """
    weight_changes = weights.diff().abs()
    daily_turnover = weight_changes.sum(axis=1)

    # only looking at days where a trade actually happened (non-zero turnover)
    trade_days = daily_turnover[daily_turnover > 1e-9]

    annual_turnover = daily_turnover.sum() / (len(daily_turnover) / config.TRADING_DAYS_YEAR)
    annual_cost_pct = annual_turnover * (config.TOTAL_COST_BPS / 10_000) * 100

    result = {
        "daily_turnover_series": daily_turnover,
        "avg_turnover_per_trade_day": trade_days.mean() if len(trade_days) > 0 else 0,
        "n_trade_days": len(trade_days),
        "annual_turnover": annual_turnover,
        "annual_cost_from_turnover_pct": annual_cost_pct,
    }

    config.log.info("Turnover | annualised: %.1fx | avg per rebalance: %.2f | annual cost: %.2f%%",
                    annual_turnover, result["avg_turnover_per_trade_day"], annual_cost_pct)

    return result


# Section 2: Exposure 

def compute_exposure(weights: pd.DataFrame) -> dict:
    """
    Exposure concepts:
    Gross exposure: sum of absolute weights. ~2.0 for a 100/100 long-short strategy.

    Net exposure: sum of weights (long minus short). Should be near 0 for market-neutral.
    If persistently non-zero → strategy has become directional (hidden beta).

    Time in market: % of days with non-zero position. Low value = low capital efficiency.
    """
    gross_exposure = weights.abs().sum(axis=1)
    net_exposure   = weights.sum(axis=1)

    pct_time_in_market = (gross_exposure > 1e-9).mean() * 100

    # per instrument exposure (how often is each instrument held, long or short?)
    per_instrument_long_pct = (weights > 0).mean(axis=0) * 100
    per_instrument_short_pct = (weights < 0).mean(axis=0) * 100

    result = {
        "gross_exposure_series": gross_exposure,
        "net_exposure_series": net_exposure,
        "avg_gross_exposure": gross_exposure.mean(),
        "avg_net_exposure": net_exposure.mean(),
        "net_exposure_std": net_exposure.std(),
        "pct_time_in_market": pct_time_in_market,
        "per_instrument_long_pct": per_instrument_long_pct,
        "per_instrument_short_pct": per_instrument_short_pct,
    }

    config.log.info("Exposure | avg gross: %.2f | avg net: %.4f (std %.4f) | time in market: %.1f%%",
                    result["avg_gross_exposure"], result["avg_net_exposure"],
                    result["net_exposure_std"], pct_time_in_market)

    if abs(result["avg_net_exposure"]) > 0.05:
        config.log.warning("Net exposure is meaningfully non-zero — "
                          "strategy may have a directional bias, not pure relative value.")

    return result



# Section 3: capacity

def compute_capacity(weights: pd.DataFrame,
                     max_participation_rate: float = 0.05) -> dict:
    """
    Capacity = the maximum AUM at which this strategy can still
    execute without moving the market too much against itself.

    Methodology (standard industry approach — "participation rate"):
    We assume we never want to be more than X% of a single instrument's
    average daily volume (ADV) on any single trade.
    max_participation_rate = 0.05 means "never more than 5% of ADV."

    For each instrument, on its biggest single-day weight CHANGE
    (worst case trade), we solve:
        AUM × |max_weight_change| ≤ max_participation_rate × ADV
        ->  AUM_max = (max_participation_rate × ADV) / |max_weight_change|

    The overall strategy capacity = the minimum across all instruments
    (the strategy is constrained by its most illiquid component (CPER), which has the
    lowest ADV of the five).
    """
    weight_changes = weights.diff().abs()
    max_weight_change_per_instrument = weight_changes.max()

    capacity_per_instrument = {}
    for ticker in weights.columns:
        adv_usd = config.AVG_DAILY_VOLUME_USD.get(ticker, np.nan) * 1_000_000
        max_wc  = max_weight_change_per_instrument[ticker]

        if max_wc > 0:
            cap = (max_participation_rate * adv_usd) / max_wc
        else:
            cap = np.nan

        capacity_per_instrument[ticker] = cap

    capacity_series = pd.Series(capacity_per_instrument)
    binding_constraint = capacity_series.idxmin()
    overall_capacity = capacity_series.min()

    config.log.info("Capacity | overall AUM limit: $%s | binding constraint: %s",
                    f"{overall_capacity:,.0f}", binding_constraint)

    for ticker, cap in capacity_series.items():
        config.log.info("  %s capacity: $%s", ticker, f"{cap:,.0f}")

    return {
        "capacity_per_instrument": capacity_series,
        "overall_capacity_usd": overall_capacity,
        "binding_constraint": binding_constraint,
        "max_participation_rate": max_participation_rate,
    }



# Section 4: visualisation 

def plot_execution_summary(turnover_result: dict,
                           exposure_result: dict,
                           capacity_result: dict,
                           save: bool = True):
    """
    Three-panel summary:
    1. Gross & net exposure over time
    2. Daily turnover over time
    3. Capacity by instrument (bar chart) — shows which instrument
       is the bottleneck for scaling the strategy
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle("Execution Analysis — Turnover / Exposure / Capacity",
                fontsize=14, fontweight="bold")

    # Panel 1
    ax1 = axes[0]
    gross = exposure_result["gross_exposure_series"]
    net = exposure_result["net_exposure_series"]
    ax1.plot(gross.index, gross.values, label="Gross Exposure", color="#4cc9f0", linewidth=1.0)
    ax1.plot(net.index, net.values, label="Net Exposure", color="#e63946", linewidth=1.0)
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax1.set_title("Portfolio Exposure Over Time")
    ax1.set_ylabel("Exposure")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2
    ax2 = axes[1]
    turnover = turnover_result["daily_turnover_series"]
    ax2.bar(turnover.index, turnover.values, width=1.0, color="#7209b7", alpha=0.6)
    ax2.set_title("Daily Turnover (portfolio fraction traded)")
    ax2.set_ylabel("Turnover")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 3
    ax3 = axes[2]
    cap_series = capacity_result["capacity_per_instrument"].sort_values()
    colors = ["#e63946" if t == capacity_result["binding_constraint"] else "#2a9d8f"
             for t in cap_series.index]
    bars = ax3.barh(cap_series.index, cap_series.values / 1_000_000, color=colors)
    ax3.set_title(f"Capacity by Instrument (red = binding constraint: "
                  f"{capacity_result['binding_constraint']})")
    ax3.set_xlabel("Max AUM ($ millions)")
    for bar, val in zip(bars, cap_series.values):
        ax3.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                 f" ${val/1e6:.1f}M", va="center", fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "6_execution_summary.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


# MAIN

def run_execution_analysis(weights: pd.DataFrame) -> dict:
    """Full execution analysis pipeline."""
    config.log.info("=" * 60)
    config.log.info("EXECUTION ANALYSIS")
    config.log.info("=" * 60)

    turnover_result = compute_turnover(weights)
    exposure_result = compute_exposure(weights)
    capacity_result = compute_capacity(weights)

    plot_execution_summary(turnover_result, exposure_result, capacity_result)

    return {
        "turnover": turnover_result,
        "exposure": exposure_result,
        "capacity": capacity_result,
    }


if __name__ == "__main__":
    import importlib.util

    def _load_module(name, filename):
        spec = importlib.util.spec_from_file_location(name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    strategy_engine = _load_module("2_strategy_engine", os.path.join(config.BASE_DIR, "2_strategy_engine.py"))

    prices_path = os.path.join(config.DATA_CLEAN, "prices.csv")

    if not os.path.exists(prices_path):
        print("Run data_pipeline.py first.")
    else:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        strat = strategy_engine.run_strategy(prices)

        results = run_execution_analysis(strat["weights"])

        print("\n── Execution Summary ──")
        print(f"Annualised turnover: {results['turnover']['annual_turnover']:.1f}x")
        print(f"Annual cost from turnover: {results['turnover']['annual_cost_from_turnover_pct']:.2f}%")
        print(f"Avg gross exposure: {results['exposure']['avg_gross_exposure']:.2f}")
        print(f"Avg net exposure: {results['exposure']['avg_net_exposure']:.4f}")
        print(f"Time in market: {results['exposure']['pct_time_in_market']:.1f}%")
        print(f"Overall capacity: ${results['capacity']['overall_capacity_usd']:,.0f}")
        print(f"Binding constraint: {results['capacity']['binding_constraint']}")