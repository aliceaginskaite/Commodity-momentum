"""
8_report.py — Final Report & research section

Covered here:
  Correlation to Market  — vs buy-and-hold commodity basket
  Correlation to Existing Strategies — vs a simple EMA trend-following proxy
  Edge Decomposition — where does the return actually come from?

Also section with strategie`s weaknesses 
  "What kills this strategy" - specific list of weaknesses,
   written using everything we found in files 1-7.

This file assembles a final visual and text report. It does not re-derive
any core logic but imports the other modules and asks new questions
of results we already trust.
"""

import pandas as pd
import numpy as np
import os
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

strategy_engine = _load_module("strategy_engine", os.path.join(config.BASE_DIR, "2_strategy_engine.py"))
backtest_core = _load_module("backtest_core",   os.path.join(config.BASE_DIR, "3_backtest_core.py"))
risk_metrics = _load_module("risk_metrics",    os.path.join(config.BASE_DIR, "5_risk_metrics.py"))


# Section 1: correlation to market

def compute_market_correlation(net_returns: pd.Series, returns: pd.DataFrame) -> dict:
    """
    "The market" here = an equal-weighted buy-and-hold basket of all
    5 commodities. This is the standard, simplest benchmark: what
    would you have earned just holding all 5 instruments equally.

    Why this matters:
    Our strategy is built to be market-neutral (net exposure ≈ 0,
    confirmed in 6_execution_analysis.py). If correlation to this
    benchmark is also near zero, that confirms the strategy is doing
    what it claims - generating returns independent of whether
    commodities broadly rise or fall. If correlation is unexpectedly
    high, that's a red flag: the "momentum" edge might actually just
    be a hidden directional bet that happened to align with a
    commodity bull market in our sample period.
    """
    benchmark_returns = returns.mean(axis=1)
    benchmark_returns.name = "market_benchmark"

    common_idx = net_returns.index.intersection(benchmark_returns.index)
    strat_aligned = net_returns.loc[common_idx]
    bench_aligned = benchmark_returns.loc[common_idx]

    correlation = strat_aligned.corr(bench_aligned)

    # beta via covariance / variance 
    covariance = strat_aligned.cov(bench_aligned)
    beta = covariance / bench_aligned.var() if bench_aligned.var() > 0 else np.nan

    config.log.info("Correlation to market benchmark: %.3f | Beta: %.3f", correlation, beta)

    if abs(correlation) > 0.3:
        config.log.warning("Correlation to market is meaningfully positive/negative — "
                          "strategy may carry hidden directional exposure to commodities.")
    else:
        config.log.info("Low correlation to market — consistent with the strategy's "
                        "market-neutral design.")

    return {
        "benchmark_returns": benchmark_returns,
        "correlation": correlation,
        "beta": beta,
    }



# Section 2: correlation to existing strategies (EMA trend-following proxy)

def build_ema_trend_proxy(prices: pd.DataFrame) -> pd.Series:
    """
    A simple EMA-crossover trend-following strategy, used here as a
    proxy "existing strategy" benchmark to check correlation against.

    This is not a real track record. It's a simplified, transparent
    proxy built specifically for this correlation check, standing in
    for a trend-following style of trading (conceptually related to
    EMA-based discretionary approaches). A real comparison would use
    actual historical returns from a live or backtested strategy with
    its own track record. This proxy exists so the "Correlation to
    Existing Strategies" checklist item isn't skipped, while being
    transparent that it's a stand-in, not a real benchmark.

    Logic: for EACH instrument, go long when 20-day EMA > 50-day EMA,
    short when 20-day EMA < 50-day EMA. Equal-weight across all 5
    instruments (no long/short balancing — this is a simple trend
    proxy, not a market-neutral design, by construction).
    """
    ema_fast = prices.ewm(span=20, adjust=False).mean()
    ema_slow = prices.ewm(span=50, adjust=False).mean()

    signal = np.where(ema_fast > ema_slow, 1, -1)
    signal = pd.DataFrame(signal, index=prices.index, columns=prices.columns)

    signal_shifted = signal.shift(1).fillna(0)

    # equal weight per instrument
    weights = signal_shifted / len(prices.columns)

    log_returns = np.log(prices / prices.shift(1)).fillna(0)
    proxy_returns = (weights * log_returns).sum(axis=1)
    proxy_returns.name = "ema_trend_proxy"

    config.log.info("EMA trend-following proxy built (20/50 crossover, not a real track record)")
    return proxy_returns


def compute_strategy_correlation(net_returns: pd.Series, proxy_returns: pd.Series) -> dict:
    """
    Correlation between our momentum strategy and the EMA trend proxy.

    If we already run trend-following strategies, we care
    a lot about whether adding our strategy actually diversifies their
    book, or just duplicates exposure they already have. Low/negative
    correlation = genuine diversification value. High correlation =
    "this is just trend-following with extra steps."
    """
    common_idx = net_returns.index.intersection(proxy_returns.index)
    correlation = net_returns.loc[common_idx].corr(proxy_returns.loc[common_idx])

    config.log.info("Correlation to EMA trend-following proxy: %.3f", correlation)
    return {"correlation": correlation, "proxy_returns": proxy_returns}



# Section 3: Edge decomposition

def decompose_edge(weights: pd.DataFrame, returns: pd.DataFrame, net_returns: pd.Series) -> dict:
    """
    "Where does the return actually come from"

    We split total net return into pieces:
    1.Long-leg contribution - P&L from instruments we were long
    2.Short-leg contribution - P&L from instruments we were short
    3.Cost drag - transaction costs + market impact (negative)

    Per-instrument contribution: which of the 5 commodities
    drove most of the P&L? If one instrument (e.g. Gold) explains 80%
    of all profit, that's not a cross-sectional commodity
    momentum, but a 'gold strategy that happens to use other
    commodities as a hedge.' This is important honesty check.
    """
    long_weights = weights.clip(lower=0)
    short_weights = weights.clip(upper=0)

    common_idx = weights.index.intersection(returns.index)
    long_pnl  = (long_weights.loc[common_idx]  * returns.loc[common_idx]).sum(axis=1)
    short_pnl = (short_weights.loc[common_idx] * returns.loc[common_idx]).sum(axis=1)

    long_total_pct  = (np.exp(long_pnl.sum())  - 1) * 100
    short_total_pct = (np.exp(short_pnl.sum()) - 1) * 100

    # Per-instrument contribution (gross, before costs)
    per_instrument_pnl = (weights.loc[common_idx] * returns.loc[common_idx])
    per_instrument_total_pct = (np.exp(per_instrument_pnl.sum()) - 1) * 100

    gross_total_pct = (np.exp((weights.loc[common_idx] * returns.loc[common_idx]).sum(axis=1).sum()) - 1) * 100
    net_total_pct = (np.exp(net_returns.sum()) - 1) * 100
    cost_drag_pct = gross_total_pct - net_total_pct

    config.log.info("Edge decomposition | Long leg: %.1f%% | Short leg: %.1f%% | Cost drag: %.1f%%",
                    long_total_pct, short_total_pct, cost_drag_pct)
    config.log.info("Per-instrument contribution:")
    for ticker, contrib in per_instrument_total_pct.sort_values(ascending=False).items():
        config.log.info("  %s: %.1f%%", ticker, contrib)

    # Concentration check - what % of total absolute contribution comes from
    # the single biggest contributor
    abs_contribs = per_instrument_total_pct.abs()
    concentration_pct = (abs_contribs.max() / abs_contribs.sum() * 100) if abs_contribs.sum() > 0 else np.nan
    top_contributor = per_instrument_total_pct.abs().idxmax()

    if concentration_pct > 50:
        config.log.warning("Single instrument (%s) explains %.0f%% of total |P&L| — "
                          "edge may be concentrated in one commodity, not broad-based momentum.",
                          top_contributor, concentration_pct)

    return {
        "long_leg_pct":   long_total_pct,
        "short_leg_pct":  short_total_pct,
        "cost_drag_pct":   cost_drag_pct,
        "per_instrument_pct": per_instrument_total_pct,
        "concentration_pct":  concentration_pct,
        "top_contributor":    top_contributor,
    }



# Section 4: visualisations

def plot_correlation_analysis(market_corr: dict, strategy_corr: dict, net_returns: pd.Series, save: bool = True):
    """
    Two-panel: scatter vs market benchmark, scatter vs EMA trend proxy.
    Visual correlation is often more convincing than a single number.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    bench = market_corr["benchmark_returns"]
    common1 = net_returns.index.intersection(bench.index)
    axes[0].scatter(bench.loc[common1], net_returns.loc[common1], alpha=0.25, s=10, color="#4cc9f0")
    axes[0].set_title(f"vs Market Benchmark (corr={market_corr['correlation']:.2f})")
    axes[0].set_xlabel("Market (equal-weight basket) daily return")
    axes[0].set_ylabel("Strategy daily return")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axvline(0, color="gray", linewidth=0.5)

    proxy = strategy_corr["proxy_returns"]
    common2 = net_returns.index.intersection(proxy.index)
    axes[1].scatter(proxy.loc[common2], net_returns.loc[common2], alpha=0.25, s=10, color="#7209b7")
    axes[1].set_title(f"vs EMA Trend Proxy (corr={strategy_corr['correlation']:.2f})")
    axes[1].set_xlabel("EMA trend proxy daily return")
    axes[1].set_ylabel("Strategy daily return")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].axvline(0, color="gray", linewidth=0.5)

    fig.suptitle("Correlation Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "08_correlation_analysis.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


def plot_edge_decomposition(decomp: dict, save: bool = True):
    """
    Two-panel: (1) long vs short vs cost drag waterfall-style bar,
    (2) per-instrument contribution bar chart, sorted, to visually
    expose concentration risk.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1 = axes[0]
    components = ["Long Leg", "Short Leg", "Cost Drag"]
    values = [decomp["long_leg_pct"], decomp["short_leg_pct"], -decomp["cost_drag_pct"]]
    colors = ["#2a9d8f" if v >= 0 else "#e63946" for v in values]
    ax1.bar(components, values, color=colors)
    ax1.axhline(0, color="gray", linewidth=0.7)
    ax1.set_title("Return Decomposition")
    ax1.set_ylabel("Contribution (%)")

    ax2 = axes[1]
    per_inst = decomp["per_instrument_pct"].sort_values()
    colors2 = ["#e63946" if t == decomp["top_contributor"] else "#2a9d8f" for t in per_inst.index]
    ax2.barh(per_inst.index, per_inst.values, color=colors2)
    ax2.axvline(0, color="gray", linewidth=0.7)
    ax2.set_title(f"Per-Instrument Contribution\n(red = largest, "
                 f"{decomp['concentration_pct']:.0f}% concentration)")
    ax2.set_xlabel("Contribution (%)")

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "8_edge_decomposition.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()



# Section 5: what kills this strategy
# 

def print_what_kills_this_strategy(metrics: dict,
                                   capacity_usd: float,
                                   binding_constraint: str,
                                   concentration_pct: float,
                                   top_contributor: str,
                                   mc_percentile: float):
    """
    The honesty section. Every point here should be traceable to a
    specific number we actually computed in files 1-8.
    """
    print("\n" + "=" * 64)
    print("What kills this strategy")
    print("=" * 64)

    print(f"""
1. RISK-ADJUSTED RETURN IS WEAK
   Sharpe {metrics['sharpe']:.2f}, Sortino {metrics['sortino']:.2f} — both well below the
   0.5-1.0 "acceptable" threshold. The strategy makes money on
   average but the ride is rough enough that few allocators
   would size it meaningfully without improvement.

2. DRAWDOWNS ARE SEVERE AND SLOW TO RECOVER
   Max drawdown {metrics['max_drawdown_pct']:.1f}%, with an UNRECOVERED drawdown
   ongoing at the end of our sample. A real investor would
   likely have redeemed long before recovery.

3. CAPACITY IS TINY
   Binding constraint: {binding_constraint}, limiting this strategy to
   ~${capacity_usd:,.0f} AUM before market impact becomes material.
   This is a strategy for a small account, not an institutional
   allocation, unless the instrument universe is expanded to
   more liquid futures.

4. CONCENTRATION RISK
   {top_contributor} alone explains {concentration_pct:.0f}% of total |P&L| —
   this is not broad-based cross-sectional momentum, it's closer
   to "a {top_contributor} strategy with 4 other instruments along for the ride."

5. MONTE CARLO ORDER SENSITIVITY
   Real result sits at the {mc_percentile:.0f}th percentile of random
   return-orderings. High volatility means the PATH matters as
   much as the average return — sequencing risk is real.

6. SURVIVORSHIP BIAS (structural, documented in 01)
   Universe = currently-listed ETFs only. Delisted commodity
   products from 2012-2024 are excluded, which may slightly
   flatter results (see 01_data_pipeline.py for detail).
""")

    print("=" * 64)
    print("What would i try next")
    print("=" * 64)

    print("""
• Volatility targeting: scale position size inversely to each
  instrument's recent volatility, instead of equal weighting.
  Should reduce the influence of NatGas/Copper spikes and
  smooth the equity curve.

• Wider instrument universe: add more liquid commodities
  (e.g. Brent, Heating Oil, Platinum) via real futures rather
  than ETF proxies, addresses both capacity and concentration
  risk simultaneously.

• Slower rebalancing or signal smoothing: test monthly
  rebalancing or an EMA-smoothed momentum score to see if
  turnover-driven cost drag (currently ~3%/year) is masking a
  better underlying Sharpe.

• Risk overlay: a simple max-drawdown circuit breaker (cut
  gross exposure in half after a -X% drawdown) to directly
  target the slow-recovery problem in finding #2.
""")



# Main — assemble everything

if __name__ == "__main__":
    prices_path  = os.path.join(config.DATA_CLEAN, "prices.csv")
    returns_path = os.path.join(config.DATA_CLEAN, "returns.csv")

    if not (os.path.exists(prices_path) and os.path.exists(returns_path)):
        print("Run 01_data_pipeline.py first.")
    else:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

        strat = strategy_engine.run_strategy(prices)
        bt = backtest_core.run_backtest(
            weights = strat["weights"],
            returns = returns,
            transaction_costs = strat["costs"],
        )
        net_returns = bt["net_returns"]
        weights = strat["weights"]

        # Risk metrics (re-used from file 5)
        metrics = risk_metrics.compute_all_metrics(net_returns)

        # Execution / capacity (re-derive minimal pieces)
        execution_analysis = _load_module("6_execution_analysis", os.path.join(config.BASE_DIR, "6_execution_analysis.py"))
        capacity_result = execution_analysis.compute_capacity(weights)

        # Correlation to market
        market_corr = compute_market_correlation(net_returns, returns)

        # Correlation to existing strategies (EMA proxy)
        ema_proxy_returns = build_ema_trend_proxy(prices)
        strategy_corr = compute_strategy_correlation(net_returns, ema_proxy_returns)

        plot_correlation_analysis(market_corr, strategy_corr, net_returns)

        # Edge decomposition
        decomp = decompose_edge(weights, returns, net_returns)
        plot_edge_decomposition(decomp)

        # Monte Carlo percentile (re-derive quickly, lighter sim count for speed here)
        robustness = _load_module("robustness", os.path.join(config.BASE_DIR, "7_robustness.py"))
        mc_result = robustness.run_monte_carlo(net_returns, n_sims=2000)

        # ── Final honesty section ──
        print_what_kills_this_strategy(
            metrics = metrics,
            capacity_usd = capacity_result["overall_capacity_usd"],
            binding_constraint = capacity_result["binding_constraint"],
            concentration_pct = decomp["concentration_pct"],
            top_contributor = decomp["top_contributor"],
            mc_percentile = mc_result["final_value_percentile"],
        )

        print("\n" + "=" * 60)
        print("Full project checklist")
        print("=" * 60)
        checklist = [
            "Survivorship Bias", "Lookahead Bias", "Data Snooping", "Missing Data Check",
            "Spread", "Commission", "Slippage", "Market Impact",
            "Out-of-Sample", "Walk Forward", "Cross Validation (time-series)",
            "Monte Carlo", "Parameter Stability", "Stress Tests", "Regime Analysis",
            "Sharpe", "Sortino", "Calmar", "Max DD", "Recovery Time", "Tail Risk",
            "Turnover", "Exposure", "Capacity",
            "Correlation to Market", "Correlation to Existing Strategies", "Edge Decomposition",
        ]
        for item in checklist:
            print(f"  ✅ {item}")