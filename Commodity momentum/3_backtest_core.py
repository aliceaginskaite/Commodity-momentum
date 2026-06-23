"""
3 - backtest_core.py — backtest engine

Covered here:
   Market Impact - cost grows with trade size relative to ADV

Logic:
  1. Take weights (from strategy engine) and instrument returns
  2. Compute gross P&L: weight[t-1] applied to return[t]
     (we earn today's return on yesterday's position (no lookahead))
  3. Subtract transaction costs (spread+commission+slippage, already computed)
  4. Subtract market impact (additional cost for large trades vs ADV)
  5. Build equity curve, log every trade
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config

# Section 1: Market Impact

def compute_market_impact(weights: pd.DataFrame,
                          portfolio_value_usd: float = 100_000) -> pd.Series:
    """
    Calculate daily market impact costs using square-root law model.
    
    Market impact (bps) = MARKET_IMPACT_COEF × sqrt(trade_size_usd / ADV_usd)
    
    This follows the standard square-root impact model (Almgren-Chriss, 
    Kyle's lambda approximation), which is widely used in execution research 
    and TCA (Transaction Cost Analysis).
    
    Notes:
    - Impact is applied only on weight changes (i.e. actual traded notional).
    - For small AUM ($100k) impact is usually negligible.
    - Becomes binding constraint at higher AUM (see capacity analysis).
    
    Parameters
    weights : pd.DataFrame
        Portfolio weights over time.
    portfolio_value_usd : float, default 100_000
        Assumed portfolio AUM used to calculate dollar trade size.
    
    Returns
    -------
    pd.Series
        Daily market impact cost as a fraction of portfolio value.
    """

    weight_changes = weights.diff().abs()

    # ADV per ticker, in USD
    adv_usd = pd.Series(
        {t: v * 1_000_000 for t, v in config.AVG_DAILY_VOLUME_USD.items()},
        index=weights.columns
    )

    trade_size_usd = weight_changes * portfolio_value_usd

    # Impact in bps per instrument = COEF × sqrt(trade_size / ADV)
    impact_bps = config.MARKET_IMPACT_COEF * np.sqrt(
        trade_size_usd.div(adv_usd, axis=1).clip(lower=0)
    )

    impact_cost_per_instrument = (impact_bps / 10_000) * weight_changes
    daily_impact = impact_cost_per_instrument.sum(axis=1)
    daily_impact.name = "market_impact"

    config.log.info("Market impact computed | avg daily impact=%.5f%% | assumed AUM=$%s",
                    daily_impact.mean() * 100, f"{portfolio_value_usd:,.0f}")
    return daily_impact


# Section 2: Portfolio pnl

def compute_portfolio_returns(weights: pd.DataFrame,
                              returns: pd.DataFrame) -> pd.Series:
    """
    Compute daily gross portfolio returns.
    
    Portfolio return on day t = ∑ (weight[t] × return[t]) across all instruments.
    
    Alignment note:
    - Weights at time t represent the positions held *during* day t (after the +1 day signal shift applied in 2_strategy_engine).
    - Therefore, we use `weights.loc[t]` directly with `returns.loc[t]`.
    - No additional lag is needed.
    
    This alignment is critical. Incorrect shifting (using weight[t-1]) 
    is a common bug that understates performance by one day.
    
    Returns
    
    pd.Series
        Daily portfolio returns (gross of transaction costs).
    """

    # Align indices — weights and returns must match exactly
    common_idx = weights.index.intersection(returns.index)
    w = weights.loc[common_idx]
    r = returns.loc[common_idx]

    gross_returns = (w * r).sum(axis=1)
    gross_returns.name = "gross_return"

    config.log.info("Gross portfolio returns computed | %d days | mean=%.5f%% daily",
                    len(gross_returns), gross_returns.mean() * 100)
    return gross_returns


def compute_net_returns(gross_returns: pd.Series,
                        transaction_costs: pd.Series,
                        market_impact: pd.Series) -> pd.Series:
    """
    Net return = gross return − transaction costs − market impact
    """
    common_idx = gross_returns.index.intersection(transaction_costs.index).intersection(market_impact.index)

    net = (gross_returns.loc[common_idx]
           - transaction_costs.loc[common_idx]
           - market_impact.loc[common_idx])
    net.name = "net_return"

    config.log.info("Net returns computed | mean=%.5f%% daily | total cost drag=%.3f%% annualised",
                    net.mean() * 100,
                    (transaction_costs.loc[common_idx].mean() + market_impact.loc[common_idx].mean())
                    * config.TRADING_DAYS_YEAR * 100)
    return net


# Section 3: Equity Curve

def build_equity_curve(returns: pd.Series, starting_value: float = 100.0) -> pd.Series:
    """
    Compound daily returns into an equity curve.

    equity[t] = starting_value × exp(cumsum(log_returns))

    Why exp(cumsum) and not (1+r).cumprod()?
      Our returns are already LOG returns (from 1_data_pipeline).
      Log returns compound via addition, not multiplication.
      exp() converts back to actual price-like terms at the end.
    """
    equity = starting_value * np.exp(returns.cumsum())
    equity.name = "equity"

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_years = len(returns) / config.TRADING_DAYS_YEAR
    config.log.info("Equity curve built | start=%.1f end=%.1f | total return=%.1f%% over %.1f years",
                    equity.iloc[0], equity.iloc[-1], total_return, n_years)
    return equity


# Section 4: Trade Log

def build_trade_log(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Extract every individual trade: date, instrument, old weight, new weight.
    """
    weight_changes = weights.diff()
    trades = []

    for date in weight_changes.index:
        row = weight_changes.loc[date]
        for ticker in weight_changes.columns:
            delta = row[ticker]
            if abs(delta) > 1e-9:   
                trades.append({
                    "date": date,
                    "ticker": ticker,
                    "weight_change": delta,
                    "new_weight": weights.loc[date, ticker],
                })

    trade_log = pd.DataFrame(trades)
    config.log.info("Trade log built | %d total trades over period", len(trade_log))
    return trade_log


# Section 5: visualize equity curve

def plot_equity_curve(equity_gross: pd.Series,
                      equity_net: pd.Series,
                      save: bool = True):
    """
    Plot gross vs net equity curve.
    The gap between them = cost of doing business (transaction costs + impact).
    A wide gap means the strategy is cost-sensitive.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(equity_gross.index, equity_gross.values,
            label="Gross (before costs)", color="#4cc9f0", linewidth=1.3, alpha=0.85)
    ax.plot(equity_net.index, equity_net.values,
            label="Net (after costs + impact)", color="#1a1a2e", linewidth=1.6)

    ax.axhline(100, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_title("Equity Curve — Cross-Sectional Commodity Momentum",
                fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (base 100)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "3_equity_curve.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


# Main - full backtest pipeline

def run_backtest(weights: pd.DataFrame,
                 returns: pd.DataFrame,
                 transaction_costs: pd.Series,
                 portfolio_value_usd: float = 100_000) -> dict:
    """
    Full backtest: weights + returns → equity curve.
    Designed to be called repeatedly (full history, IS, OOS,
    walk-forward windows, stress periods) for comprehensive performance analysis.
    """
    config.log.info("Running backtest...")

    market_impact = compute_market_impact(weights, portfolio_value_usd)
    gross_returns = compute_portfolio_returns(weights, returns)
    net_returns = compute_net_returns(gross_returns, transaction_costs, market_impact)

    equity_gross = build_equity_curve(gross_returns)
    equity_net = build_equity_curve(net_returns)

    trade_log = build_trade_log(weights)

    config.log.info("Backtest complete.")

    return {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "equity_gross": equity_gross,
        "equity_net": equity_net,
        "market_impact": market_impact,
        "trade_log": trade_log,
    }


if __name__ == "__main__":
    import importlib.util

    # Load strategy engine module 
    spec = importlib.util.spec_from_file_location("2_strategy_engine", "2_strategy_engine.py")
    strategy_engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_engine)

    prices_path  = os.path.join(config.DATA_CLEAN, "prices.csv")
    returns_path = os.path.join(config.DATA_CLEAN, "returns.csv")

    if not (os.path.exists(prices_path) and os.path.exists(returns_path)):
        print("Run 1_data_pipeline.py first.")
    else:
        prices  = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

        strategy_result = strategy_engine.run_strategy(prices)
        backtest_result  = run_backtest(
            weights           = strategy_result["weights"],
            returns           = returns,
            transaction_costs = strategy_result["costs"],
        )

        plot_equity_curve(backtest_result["equity_gross"], backtest_result["equity_net"])

        print("\n── Summary ──")
        gross_total = (backtest_result["equity_gross"].iloc[-1] / 100 - 1) * 100
        net_total   = (backtest_result["equity_net"].iloc[-1] / 100 - 1) * 100
        print(f"Gross total return: {gross_total:.1f}%")
        print(f"Net total return:   {net_total:.1f}%")
        print(f"Cost drag:           {gross_total - net_total:.1f}% over full period")
        print(f"Total trades:        {len(backtest_result['trade_log'])}")