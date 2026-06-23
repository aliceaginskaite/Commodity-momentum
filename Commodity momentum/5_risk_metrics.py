"""
5_risk_metrics.py — risk & performance metrics

Covered here:
  Sharpe - risk-adjusted return vs total volatility
  Sortino - risk-adjusted return vs downside volatility only
  Calmar - return vs worst drawdown
  Max Drawdown - biggest peak-to-trough loss
  Recovery Time - how long it took to climb back out of each drawdown
  Tail Risk - CVaR/Expected Shortfall: average loss in the worst days

This file takes a returns series and produces a single dict of numbers
that becomes the "report card" for the strategy.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config


# Section 1: sharpe ratio

def compute_sharpe(returns: pd.Series,
                   risk_free_rate: float = config.RISK_FREE_RATE) -> float:
    """
    Sharpe Ratio = (annualised return − risk-free rate) / annualised volatility

    Interpretation:
    <0.5 - weak
    0.5-1.0 - acceptable
    1.0-2.0 - good
    >2.0 - excellent (or suspicious — check for overfitting/bugs)

    Limitation this metric has (important to know):
    Sharpe penalises upside volatility the same as downside volatility.
    A strategy with occasional huge winning days gets "punished" by Sharpe.
    This is why Sortino (below) exists.
    """

    annual_return = returns.mean() * config.TRADING_DAYS_YEAR
    annual_vol = returns.std()  * np.sqrt(config.TRADING_DAYS_YEAR)

    if annual_vol == 0:
        return np.nan

    sharpe = (annual_return - risk_free_rate) / annual_vol
    return sharpe


# Section 2: sortino ratio

def compute_sortino(returns: pd.Series,
                    risk_free_rate: float = config.RISK_FREE_RATE) -> float:
    """
    Sortino Ratio = (annualised return − risk-free rate) / downside deviation

    Difference from Sharpe: the denominator only counts negative returns.
    Downside deviation = std dev of returns that are below 0 (or below
    a target return, we use 0 here for simplicity).

    Why this matters for a long-short strategy:
    If our strategy has volatile UP days (good volatility) but calm
    down days, Sortino will reward it more than Sharpe does.
    """

    daily_target = 0.0 

    downside_returns = returns[returns < daily_target]

    if len(downside_returns) == 0:
        return np.nan 

    downside_deviation = downside_returns.std() * np.sqrt(config.TRADING_DAYS_YEAR)
    annual_return = returns.mean() * config.TRADING_DAYS_YEAR

    if downside_deviation == 0:
        return np.nan

    sortino = (annual_return - risk_free_rate) / downside_deviation
    return sortino


# Section 3: drawdown series 

def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Drawdown[t] = (equity[t] / running_max_equity_up_to_t) − 1

    Always ≤ 0. A value of -0.20 means "currently 20% below the
    highest point ever reached."

    This single series is the foundation for three metrics below:
    Max Drawdown, Calmar Ratio, and Recovery Time. Computing it once
    here avoids repeating the same logic three times.
    """
    equity = np.exp(returns.cumsum())
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1
    drawdown.name = "drawdown"
    return drawdown

# Section 4: max drawdown

def compute_max_drawdown(returns: pd.Series) -> dict:
    """
    Max Drawdown = the single worst peak-to-trough decline in the
    entire equity curve.

    We return not just the number but when it happened. The date
    of the peak and the date of the trough. This context matters:
    a -30% drawdown during the 2020 COVID crash has a different meaning
    than the same number happening for no clear reason.
    """
    drawdown = compute_drawdown_series(returns)

    trough_date = drawdown.idxmin()
    max_dd = drawdown.min()

    pre_trough = drawdown.loc[:trough_date]
    peak_date = pre_trough[pre_trough == 0].index[-1] if (pre_trough == 0).any() else pre_trough.index[0]

    return {
        "max_drawdown_pct": max_dd * 100,
        "peak_date":        peak_date,
        "trough_date":      trough_date,
    }


# Section 5: Calmar ratio

def compute_calmar(returns: pd.Series) -> float:
    """
    Calmar Ratio = annualised return / |max drawdown|

    Why this matters alongside Sharpe:
    Sharpe uses volatility.
    Calmar uses max drawdown.
    A strategy can have great Sharpe but one catastrophic drawdown
    that Sharpe smooths over, and Calmar catches that.
    """
    annual_return = (returns.mean() * config.TRADING_DAYS_YEAR)
    max_dd = compute_max_drawdown(returns)["max_drawdown_pct"] / 100

    if max_dd == 0:
        return np.nan

    calmar = annual_return / abs(max_dd)
    return calmar


# Section 6: recovery time

def compute_recovery_times(returns: pd.Series, threshold: float = -0.05) -> pd.DataFrame:
    """
    For every drawdown deeper than `threshold` (default -5%), measure:
    1. How many trading days it took to recover back to the previous peak
    2. If it never recovered, mark as "ongoing" with days-so-far

    """
    drawdown = compute_drawdown_series(returns)

    in_drawdown = False
    episodes = []
    peak_idx = 0

    for i, (date, dd) in enumerate(drawdown.items()):
        if not in_drawdown and dd < threshold:
            in_drawdown = True
            peak_idx = i

            while peak_idx > 0 and drawdown.iloc[peak_idx] < 0:
                peak_idx -= 1
            trough_idx = i
            trough_dd  = dd

        elif in_drawdown:
            if dd < trough_dd:
                trough_idx = i
                trough_dd  = dd
            if dd >= 0:   # fully recovered (back to or above previous peak)
                episodes.append({
                    "peak_date": drawdown.index[peak_idx],
                    "trough_date": drawdown.index[trough_idx],
                    "recovery_date": date,
                    "max_drawdown_pct": trough_dd * 100,
                    "days_to_trough": trough_idx - peak_idx,
                    "days_to_recover": i - trough_idx,
                    "total_days_under": i - peak_idx,
                    "status": "recovered",
                })
                in_drawdown = False

    # handle an ongoing drawdown at the end of the series
    if in_drawdown:
        episodes.append({
            "peak_date": drawdown.index[peak_idx],
            "trough_date": drawdown.index[trough_idx],
            "recovery_date": None,
            "max_drawdown_pct": trough_dd * 100,
            "days_to_trough": trough_idx - peak_idx,
            "days_to_recover": None,
            "total_days_underwater": len(drawdown) - 1 - peak_idx,
            "status": "ongoing",
        })

    episodes_df = pd.DataFrame(episodes)
    if len(episodes_df) > 0:
        config.log.info("Recovery analysis | %d drawdown episodes deeper than %.0f%%",
                        len(episodes_df), threshold * 100)
        config.log.info("Avg days to recover: %.0f | Longest underwater: %d days",
                        episodes_df["days_to_recover"].dropna().mean() if episodes_df["days_to_recover"].notna().any() else np.nan,
                        episodes_df["total_days_underwater"].max())
    else:
        config.log.info("No drawdown episodes deeper than %.0f%% found", threshold * 100)

    return episodes_df


# Section 7: tail risk 

def compute_tail_risk(returns: pd.Series,
                      alpha: float = config.TAIL_RISK_ALPHA) -> dict:
    """
    VaR (Value at Risk) at 95% confidence = the loss threshold that
    is only exceeded 5% of the time. E.g. VaR_95 = -2% means
    "on 95% of days, you lose less than 2%."

    CVaR / Expected Shortfall (the more useful number) = the average
    loss on those worst 5% of days.

    Why CVaR over VaR for risk reporting:
    VaR can be the same for a strategy with a small tail and one with
    a catastrophic tail, as long as the 5% threshold itself is similar.
    CVaR captures the difference because it looks inside the tail.
    """
    var_95  = returns.quantile(alpha)
    tail_returns = returns[returns <= var_95]
    cvar_95 = tail_returns.mean()

    return {
        "var_95_daily_pct": var_95 * 100,
        "cvar_95_daily_pct": cvar_95 * 100,
        "var_95_annualised_pct": var_95 * np.sqrt(config.TRADING_DAYS_YEAR) * 100,
        "cvar_95_annualised_pct": cvar_95 * np.sqrt(config.TRADING_DAYS_YEAR) * 100,
        "n_tail_days": len(tail_returns),
    }


# Section 8: full metrics summary

def compute_all_metrics(returns: pd.Series) -> dict:
    """
    Run every metric above and assemble a single report-card dict.
    This is the function other files will call.
    """
    max_dd_info = compute_max_drawdown(returns)
    tail_info = compute_tail_risk(returns)

    total_return = (np.exp(returns.sum()) - 1) * 100
    n_years = len(returns) / config.TRADING_DAYS_YEAR
    cagr = ((np.exp(returns.sum())) ** (1 / n_years) - 1) * 100 if n_years > 0 else np.nan

    metrics = {
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "annual_return_pct": returns.mean() * config.TRADING_DAYS_YEAR * 100,
        "annual_vol_pct": returns.std()  * np.sqrt(config.TRADING_DAYS_YEAR) * 100,
        "sharpe":  compute_sharpe(returns),
        "sortino": compute_sortino(returns),
        "calmar": compute_calmar(returns),
        "max_drawdown_pct": max_dd_info["max_drawdown_pct"],
        "max_dd_peak_date": max_dd_info["peak_date"],
        "max_dd_trough_date": max_dd_info["trough_date"],
        "var_95_daily_pct": tail_info["var_95_daily_pct"],
        "cvar_95_daily_pct": tail_info["cvar_95_daily_pct"],
        "win_rate_pct": (returns > 0).mean() * 100,
        "n_days": len(returns),
        "n_years": n_years,
    }

    return metrics


def print_metrics_table(metrics: dict, title: str = "Risk metrics & performance summary"):
    """Print the metrics dict as a readable table"""
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)
    print(f"{'Total Return':<28}{metrics['total_return_pct']:>10.1f}%")
    print(f"{'CAGR':<28}{metrics['cagr_pct']:>10.1f}%")
    print(f"{'Annualised Return':<28}{metrics['annual_return_pct']:>10.1f}%")
    print(f"{'Annualised Volatility':<28}{metrics['annual_vol_pct']:>10.1f}%")
    print("-" * 55)
    print(f"{'Sharpe Ratio':<28}{metrics['sharpe']:>10.2f}")
    print(f"{'Sortino Ratio':<28}{metrics['sortino']:>10.2f}")
    print(f"{'Calmar Ratio':<28}{metrics['calmar']:>10.2f}")
    print("-" * 55)
    print(f"{'Max Drawdown':<28}{metrics['max_drawdown_pct']:>10.1f}%")
    print(f"{'  Peak date':<28}{str(metrics['max_dd_peak_date'].date()):>10}")
    print(f"{'  Trough date':<28}{str(metrics['max_dd_trough_date'].date()):>10}")
    print("-" * 55)
    print(f"{'VaR 95% (daily)':<28}{metrics['var_95_daily_pct']:>10.2f}%")
    print(f"{'CVaR 95% (daily)':<28}{metrics['cvar_95_daily_pct']:>10.2f}%")
    print("-" * 55)
    print(f"{'Win Rate':<28}{metrics['win_rate_pct']:>10.1f}%")
    print(f"{'Period':<28}{metrics['n_years']:>10.1f} years")
    print("=" * 55)


# Section 9: visualisations

def plot_drawdown_chart(returns: pd.Series, save: bool = True):
    """
    Underwater chart - shows drawdown over time as a filled area below zero.
    Classic risk visualisation: the deeper and wider the red zones,
    the more painful the strategy is to hold through.
    """
    drawdown = compute_drawdown_series(returns)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                    color="#e63946", alpha=0.6)
    ax.plot(drawdown.index, drawdown.values * 100, color="#9d0208", linewidth=0.8)
    ax.set_title("Drawdown chart", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "05_drawdown_chart.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


def plot_returns_distribution(returns: pd.Series, save: bool = True):
    """
    Histogram of daily returns with VaR/CVaR marked.
    Shows the shape of the return distribution and exactly where the tail risk threshold sits.
    Makes "tail risk" tangible by showing the actual returns that fall into that worst 5% bucket.
    """
    tail_info = compute_tail_risk(returns)
    var_95  = tail_info["var_95_daily_pct"] / 100
    cvar_95 = tail_info["cvar_95_daily_pct"] / 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(returns, bins=80, color="#4cc9f0", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(var_95, color="#f77f00", linewidth=2, linestyle="--",
              label=f"VaR 95%: {var_95*100:.2f}%")
    ax.axvline(cvar_95, color="#e63946", linewidth=2, linestyle="--",
              label=f"CVaR 95%: {cvar_95*100:.2f}%")
    ax.set_title("Daily Returns Distribution & Tail Risk", fontsize=14, fontweight="bold")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "5_returns_distribution.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


# Main

if __name__ == "__main__":
    import importlib.util

    def _load_module(name, filename):
        spec = importlib.util.spec_from_file_location(name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    strategy_engine = _load_module("strategy_engine", os.path.join(config.BASE_DIR, "2_strategy_engine.py"))
    backtest_core = _load_module("backtest_core", os.path.join(config.BASE_DIR, "3_backtest_core.py"))

    prices_path  = os.path.join(config.DATA_CLEAN, "prices.csv")
    returns_path = os.path.join(config.DATA_CLEAN, "returns.csv")

    if not (os.path.exists(prices_path) and os.path.exists(returns_path)):
        print("Run 01_data_pipeline.py first.")
    else:
        prices  = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        returns_data = pd.read_csv(returns_path, index_col=0, parse_dates=True)

        strat = strategy_engine.run_strategy(prices)
        bt = backtest_core.run_backtest(
            weights  = strat["weights"],
            returns = returns_data,
            transaction_costs = strat["costs"],
        )

        net_returns = bt["net_returns"]

        metrics = compute_all_metrics(net_returns)
        print_metrics_table(metrics)

        recovery_df = compute_recovery_times(net_returns)
        if len(recovery_df) > 0:
            print("\n── Drawdown Episodes (>5%) ──")
            print(recovery_df.to_string(index=False))

        plot_drawdown_chart(net_returns)
        plot_returns_distribution(net_returns)