"""
7 - robustness.py - robustness & Stress testing

Checklist items covered here:
  Monte Carlo — shuffle returns, see if real result beats random luck
  Parameter Stability — sweep momentum window, look for a smooth surface
  Stress Tests — performance during known historical crises
  Regime Analysis — performance split by risk-on/risk-off regimes

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

strategy_engine = _load_module("2_strategy_engine", os.path.join(config.BASE_DIR, "2_strategy_engine.py"))
backtest_core = _load_module("3_backtest_core",   os.path.join(config.BASE_DIR, "3_backtest_core.py"))


def _run_on_slice(prices_slice: pd.DataFrame,
                  returns_slice: pd.DataFrame,
                  window: int = config.MOMENTUM_WINDOW) -> dict:
    """Same reusable helper pattern as in 4_validation.py."""
    strat = strategy_engine.run_strategy(prices_slice, window=window)
    bt = backtest_core.run_backtest(
        weights = strat["weights"],
        returns = returns_slice,
        transaction_costs = strat["costs"],
    )
    net_returns = bt["net_returns"]

    if len(net_returns) < 5 or net_returns.std() == 0:
        return {"sharpe": np.nan, "total_return": np.nan, "net_returns": net_returns}

    annual_return = net_returns.mean() * config.TRADING_DAYS_YEAR
    annual_vol = net_returns.std()  * np.sqrt(config.TRADING_DAYS_YEAR)
    sharpe = (annual_return - config.RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
    total_return = (np.exp(net_returns.sum()) - 1) * 100

    return {"sharpe": sharpe, "total_return": total_return, "net_returns": net_returns}



# Section 1: Monte Carlo 

def run_monte_carlo(net_returns: pd.Series,
                    n_sims: int = config.MONTE_CARLO_SIMS,
                    seed: int = config.MC_RANDOM_SEED) -> dict:
    """
    Take our actual daily returns and shuffle the ORDER randomly,
    many times. Build an equity curve for each shuffle.

    The logic:
    If our strategy's edge is real, the actual (unshuffled) sequence
    shouldn't be dramatically better than a typical random reordering
    of the same returns because shuffling preserves the mean,
    std, and overall distribution of returns, it just breaks the
    time order. If real performance only looks good because of the
    lucky order returns happened to occur in, that's a red flag.

    What we specifically check:
    - Where does our real final equity value rank among all the
      shuffled versions? (percentile)
    - Is our real max drawdown unusually mild compared to the
      distribution of shuffled max drawdowns?

    Note: this version shuffles non-overlapping daily returns
    (assumes low autocorrelation, reasonable for a weekly-rebalanced
    long-short strategy). It does not resample the strategy logic itself —
    it's a check on the outcome sequence, not a full re-simulation.
    """
    config.log.info("=" * 60)
    config.log.info("Monte Carlo Simulation (%d sims)", n_sims)
    config.log.info("=" * 60)

    rng = np.random.default_rng(seed)
    returns_array = net_returns.values
    n_days = len(returns_array)

    simulated_curves = np.zeros((n_sims, n_days))
    final_values = np.zeros(n_sims)
    max_drawdowns = np.zeros(n_sims)

    for i in range(n_sims):
        shuffled = rng.permutation(returns_array)
        equity = 100 * np.exp(np.cumsum(shuffled))
        simulated_curves[i] = equity

        running_max = np.maximum.accumulate(equity)
        dd = (equity / running_max) - 1
        max_drawdowns[i] = dd.min()
        final_values[i] = equity[-1]

    # real (actual order) equity curve for comparison
    real_equity = 100 * np.exp(np.cumsum(returns_array))
    real_final = real_equity[-1]
    real_running_max = np.maximum.accumulate(real_equity)
    real_dd = ((real_equity / real_running_max) - 1).min()

    final_value_percentile = (final_values < real_final).mean() * 100
    dd_percentile = (max_drawdowns < real_dd).mean() * 100 

    config.log.info("Real final equity: %.1f | percentile vs %d random shuffles: %.0f%%",
                    real_final, n_sims, final_value_percentile)
    config.log.info("Real max drawdown: %.1f%% | percentile vs shuffles: %.0f%%",
                    real_dd * 100, dd_percentile)

    if final_value_percentile < 50:
        config.log.warning("Real result is WORSE than the median random shuffle — "
                          "order of returns didn't help us. Edge (if any) doesn't "
                          "depend on timing luck, which is actually reassuring for "
                          "robustness, though it also means raw return ordering isn't special.")
    else:
        config.log.info("Real result beats the median random shuffle - consistent with "
                        "a real, order-independent edge (not just a lucky sequence).")

    return {
        "simulated_curves": simulated_curves,   # shape (n_sims, n_days)
        "real_equity": real_equity,
        "final_values": final_values,
        "max_drawdowns": max_drawdowns,
        "real_final": real_final,
        "real_max_dd": real_dd,
        "final_value_percentile": final_value_percentile,
        "dd_percentile": dd_percentile,
        "dates": net_returns.index,
    }


def plot_monte_carlo_fan_chart(mc_result: dict, save: bool = True):
    """
    The classic "fan chart" is a hundreds of thin, semi-transparent
    random-shuffle equity curves plotted together with the real equity curve drawn on top as
    a single bold line so it's easy to see where reality landed
    relative to the cloud of random possibilities.
    """
    sim_curves = mc_result["simulated_curves"]
    real_eq = mc_result["real_equity"]
    dates = mc_result["dates"]

    fig, ax = plt.subplots(figsize=(13, 7))

    # plot a sample of simulations
    n_to_plot = min(500, sim_curves.shape[0])
    for i in range(n_to_plot):
        ax.plot(dates, sim_curves[i], color="#5652d5", alpha=0.035, linewidth=0.7)

    # overlay percentile bands for extra clarity
    p05 = np.percentile(sim_curves, 5, axis=0)
    p95 = np.percentile(sim_curves, 95, axis=0)
    p50 = np.percentile(sim_curves, 50, axis=0)
    ax.plot(dates, p50, color="#8a0251", linewidth=1.2, linestyle="--",
           label="Median of random shuffles", alpha=0.8)
    ax.fill_between(dates, p05, p95, color="#0077b6", alpha=0.07,
                    label="5th–95th percentile band")

    # real equity curve on top, bold
    ax.plot(dates, real_eq, color="#39e656", linewidth=2.2,
           label="Actual strategy (real order)", zorder=10)

    ax.set_title(f"Monte Carlo Fan Chart — {sim_curves.shape[0]} Random Reorderings\n"
                f"Real result percentile: {mc_result['final_value_percentile']:.0f}%",
                fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value (base 100)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "7_monte_carlo_fan_chart.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


# Section 2: parameter stability

def run_parameter_stability(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Sweep the momentum window from PARAM_WINDOW_MIN to PARAM_WINDOW_MAX.
    For each window value, run the full strategy + backtest and record sharpe.

    We test the value we actually use (config.MOMENTUM_WINDOW = 63) by
    making sure it sits within this swept range.
    """
    config.log.info("=" * 60)
    config.log.info("Parameter stability sweep")
    config.log.info("=" * 60)

    windows = range(config.PARAM_WINDOW_MIN, config.PARAM_WINDOW_MAX + 1, config.PARAM_WINDOW_STEP)
    results = []

    for window in windows:
        result = _run_on_slice(prices, returns, window=window)
        results.append({
            "window": window,
            "sharpe": result["sharpe"],
            "total_return": result["total_return"],
        })
        config.log.info("Window=%3d days | Sharpe: %6.2f | Total Return: %7.1f%%",
                        window, result["sharpe"], result["total_return"])

    stability_df = pd.DataFrame(results)

    # Smoothness check: how much does Sharpe change between adjacent windows?
    sharpe_diffs = stability_df["sharpe"].diff().abs()
    config.log.info("Avg Sharpe change between adjacent windows: %.3f "
                    "(smaller = smoother = more robust)", sharpe_diffs.mean())

    return stability_df


def plot_parameter_stability(stability_df: pd.DataFrame, save: bool = True):
    """
    Line chart: Sharpe ratio vs momentum window.
    Highlight our chosen window (config.MOMENTUM_WINDOW = 63) with a
    vertical line, so it's visually obvious whether we are on a stable
    base or a fragile spike.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(stability_df["window"], stability_df["sharpe"],
           marker="o", markersize=4, color="#1a1a2e", linewidth=1.5)
    ax.axvline(config.MOMENTUM_WINDOW, color="#e63946", linewidth=1.5, linestyle="--",
              label=f"Chosen window: {config.MOMENTUM_WINDOW} days")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_title("Parameter Stability — Sharpe Ratio vs Momentum Window",
                fontsize=14, fontweight="bold")
    ax.set_xlabel("Momentum Window (trading days)")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "07_parameter_stability.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()



# Section 3: stress tests

def run_stress_tests(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Run the strategy specifically during known historical crisis windows
    (defined in config.STRESS_PERIODS). These are not cherry-picked good
    periods. They're picked because they were hard for most strategies,
    so seeing how ours behaved is informative regardless of outcome.

    For each period we include a buffer of history before it so the
    momentum calculation has enough lookback at the start of the window.
    """
    config.log.info("=" * 60)
    config.log.info("Stress tests")
    config.log.info("=" * 60)

    results = []
    for name, (start, end) in config.STRESS_PERIODS.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        buffer_start = start_dt - pd.Timedelta(days=config.MOMENTUM_WINDOW * 2)

      # Rebuild mask for each DataFrame — returns has one fewer row (lost to .diff()), so indices differ.
        prices_mask  = (prices.index >= buffer_start) & (prices.index <= end_dt)
        returns_mask = (returns.index >= buffer_start) & (returns.index <= end_dt)

        prices_window  = prices.loc[prices_mask]
        returns_window = returns.loc[returns_mask]

        if len(prices_window) < config.MOMENTUM_WINDOW + 10:
            config.log.warning("Skipping '%s' — not enough data in this period", name)
            continue

        result = _run_on_slice(prices_window, returns_window)

        # Trim to just the actual stress period (not the buffer) for reporting
        actual_mask = (result["net_returns"].index >= start_dt) & (result["net_returns"].index <= end_dt)
        period_returns = result["net_returns"].loc[actual_mask]
        period_total_return = (np.exp(period_returns.sum()) - 1) * 100 if len(period_returns) > 0 else np.nan

        results.append({
            "period": name,
            "start": start_dt,
            "end": end_dt,
            "total_return_pct": period_total_return,
            "n_days": len(period_returns),
        })

        config.log.info("%-25s | %s → %s | Return: %7.1f%%",
                        name, start_dt.date(), end_dt.date(), period_total_return)

    stress_df = pd.DataFrame(results)
    return stress_df


def plot_stress_tests(stress_df: pd.DataFrame, save: bool = True):
    """Bar chart of returns during each stress period. Red = lost money, green = held up."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2a9d8f" if r > 0 else "#e63946" for r in stress_df["total_return_pct"]]
    bars = ax.barh(stress_df["period"], stress_df["total_return_pct"], color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title("Performance During Historical Stress Periods", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Return During Period (%)")
    for bar, val in zip(bars, stress_df["total_return_pct"]):
        ax.text(val, bar.get_y() + bar.get_height()/2, f" {val:.1f}%",
               va="center", ha="left" if val >= 0 else "right", fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "7_stress_tests.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()



# Section 4: regime analysis

def run_regime_analysis(prices: pd.DataFrame, returns: pd.DataFrame, net_returns: pd.Series) -> pd.DataFrame:
    """
    Split history into "risk-on" vs "risk-off" regimes using a simple
    rule: rolling realised volatility of an equal-weighted
    commodity basket. High vol = risk-off / turbulent, low vol = risk-on / calm.

    Why this specific regime definition?
    It is simple and doesn't use any external regime
    classification service. Anyone reading the code can verify exactly
    how "risk-off" was defined. More sophisticated regime models (HMM,
    GARCH) are documented as future work in the report.

    For each regime, we compute Sharpe and total return separately.
    If the strategy only works in one regime, that's a negative finding. 
    It means the "edge" isn't momentum in general, it's momentum
    inder specific conditions, which is a much narrower (but still useful)
    claim.
    """
    config.log.info("=" * 60)
    config.log.info("Regime analysis")
    config.log.info("=" * 60)

    # equal weighted basket return as market proxy
    basket_returns = returns.mean(axis=1)
    rolling_vol = basket_returns.rolling(21).std() * np.sqrt(config.TRADING_DAYS_YEAR)

    vol_median = rolling_vol.median()
    regime = pd.Series(
        np.where(rolling_vol > vol_median, "risk_off_high_vol", "risk_on_low_vol"),
        index=rolling_vol.index
    )

    common_idx = net_returns.index.intersection(regime.index)
    net_aligned = net_returns.loc[common_idx]
    regime_aligned = regime.loc[common_idx]

    results = []
    for regime_name in ["risk_on_low_vol", "risk_off_high_vol"]:
        mask = regime_aligned == regime_name
        regime_returns = net_aligned[mask]

        if len(regime_returns) < 5 or regime_returns.std() == 0:
            continue

        annual_return = regime_returns.mean() * config.TRADING_DAYS_YEAR
        annual_vol = regime_returns.std()  * np.sqrt(config.TRADING_DAYS_YEAR)
        sharpe = (annual_return - config.RISK_FREE_RATE) / annual_vol

        results.append({
            "regime": regime_name,
            "n_days": len(regime_returns),
            "pct_of_history": len(regime_returns) / len(net_aligned) * 100,
            "annual_return_pct": annual_return * 100,
            "sharpe": sharpe,
        })

        config.log.info("%-20s | %d days (%.0f%%) | Annual Return: %6.1f%% | Sharpe: %5.2f",
                        regime_name, len(regime_returns),
                        len(regime_returns) / len(net_aligned) * 100,
                        annual_return * 100, sharpe)

    regime_df = pd.DataFrame(results)
    return regime_df


def plot_regime_analysis(regime_df: pd.DataFrame, save: bool = True):
    """Side by side bar comparison: sharpe and annual return, risk-on vs risk-off."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    colors = ["#2a9d8f", "#e63946"]

    ax1 = axes[0]
    ax1.bar(regime_df["regime"], regime_df["sharpe"], color=colors)
    ax1.axhline(0, color="gray", linewidth=0.7)
    ax1.set_title("Sharpe Ratio by Regime")
    ax1.set_ylabel("Sharpe Ratio")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15)

    ax2 = axes[1]
    ax2.bar(regime_df["regime"], regime_df["annual_return_pct"], color=colors)
    ax2.axhline(0, color="gray", linewidth=0.7)
    ax2.set_title("Annualised Return by Regime")
    ax2.set_ylabel("Annual Return (%)")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "7_regime_analysis.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


# Main

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

        #  Monte Carlo
        mc_result = run_monte_carlo(net_returns)
        plot_monte_carlo_fan_chart(mc_result)

        # Parameter Stability
        stability_df = run_parameter_stability(prices, returns)
        plot_parameter_stability(stability_df)

        # Stress Tests
        stress_df = run_stress_tests(prices, returns)
        plot_stress_tests(stress_df)

        # Regime Analysis
        regime_df = run_regime_analysis(prices, returns, net_returns)
        plot_regime_analysis(regime_df)

        print("\n" + "=" * 60)
        print("Robustness summary")
        print("=" * 60)
        print(f"Monte Carlo percentile (final equity): {mc_result['final_value_percentile']:.0f}%")
        print(f"Parameter stability (avg adjacent Sharpe change): "
             f"{stability_df['sharpe'].diff().abs().mean():.3f}")
        print(f"\nStress test results:")
        print(stress_df[["period", "total_return_pct"]].to_string(index=False))
        print(f"\nRegime analysis:")
        print(regime_df[["regime", "annual_return_pct", "sharpe"]].to_string(index=False))