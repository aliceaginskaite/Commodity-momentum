"""
4_validation.py - validation suite

Three validation methods:
- Out-of-Sample (OOS): hard 70/30 split. OOS is locked and not examined 
  until all in-sample analysis is complete and parameters are frozen.
- Walk-Forward: rolling train/test windows across the full history.
- Time-Series cross-validation: sklearn TimeSeriesSplit with no future leakage.

Why three methods?
Each addresses a distinct failure mode:
- OOS - simple overfitting (does the strategy work on unseen data at all?)
- Walk-Forward - regime dependence (is performance consistent across 
  different time periods, not just one lucky split?)
- Time-Series CV - distribution of Sharpe ratios (not just a single pass/fail; 
  gives expected variance and robustness).
"""

import pandas as pd
import numpy as np
import os
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import TimeSeriesSplit
import config


# Load sibling modules 
def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

strategy_engine = _load_module("strategy_engine", os.path.join(config.BASE_DIR, "2_strategy_engine.py"))
backtest_core = _load_module("backtest_core", os.path.join(config.BASE_DIR, "3_backtest_core.py"))

# Helper: run strategy + backtest on any price slice, return key metrics

def _run_on_slice(prices_slice: pd.DataFrame,
                  returns_slice: pd.DataFrame,
                  window: int = config.MOMENTUM_WINDOW) -> dict:
    """
    Run the full strategy + backtest pipeline on a given price slice.
    This is the core reusable unit. Every validation method below
    calls this on different slices of history.
    """
    strat = strategy_engine.run_strategy(prices_slice, window=window)
    bt = backtest_core.run_backtest(
        weights           = strat["weights"],
        returns           = returns_slice,
        transaction_costs = strat["costs"],
    )

    net_returns = bt["net_returns"]

    if len(net_returns) < 5 or net_returns.std() == 0:
        # Not enough data/no variance in this slice = return NaN
        return {"sharpe": np.nan, "total_return": np.nan,
               "n_days": len(net_returns), "net_returns": net_returns}

    annual_return = net_returns.mean() * config.TRADING_DAYS_YEAR
    annual_vol = net_returns.std() * np.sqrt(config.TRADING_DAYS_YEAR)
    sharpe = (annual_return - config.RISK_FREE_RATE) / annual_vol if annual_vol > 0 else np.nan
    total_return = (np.exp(net_returns.sum()) - 1) * 100

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "annual_return": annual_return * 100,
        "annual_vol": annual_vol * 100,
        "n_days": len(net_returns),
        "net_returns": net_returns,
    }

# Section 1: out of sample

def run_out_of_sample_test(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """
    Hard 70/30 split. Run strategy independently on each half.

    What we're looking for:
    If IS Sharpe is great (e.g. 1.5) and OOS Sharpe collapses (e.g. 0.1),
    that's a red flag.Indicates overfitting to the specific quirks of the IS period; 
    edge does not generalise.

    If IS and OOS Sharpe are similar (within ~30-40% of each other),
    that's a good sign of genuine, generalisable edge.
    """
    config.log.info("=" * 60)
    config.log.info("Out-Of-Sample Test")
    config.log.info("=" * 60)

    split_idx  = int(len(prices) * config.TRAIN_RATIO)
    split_date = prices.index[split_idx]

    prices_is,  prices_oos  = prices.iloc[:split_idx],  prices.iloc[split_idx:]
    returns_is, returns_oos = returns.iloc[:split_idx], returns.iloc[split_idx:]

    config.log.info("Split date: %s | IS: %d days | OOS: %d days",
                    split_date.date(), len(prices_is), len(prices_oos))

    result_is  = _run_on_slice(prices_is,  returns_is)
    result_oos = _run_on_slice(prices_oos, returns_oos)

    config.log.info("IS  Sharpe: %.2f | Total Return: %.1f%%",
                    result_is["sharpe"], result_is["total_return"])
    config.log.info("OOS Sharpe: %.2f | Total Return: %.1f%%",
                    result_oos["sharpe"], result_oos["total_return"])

    # Degradation ratio (how much of the IS edge survives into OOS)?
    if result_is["sharpe"] and not np.isnan(result_is["sharpe"]) and result_is["sharpe"] != 0:
        degradation = result_oos["sharpe"] / result_is["sharpe"]
        config.log.info("OOS/IS Sharpe ratio: %.2f  (1.0 = perfect, <0 = OOS lost money while IS made money)",
                        degradation)
    else:
        degradation = np.nan

    return {
        "split_date":   split_date,
        "in_sample":    result_is,
        "out_of_sample": result_oos,
        "degradation_ratio": degradation,
    }


# ══════════════════════════════════════════════════════════════
# SECTION 2: WALK-FORWARD  ✅
# ══════════════════════════════════════════════════════════════

def run_walk_forward_test(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """
    Walk-forward analysis: slide a train/test window across the full history.

    Window structure (configured in config.py):
      WFO_TRAIN_MONTHS = 24 - not actually used for parameter fitting here
      (we use the same fixed params throughout, WFO tests stability, not re optimization)
      WFO_TEST_MONTHS = 6 - each test window is 6 months
      WFO_N_SPLITS = 8 - repeat 8 times, sliding forward

    Why this matters more than a single OOS split:
    A single split could be lucky or unlucky. Walk-forward gives us
    8 independent "out-of-sample" tests across different market regimes
    (2014 oil crash, 2020 COVID, 2022 inflation, etc).
    If performance is consistently positive across most windows,
    that's much stronger evidence than one good OOS period.

    Note on methodology: a "true" walk-forward also re-optimizes parameters
    on each training window. We keep parameters fixed here and use walk-forward
    purely to test stability across time, not to re fit. Re-optimizing per window would
    reintroduce the snooping risk we're trying to avoid.
    """
    config.log.info("=" * 60)
    config.log.info("Walk-Forward test")
    config.log.info("=" * 60)

    test_days = config.WFO_TEST_MONTHS * 21 #days per month
    n_splits  = config.WFO_N_SPLITS

    total_needed = test_days * n_splits
    if total_needed > len(prices):
        n_splits = len(prices) // test_days
        config.log.warning("Not enough data for %d splits — reduced to %d",
                          config.WFO_N_SPLITS, n_splits)

    start_offset = len(prices) - (test_days * n_splits)
    start_offset = max(start_offset, config.MOMENTUM_WINDOW + 10) #leaves room for momentum

    window_results = []
    for i in range(n_splits):
        test_start_idx = start_offset + i * test_days
        test_end_idx = test_start_idx + test_days
        buffer_start_idx = max(0, test_start_idx - config.MOMENTUM_WINDOW - 30)

        if test_end_idx > len(prices):
            break

        prices_window  = prices.iloc[buffer_start_idx:test_end_idx]
        returns_window = returns.iloc[buffer_start_idx:test_end_idx]

        result = _run_on_slice(prices_window, returns_window)

        # only count the test portion's performance not the buffer
        test_start_date = prices.index[test_start_idx]
        test_end_date = prices.index[min(test_end_idx - 1, len(prices) - 1)]

        window_results.append({
            "window": i + 1,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "sharpe": result["sharpe"],
            "total_return": result["total_return"],
        })

        config.log.info("Window %d/%d | %s → %s | Sharpe: %.2f | Return: %.1f%%",
                        i + 1, n_splits,
                        test_start_date.date(), test_end_date.date(),
                        result["sharpe"], result["total_return"])

    wfo_df = pd.DataFrame(window_results)

    pct_positive = (wfo_df["total_return"] > 0).mean() * 100
    avg_sharpe   = wfo_df["sharpe"].mean()

    config.log.info("Walk-forward summary | avg Sharpe: %.2f | %% positive windows: %.0f%%",
                    avg_sharpe, pct_positive)

    return {
        "windows": wfo_df,
        "avg_sharpe": avg_sharpe,
        "pct_positive_windows": pct_positive,
    }


# Section 3: time-series cross validation 

def run_time_series_cv(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """
    sklearn's TimeSeriesSplit — the key difference from regular K-Fold CV:
    regular CV would randomly shuffle data into folds, which means a
    "training" fold could contain data from after a "test" fold.
    That's lookahead bias putted into the validation method itself.

    """
    config.log.info("=" * 60)
    config.log.info("Time-series cross validation")
    config.log.info("=" * 60)

    tscv = TimeSeriesSplit(n_splits=config.CV_N_SPLITS)
    n = len(prices)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(prices), start=1):
        # We need the test slice (plus a small buffer before it for momentum calc)
        buffer_start = max(0, test_idx[0] - config.MOMENTUM_WINDOW - 30)
        test_end = test_idx[-1] + 1

        prices_test = prices.iloc[buffer_start:test_end]
        returns_test = returns.iloc[buffer_start:test_end]

        result = _run_on_slice(prices_test, returns_test)

        fold_results.append({
            "fold": fold_i,
            "test_start": prices.index[test_idx[0]],
            "test_end": prices.index[test_idx[-1]],
            "sharpe": result["sharpe"],
            "total_return": result["total_return"],
        })

        config.log.info("Fold %d | %s → %s | Sharpe: %.2f",
                        fold_i,
                        prices.index[test_idx[0]].date(),
                        prices.index[test_idx[-1]].date(),
                        result["sharpe"])

    cv_df = pd.DataFrame(fold_results)
    sharpe_mean = cv_df["sharpe"].mean()
    sharpe_std = cv_df["sharpe"].std()

    config.log.info("CV Sharpe distribution | mean: %.2f | std: %.2f | range: [%.2f, %.2f]",
                    sharpe_mean, sharpe_std, cv_df["sharpe"].min(), cv_df["sharpe"].max())

    return {
        "folds": cv_df,
        "sharpe_mean": sharpe_mean,
        "sharpe_std":  sharpe_std,
    }


# Section 4: visualizing

def plot_validation_summary(oos_result: dict,
                            wfo_result: dict,
                            cv_result: dict,
                            save: bool = True):
    """
    Three-panel chart
    1. IS vs OOS bar comparison (Sharpe)
    2. Walk-forward sharpe across windows (line chart over time)
    3. Time-series CV Sharpe distribution (box/strip plot across folds)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Validation Suite — OOS / Walk-Forward / Time-Series CV",
                fontsize=14, fontweight="bold")

    # Panel 1
    ax1 = axes[0]
    is_sharpe  = oos_result["in_sample"]["sharpe"]
    oos_sharpe = oos_result["out_of_sample"]["sharpe"]
    bars = ax1.bar(["In-Sample", "Out-of-Sample"], [is_sharpe, oos_sharpe],
                   color=["#4cc9f0", "#1a1a2e"])
    ax1.axhline(0, color="gray", linewidth=0.7)
    ax1.set_title("Out-of-Sample Test")
    ax1.set_ylabel("Sharpe Ratio")
    for bar, val in zip(bars, [is_sharpe, oos_sharpe]):
        ax1.text(bar.get_x() + bar.get_width()/2, val, f"{val:.2f}",
                 ha="center", va="bottom" if val >= 0 else "top", fontsize=10)

    # Panel 2
    ax2 = axes[1]
    wfo_df = wfo_result["windows"]
    colors = ["#2a9d8f" if s > 0 else "#e63946" for s in wfo_df["sharpe"]]
    ax2.bar(wfo_df["window"], wfo_df["sharpe"], color=colors)
    ax2.axhline(0, color="gray", linewidth=0.7)
    ax2.axhline(wfo_result["avg_sharpe"], color="black", linewidth=1, linestyle="--",
               label=f"Avg: {wfo_result['avg_sharpe']:.2f}")
    ax2.set_title("Walk-Forward Windows")
    ax2.set_xlabel("Window #")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.legend(fontsize=8)

    # Panel 3
    ax3 = axes[2]
    cv_df = cv_result["folds"]
    ax3.scatter(cv_df["fold"], cv_df["sharpe"], s=80, color="#f77f00", zorder=3)
    ax3.axhline(cv_result["sharpe_mean"], color="black", linewidth=1, linestyle="--",
               label=f"Mean: {cv_result['sharpe_mean']:.2f}")
    ax3.fill_between(
        [cv_df["fold"].min() - 0.5, cv_df["fold"].max() + 0.5],
        cv_result["sharpe_mean"] - cv_result["sharpe_std"],
        cv_result["sharpe_mean"] + cv_result["sharpe_std"],
        alpha=0.15, color="gray", label="±1 std"
    )
    ax3.axhline(0, color="gray", linewidth=0.7)
    ax3.set_title("Time-Series CV Folds")
    ax3.set_xlabel("Fold #")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "04_validation_summary.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


# Main

def run_full_validation(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Run all three validation methods and return combined results."""
    oos_result = run_out_of_sample_test(prices, returns)
    wfo_result = run_walk_forward_test(prices, returns)
    cv_result = run_time_series_cv(prices, returns)

    plot_validation_summary(oos_result, wfo_result, cv_result)

    return {
        "out_of_sample": oos_result,
        "walk_forward": wfo_result,
        "cross_validation": cv_result,
    }


if __name__ == "__main__":
    prices_path  = os.path.join(config.DATA_CLEAN, "prices.csv")
    returns_path = os.path.join(config.DATA_CLEAN, "returns.csv")

    if not (os.path.exists(prices_path) and os.path.exists(returns_path)):
        print("Run 01_data_pipeline.py first.")
    else:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

        results = run_full_validation(prices, returns)

        print("\n" + "=" * 60)
        print("Validation summary")
        print("=" * 60)
        print(f"OOS Sharpe (IS → OOS):     {results['out_of_sample']['in_sample']['sharpe']:.2f} → "
             f"{results['out_of_sample']['out_of_sample']['sharpe']:.2f}")
        print(f"Walk-Forward avg Sharpe:    {results['walk_forward']['avg_sharpe']:.2f} "
             f"({results['walk_forward']['pct_positive_windows']:.0f}% positive windows)")
        print(f"Time-Series CV Sharpe:      {results['cross_validation']['sharpe_mean']:.2f} "
             f"± {results['cross_validation']['sharpe_std']:.2f}")