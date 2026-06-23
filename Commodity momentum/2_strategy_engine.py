'''
# 2 - strategy_engine.py — Strategy Signal Engine

# Covered here:
# Lookahead Bias - signals shifted +1 day
# Spread / Commission / Slippage - cost applied on every rebalance

# Logic:
#   1. Compute 63-day momentum score for each instrument
#   2. Rank instruments daily
#   3. Long top-2, Short bottom-2, Neutral middle-1
#   4. Rebalance weekly (not daily — reduces turnover and costs)
#   5. Shift signals +1 day → execution happens next day open
#   6. Apply transaction costs on every position change
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config

# Section 1: momentum score

def compute_momentum_scores(prices: pd.DataFrame,
                             window: int = config.MOMENTUM_WINDOW) -> pd.DataFrame:
    """
Momentum score: log return over window days.

Log returns used for time-additivity (simple returns compound multiplicatively, logs sum linearly).

Critical lookahead protection: signal at day T uses prices up to T-1 only.
Today's close (T) is excluded, signal generated after market close, executed at next open.
Implementation: prices.shift(1) aligns scores to close-of-day timing.

Returns

pd.DataFrame : momentum scores, same shape as input prices.
    """

    # Use log of price ratio over window:
    # score[t] = log(price[t-1]) - log(price[t-1-window])
    log_prices = np.log(prices.shift(1)) 
    momentum = log_prices - log_prices.shift(window)

    momentum.index.name = "Date"
    config.log.info("Momentum scores computed | window=%d days | shape=%s",
                    window, momentum.shape)
    return momentum


# Section 2: rank and generate signals

def generate_signals(momentum: pd.DataFrame,
                     n_long:  int = config.N_LONG,
                     n_short: int = config.N_SHORT) -> pd.DataFrame:
    """
    Rank instruments by momentum score each day.
    Assign: top-N - +1 (long), bottom-N - -1 (short), rest - 0.

    With 5 instruments, N_LONG=2, N_SHORT=2:
        Rank 1 (highest momentum) - +1
        Rank 2 - +1
        Rank 3 (middle) - 0 - out
        Rank 4 - -1
        Rank 5 (lowest momentum) - -1

  Equal weights used for simplicity and robustness. 
  Risk-weighting (e.g., inverse volatility) is reserved as a documented enhancement.

    Returns
   
   pd.DataFrame  same shape as momentum, values ∈ {-1, 0, +1}
    """
    n_instruments = len(momentum.columns)

    ranks = momentum.rank(axis=1, ascending=False, method="first")

    signals = pd.DataFrame(0, index=momentum.index, columns=momentum.columns)
    signals[ranks <= n_long]                      = 1    # top performers → long
    signals[ranks > (n_instruments - n_short)]    = -1   # worst performers → short

    config.log.info("Signals generated | long=%d short=%d neutral=%d",
                    n_long, n_short, n_instruments - n_long - n_short)
    return signals


# Section 3: rebalance schedule

def apply_rebalance_schedule(signals: pd.DataFrame,
                              freq: str = config.REBALANCE_FREQ) -> pd.DataFrame:
    """
    Positions updated only on rebalance days; held constant between.
    Weekly rebalancing used over daily to control transaction costs:
    ~52 rebalances/year at ~10 bps each = ~0.52% annual cost.
    Daily would cost ~2.52%, significantly eroding net returns.

    Implementation: signals resampled to weekly (last value), 
    then forward-filled to daily — creating a step-function.

  Parameters

  freq : str - 'D' daily, 'W' weekly (Friday), 'M' month-end
    """
    if freq == "D":
        config.log.info("Rebalance: DAILY (high turnover warning)")
        return signals

    # Resample to desired frequency, forward-fill back to daily
    
    freq_anchor = "W-FRI" if freq == "W" else freq
    signals_resampled = (
        signals
        .resample(freq_anchor)
        .last()                         
        .reindex(signals.index)         
        .ffill()                       
        .fillna(0)                     
    )

    config.log.info("Rebalance frequency: %s | rebalance days per year: ~%d",
                    freq, {"W": 52, "M": 12}.get(freq, "?"))
    return signals_resampled


# Section 4: lookahead shift

def shift_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """
   Signals are shifted forward by 1 day — the single most critical 
   lookahead safeguard in the pipeline.

   Signal at day T is generated using close-of-day prices at T, 
   but executed at T+1 open, reflecting real-world timing 
   (calculation after close, order placed overnight, filled at open).
    """
    shifted = signals.shift(1).fillna(0)
    config.log.info("Signals shifted +1 day (lookahead bias protection)")
    return shifted


# Section 5: position sizing

def compute_weights(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Equal-weight within legs, balanced long/short.

    Example (N_LONG=2, N_SHORT=2):
    Long leg: +0.5 each (sum +1.0)
    Short leg: -0.5 each (sum -1.0)
    Net exposure: 0.0

    Market-neutral by construction expresses relative strength view (winners outperform losers), 
    not directional commodity view.

  Returns

  pd.DataFrame : weights in {-0.5, 0, +0.5}, same shape as signals.
    """
    weights = signals.copy().astype(float)

    # Count long and short positions each day

    n_long  = (signals == 1).sum(axis=1)
    n_short = (signals == -1).sum(axis=1)

    for date in signals.index:
        row  = signals.loc[date]
        nl   = n_long[date]
        ns   = n_short[date]
        for ticker in signals.columns:
            sig = row[ticker]
            if sig == 1 and nl > 0:
                weights.loc[date, ticker] = 1.0 / nl
            elif sig == -1 and ns > 0:
                weights.loc[date, ticker] = -1.0 / ns
            else:
                weights.loc[date, ticker] = 0.0

    config.log.info("Weights computed | avg gross exposure: %.2f",
                    weights.abs().sum(axis=1).mean())
    return weights


def compute_weights_fast(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised version of compute_weights (much faster for long histories).
    Same result as compute_weights() but uses matrix operations instead of loops.

    Learning note:
    The loop version above is easier to read and understand.
    The vectorised version below is how you'd write it in production.
    Both are included so you can compare.
    """
    weights = signals.astype(float).copy()

    long_mask    = signals == 1
    long_counts  = long_mask.sum(axis=1).replace(0, np.nan) 
    weights[long_mask] = (long_mask.div(long_counts, axis=0))[long_mask]

    short_mask   = signals == -1
    short_counts = short_mask.sum(axis=1).replace(0, np.nan)
    weights[short_mask] = -(short_mask.div(short_counts, axis=0))[short_mask]

    weights = weights.fillna(0)
    return weights


# Section 6: transaction costs  spread / commission / slippage


def compute_transaction_costs(weights: pd.DataFrame) -> pd.Series:
    """
    Cost is charged when weights change (i.e. on rebalance days):
    cost = |delta_weight| * total_cost_bps / 10000
 
    Total cost is 10 bps per round-trip trade, made up of three
    components: spread (5bp, the bid-ask gap — you buy at ask and
    sell at bid), commission (2bp, broker fee per side), and slippage
    (3bp, the price moving against you while an order fills).
 
    Returns
    -------
    pd.Series  daily cost as fraction of portfolio value (e.g. 0.001 = 0.1%)
    """
    # Weight change each day = amount we need to trade
    weight_changes = weights.diff().abs()   # |w[t] - w[t-1]|
 
    # Sum across all instruments → total portfolio turnover on that day
    daily_turnover = weight_changes.sum(axis=1)
 
    # Cost = turnover × cost per unit traded
    cost_per_unit  = config.TOTAL_COST_BPS / 10_000   # convert bps → fraction
 
    daily_costs    = daily_turnover * cost_per_unit
    daily_costs.name = "transaction_costs"
 
    config.log.info("Transaction costs | total_cost=%.0fbps | avg daily cost=%.4f%%",
                    config.TOTAL_COST_BPS, daily_costs.mean() * 100)
    return daily_costs


# Section 7: visualise signals

def plot_signals(signals: pd.DataFrame,
                 weights: pd.DataFrame,
                 save: bool = True):
    """
    Two charts:
    1 Signal heatmap: shows which instrument is long/short/neutral each week
    2 Weight over time per instrument

    What to look for:
    Positions rotate over time (not stuck in one instrument forever)
    Never more than N_LONG instruments are long at once
    Long and short legs look balanced
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Strategy Signals & Weights", fontsize=14, fontweight="bold")

    # Chart 1: signal heatmap 
    ax1 = axes[0]
    sig_weekly = signals.resample("W").last()
    im = ax1.imshow(
        sig_weekly.T.values,
        aspect="auto",
        cmap="RdYlGn",  
        vmin=-1, vmax=1,
        interpolation="none",
    )
    ax1.set_yticks(range(len(signals.columns)))
    ax1.set_yticklabels(signals.columns, fontsize=9)
    ax1.set_title("Position Signals (green=long, red=short, yellow=neutral)")

    # X-axis (show years)
    n_weeks = len(sig_weekly)
    year_ticks = [i for i, d in enumerate(sig_weekly.index) if d.month == 1 and d.day <= 7]
    ax1.set_xticks(year_ticks)
    ax1.set_xticklabels([str(sig_weekly.index[i].year) for i in year_ticks], rotation=45)

    plt.colorbar(im, ax=ax1, orientation="vertical", fraction=0.02, pad=0.04)

    # Chart 2: weights over time
    ax2 = axes[1]
    weights_weekly = weights.resample("W").last()
    for ticker in weights.columns:
        ax2.plot(weights_weekly.index, weights_weekly[ticker],
                 label=ticker, linewidth=1.0, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Portfolio Weights Over Time")
    ax2.set_ylabel("Weight")
    ax2.legend(loc="upper right", fontsize=8, ncol=3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "2_signals_weights.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


# MAIN — callable from other modules

def run_strategy(prices: pd.DataFrame,
                 window: int = config.MOMENTUM_WINDOW) -> dict:
    """
   Full signal generation pipeline.

   Callable from 3_backtest_core with any price DataFrame 
   (full history, IS/OOS, stress periods, etc.).

   Returns
    momentum: raw momentum scores
    signals: raw integer signals (-1/0/+1) pre-shift
    signals_shifted: signals shifted +1 day - use this for backtest
    weights: portfolio weights post-shift
    costs: daily transaction cost series
    """
    config.log.info("Running strategy engine...")

    momentum = compute_momentum_scores(prices, window=window)
    signals_raw = generate_signals(momentum)
    signals_rebal = apply_rebalance_schedule(signals_raw)
    signals_shifted = shift_signals(signals_rebal)          # ← the crucial step
    weights = compute_weights_fast(signals_shifted)
    costs = compute_transaction_costs(weights)

    config.log.info("Strategy engine complete")

    return {
        "momentum": momentum,
        "signals_raw": signals_raw,
        "signals_shifted": signals_shifted,
        "weights": weights,
        "costs": costs,
    }


if __name__ == "__main__":

    prices_path = os.path.join(config.DATA_CLEAN, "prices.csv")

    if not os.path.exists(prices_path):
        print("Run 1_data_pipeline.py first to generate clean prices.")
    else:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        result = run_strategy(prices)

        plot_signals(result["signals_shifted"], result["weights"])

        print("\n── Momentum scores (last 5 rows) ──")
        print(result["momentum"].tail())
        print("\n── Signals (last 5 rows) ──")
        print(result["signals_shifted"].tail())
        print("\n── Weights (last 5 rows) ──")
        print(result["weights"].tail())
        print("\n── Daily costs (last 5 rows) ──")
        print(result["costs"].tail())
        print(f"\nAvg annual cost estimate: {result['costs'].mean() * 252 * 100:.2f}%")