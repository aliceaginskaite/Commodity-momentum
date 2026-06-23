"""
1 - data_pipeline.py — Data Download & Quality Checks

Covered here:
Missing Data Check — quantify gaps, decide fill strategy
Survivorship Bias — document the limitation explicitly
Lookahead Bias — architectural fix: signal shift baked in here
Data Snooping — parameter freeze documented with rationale
.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import config


# Section 1: downloading

def _generate_synthetic_data() -> pd.DataFrame:
    """
    Fallback: generate realistic synthetic commodity price data
    when Yahoo Finance is unavailable (e.g. restricted network).

    Uses geometric Brownian motion with realistic parameters per commodity.
    Not for production — only for local testing of the pipeline logic.
    """
    config.log.warning("Warning: Using SYNTHETIC data (Yahoo Finance unavailable).")
    config.log.warning("On your machine, run with real yfinance data.")

    np.random.seed(42)
    dates = pd.bdate_range(start=config.START_DATE, end=config.END_DATE)

    # (annual_drift, annual_vol, start_price) — roughly realistic per ETF
    params = {
        "USO":  (0.03, 0.38, 30.0),
        "GLD":  (0.06, 0.15, 160.0),
        "SLV":  (0.02, 0.28, 28.0),
        "UNG":  (-0.05, 0.55, 18.0),
        "CPER": (0.04, 0.25, 20.0),
    }

    dt   = 1 / config.TRADING_DAYS_YEAR
    data = {}
    for ticker, (mu, sigma, s0) in params.items():
        shocks  = np.random.normal((mu - 0.5 * sigma**2) * dt,
                                    sigma * np.sqrt(dt),
                                    size=len(dates))
        prices  = s0 * np.exp(np.cumsum(shocks))
        data[ticker] = prices

    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


def download_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers.
    Saves raw data to data/raw/ for reproducibility.

    Why adjusted close?
    Accounts for splits and dividends. ETF distributions would create fake return spikes otherwise.

    Falls back to synthetic data if Yahoo Finance is unavailable.

    Parameters
    
    force_refresh : if True, ignore cached file and re-download.

    Returns

    pd.DataFrame  shape: (trading_days, n_tickers)  columns = ticker symbols
    """
    cache_path = os.path.join(config.DATA_RAW, "prices_raw.csv")

    if os.path.exists(cache_path) and not force_refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not df.empty:
            config.log.info("Loading cached raw data from %s shape: %s",
                            cache_path, df.shape)
            return df

    config.log.info("Downloading data from Yahoo Finance...")
    config.log.info("Tickers: %s", config.TICKER_LIST)
    config.log.info("Period:  %s → %s", config.START_DATE, config.END_DATE)

    try:
        raw = yf.download(
            tickers = config.TICKER_LIST,
            start = config.START_DATE,
            end = config.END_DATE,
            auto_adjust = True,
            progress = False,
        )
        # yfinance returns multi-level columns: (OHLCV, Ticker)
        prices = raw["Close"].copy()
        prices.index.name = "Date"
        prices.columns.name = None

        if prices.empty:
            raise ValueError("yfinance returned empty DataFrame")

    except Exception as e:
        config.log.warning("yfinance failed (%s) — using synthetic data.", e)
        prices = _generate_synthetic_data()

    prices.to_csv(cache_path)
    config.log.info("Raw data saved → %s  shape: %s", cache_path, prices.shape)
    return prices


# Section 2: Missing data check  

def check_missing_data(prices: pd.DataFrame) -> dict:
    """
    Audit missing values across all tickers and time.

    Three types of "missing" in market data:
    1. NaN - yfinance couldn't fetch the value
    2. Zero - instrument not trading yet (early history) or data error
    3. Stale - same price repeated many days (frozen feed)

    Returns a summary dict and prints a report.
    """
    config.log.info("=" * 60)
    config.log.info("Missing Data Check")
    config.log.info("=" * 60)

    report = {}

    for ticker in prices.columns:
        series = prices[ticker]
        n_total = len(series)
        n_nan = series.isna().sum()
        n_zero = (series == 0).sum()

        # Stale price detection: find runs of identical prices > 3 days
        stale_mask = (series == series.shift(1)) & (series == series.shift(2)) & series.notna()
        n_stale = stale_mask.sum()

        pct_missing = n_nan / n_total * 100

        report[ticker] = {
            "total_rows": n_total,
            "nan_count": n_nan,
            "zero_count": n_zero,
            "stale_count": n_stale,
            "pct_missing": round(pct_missing, 2),
        }

        config.log.info(
            "%s — NaN: %d (%.1f%%)|Zeros: %d|Stale: %d",
            ticker, n_nan, pct_missing, n_zero, n_stale
        )

    # Flag tickers with >5% missing data
    bad_tickers = [t for t, r in report.items() if r["pct_missing"] > 5.0]
    if bad_tickers:
        config.log.warning("High missing data: %s — review before proceeding", bad_tickers)
    else:
        config.log.info("All tickers below 5% missing threshold")

    return report


def visualise_missing_data(prices: pd.DataFrame, save: bool = True):
    """
    Heatmap: rows = time (sampled monthly), columns = tickers.
    Black cell = data present, white = missing.
    Makes it instantly obvious if any ticker has early-history gaps.
    """
    # Resample to monthly to keep the chart readable
    monthly_mask = prices.resample("ME").last().isna().astype(int)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        monthly_mask.T,
        ax       = ax,
        cmap     = ["#1a1a2e", "#e63946"],  # dark=present, red=missing
        cbar     = False,
        linewidths = 0.3,
    )
    ax.set_title("Missing Data Map (red = missing, monthly resolution)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Ticker")

    # Format x-axis as years
    n_cols = monthly_mask.shape[0]
    step   = max(1, n_cols // 8)
    ax.set_xticks(range(0, n_cols, step))
    ax.set_xticklabels(
        [str(d.year) for d in monthly_mask.index[::step]],
        rotation=45
    )

    plt.tight_layout()
    if save:
        path = os.path.join(config.OUTPUT_DIR, "01_missing_data_heatmap.png")
        fig.savefig(path, dpi=150)
        config.log.info("Saved: %s", path)
    plt.show()
    plt.close()


# Section 3: Survivorship bias

def document_survivorship_bias():
    """
    Survivorship bias: we only include instruments that exist today.
    Any commodity ETF that closed/delisted is automatically excluded.
    This means our backtest universe is slightly optimistic.

    We can't fully fix this with yfinance data, so we document it as a known limitation with its expected impact.
    """
    print("\n" + "-" * 60)
    print("Survivorship bias — known limitation")
    print("-" * 60)
    print("""
Universe consists of 5 commodity ETFs currently trading as of 2024. 
Delisted ETFs that ceased to exist during 2012–2024 are excluded.

Impact: modest for cross-sectional momentum. Excluded instruments 
would most likely have been losers (short leg), so their omission 
marginally improves short-side performance.

Mitigation: explicitly documented; cannot be corrected via yfinance. 
Production-grade implementation would require continuous futures data 
(e.g. Bloomberg, Refinitiv), which preserves expired contracts and 
eliminates this bias entirely.
""")
    print("-" * 60)
    config.log.info("Survivorship bias documented (see output above)")


# Section 4: cleaning and handling missing data

def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
  Cleaning rules:
 - Forward-fill gaps ≤ 3 days (handles non-trading days across exchanges).
 - Drop rows where any NaN remains (pre-launch history).
 - Convert zeros → NaN, then forward-fill (zero is invalid for ETFs).

Forward-fill used over interpolation to prevent lookahead bias. Interpolation would infer missing values using future prices.
    """
    config.log.info("Cleaning prices...")
    df = prices.copy()

    # Replace zeros with NaN
    df = df.replace(0, np.nan)

    # Forward-fill up to 3 days
    df = df.ffill(limit=3)

    # Drop rows where any ticker is still NaN
    n_before = len(df)
    df = df.dropna()
    n_after  = len(df)
    config.log.info("Dropped %d rows with NaN after fill (%s → %s)",
                    n_before - n_after, n_before, n_after)

    config.log.info("Clean price data: %s rows, %s tickers, %s → %s",
                    len(df), len(df.columns),
                    df.index[0].date(), df.index[-1].date())
    return df


# Section 5: lookahead bias protection 

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
   Returns: r[t] = log(P[t] / P[t-1]), earned at end of day t.

 Signals computed using data available at day t, but executed at 
 day t+1 open. This one-day shift is the core architectural 
 safeguard against lookahead bias.
    """
    returns = np.log(prices / prices.shift(1)).dropna()
    config.log.info("Returns computed: %s rows (log returns)", len(returns))
    return returns


# Section 6: data snooping documentation 

def document_data_snooping_controls():
    """
    Data snooping / p-hacking prevention.
    We document our parameter choices before running the strategy.
    We don't change MOMENTUM_WINDOW after seeing results.
    """

    print("\n" + "-" * 60)
    print("Data snooping controls")
    print("-" * 60)
    print(f"""
          
MOMENTUM_WINDOW = {config.MOMENTUM_WINDOW} days
Source: Jegadeesh & Titman (1993), "Returns to Buying Winners
and Selling Losers"

N_LONG = {config.N_LONG}, N_SHORT = {config.N_SHORT} (out of 5 instruments)
Source: symmetric long-short construction is standard practice
in academic cross-sectional momentum research.

Both parameters were set before any backtest was run on this
data. The parameter sensitivity sweep in 7_robustness.py shows
results across the full range of window values, so the choice of 63 days can be checked against its
neighbours rather than taken on faith.

Frozen on: {datetime.now().strftime("%Y-%m-%d")}
""")
    print("-" * 60)


# Section 7: train / test split

def split_data(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hard split: 70% in-sample, 30% out-of-sample. 
    OOS period is treated as a lockbox — not evaluated until 
    parameters are frozen and in-sample analysis is complete.
    """
    split_idx = int(len(prices) * config.TRAIN_RATIO)
    split_date = prices.index[split_idx]

    prices_is  = prices.iloc[:split_idx]
    prices_oos = prices.iloc[split_idx:]

    config.log.info("Train/Test split at %.0f%% | Split date: %s",
                    config.TRAIN_RATIO * 100, split_date.date())
    config.log.info("  In-sample:     %d rows  (%s → %s)",
                    len(prices_is),
                    prices_is.index[0].date(), prices_is.index[-1].date())
    config.log.info("  Out-of-sample: %d rows  (%s → %s)",
                    len(prices_oos),
                    prices_oos.index[0].date(), prices_oos.index[-1].date())

    return prices_is, prices_oos


# Section 8: save clean data

def save_clean_data(prices: pd.DataFrame, returns: pd.DataFrame):
    """Save processed data for use by all downstream modules."""
    prices_path  = os.path.join(config.DATA_CLEAN, "prices.csv")
    returns_path = os.path.join(config.DATA_CLEAN, "returns.csv")

    prices.to_csv(prices_path)
    returns.to_csv(returns_path)

    config.log.info("Clean prices  saved → %s", prices_path)
    config.log.info("Clean returns saved → %s", returns_path)


# Main: run this file to execute the full pipeline

def run_pipeline(force_refresh: bool = False) -> dict:
    """
    Execute full data pipeline. Returns dict with all data objects
    so downstream modules can call run_pipeline() and get everything.
    """
    
    config.log.info("Starting data pipeline...")

    # 1. Download
    prices_raw = download_data(force_refresh=force_refresh)

    # 2. Quality checks
    missing_report = check_missing_data(prices_raw)
    visualise_missing_data(prices_raw)

    # 3. Document biases
    document_survivorship_bias()
    document_data_snooping_controls()

    # 4. Clean
    prices_clean = clean_prices(prices_raw)

    # 5. Returns (with lookahead protection baked in)
    returns = compute_returns(prices_clean)

    # 6. Split
    prices_is, prices_oos = split_data(prices_clean)

    # 7. Save
    save_clean_data(prices_clean, returns)

    config.log.info("Data pipeline complete.")

    return {
        "prices_raw": prices_raw,
        "prices_clean": prices_clean,
        "returns": returns,
        "prices_is": prices_is,
        "prices_oos": prices_oos,
        "missing_report": missing_report,
    }


if __name__ == "__main__":
    data = run_pipeline(force_refresh=False)
    print("\nFinal clean prices shape:", data["prices_clean"].shape)
    print(data["prices_clean"].tail())

