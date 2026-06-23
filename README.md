Commodity Cross-Sectional Momentum — Full Validation Backtest

The objective of this project is not to discover a profitable commodity momentum strategy. The objective is to demonstrate a complete quantitative research workflow: data quality controls, bias detection, realistic execution modelling, robustness testing, risk analysis, and strategy validation. The commodity momentum strategy serves only as a case study for that process.

📌 Description

A complete backtest of a cross-sectional momentum strategy on a basket of 5 commodity ETFs (oil, gold, silver, natural gas, copper). The strategy buys the top-2 momentum leaders and shorts the bottom-2 losers over a 3-month lookback, remaining market-neutral.
The main goal of this project is to demonstrate a full, honest validation process following a 25-point checklist: data bias checks, realistic transaction costs, three validation methods, a full suite of risk metrics, capacity analysis, and stress tests for robustnessБ applied to a strategy that ultimately does not survive that process unscathed, which is itself the point.
<img width="1614" height="751" alt="image" src="https://github.com/user-attachments/assets/84728d6c-89ab-4c12-8812-56acc13a1ec4" />



🛠️ Technologies Used

Python 3 — core language
Pandas, NumPy — time series processing, vectorized computations
yfinance — historical price data
scikit-learn (TimeSeriesSplit) — time-series cross-validation without lookahead leakage
Matplotlib, Seaborn — visualizations (equity curves, heatmaps, fan charts, drawdown charts)
scipy — statistical calculations for risk metrics


📊 Key Results

Test Period: 13 years (2012–2024), 3,392 trading days
Net Total Return: 266% (Gross: 466%, difference reflects realistic costs)
Sharpe Ratio: 0.18 | Sortino Ratio: 0.28 | Calmar Ratio: 0.17
Max Drawdown: −59.6%
Out-of-Sample Sharpe is nearly identical to In-Sample (0.12 vs 0.14) — the strategy is not overfitted
Correlation to market benchmark: −0.11 — confirms the market-neutral nature of the strategy
Capacity: limited to ~$250,000 AUM due to liquidity constraints (copper/CPER)

<img width="1616" height="501" alt="image" src="https://github.com/user-attachments/assets/3aa91062-9275-4cca-8c6d-d875e6d2293a" />


📁 Project Structure

commodity_momentum/
├── config.py                    - all strategy parameters in one place
├── 01_data_pipeline.py          - data loading + quality checks
├── 02_strategy_engine.py        - momentum signal logic
├── 03_backtest_core.py          - backtest engine with transaction costs
├── 04_validation.py             - OOS, Walk-Forward, Time-Series CV
├── 05_risk_metrics.py           - Sharpe, Sortino, Calmar, drawdowns, tail risk
├── 06_execution_analysis.py     - turnover, exposure, capacity
├── 07_robustness.py             - Monte Carlo, stress tests, regime analysis
└── 08_report.py                 - correlation, edge decomposition, final conclusions

✅ Validation Checklist (25/25 items)
- Data: Survivorship Bias · Lookahead Bias · Data Snooping · Missing Data Check
- Realism: Spread · Commission · Slippage · Market Impact
- Validation: Out-of-Sample · Walk-Forward · Time-Series Cross-Validation
- Robustness: Monte Carlo · Parameter Stability · Stress Tests · Regime Analysis
- Risk: Sharpe · Sortino · Calmar · Max Drawdown · Recovery Time · Tail Risk (CVaR)
- Execution: Turnover · Exposure · Capacity
- Research: Correlation to Market · Correlation to Existing Strategies · Edge Decomposition
- Every item is implemented as a separate function with explicit comments explaining the logic.

🚀 How to Run

bashgit clone https://github.com/aliceaginskaite/Commodity-momentum.git

cd commodity_momentum

pip install -r requirements.txt

python 01_data_pipeline.py    - downloads data, builds missing-data heatmap

python 02_strategy_engine.py  - generates signals, visualizes positions

python 03_backtest_core.py    - builds equity curve

python 04_validation.py       - OOS / Walk-Forward / CV

python 05_risk_metrics.py     - full risk table

python 06_execution_analysis.py - turnover, exposure, capacity

python 07_robustness.py       - Monte Carlo fan chart, stress tests

python 08_report.py           - final report + "what kills the strategy"


🔍 What Works Well

Edge is not fabricated. OOS Sharpe (0.12) closely matches IS Sharpe (0.14), so the strategy is not curve-fitted; the momentum effect generalizes to unseen data.
Market-neutral design works as intended. Net exposure stays near 0.0000, correlation to the buy-and-hold commodity basket is only −0.11. The strategy does not take directional bets on commodities.
5 out of 8 walk-forward windows are positive, showing the effect persists across different macro regimes, not just one lucky period.

<img width="1594" height="852" alt="image" src="https://github.com/user-attachments/assets/ae91d4e4-135c-422d-8054-fc099f932152" />

⚠️ What Kills This Strategy

- Weak risk-adjusted returns. Sharpe 0.18 and Sortino 0.28 are well below institutional acceptability thresholds (0.5–1.0).
- Drawdown is severe and recovery is slow. −59.6% max drawdown, and the strategy remains in an unrecovered drawdown for over a year at the end of the test period.
- Volatility is structurally mismatched across instruments, and equal weighting doesn't correct for it. Annualised volatility ranges from 15% (gold) to 54% (natural gas) across the basket - a roughly 3.5x spread. Because the strategy assigns the same ±0.5 weight to any instrument regardless of which leg it lands in, a natural gas position contributes disproportionately more risk than a gold position in the exact same slot. This is a direct, quantifiable driver of the weak Sharpe above, not a vague volatility comment, but a specific imbalance visible in the per-instrument numbers.
- Capacity is critically low. The least liquid instrument (copper/CPER) limits the strategy to ~$250k AUM - retail scale, not institutional.
- Concentration risk. A significant portion of P&L comes from one specific instrument, not evenly distributed across all 5 commodities. This is more of a single-asset effect disguised as a portfolio strategy.
- High correlation with trend-following (0.62). If an investor already holds trend-following strategies, this momentum strategy adds limited diversification.
- Survivorship bias - only currently traded ETFs are used, which slightly inflates results on the short side structurally.

<img width="1612" height="619" alt="image" src="https://github.com/user-attachments/assets/f90a0290-861a-4711-8b70-9303b55968cc" />


🔧 What I Plan to Improve Next

- Volatility targeting - weight positions inversely to recent instrument volatility instead of equal weights. Given the 15%–54% volatility spread documented above, this single change should meaningfully reduce the impact of "explosive" assets (natural gas, copper) on overall portfolio risk and could lift Sharpe without touching the underlying momentum signal at all.
- Expand instrument universe - move from 5 ETFs to a broader basket of actual futures (Brent, Heating Oil, Platinum, etc.), simultaneously solving both capacity and concentration risk.
- Risk overlay - a simple circuit breaker that halves gross exposure after exceeding a drawdown threshold, directly addressing the slow recovery issue.
- Signal smoothing - test monthly rebalancing or EMA-smoothed momentum scores to check if the high turnover (~32x annually) is masking a better underlying Sharpe.

📝 Methodological Notes

Strategy parameters (momentum window = 63 days, top-2/bottom-2 long-short) were fixed before any backtest was run, based on academic literature (Jegadeesh & Titman, 1993), not optimized to fit results. This is explicitly documented in the code as protection against data snooping. The walk-forward analysis uses fixed parameters specifically to test temporal stability, not re-optimization at each window, re-optimizing parameters on rolling windows would reintroduce the very data snooping risk we aim to avoid.
