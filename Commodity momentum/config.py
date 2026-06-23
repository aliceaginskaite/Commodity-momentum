
# config.py — Central Configuration
# All parameters live here


# Instruments
# Commodity futures ETFs/proxies via yfinance

TICKERS = {
    "WTI_OIL": "USO",   
    "GOLD": "GLD", 
    "SILVER": "SLV", 
    "NAT_GAS": "UNG", 
    "COPPER": "CPER", 
}


TICKER_LIST = list(TICKERS.values()) 

# Date range

START_DATE = "2012-01-01"   # CPER launched 2011, 2012 gives us clean data
END_DATE = "2024-12-31"

# Data split (70/30)

TRAIN_RATIO = 0.70

# Parameters

MOMENTUM_WINDOW = 63 
N_LONG = 2 
N_SHORT = 2 
REBALANCE_FREQ = "W" 

# Transaction costs (1 bp = 0.01%)

SPREAD_BPS = 5 
COMMISSION_BPS = 2 
SLIPPAGE_BPS = 3 
TOTAL_COST_BPS = SPREAD_BPS + COMMISSION_BPS + SLIPPAGE_BPS 

# Market impact model: cost grows with trade size relative to ADV

MARKET_IMPACT_COEF = 0.1 
AVG_DAILY_VOLUME_USD = {  
"USO": 150,
    "GLD": 1200,
    "SLV": 300,
    "UNG": 60,
    "CPER": 5,
}
  
# Risk/perfomance

RISK_FREE_RATE = 0.04 
TRADING_DAYS_YEAR = 252
TAIL_RISK_ALPHA = 0.05

# Validation

WFO_N_SPLITS = 8 
WFO_TRAIN_MONTHS = 24 
WFO_TEST_MONTHS = 6 
CV_N_SPLITS = 5 

# Robustness

MONTE_CARLO_SIMS = 100000 
MC_RANDOM_SEED = 42

# Parameter stability grid

PARAM_WINDOW_MIN = 20
PARAM_WINDOW_MAX = 120
PARAM_WINDOW_STEP = 5

# Stress test periods

STRESS_PERIODS = {
    "COVID Crash":("2020-02-01", "2020-04-30"),
    "Oil Crash 2020":("2020-03-01", "2020-06-30"),
    "Inflation Shock 2022":("2022-01-01", "2022-12-31"),
    "China Slowdown 2015":("2015-06-01", "2015-12-31"),
}

# Paths

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_CLEAN = os.path.join(BASE_DIR, "data", "clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for _dir in [DATA_RAW, DATA_CLEAN, OUTPUT_DIR]:
    os.makedirs(_dir, exist_ok=True)


# Logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)