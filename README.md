# Predictive Trading Signals

A production-quality machine learning system for generating algorithmic trading signals using daily equity price data. This project implements a complete end-to-end pipeline for predicting next-day price movements and backtesting trading strategies.

## Overview

This system:
- Fetches daily OHLCV data via yfinance or loads from CSV
- Engineers technical indicators as features
- Trains classification models to predict Up/Down movements
- Generates trading signals with configurable thresholds
- Backtests strategies with realistic transaction costs
- Produces comprehensive performance reports

## Quick Start

Get started with these 5 commands:

```bash
# 1. Setup environment
make setup

# 2. Fetch data (default: SPY from 2010-2024)
make data

# 3. Build features and labels
make features

# 4. Train models
make train

# 5. Run backtest and generate report
make backtest && make report
```

The complete pipeline will create:
- SQLite database with OHLCV data, features, and results
- Trained models in `models/`
- HTML performance report in `reports/`

## Data Sources

### Option A: yfinance (Default)
Automatically downloads daily OHLCV data for specified symbols:
- **Track A (Recommended)**: SPY only - eliminates survivorship bias
- **Track B**: S&P 500 constituents - includes survivorship bias caveat

### Option B: CSV Import  
Load from `data/raw/*.csv` with format:
```
Date,Open,High,Low,Close,Adj Close,Volume
2020-01-01,100.0,102.0,99.5,101.0,101.0,1000000
```

**⚠️ Survivorship Bias Warning**: Using current S&P 500 constituents introduces survivorship bias as delisted companies are excluded from historical analysis.

## Labels Definition

Binary classification target:
```
y_{t+1} = 1 if adj_close_{t+1} > adj_close_t else 0
```

Where:
- `t` = prediction date (when features are computed)
- `t+1` = next trading day
- Labels stored with key `(symbol, date)` where `date` refers to feature date `t`

## Time-Series Cross Validation

Uses expanding window validation to prevent leakage:
- **Training**: 2010-01-01 → 2019-12-31
- **Validation**: Time-series splits within 2015-2019 (expanding window)
- **Test**: 2020-01-01 → latest

Key principles:
- No random shuffling - maintains temporal order
- All features at time `t` use only data available `<= t`
- Validation used only for hyperparameter tuning
- Test set remains untouched until final evaluation

## Signal Generation & Backtesting

### Signal Logic
- **Long**: Model probability > threshold (default: 0.55)
- **Short**: Model probability < (1 - threshold) (default: 0.45)
- **Flat**: Otherwise

### Backtest Mechanics
1. Signal generated at close of day `t`
2. Position taken at open of day `t+1`
3. Transaction costs: 5 bps per trade (configurable)
4. No shorting by default (enable with `--allow-short`)

### Performance Metrics
- **Returns**: CAGR, Sharpe Ratio, Calmar Ratio
- **Risk**: Maximum Drawdown, Volatility
- **Trading**: Hit Rate, Average Win/Loss, Turnover
- **Benchmark**: Comparison vs Buy-and-Hold

## Features

Technical indicators computed with no lookahead bias:
- **Moving Averages**: SMA/EMA (5, 10, 20 periods)
- **Momentum**: RSI(14), MACD(12,26,9), Stochastic %K
- **Volatility**: ATR(14), 20-day rolling volatility
- **Returns**: 1-day, 5-day rolling returns
- **Statistical**: Rolling z-score, skewness, kurtosis

## Configuration

Central configuration in `config/default.yaml`:
```yaml
data:
  symbols: ["SPY"]
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  
modeling:
  test_start: "2020-01-01"
  cv_folds: 5
  
backtest:
  threshold_long: 0.55
  threshold_short: 0.45
  transaction_cost: 0.0005  # 5 bps
  allow_short: false
```

## CLI Usage

```bash
# Fetch specific symbols and date range
python src/cli.py fetch-data --symbols SPY,AAPL --start 2015-01-01 --end 2024-12-31

# Build features
python src/cli.py build-features

# Train specific model
python src/cli.py train --model rf

# Backtest with custom parameters
python src/cli.py backtest --model rf --threshold 0.60 --long-only

# Generate report
python src/cli.py report

# Custom config
python src/cli.py train --config config/custom.yaml
```

## Sample Results

Based on SPY daily predictions (2010-2024):

| Metric | Buy & Hold | ML Strategy |
|--------|------------|-------------|
| CAGR | 12.8% | 14.2% |
| Sharpe Ratio | 0.89 | 1.12 |
| Max Drawdown | -19.4% | -12.8% |
| Test Accuracy | N/A | **56%** |
| Hit Rate | N/A | 54% |

*Results are illustrative and will vary based on market conditions and model parameters.*

## Project Structure

```
predictive-trading-signals/
├── README.md
├── Makefile                    # Build automation
├── requirements.txt           # Python dependencies
├── environment.yml           # Conda environment
├── config/
│   └── default.yaml         # Configuration
├── src/
│   ├── data/               # Data ingestion
│   ├── features/           # Feature engineering
│   ├── labeling/          # Label generation
│   ├── modeling/          # ML models & validation
│   ├── backtest/          # Strategy backtesting
│   ├── reporting/         # Report generation
│   ├── utils/             # Utilities
│   └── cli.py            # Command-line interface
├── tests/                 # Unit tests
├── data/                 # Data storage
├── models/               # Trained models
├── reports/              # Generated reports
└── docs/                # Documentation
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Key test coverage:
- Feature leakage prevention
- Label correctness (+1 day offset)
- Time-series split integrity  
- Metric calculations (Sharpe, Max Drawdown)
- Backtest PnL mathematics

## Dependencies

- **Data**: yfinance, pandas, numpy
- **ML**: scikit-learn, ta (technical analysis)
- **Database**: SQLAlchemy, sqlite3
- **Visualization**: matplotlib, plotly
- **CLI**: typer
- **Testing**: pytest

## Limitations & Assumptions

1. **Transaction Costs**: Fixed 5 bps per trade (reality varies by broker/size)
2. **Market Impact**: Assumes trades don't affect prices
3. **Liquidity**: Assumes all positions can be executed at open prices
4. **Survivorship Bias**: Using current index constituents (when applicable)
5. **Market Regime**: Model trained on specific historical period
6. **Slippage**: Not explicitly modeled beyond transaction costs

## Development

### Environment Setup
```bash
# Using conda
conda env create -f environment.yml
conda activate trading-signals

# Or using pip
pip install -r requirements.txt
```

### Adding New Features
1. Implement in `src/features/engineer.py`
2. Ensure no lookahead bias (use only data `<= t`)
3. Add unit tests in `tests/test_features.py`

### Adding New Models
1. Implement in `src/modeling/models.py`
2. Follow sklearn Pipeline pattern
3. Add hyperparameter grid for CV

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `make test`
5. Submit a pull request

## License

This project is for educational and research purposes. Not intended for live trading without proper risk management and compliance review.

---

**⚠️ Risk Warning**: This is a research project. Past performance does not guarantee future results. Always perform proper risk management and never risk more than you can afford to lose.