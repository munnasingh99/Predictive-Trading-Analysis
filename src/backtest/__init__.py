"""
Backtest package for predictive trading signals.

This package handles strategy backtesting and performance analysis including:
- Realistic trade execution simulation
- Transaction cost and slippage modeling
- Position sizing and risk management
- Comprehensive performance metrics calculation
- Benchmark comparison analysis
- Trade-level analysis and reporting

Key features:
- Signal generation from ML predictions
- Long-only and long/short strategies
- Risk-adjusted performance metrics
- Drawdown analysis and risk assessment
- Rolling performance analysis

Main components:
- BacktestEngine: Comprehensive backtesting framework
- PerformanceAnalyzer: Performance analysis and metrics calculation
"""

from src.backtest.engine import BacktestEngine
from src.backtest.perf import PerformanceAnalyzer

__all__ = [
    "BacktestEngine",
    "PerformanceAnalyzer",
]
