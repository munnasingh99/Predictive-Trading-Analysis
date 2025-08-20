"""
Predictive Trading Signals

A production-quality machine learning system for generating algorithmic trading signals
using daily equity price data.

This package provides:
- Data ingestion from yfinance and CSV sources
- Technical indicator feature engineering
- ML model training with time-series cross-validation
- Strategy backtesting with realistic transaction costs
- Performance reporting and visualization

Example:
    >>> from src.cli import main
    >>> # Run full pipeline via CLI

    >>> from src.data.ingest import DataIngestor
    >>> from src.features.engineer import FeatureEngineer
    >>> # Use components directly
"""

__version__ = "1.0.0"
__author__ = "Predictive Trading Signals Team"

# Core components
from src.data.db import DatabaseManager
from src.data.ingest import DataIngestor
from src.features.engineer import FeatureEngineer
from src.labeling.labels import LabelGenerator
from src.modeling.models import ModelTrainer
from src.backtest.engine import BacktestEngine
from src.reporting.report import ReportGenerator
from src.utils.config import ConfigManager

__all__ = [
    "DatabaseManager",
    "DataIngestor",
    "FeatureEngineer",
    "LabelGenerator",
    "ModelTrainer",
    "BacktestEngine",
    "ReportGenerator",
    "ConfigManager",
]
