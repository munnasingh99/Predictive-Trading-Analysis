"""
Features package for predictive trading signals.

This package handles feature engineering operations including:
- Technical indicator computation (SMA, EMA, RSI, MACD, etc.)
- Statistical feature generation
- No-lookahead validation
- Feature quality assessment

Main components:
- FeatureEngineer: Computes technical indicators with no lookahead bias
"""

from src.features.engineer import FeatureEngineer

__all__ = [
    "FeatureEngineer",
]
