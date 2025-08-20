"""
Modeling package for predictive trading signals.

This package handles machine learning model training, evaluation, and prediction
with time-series cross-validation to prevent lookahead bias.

Key features:
- Time-series aware cross-validation
- Multiple model architectures (Logistic Regression, Random Forest)
- Hyperparameter optimization
- Comprehensive evaluation metrics
- Model persistence and versioning

Main components:
- ModelTrainer: Handles model training and evaluation
- TimeSeriesSplitter: Provides time-series cross-validation
- ModelMetrics: Comprehensive evaluation metrics
"""

from src.modeling.models import ModelTrainer
from src.modeling.splits import TimeSeriesSplitter
from src.modeling.metrics import ModelMetrics

__all__ = [
    "ModelTrainer",
    "TimeSeriesSplitter",
    "ModelMetrics",
]
