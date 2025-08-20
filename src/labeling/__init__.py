"""
Labeling package for predictive trading signals.

This package handles target variable generation including:
- Binary classification labels (Up/Down price movements)
- Multiclass labels (Down/Flat/Up)
- Regression labels (continuous returns)
- Temporal alignment validation
- Label quality assessment

Main components:
- LabelGenerator: Creates target variables with proper temporal alignment
"""

from src.labeling.labels import LabelGenerator

__all__ = [
    "LabelGenerator",
]
