"""
Time series cross-validation module for predictive trading signals.

This module provides time-series aware data splitting functionality that prevents
lookahead bias in model training and evaluation. It implements expanding window
and sliding window cross-validation strategies appropriate for financial time series.

Key features:
- Expanding window cross-validation
- Configurable train/validation/test splits
- Gap handling to prevent leakage
- Symbol-aware splitting for multi-asset portfolios
"""

import logging
from datetime import datetime, timedelta
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


class TimeSeriesSplitter:
    """Time series data splitter with no lookahead bias."""

    def __init__(self,
                 train_end: str = "2019-12-31",
                 validation_start: str = "2015-01-01",
                 test_start: str = "2020-01-01",
                 cv_folds: int = 5,
                 gap_days: int = 1):
        """Initialize time series splitter.

        Args:
            train_end: End date for training data (YYYY-MM-DD)
            validation_start: Start date for validation period (YYYY-MM-DD)
            test_start: Start date for test data (YYYY-MM-DD)
            cv_folds: Number of cross-validation folds
            gap_days: Gap between train and validation to prevent leakage
        """
        self.train_end = pd.to_datetime(train_end)
        self.validation_start = pd.to_datetime(validation_start)
        self.test_start = pd.to_datetime(test_start)
        self.cv_folds = cv_folds
        self.gap_days = gap_days

        # Validate date logic
        if self.validation_start >= self.train_end:
            raise ValueError("validation_start must be before train_end")
        if self.test_start <= self.train_end:
            raise ValueError("test_start must be after train_end")

        logger.info(f"TimeSeriesSplitter initialized: train_end={train_end}, "
                   f"validation_start={validation_start}, test_start={test_start}, "
                   f"cv_folds={cv_folds}, gap_days={gap_days}")

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets.

        Args:
            data: DataFrame with date column and other features

        Returns:
            Tuple of (train_data, validation_data, test_data)
        """
        if 'date' not in data.columns:
            raise ValueError("Data must contain 'date' column")

        # Ensure date is datetime
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])

        # Sort by date
        data = data.sort_values(['symbol', 'date'] if 'symbol' in data.columns else 'date')

        # Split into train/validation/test
        train_data = data[data['date'] <= self.train_end].copy()
        validation_data = data[
            (data['date'] >= self.validation_start) &
            (data['date'] <= self.train_end)
        ].copy()
        test_data = data[data['date'] >= self.test_start].copy()

        logger.info(f"Data split: Train={len(train_data)}, Val={len(validation_data)}, Test={len(test_data)}")

        return train_data, validation_data, test_data

    def get_cv_splits(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate cross-validation splits for time series data.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            List of (train_indices, validation_indices) tuples
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        # Assume X has the same index as the original data
        # and we need to respect temporal ordering
        dates = X.index if isinstance(X.index, pd.DatetimeIndex) else None

        if dates is None:
            # If no datetime index, create sequential splits
            return self._sequential_splits(len(X))
        else:
            return self._date_based_splits(X, dates)

    def _sequential_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate sequential expanding window splits.

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_indices, validation_indices) tuples
        """
        splits = []

        # Calculate validation size for each fold
        val_size = n_samples // (self.cv_folds + 1)

        for fold in range(self.cv_folds):
            # Expanding window: train size grows with each fold
            train_start = 0
            train_end = val_size * (fold + 1)

            # Validation window follows training with gap
            val_start = train_end + self.gap_days
            val_end = val_start + val_size

            # Ensure we don't exceed sample bounds
            if val_end > n_samples:
                break

            train_indices = np.arange(train_start, train_end)
            val_indices = np.arange(val_start, val_end)

            splits.append((train_indices, val_indices))

            logger.debug(f"Fold {fold}: Train[{train_start}:{train_end}], "
                        f"Val[{val_start}:{val_end}]")

        return splits

    def _date_based_splits(self, X: pd.DataFrame, dates: pd.DatetimeIndex) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate date-based expanding window splits.

        Args:
            X: Features DataFrame
            dates: DatetimeIndex

        Returns:
            List of (train_indices, validation_indices) tuples
        """
        splits = []

        # Get validation period
        val_start_date = self.validation_start
        val_end_date = self.train_end

        # Calculate validation period length
        val_period = (val_end_date - val_start_date).days
        fold_days = val_period // self.cv_folds

        for fold in range(self.cv_folds):
            # Current fold validation period
            fold_val_start = val_start_date + timedelta(days=fold * fold_days)
            fold_val_end = fold_val_start + timedelta(days=fold_days)

            # Training period (expanding window)
            train_mask = dates < (fold_val_start - timedelta(days=self.gap_days))
            val_mask = (dates >= fold_val_start) & (dates < fold_val_end)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]

            if len(train_indices) == 0 or len(val_indices) == 0:
                continue

            splits.append((train_indices, val_indices))

            logger.debug(f"Fold {fold}: Train until {fold_val_start - timedelta(days=self.gap_days)}, "
                        f"Val [{fold_val_start} to {fold_val_end}]")

        return splits

    def validate_splits(self, data: pd.DataFrame, splits: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Validate that splits don't contain lookahead bias.

        Args:
            data: Original data
            splits: List of (train_indices, val_indices) tuples

        Returns:
            True if splits are valid

        Raises:
            ValueError: If lookahead bias is detected
        """
        if 'date' not in data.columns:
            logger.warning("Cannot validate splits without date column")
            return True

        data_with_index = data.reset_index(drop=True)
        dates = pd.to_datetime(data_with_index['date'])

        for fold, (train_idx, val_idx) in enumerate(splits):
            train_dates = dates.iloc[train_idx]
            val_dates = dates.iloc[val_idx]

            # Check that all training dates are before validation dates
            max_train_date = train_dates.max()
            min_val_date = val_dates.min()

            if max_train_date >= min_val_date - timedelta(days=self.gap_days):
                error_msg = (f"Lookahead bias detected in fold {fold}: "
                           f"max train date ({max_train_date}) >= "
                           f"min val date - gap ({min_val_date - timedelta(days=self.gap_days)})")
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Check for overlapping periods
            train_date_set = set(train_dates.dt.date)
            val_date_set = set(val_dates.dt.date)
            overlap = train_date_set.intersection(val_date_set)

            if overlap:
                error_msg = f"Date overlap detected in fold {fold}: {overlap}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.info("Split validation passed: No lookahead bias detected")
        return True


class TimeSeriesCV(BaseCrossValidator):
    """Scikit-learn compatible time series cross-validator.

    This class provides a sklearn-compatible interface for time series
    cross-validation that can be used directly with GridSearchCV and
    other sklearn model selection tools.
    """

    def __init__(self, n_splits: int = 5, gap: int = 1, max_train_size: Optional[int] = None):
        """Initialize TimeSeriesCV.

        Args:
            n_splits: Number of splits
            gap: Gap between train and test to prevent leakage
            max_train_size: Maximum size of training set
        """
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Args:
            X: Features
            y: Target (optional)
            groups: Group labels (optional)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = X.shape[0]

        # Calculate test size for each fold
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Test set for this fold
            test_start = (i + 1) * test_size
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            # Training set (expanding window)
            train_start = 0
            train_end = test_start - self.gap

            if train_end <= train_start:
                continue

            # Apply max_train_size if specified
            if self.max_train_size is not None and (train_end - train_start) > self.max_train_size:
                train_start = train_end - self.max_train_size

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits.

        Args:
            X: Features (optional)
            y: Target (optional)
            groups: Groups (optional)

        Returns:
            Number of splits
        """
        return self.n_splits


class WalkForwardAnalysis:
    """Walk-forward analysis for time series models.

    This class implements walk-forward analysis, which is useful for
    evaluating model performance in a more realistic trading simulation.
    """

    def __init__(self, initial_train_size: int, refit_frequency: int = 1):
        """Initialize walk-forward analysis.

        Args:
            initial_train_size: Initial training window size
            refit_frequency: How often to refit the model (1 = every period)
        """
        self.initial_train_size = initial_train_size
        self.refit_frequency = refit_frequency

    def split(self, X: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate walk-forward splits.

        Args:
            X: Features DataFrame

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        if n_samples <= self.initial_train_size:
            raise ValueError("Not enough samples for initial training window")

        for i in range(self.initial_train_size, n_samples):
            # Training window (expanding)
            train_indices = np.arange(0, i)

            # Test is just the next period
            test_indices = np.array([i])

            # Only yield if it's time to refit
            if (i - self.initial_train_size) % self.refit_frequency == 0:
                yield train_indices, test_indices

    def evaluate_walk_forward(self, model, X: pd.DataFrame, y: pd.Series,
                            metric_func) -> List[float]:
        """Evaluate model using walk-forward analysis.

        Args:
            model: Sklearn-compatible model
            X: Features
            y: Target
            metric_func: Function to calculate performance metric

        Returns:
            List of performance scores for each period
        """
        scores = []

        for train_idx, test_idx in self.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metric
            score = metric_func(y_test, y_pred)
            scores.append(score)

        return scores


def validate_time_series_split(data: pd.DataFrame,
                             train_indices: np.ndarray,
                             val_indices: np.ndarray,
                             date_column: str = 'date',
                             gap_days: int = 1) -> bool:
    """Validate a single time series split for lookahead bias.

    Args:
        data: DataFrame with time series data
        train_indices: Training set indices
        val_indices: Validation set indices
        date_column: Name of date column
        gap_days: Required gap between train and validation

    Returns:
        True if split is valid

    Raises:
        ValueError: If lookahead bias is detected
    """
    if date_column not in data.columns:
        raise ValueError(f"Date column '{date_column}' not found in data")

    dates = pd.to_datetime(data[date_column])

    train_dates = dates.iloc[train_indices]
    val_dates = dates.iloc[val_indices]

    max_train_date = train_dates.max()
    min_val_date = val_dates.min()

    # Check gap requirement
    required_gap = timedelta(days=gap_days)
    actual_gap = min_val_date - max_train_date

    if actual_gap < required_gap:
        raise ValueError(f"Insufficient gap between train and validation: "
                        f"required {gap_days} days, got {actual_gap.days} days")

    # Check for date overlap
    train_date_set = set(train_dates.dt.date)
    val_date_set = set(val_dates.dt.date)
    overlap = train_date_set.intersection(val_date_set)

    if overlap:
        raise ValueError(f"Date overlap detected: {overlap}")

    return True


def create_purged_splits(data: pd.DataFrame,
                        test_start_dates: List[str],
                        train_window_days: int = 252,
                        gap_days: int = 1,
                        purge_days: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create purged time series splits.

    Purging removes samples around the test set to prevent leakage
    due to overlapping information or label generation periods.

    Args:
        data: DataFrame with date column
        test_start_dates: List of test period start dates
        train_window_days: Size of training window in days
        gap_days: Gap between train and test
        purge_days: Days to purge around test set

    Returns:
        List of (train_indices, test_indices) tuples
    """
    if 'date' not in data.columns:
        raise ValueError("Data must contain 'date' column")

    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)

    splits = []

    for test_start_str in test_start_dates:
        test_start = pd.to_datetime(test_start_str)

        # Define test period (could be extended for longer test periods)
        test_end = test_start + timedelta(days=30)  # 30-day test window

        # Define purge period
        purge_start = test_start - timedelta(days=purge_days)
        purge_end = test_end + timedelta(days=purge_days)

        # Define training period
        train_end = purge_start - timedelta(days=gap_days)
        train_start = train_end - timedelta(days=train_window_days)

        # Create masks
        train_mask = (data['date'] >= train_start) & (data['date'] <= train_end)
        test_mask = (data['date'] >= test_start) & (data['date'] <= test_end)

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))

            logger.debug(f"Purged split: Train[{train_start} to {train_end}], "
                        f"Test[{test_start} to {test_end}], "
                        f"Purged[{purge_start} to {purge_end}]")

    return splits
