"""
Tests for preventing future information leakage in features and labels.

This module contains tests to ensure that no future information is used
when generating features or labels, which would invalidate backtesting results.
Key tests include:
- Feature computation uses only historical data
- Labels are correctly offset by prediction horizon
- Time series splits maintain temporal order
- No data from future periods leaks into training
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineer import FeatureEngineer
from src.labeling.labels import LabelGenerator
from src.modeling.splits import TimeSeriesSplitter
from src.utils.config import ConfigManager


class TestFeatureLeakage:
    """Test feature engineering for future information leakage."""

    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'features': {
                'sma_periods': [5, 10, 20],
                'ema_periods': [5, 10],
                'rsi_period': 14,
                'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
                'returns_periods': [1, 5],
                'volatility_period': 20,
                'zscore_period': 20
            }
        }

        # Mock database
        self.mock_db = Mock()
        self.feature_engineer = FeatureEngineer(self.config, self.mock_db)

    def create_sample_data(self, n_days=100):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        np.random.seed(42)

        # Generate realistic price series
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [initial_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        closes = np.array(prices)
        opens = np.roll(closes, 1)
        opens[0] = initial_price

        # Generate OHLC with some noise
        highs = closes * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
        volumes = np.random.lognormal(10, 0.5, n_days).astype(int)

        return pd.DataFrame({
            'symbol': ['TEST'] * n_days,
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'adj_close': closes,
            'volume': volumes
        })

    def test_sma_no_lookahead(self):
        """Test that SMA uses only historical data."""
        data = self.create_sample_data(50)
        features_df = self.feature_engineer.compute_moving_averages(data)

        # Check that SMA at position i uses only data up to position i
        for period in [5, 10, 20]:
            sma_col = f'sma_{period}'
            if sma_col in features_df.columns:
                sma_values = features_df[sma_col].values

                # First period-1 values should be NaN
                assert pd.isna(sma_values[0:period-1]).all(), f"SMA_{period} should be NaN for first {period-1} values"

                # Manual calculation for validation
                for i in range(period, min(len(data), period + 10)):
                    expected_sma = data['adj_close'].iloc[i-period:i].mean()
                    actual_sma = sma_values[i]
                    if not pd.isna(actual_sma):
                        assert abs(actual_sma - expected_sma) < 1e-10, f"SMA calculation incorrect at position {i}"

    def test_rsi_no_lookahead(self):
        """Test that RSI calculation doesn't use future data."""
        data = self.create_sample_data(50)
        features_df = self.feature_engineer.compute_momentum_indicators(data)

        if 'rsi_14' in features_df.columns:
            rsi_values = features_df['rsi_14'].values

            # RSI should be NaN for early periods
            assert pd.isna(rsi_values[0:13]).all(), "RSI should be NaN for first 13 values"

            # RSI should be between 0 and 100
            valid_rsi = rsi_values[~pd.isna(rsi_values)]
            assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), "RSI values should be between 0 and 100"

    def test_feature_temporal_order(self):
        """Test that features maintain temporal order."""
        data = self.create_sample_data(100)
        # Shuffle data to test if features handle ordering correctly
        shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Features should handle unordered data by sorting internally
        features_df = self.feature_engineer.compute_all_features(shuffled_data)

        # Verify output is sorted by date
        assert features_df['date'].is_monotonic_increasing, "Features output should be sorted by date"

        # Verify no future leakage by checking a specific feature
        if 'sma_10' in features_df.columns:
            for i in range(10, len(features_df)):
                # SMA at position i should only use data from positions 0 to i
                expected_sma = features_df['adj_close'].iloc[i-10:i].mean()
                actual_sma = features_df['sma_10'].iloc[i]
                if not pd.isna(actual_sma):
                    assert abs(actual_sma - expected_sma) < 1e-10, f"Temporal order violation at position {i}"

    def test_cross_symbol_contamination(self):
        """Test that features for one symbol don't use data from another."""
        # Create data for two symbols
        data1 = self.create_sample_data(50)
        data1['symbol'] = 'SYMBOL1'

        data2 = self.create_sample_data(50)
        data2['symbol'] = 'SYMBOL2'
        data2['adj_close'] = data2['adj_close'] * 2  # Make prices different

        combined_data = pd.concat([data1, data2], ignore_index=True)

        features_df = self.feature_engineer.compute_all_features(combined_data)

        # Extract features for each symbol
        symbol1_features = features_df[features_df['symbol'] == 'SYMBOL1']
        symbol2_features = features_df[features_df['symbol'] == 'SYMBOL2']

        # SMA should be different for different symbols with different prices
        if 'sma_10' in features_df.columns:
            s1_sma = symbol1_features['sma_10'].dropna()
            s2_sma = symbol2_features['sma_10'].dropna()

            if len(s1_sma) > 0 and len(s2_sma) > 0:
                # SMAs should be significantly different due to price differences
                mean_diff = abs(s1_sma.mean() - s2_sma.mean())
                assert mean_diff > 10, "Features appear to be contaminated across symbols"


class TestLabelLeakage:
    """Test label generation for future information leakage."""

    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'labeling': {
                'target_type': 'binary',
                'lookahead_days': 1,
                'min_return_threshold': 0.0
            }
        }

        self.mock_db = Mock()
        self.label_generator = LabelGenerator(self.config, self.mock_db)

    def create_sample_data(self, n_days=50):
        """Create sample price data."""
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        np.random.seed(42)

        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))

        return pd.DataFrame({
            'symbol': ['TEST'] * n_days,
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'adj_close': prices,
            'volume': np.random.randint(1000, 10000, n_days)
        })

    def test_label_offset_correct(self):
        """Test that labels are correctly offset by prediction horizon."""
        data = self.create_sample_data(20)
        labels_df = self.label_generator.generate_binary_labels(data)

        # Should have fewer labels than input data due to lookahead
        assert len(labels_df) == len(data) - self.config['labeling']['lookahead_days']

        # Verify label calculation manually
        for i in range(len(labels_df)):
            current_price = data['adj_close'].iloc[i]
            future_price = data['adj_close'].iloc[i + 1]  # Next day price

            expected_label = 1 if future_price > current_price else 0
            actual_label = labels_df['y_next'].iloc[i]

            assert actual_label == expected_label, f"Label incorrect at position {i}"

    def test_no_future_price_in_features_date(self):
        """Test that labels don't use prices beyond the feature date."""
        data = self.create_sample_data(30)
        labels_df = self.label_generator.generate_binary_labels(data)

        # Each label should correspond to a prediction made on that date
        # using only data available up to that date
        for i, row in labels_df.iterrows():
            feature_date = row['date']

            # Find corresponding row in original data
            data_row_idx = data[data['date'] == feature_date].index[0]

            # Ensure we're not using future data
            assert data_row_idx < len(data) - 1, "Cannot generate label for last day of data"

            # Label should be based on next day's price movement
            current_price = data.loc[data_row_idx, 'adj_close']
            next_price = data.loc[data_row_idx + 1, 'adj_close']

            expected_label = 1 if next_price > current_price else 0
            assert row['y_next'] == expected_label, f"Label leakage detected at {feature_date}"

    def test_multiday_lookahead(self):
        """Test labels with multi-day lookahead."""
        self.config['labeling']['lookahead_days'] = 3

        data = self.create_sample_data(20)
        labels_df = self.label_generator.generate_binary_labels(data)

        # Should have 3 fewer labels due to 3-day lookahead
        assert len(labels_df) == len(data) - 3

        # Verify 3-day lookahead calculation
        for i in range(len(labels_df)):
            current_price = data['adj_close'].iloc[i]
            future_price = data['adj_close'].iloc[i + 3]  # 3 days ahead

            expected_label = 1 if future_price > current_price else 0
            actual_label = labels_df['y_next'].iloc[i]

            assert actual_label == expected_label, f"3-day lookahead label incorrect at position {i}"


class TestTimeSeriesSplitting:
    """Test time series splitting for temporal leakage."""

    def setup_method(self):
        """Setup test environment."""
        self.splitter = TimeSeriesSplitter(
            train_end='2019-12-31',
            validation_start='2015-01-01',
            test_start='2020-01-01',
            cv_folds=3,
            gap_days=1
        )

    def create_sample_timeseries(self, start_date='2010-01-01', end_date='2022-12-31'):
        """Create sample time series data."""
        dates = pd.date_range(start_date, end_date, freq='D')
        n_samples = len(dates)

        return pd.DataFrame({
            'date': dates,
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })

    def test_train_test_split_no_leakage(self):
        """Test that train/test split maintains temporal order."""
        data = self.create_sample_timeseries()

        train_data, val_data, test_data = self.splitter.split_data(data)

        # Check temporal ordering
        if not train_data.empty:
            max_train_date = train_data['date'].max()
            assert max_train_date <= pd.to_datetime('2019-12-31'), "Training data extends beyond train_end"

        if not test_data.empty:
            min_test_date = test_data['date'].min()
            assert min_test_date >= pd.to_datetime('2020-01-01'), "Test data starts before test_start"

        # Ensure no overlap
        if not train_data.empty and not test_data.empty:
            assert max_train_date < min_test_date, "Training and test data overlap"

    def test_cv_splits_temporal_order(self):
        """Test that cross-validation splits maintain temporal order."""
        data = self.create_sample_timeseries()
        X = data[['feature1', 'feature2']]
        y = data['target']

        # Add date index for proper splitting
        X.index = data['date']

        splits = self.splitter.get_cv_splits(X, y)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Get dates for this fold
            train_dates = X.iloc[train_idx].index
            val_dates = X.iloc[val_idx].index

            if len(train_dates) > 0 and len(val_dates) > 0:
                max_train_date = train_dates.max()
                min_val_date = val_dates.min()

                # Validation should come after training
                assert max_train_date < min_val_date, f"Temporal leakage in CV fold {fold_idx}"

                # Check gap requirement
                gap = (min_val_date - max_train_date).days
                assert gap >= self.splitter.gap_days, f"Insufficient gap in CV fold {fold_idx}: {gap} days"

    def test_expanding_window_property(self):
        """Test that CV uses expanding window (training set grows)."""
        data = self.create_sample_timeseries('2015-01-01', '2019-12-31')
        X = data[['feature1', 'feature2']]
        y = data['target']
        X.index = data['date']

        splits = self.splitter.get_cv_splits(X, y)

        prev_train_size = 0
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            current_train_size = len(train_idx)

            if fold_idx > 0:
                # Training set should grow or stay same (expanding window)
                assert current_train_size >= prev_train_size, \
                    f"Training set shrunk in fold {fold_idx}: {current_train_size} vs {prev_train_size}"

            prev_train_size = current_train_size

    def test_no_data_leakage_in_splits(self):
        """Test comprehensive check for data leakage in splits."""
        data = self.create_sample_timeseries()

        # This should pass validation
        train_data, val_data, test_data = self.splitter.split_data(data)

        # Create fake splits with leakage
        leaky_splits = [
            (np.array([10, 11, 12, 15, 16]), np.array([13, 14])),  # Overlap in indices
        ]

        # Validation should catch the leakage
        with pytest.raises(ValueError, match="lookahead bias"):
            self.splitter.validate_splits(data, leaky_splits)


class TestEndToEndLeakage:
    """End-to-end tests for leakage prevention."""

    def setup_method(self):
        """Setup complete test environment."""
        self.config = ConfigManager()

    def test_complete_pipeline_no_leakage(self):
        """Test entire pipeline for any potential leakage."""
        # Create synthetic dataset with known future information
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        n_samples = len(dates)

        # Create price series with specific pattern
        prices = 100 + np.sin(np.arange(n_samples) * 0.1) * 10 + np.random.normal(0, 1, n_samples)

        data = pd.DataFrame({
            'symbol': ['TEST'] * n_samples,
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'adj_close': prices,
            'volume': np.random.randint(1000, 10000, n_samples)
        })

        # Process through pipeline
        mock_db = Mock()

        # Features
        feature_engineer = FeatureEngineer(self.config.config, mock_db)
        features_df = feature_engineer.compute_all_features(data)

        # Labels
        label_generator = LabelGenerator(self.config.config, mock_db)
        labels_df = label_generator.generate_binary_labels(data)

        # Verify no leakage by checking that features on day i
        # don't correlate suspiciously with labels on day i
        merged = features_df.merge(labels_df, on=['symbol', 'date'], how='inner')

        if not merged.empty and 'sma_5' in merged.columns:
            # Features should not perfectly predict labels (that would indicate leakage)
            feature_cols = [col for col in merged.columns
                          if col not in ['symbol', 'date', 'y_next']]

            for col in feature_cols[:5]:  # Test first 5 features
                if merged[col].notna().sum() > 10:  # Enough data points
                    correlation = merged[col].corr(merged['y_next'])

                    # Correlation should not be suspiciously high
                    assert abs(correlation) < 0.9, \
                        f"Suspiciously high correlation between {col} and labels: {correlation}"

    def test_feature_label_alignment(self):
        """Test that features and labels are properly aligned temporally."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        n_samples = len(dates)

        # Create data with clear trend
        trend = np.arange(n_samples) * 0.5
        noise = np.random.normal(0, 0.1, n_samples)
        prices = 100 + trend + noise

        data = pd.DataFrame({
            'symbol': ['TEST'] * n_samples,
            'date': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'adj_close': prices,
            'volume': np.random.randint(1000, 10000, n_samples)
        })

        # Generate features and labels
        mock_db = Mock()
        feature_engineer = FeatureEngineer({'features': {'sma_periods': [5]}}, mock_db)
        label_generator = LabelGenerator({'labeling': {'lookahead_days': 1}}, mock_db)

        features_df = feature_engineer.compute_all_features(data)
        labels_df = label_generator.generate_binary_labels(data)

        # Merge and verify alignment
        merged = features_df.merge(labels_df, on=['symbol', 'date'], how='inner')

        for i, row in merged.iterrows():
            feature_date = row['date']

            # Find the original data row
            orig_idx = data[data['date'] == feature_date].index[0]

            # Verify that the label corresponds to next day's movement
            if orig_idx < len(data) - 1:
                current_price = data.loc[orig_idx, 'adj_close']
                next_price = data.loc[orig_idx + 1, 'adj_close']
                expected_label = 1 if next_price > current_price else 0

                assert row['y_next'] == expected_label, \
                    f"Feature-label misalignment at {feature_date}"
