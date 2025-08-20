"""
Label generation module for creating target variables.

This module generates binary classification labels for predicting next-day
price movements. Key principles:
- Labels are based on future price movements (y_t predicts movement from t to t+1)
- Strict temporal alignment to prevent leakage
- Configurable labeling strategies
- Comprehensive validation of label correctness
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.db import DatabaseManager
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generates target labels for machine learning models."""

    def __init__(self, config: Union[Dict, ConfigManager], db: Optional[DatabaseManager] = None):
        """Initialize label generator.

        Args:
            config: Configuration dict or ConfigManager instance
            db: Optional DatabaseManager instance
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.config

        self.db = db or DatabaseManager(self.config.get('data', {}).get('db_path', 'data/trading.db'))

        # Labeling configuration
        self.labeling_config = self.config.get('labeling', {})
        self.target_type = self.labeling_config.get('target_type', 'binary')
        self.lookahead_days = self.labeling_config.get('lookahead_days', 1)
        self.min_return_threshold = self.labeling_config.get('min_return_threshold', 0.0)

        logger.info(f"LabelGenerator initialized: target_type={self.target_type}, lookahead={self.lookahead_days}")

    def generate_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate binary classification labels (Up/Down).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with labels added
        """
        if df.empty:
            return df

        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])

        # Sort by symbol and date to ensure proper ordering
        data = data.sort_values(['symbol', 'date']).reset_index(drop=True)

        all_labels = []

        # Allow dynamic updates to configuration after initialization
        lookahead_days = self.labeling_config.get('lookahead_days', self.lookahead_days)
        threshold = self.labeling_config.get('min_return_threshold', self.min_return_threshold)

        # Process each symbol separately
        for symbol in data['symbol'].unique():
            logger.debug(f"Generating labels for {symbol}")

            symbol_data = data[data['symbol'] == symbol].copy().reset_index(drop=True)

            if len(symbol_data) < lookahead_days + 1:
                logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} rows")
                continue

            # Calculate future returns
            symbol_data['future_price'] = symbol_data['adj_close'].shift(-lookahead_days)

            # Calculate return from current price to future price
            symbol_data['future_return'] = (
                (symbol_data['future_price'] - symbol_data['adj_close']) / symbol_data['adj_close']
            )

            # Generate binary labels
            # y_next = 1 if future_return > threshold, 0 otherwise
            symbol_data['y_next'] = (symbol_data['future_return'] > threshold).astype(int)

            # Remove the last lookahead_days rows as they don't have valid labels
            symbol_data = symbol_data.iloc[:-lookahead_days].copy()

            # Select only required columns for labels table
            label_data = symbol_data[['symbol', 'date', 'y_next']].copy()

            # Validate labels
            if self._validate_labels(label_data, symbol):
                all_labels.append(label_data)

        if not all_labels:
            raise ValueError("No valid labels were generated")

        # Combine all labels
        result_df = pd.concat(all_labels, ignore_index=True)

        logger.info(f"Generated {len(result_df)} labels for {result_df['symbol'].nunique()} symbols")

        return result_df

    def _validate_labels(self, labels_df: pd.DataFrame, symbol: str) -> bool:
        """Validate generated labels for correctness.

        Args:
            labels_df: DataFrame with labels
            symbol: Symbol name for logging

        Returns:
            True if validation passes, False otherwise
        """
        if labels_df.empty:
            logger.warning(f"No labels generated for {symbol}")
            return False

        # Check for missing labels
        if labels_df['y_next'].isna().any():
            logger.warning(f"Missing labels found for {symbol}")
            return False

        # Check label distribution
        label_counts = labels_df['y_next'].value_counts()
        total_labels = len(labels_df)

        if len(label_counts) < 2:
            logger.warning(f"Only one class found in labels for {symbol}: {label_counts.to_dict()}")
            logger.debug(f"Label distribution for {symbol}: {label_counts.to_dict()}")
            return True

        # Check for extreme class imbalance (less than 5% minority class)
        minority_pct = label_counts.min() / total_labels
        if minority_pct < 0.05:
            logger.warning(
                f"Extreme class imbalance for {symbol}: {label_counts.to_dict()} "
                f"({minority_pct:.1%} minority class)"
            )

        pos_pct = label_counts.get(1, 0) / total_labels
        logger.debug(
            f"Label distribution for {symbol}: {label_counts.to_dict()} "
            f"({pos_pct:.1%} positive)"
        )

        return True

    def validate_temporal_alignment(self, bars_df: pd.DataFrame, labels_df: pd.DataFrame) -> bool:
        """Validate that labels are correctly aligned temporally.

        Args:
            bars_df: DataFrame with OHLCV data
            labels_df: DataFrame with labels

        Returns:
            True if alignment is correct

        Raises:
            ValueError: If temporal alignment is incorrect
        """
        # Merge bars and labels to check alignment
        merged = bars_df.merge(
            labels_df,
            on=['symbol', 'date'],
            how='inner'
        )

        if merged.empty:
            raise ValueError("No matching dates between bars and labels")

        # For each row, verify that the label correctly predicts future movement
        validation_errors = []

        for symbol in merged['symbol'].unique():
            symbol_data = merged[merged['symbol'] == symbol].copy()
            symbol_bars = bars_df[bars_df['symbol'] == symbol].copy()

            # Sort by date
            symbol_data = symbol_data.sort_values('date')
            symbol_bars = symbol_bars.sort_values('date')

            for idx, row in symbol_data.iterrows():
                current_date = row['date']
                current_price = row['adj_close']
                predicted_label = row['y_next']

                # Find the future date (lookahead_days ahead)
                future_bars = symbol_bars[symbol_bars['date'] > current_date].head(self.lookahead_days)

                if len(future_bars) < self.lookahead_days:
                    continue

                future_price = future_bars.iloc[-1]['adj_close']
                actual_return = (future_price - current_price) / current_price
                expected_label = int(actual_return > self.min_return_threshold)

                if predicted_label != expected_label:
                    validation_errors.append({
                        'symbol': symbol,
                        'date': current_date,
                        'predicted_label': predicted_label,
                        'expected_label': expected_label,
                        'actual_return': actual_return
                    })

        if validation_errors:
            error_count = len(validation_errors)
            total_count = len(merged)
            error_rate = error_count / total_count

            if error_rate > 0.01:  # More than 1% errors
                sample_errors = validation_errors[:5]  # Show first 5 errors
                error_msg = (f"Temporal alignment validation failed: {error_count}/{total_count} "
                           f"({error_rate:.1%}) misaligned labels. Sample errors: {sample_errors}")
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                logger.warning(f"Minor temporal alignment issues: {error_count}/{total_count} "
                             f"({error_rate:.1%}) misaligned labels (within tolerance)")

        logger.info("Temporal alignment validation passed")
        return True

    def generate_multiclass_labels(self, df: pd.DataFrame,
                                 thresholds: List[float] = [-0.02, 0.02]) -> pd.DataFrame:
        """Generate multiclass labels (Down/Flat/Up).

        Args:
            df: DataFrame with OHLCV data
            thresholds: List of [down_threshold, up_threshold]

        Returns:
            DataFrame with multiclass labels
        """
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values(['symbol', 'date']).reset_index(drop=True)

        down_threshold, up_threshold = thresholds
        all_labels = []

        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy().reset_index(drop=True)

            if len(symbol_data) < self.lookahead_days + 1:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Calculate future returns
            symbol_data['future_price'] = symbol_data['adj_close'].shift(-self.lookahead_days)
            symbol_data['future_return'] = (
                (symbol_data['future_price'] - symbol_data['adj_close']) / symbol_data['adj_close']
            )

            # Generate multiclass labels
            # 0: Down, 1: Flat, 2: Up
            conditions = [
                symbol_data['future_return'] <= down_threshold,
                (symbol_data['future_return'] > down_threshold) & (symbol_data['future_return'] <= up_threshold),
                symbol_data['future_return'] > up_threshold
            ]
            choices = [0, 1, 2]
            symbol_data['y_next'] = np.select(conditions, choices, default=1)

            # Remove last rows without valid labels
            symbol_data = symbol_data.iloc[:-self.lookahead_days].copy()
            label_data = symbol_data[['symbol', 'date', 'y_next']].copy()

            if not label_data.empty:
                all_labels.append(label_data)

        if not all_labels:
            raise ValueError("No valid multiclass labels were generated")

        result_df = pd.concat(all_labels, ignore_index=True)
        logger.info(f"Generated {len(result_df)} multiclass labels")

        return result_df

    def generate_regression_labels(self, df: pd.DataFrame,
                                 clip_percentiles: Tuple[float, float] = (1, 99)) -> pd.DataFrame:
        """Generate regression labels (continuous returns).

        Args:
            df: DataFrame with OHLCV data
            clip_percentiles: Tuple of (lower, upper) percentiles to clip outliers

        Returns:
            DataFrame with regression labels
        """
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values(['symbol', 'date']).reset_index(drop=True)

        all_labels = []

        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy().reset_index(drop=True)

            if len(symbol_data) < self.lookahead_days + 1:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Calculate future returns
            symbol_data['future_price'] = symbol_data['adj_close'].shift(-self.lookahead_days)
            symbol_data['future_return'] = (
                (symbol_data['future_price'] - symbol_data['adj_close']) / symbol_data['adj_close']
            )

            # Clip outliers
            lower_clip = np.percentile(symbol_data['future_return'].dropna(), clip_percentiles[0])
            upper_clip = np.percentile(symbol_data['future_return'].dropna(), clip_percentiles[1])
            symbol_data['y_next'] = symbol_data['future_return'].clip(lower_clip, upper_clip)

            # Remove last rows without valid labels
            symbol_data = symbol_data.iloc[:-self.lookahead_days].copy()
            label_data = symbol_data[['symbol', 'date', 'y_next']].copy()

            if not label_data.empty:
                all_labels.append(label_data)

        if not all_labels:
            raise ValueError("No valid regression labels were generated")

        result_df = pd.concat(all_labels, ignore_index=True)
        logger.info(f"Generated {len(result_df)} regression labels")

        return result_df

    def store_labels(self, labels_df: pd.DataFrame, replace: bool = False) -> None:
        """Store generated labels in database.

        Args:
            labels_df: DataFrame with labels
            replace: Whether to replace existing labels
        """
        if replace:
            logger.info("Clearing existing labels data")
            self.db.clear_table('labels')

        # Check for existing data
        if not replace:
            existing_symbols = []
            for symbol in labels_df['symbol'].unique():
                existing_data = self.db.execute_query(
                    "SELECT COUNT(*) as count FROM labels WHERE symbol = :symbol",
                    {'symbol': symbol}
                )
                if existing_data.iloc[0]['count'] > 0:
                    existing_symbols.append(symbol)

            if existing_symbols:
                logger.warning(f"Labels already exist for {len(existing_symbols)} symbols: {existing_symbols}")
                logger.info("Use replace=True to overwrite existing labels")
                labels_df = labels_df[~labels_df['symbol'].isin(existing_symbols)]

        if not labels_df.empty:
            self.db.insert_labels(labels_df)
            logger.info(f"Stored {len(labels_df)} labels for {labels_df['symbol'].nunique()} symbols")
        else:
            logger.info("No new labels to store")

    def generate_labels_from_db(self, symbols: Optional[List[str]] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              replace: bool = False) -> pd.DataFrame:
        """Generate labels from OHLCV data in database.

        Args:
            symbols: Optional list of symbols to process
            start_date: Optional start date filter
            end_date: Optional end date filter
            replace: Whether to replace existing labels

        Returns:
            DataFrame with generated labels
        """
        # Get OHLCV data from database
        bars_df = self.db.get_bars(symbols=symbols, start_date=start_date, end_date=end_date)

        if bars_df.empty:
            raise ValueError("No OHLCV data found in database")

        logger.info(f"Generating labels from {len(bars_df)} OHLCV rows")

        # Generate labels based on target type
        if self.target_type == 'binary':
            labels_df = self.generate_binary_labels(bars_df)
        elif self.target_type == 'multiclass':
            labels_df = self.generate_multiclass_labels(bars_df)
        elif self.target_type == 'regression':
            labels_df = self.generate_regression_labels(bars_df)
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")

        # Validate temporal alignment
        self.validate_temporal_alignment(bars_df, labels_df)

        # Store in database
        self.store_labels(labels_df, replace=replace)

        return labels_df

    def get_label_summary(self) -> Dict:
        """Get summary statistics of labels in database.

        Returns:
            Dictionary with label summary
        """
        try:
            labels_data = self.db.execute_query("SELECT * FROM labels")

            if labels_data.empty:
                return {'message': 'No labels found in database'}

            # Calculate summary statistics
            summary = {
                'total_labels': len(labels_data),
                'symbols_count': labels_data['symbol'].nunique(),
                'symbols_list': sorted(labels_data['symbol'].unique().tolist()),
                'date_range': {
                    'start': labels_data['date'].min(),
                    'end': labels_data['date'].max()
                },
                'label_distribution': labels_data['y_next'].value_counts().to_dict(),
                'positive_rate': labels_data['y_next'].mean(),
                'labels_per_symbol': labels_data.groupby('symbol')['y_next'].count().to_dict()
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to generate label summary: {e}")
            return {'error': str(e)}

    def validate_label_quality(self, labels_df: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
        """Validate quality of generated labels.

        Args:
            labels_df: DataFrame with labels

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        # Check for missing labels
        if labels_df['y_next'].isna().any():
            issues.append("Missing labels found")

        # Check for valid label values
        unique_labels = labels_df['y_next'].dropna().unique()

        if self.target_type == 'binary':
            expected_labels = {0, 1}
            if not set(unique_labels).issubset(expected_labels):
                issues.append(f"Invalid binary labels found: {unique_labels}")
        elif self.target_type == 'multiclass':
            expected_labels = {0, 1, 2}
            if not set(unique_labels).issubset(expected_labels):
                issues.append(f"Invalid multiclass labels found: {unique_labels}")

        # Check label distribution
        label_counts = labels_df['y_next'].value_counts()
        total_labels = len(labels_df)

        # Check for extreme class imbalance
        if len(label_counts) > 1:
            minority_pct = label_counts.min() / total_labels
            if minority_pct < 0.01:
                warnings.append(f"Extreme class imbalance: {minority_pct:.1%} minority class")
            elif minority_pct < 0.05:
                warnings.append(f"High class imbalance: {minority_pct:.1%} minority class")

        # Check for temporal consistency
        symbols_with_gaps = []
        for symbol in labels_df['symbol'].unique():
            symbol_data = labels_df[labels_df['symbol'] == symbol].sort_values('date')
            date_diffs = symbol_data['date'].diff().dt.days

            # Check for large gaps (more than 10 business days)
            large_gaps = date_diffs[date_diffs > 10]
            if len(large_gaps) > 0:
                symbols_with_gaps.append(symbol)

        if symbols_with_gaps:
            warnings.append(f"Large temporal gaps found in symbols: {symbols_with_gaps}")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_labels': len(labels_df),
            'symbols_count': labels_df['symbol'].nunique(),
            'label_distribution': label_counts.to_dict()
        }

    def create_balanced_labels(self, labels_df: pd.DataFrame,
                             balance_method: str = 'undersample') -> pd.DataFrame:
        """Create balanced labels to address class imbalance.

        Args:
            labels_df: DataFrame with labels
            balance_method: Method to balance ('undersample', 'oversample')

        Returns:
            DataFrame with balanced labels
        """
        if self.target_type != 'binary':
            logger.warning("Balancing only implemented for binary classification")
            return labels_df

        balanced_data = []

        for symbol in labels_df['symbol'].unique():
            symbol_data = labels_df[labels_df['symbol'] == symbol].copy()

            # Get class counts
            class_counts = symbol_data['y_next'].value_counts()

            if len(class_counts) < 2:
                balanced_data.append(symbol_data)
                continue

            min_count = class_counts.min()
            max_count = class_counts.max()

            if balance_method == 'undersample':
                # Undersample majority class
                minority_class = class_counts.idxmin()
                majority_class = class_counts.idxmax()

                minority_data = symbol_data[symbol_data['y_next'] == minority_class]
                majority_data = symbol_data[symbol_data['y_next'] == majority_class].sample(
                    n=min_count, random_state=42
                )

                balanced_symbol_data = pd.concat([minority_data, majority_data])

            elif balance_method == 'oversample':
                # Simple oversampling (duplicate minority samples)
                minority_class = class_counts.idxmin()
                majority_class = class_counts.idxmax()

                minority_data = symbol_data[symbol_data['y_next'] == minority_class]
                majority_data = symbol_data[symbol_data['y_next'] == majority_class]

                # Oversample minority class
                oversample_count = max_count - min_count
                oversampled_minority = minority_data.sample(
                    n=oversample_count, replace=True, random_state=42
                )

                balanced_symbol_data = pd.concat([minority_data, majority_data, oversampled_minority])

            else:
                raise ValueError(f"Unsupported balance method: {balance_method}")

            balanced_data.append(balanced_symbol_data.sort_values('date'))

        result_df = pd.concat(balanced_data, ignore_index=True)

        logger.info(f"Balanced labels: {len(labels_df)} -> {len(result_df)} rows using {balance_method}")

        return result_df
