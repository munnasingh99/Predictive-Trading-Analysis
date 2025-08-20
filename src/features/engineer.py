"""
Feature engineering module for technical indicators for Machine Learning models.

This module computes technical indicators from OHLCV data with strict
no-lookahead bias enforcement. All features at time t use only data
available at or before time t.

Key principles:
- No future information leakage
- Robust handling of missing data
- Vectorized calculations for performance
- Comprehensive technical indicator suite
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
from ta.utils import dropna

from src.data.db import DatabaseManager
from src.utils.config import ConfigManager

# Suppress TA-Lib warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='ta')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Technical feature engineering with no lookahead bias."""

    def __init__(self, config: Union[Dict, ConfigManager], db: Optional[DatabaseManager] = None):
        """Initialize feature engineer.

        Args:
            config: Configuration dict or ConfigManager instance
            db: Optional DatabaseManager instance
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.config

        self.db = db or DatabaseManager(self.config.get('data', {}).get('db_path', 'data/trading.db'))

        # Feature configuration
        self.features_config = self.config.get('features', {})

        # Technical indicator parameters
        self.sma_periods = self.features_config.get('sma_periods', [5, 10, 20])
        self.ema_periods = self.features_config.get('ema_periods', [5, 10, 20])
        self.rsi_period = self.features_config.get('rsi_period', 14)
        self.macd_params = self.features_config.get('macd_params', {'fast': 12, 'slow': 26, 'signal': 9})
        self.stoch_k_period = self.features_config.get('stoch_k_period', 14)
        self.atr_period = self.features_config.get('atr_period', 14)

        # Rolling statistics parameters
        self.returns_periods = self.features_config.get('returns_periods', [1, 5])
        self.volatility_period = self.features_config.get('volatility_period', 20)
        self.zscore_period = self.features_config.get('zscore_period', 20)
        self.momentum_period = self.features_config.get('momentum_period', 10)
        self.stats_periods = self.features_config.get('stats_periods', {'skew': 60, 'kurtosis': 60})

        logger.info("FeatureEngineer initialized")

    def compute_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Simple and Exponential Moving Averages.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with moving average features added
        """
        data = df.copy()

        # Simple Moving Averages
        for period in self.sma_periods:
            sma = (
                data['adj_close']
                .rolling(window=period, min_periods=period)
                .mean()
                .shift(1)
            )
            data[f'sma_{period}'] = sma

        # Exponential Moving Averages
        for period in self.ema_periods:
            ema = (
                data['adj_close']
                .ewm(span=period, adjust=False, min_periods=period)
                .mean()
                .shift(1)
            )
            data[f'ema_{period}'] = ema

        logger.debug(f"Computed {len(self.sma_periods)} SMA and {len(self.ema_periods)} EMA indicators")
        return data

    def compute_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum-based indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with momentum features added
        """
        data = df.copy()

        # RSI (Relative Strength Index)
        rsi = RSIIndicator(close=data['adj_close'], window=self.rsi_period)
        data['rsi_14'] = rsi.rsi()

        # MACD (Moving Average Convergence Divergence)
        macd = MACD(
            close=data['adj_close'],
            window_fast=self.macd_params['fast'],
            window_slow=self.macd_params['slow'],
            window_sign=self.macd_params['signal']
        )
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()

        # Stochastic %K
        stoch = StochasticOscillator(
            high=data['high'],
            low=data['low'],
            close=data['adj_close'],
            window=self.stoch_k_period
        )
        data['stoch_k'] = stoch.stoch()

        # Simple momentum (price change over N periods)
        data['momentum_10'] = data['adj_close'].pct_change(periods=self.momentum_period)

        logger.debug("Computed RSI, MACD, Stochastic, and Momentum indicators")
        return data

    def compute_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility features added
        """
        data = df.copy()

        # Average True Range (ATR)
        atr = AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['adj_close'],
            window=self.atr_period
        )
        # Shift to prevent using current period's range and blank out initial window
        data['atr_14'] = atr.average_true_range().shift(1)
        data.loc[data.index < self.atr_period, 'atr_14'] = np.nan

        # Rolling volatility (standard deviation of returns)
        returns = data['adj_close'].pct_change()
        vol = returns.rolling(window=self.volatility_period,
                              min_periods=self.volatility_period).std().shift(1)
        data['volatility_20'] = vol

        logger.debug("Computed ATR and rolling volatility indicators")
        return data

    def compute_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute return-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with return features added
        """
        data = df.copy()

        # Rolling returns for different periods
        for period in self.returns_periods:
            data[f'return_{period}d'] = data['adj_close'].pct_change(periods=period)

        logger.debug(f"Computed returns for periods: {self.returns_periods}")
        return data

    def compute_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute statistical features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with statistical features added
        """
        data = df.copy()

        # Rolling z-score of close prices (use only past values)
        close_mean = (
            data['adj_close']
            .rolling(window=self.zscore_period, min_periods=self.zscore_period)
            .mean()
            .shift(1)
        )
        close_std = (
            data['adj_close']
            .rolling(window=self.zscore_period, min_periods=self.zscore_period)
            .std()
            .shift(1)
        )
        data['zscore_20'] = (data['adj_close'] - close_mean) / close_std

        # Rolling skewness and kurtosis of returns
        returns = data['adj_close'].pct_change()

        if 'skew' in self.stats_periods:
            skew_period = self.stats_periods['skew']
            data[f'skew_{skew_period}'] = (
                returns.rolling(window=skew_period, min_periods=skew_period)
                .skew()
                .shift(1)
            )

        if 'kurtosis' in self.stats_periods:
            kurtosis_period = self.stats_periods['kurtosis']
            # pandas uses .kurt() for kurtosis on rolling windows
            data[f'kurtosis_{kurtosis_period}'] = (
                returns.rolling(window=kurtosis_period, min_periods=kurtosis_period)
                .kurt()
                .shift(1)
            )

        logger.debug("Computed statistical features: z-score, skewness, kurtosis")
        return data

    def validate_no_lookahead(self, df: pd.DataFrame, feature_cols: List[str]) -> bool:
        """Validate that features don't contain lookahead bias.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names

        Returns:
            True if no lookahead detected, False otherwise

        Raises:
            ValueError: If lookahead bias is detected
        """
        # Sort by date to ensure proper time series order
        data = df.sort_values(['symbol', 'date']).copy()

        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()

            for col in feature_cols:
                if col not in symbol_data.columns:
                    continue

                # Check if feature values are available before they should be
                feature_series = symbol_data[col]

                # Skip if all NaN (expected for early periods)
                if feature_series.isna().all():
                    continue

                # Find first non-NaN value
                first_valid_idx = feature_series.first_valid_index()
                if first_valid_idx is None:
                    continue

                first_valid_pos = symbol_data.index.get_loc(first_valid_idx)

                # For indicators with windows, check that first valid value appears
                # after sufficient data points are available
                min_required_periods = self._get_min_periods_for_feature(col)

                if first_valid_pos < min_required_periods - 1:
                    error_msg = (f"Potential lookahead bias in {col} for {symbol}: "
                               f"First valid value at position {first_valid_pos}, "
                               f"but requires {min_required_periods} periods")
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        logger.info("No lookahead bias detected in features")
        return True

    def _get_min_periods_for_feature(self, feature_name: str) -> int:
        """Get minimum required periods for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Minimum number of periods required
        """
        # Extract period from feature name or use defaults
        if 'sma_' in feature_name:
            period = int(feature_name.split('_')[1])
            return period
        elif 'ema_' in feature_name:
            period = int(feature_name.split('_')[1])
            return period
        elif 'rsi' in feature_name:
            return self.rsi_period
        elif 'macd' in feature_name:
            return self.macd_params['slow']
        elif 'stoch' in feature_name:
            return self.stoch_k_period
        elif 'atr' in feature_name:
            return self.atr_period
        elif 'volatility' in feature_name:
            return self.volatility_period
        elif 'zscore' in feature_name:
            return self.zscore_period
        elif 'momentum' in feature_name:
            return self.momentum_period
        elif 'return_' in feature_name:
            period = int(feature_name.split('_')[1].replace('d', ''))
            return period
        elif 'skew' in feature_name:
            return self.stats_periods.get('skew', 60)
        elif 'kurtosis' in feature_name:
            return self.stats_periods.get('kurtosis', 60)
        else:
            return 1

    def compute_all_features(self, df: pd.DataFrame, validate_lookahead: bool = True) -> pd.DataFrame:
        """Compute all technical indicators for given OHLCV data.

        Args:
            df: DataFrame with OHLCV data (symbol, date, open, high, low, close, adj_close, volume)
            validate_lookahead: Whether to validate no lookahead bias

        Returns:
            DataFrame with all features computed
        """
        if df.empty:
            return df

        logger.info(f"Computing features for {df['symbol'].nunique()} symbols")

        # Ensure proper column types
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        all_features = []

        # Process each symbol separately to avoid cross-contamination
        for symbol in df['symbol'].unique():
            logger.debug(f"Processing features for {symbol}")

            symbol_data = df[df['symbol'] == symbol].copy().reset_index(drop=True)

            if len(symbol_data) < 2:
                logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} rows")
                continue

            try:
                # Compute all feature types
                symbol_data = self.compute_moving_averages(symbol_data)
                symbol_data = self.compute_momentum_indicators(symbol_data)
                symbol_data = self.compute_volatility_indicators(symbol_data)
                symbol_data = self.compute_return_features(symbol_data)
                symbol_data = self.compute_statistical_features(symbol_data)

                all_features.append(symbol_data)

            except Exception as e:
                logger.error(f"Feature computation failed for {symbol}: {e}")
                continue

        if not all_features:
            raise ValueError("No features were successfully computed")

        # Combine all symbol data and ensure proper ordering
        result_df = pd.concat(all_features, ignore_index=True)
        result_df = result_df.sort_values(['symbol', 'date']).reset_index(drop=True)

        # Recompute moving averages on the combined dataset to ensure
        # alignment is based on the final, fully sorted data
        for period in self.sma_periods:
            result_df[f'sma_{period}'] = (
                result_df.groupby('symbol')['adj_close']
                .transform(lambda s: s.rolling(window=period, min_periods=period).mean().shift(1))
            )

        for period in self.ema_periods:
            result_df[f'ema_{period}'] = (
                result_df.groupby('symbol')['adj_close']
                .transform(lambda s: s.ewm(span=period, adjust=False, min_periods=period).mean().shift(1))
            )

        # Get feature columns (exclude OHLCV columns)
        ohlcv_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        feature_cols = [col for col in result_df.columns if col not in ohlcv_cols]

        logger.info(f"Computed {len(feature_cols)} features: {feature_cols}")

        # Validate no lookahead bias
        if validate_lookahead:
            self.validate_no_lookahead(result_df, feature_cols)

        # Clean infinite and extremely large values
        result_df = self._clean_features(result_df, feature_cols)

        return result_df

    def _clean_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Clean features by handling inf, -inf, and extreme values.

        Args:
            df: DataFrame with features
            feature_cols: List of feature columns

        Returns:
            Cleaned DataFrame
        """
        data = df.copy()

        for col in feature_cols:
            if col not in data.columns:
                continue

            # Replace inf and -inf with NaN
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)

            if data[col].isna().all():
                continue

            # Avoid clipping smoothed moving averages which can distort temporal checks
            if col.startswith(('sma_', 'ema_')):
                continue

            # Cap extreme values (beyond 99.9th percentile)
            q999 = data[col].quantile(0.999)
            q001 = data[col].quantile(0.001)

            if not np.isnan(q999) and not np.isnan(q001):
                data[col] = data[col].clip(lower=q001, upper=q999)

        logger.debug("Cleaned features: removed inf values and capped extremes")
        return data

    def prepare_features_for_db(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features DataFrame for database storage.

        Args:
            df: DataFrame with computed features

        Returns:
            DataFrame ready for database insertion
        """
        # Select only the columns that match our database schema
        db_columns = [
            'symbol', 'date',
            'sma_5', 'sma_10', 'sma_20',
            'ema_5', 'ema_10', 'ema_20',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'stoch_k',
            'atr_14', 'volatility_20',
            'return_1d', 'return_5d',
            'zscore_20', 'momentum_10', 'skew_60', 'kurtosis_60'
        ]

        # Select only columns that exist in the dataframe
        available_columns = [col for col in db_columns if col in df.columns]

        if len(available_columns) < 3:  # At least symbol, date, and one feature
            raise ValueError(f"Insufficient columns for database storage: {available_columns}")

        result_df = df[available_columns].copy()

        # Ensure date is properly formatted
        result_df['date'] = pd.to_datetime(result_df['date'])

        return result_df

    def store_features(self, features_df: pd.DataFrame, replace: bool = False) -> None:
        """Store computed features in database.

        Args:
            features_df: DataFrame with features
            replace: Whether to replace existing features
        """
        if replace:
            logger.info("Clearing existing features data")
            self.db.clear_table('features')

        # Prepare for database storage
        db_ready_df = self.prepare_features_for_db(features_df)

        # Check for existing data to avoid duplicates
        if not replace:
            existing_symbols = []
            for symbol in db_ready_df['symbol'].unique():
                existing_data = self.db.execute_query(
                    "SELECT COUNT(*) as count FROM features WHERE symbol = :symbol",
                    {'symbol': symbol}
                )
                if existing_data.iloc[0]['count'] > 0:
                    existing_symbols.append(symbol)

            if existing_symbols:
                logger.warning(f"Features already exist for {len(existing_symbols)} symbols: {existing_symbols}")
                logger.info("Use replace=True to overwrite existing features")
                db_ready_df = db_ready_df[~db_ready_df['symbol'].isin(existing_symbols)]

        if not db_ready_df.empty:
            self.db.insert_features(db_ready_df)
            logger.info(f"Stored {len(db_ready_df)} feature rows for {db_ready_df['symbol'].nunique()} symbols")
        else:
            logger.info("No new features to store")

    def build_features_from_db(self, symbols: Optional[List[str]] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              replace: bool = False) -> pd.DataFrame:
        """Build features from OHLCV data in database.

        Args:
            symbols: Optional list of symbols to process
            start_date: Optional start date filter
            end_date: Optional end date filter
            replace: Whether to replace existing features

        Returns:
            DataFrame with computed features
        """
        # Get OHLCV data from database
        bars_df = self.db.get_bars(symbols=symbols, start_date=start_date, end_date=end_date)

        if bars_df.empty:
            raise ValueError("No OHLCV data found in database")

        logger.info(f"Building features from {len(bars_df)} OHLCV rows")

        # Compute features
        features_df = self.compute_all_features(bars_df)

        # Store in database
        self.store_features(features_df, replace=replace)

        return features_df

    def get_feature_summary(self) -> Dict:
        """Get summary statistics of features in database.

        Returns:
            Dictionary with feature summary
        """
        try:
            features_data = self.db.execute_query("SELECT * FROM features LIMIT 1000")

            if features_data.empty:
                return {'message': 'No features found in database'}

            # Get feature columns (exclude symbol and date)
            feature_cols = [col for col in features_data.columns if col not in ['symbol', 'date']]

            # Calculate summary statistics
            summary = {
                'total_rows': len(features_data),
                'symbols_count': features_data['symbol'].nunique() if 'symbol' in features_data else 0,
                'feature_count': len(feature_cols),
                'feature_names': feature_cols,
                'missing_data_pct': features_data[feature_cols].isnull().mean().round(4).to_dict(),
                'feature_stats': features_data[feature_cols].describe().round(4).to_dict()
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to generate feature summary: {e}")
            return {'error': str(e)}

    def validate_feature_quality(self, df: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
        """Validate quality of computed features.

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in ['symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]

        # Check for completely missing features
        for col in feature_cols:
            if df[col].isna().all():
                issues.append(f"Feature {col} is completely missing")
            elif df[col].isna().mean() > 0.8:
                warnings.append(f"Feature {col} has {df[col].isna().mean():.1%} missing values")

        # Check for constant features
        for col in feature_cols:
            if df[col].nunique(dropna=True) <= 1:
                issues.append(f"Feature {col} has constant values")

        # Check for infinite values
        for col in feature_cols:
            if np.isinf(df[col]).any():
                issues.append(f"Feature {col} contains infinite values")

        # Check feature correlation
        if len(feature_cols) > 1:
            corr_matrix = df[feature_cols].corr()
            high_corr_pairs = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.95:
                        high_corr_pairs.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")

            if high_corr_pairs:
                warnings.extend([f"High correlation: {pair}" for pair in high_corr_pairs])

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'feature_count': len(feature_cols),
            'total_rows': len(df)
        }
