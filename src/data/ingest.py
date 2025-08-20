"""
Data ingestion module for fetching market data from various sources.

This module provides a unified interface for fetching OHLCV data from:
- yfinance (Yahoo Finance API)
- CSV files
- Other potential data sources

All data is validated, cleaned, and stored in the SQLite database.
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.data.db import DatabaseManager
from src.utils.config import ConfigManager

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


class DataIngestor:
    """Handles data ingestion from multiple sources."""

    def __init__(self, config: Union[Dict, ConfigManager], db: Optional[DatabaseManager] = None):
        """Initialize data ingestor.

        Args:
            config: Configuration dict or ConfigManager instance
            db: Optional DatabaseManager instance
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.config

        self.db = db or DatabaseManager(self.config.get('data', {}).get('db_path', 'data/trading.db'))

        # Data source configuration
        self.data_config = self.config.get('data', {})
        self.source = self.data_config.get('source', 'yfinance')
        self.csv_dir = Path(self.data_config.get('csv_dir', 'data/raw'))

        logger.info(f"DataIngestor initialized with source: {self.source}")

    def get_sp500_symbols(self) -> List[str]:
        """Fetch current S&P 500 symbols from Wikipedia.

        Returns:
            List of S&P 500 ticker symbols

        Note:
            This introduces survivorship bias as it uses current constituents.
        """
        try:
            # Fetch S&P 500 list from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]

            # Extract symbols and clean them
            symbols = sp500_table['Symbol'].tolist()

            # Clean symbols (remove dots and other special characters that yfinance doesn't like)
            cleaned_symbols = []
            for symbol in symbols:
                # Replace common problematic characters
                clean_symbol = symbol.replace('.', '-')
                cleaned_symbols.append(clean_symbol)

            logger.info(f"Retrieved {len(cleaned_symbols)} S&P 500 symbols")
            logger.warning("Using current S&P 500 constituents introduces survivorship bias")

            return cleaned_symbols

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 symbols: {e}")
            # Fallback to a small set of major symbols
            fallback_symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
            logger.info(f"Using fallback symbols: {fallback_symbols}")
            return fallback_symbols

    def fetch_yfinance_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        failed_symbols = []

        logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")

        for symbol in tqdm(symbols, desc="Fetching market data"):
            try:
                # Create yfinance ticker
                ticker = yf.Ticker(symbol)

                # Fetch historical data
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,  # We want both Close and Adj Close
                    back_adjust=False,
                    repair=True,
                    keepna=False
                )

                if data.empty:
                    logger.warning(f"No data available for {symbol}")
                    failed_symbols.append(symbol)
                    continue

                # Reset index to make Date a column
                data = data.reset_index()

                # Rename columns to match our schema
                data.columns = data.columns.str.lower().str.replace(' ', '_')

                # Add symbol column
                data['symbol'] = symbol

                # Ensure we have the required columns
                required_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                if not all(col in data.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {symbol}: {data.columns.tolist()}")
                    failed_symbols.append(symbol)
                    continue

                # Select only required columns
                data = data[['symbol'] + required_cols]

                # Data validation
                if len(data) < self.data_config.get('min_trading_days', 252):
                    logger.warning(f"Insufficient data for {symbol}: {len(data)} days")
                    failed_symbols.append(symbol)
                    continue

                # Check for missing values
                if data[['open', 'high', 'low', 'close', 'adj_close', 'volume']].isnull().any().any():
                    logger.warning(f"Missing values found in {symbol}, cleaning...")
                    # Forward fill missing values
                    data = data.fillna(method='ffill').dropna()

                # Validate price relationships
                invalid_prices = (
                    (data['high'] < data['low']) |
                    (data['high'] < data['open']) |
                    (data['high'] < data['close']) |
                    (data['low'] > data['open']) |
                    (data['low'] > data['close']) |
                    (data['volume'] < 0)
                )

                if invalid_prices.any():
                    logger.warning(f"Invalid price relationships found in {symbol}, removing {invalid_prices.sum()} rows")
                    data = data[~invalid_prices]

                all_data.append(data)
                logger.debug(f"Successfully fetched {len(data)} days of data for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")

        if not all_data:
            raise ValueError("No data was successfully fetched")

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Sort by symbol and date
        combined_data = combined_data.sort_values(['symbol', 'date']).reset_index(drop=True)

        logger.info(f"Successfully fetched {len(combined_data)} total rows for {len(all_data)} symbols")

        return combined_data

    def load_csv_data(self, csv_dir: Optional[Path] = None) -> pd.DataFrame:
        """Load OHLCV data from CSV files.

        Args:
            csv_dir: Directory containing CSV files (optional)

        Returns:
            DataFrame with OHLCV data

        Expected CSV format:
            Date,Open,High,Low,Close,Adj Close,Volume
            2020-01-01,100.0,102.0,99.5,101.0,101.0,1000000
        """
        csv_dir = csv_dir or self.csv_dir
        csv_dir = Path(csv_dir)

        if not csv_dir.exists():
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

        all_data = []
        csv_files = list(csv_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")

        logger.info(f"Loading data from {len(csv_files)} CSV files")

        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            try:
                # Extract symbol from filename (assume filename is SYMBOL.csv)
                symbol = csv_file.stem.upper()

                # Load CSV
                data = pd.read_csv(csv_file)

                # Standardize column names
                data.columns = data.columns.str.lower().str.replace(' ', '_')

                # Add symbol column
                data['symbol'] = symbol

                # Ensure date is properly formatted
                data['date'] = pd.to_datetime(data['date'])

                # Rename adj_close if needed
                if 'adj_close' not in data.columns and 'adj_close' in data.columns:
                    data.rename(columns={'adj_close': 'adj_close'}, inplace=True)

                # Validate required columns
                required_cols = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                if not all(col in data.columns for col in required_cols):
                    logger.error(f"CSV {csv_file} missing required columns: {required_cols}")
                    logger.error(f"Available columns: {data.columns.tolist()}")
                    continue

                # Select and reorder columns
                data = data[['symbol'] + required_cols]

                # Data validation
                data = self._validate_ohlcv_data(data, symbol)

                if len(data) > 0:
                    all_data.append(data)
                    logger.debug(f"Loaded {len(data)} rows from {csv_file}")

            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")

        if not all_data:
            raise ValueError("No valid CSV data was loaded")

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(['symbol', 'date']).reset_index(drop=True)

        logger.info(f"Successfully loaded {len(combined_data)} total rows from CSV files")

        return combined_data

    def _validate_ohlcv_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean OHLCV data.

        Args:
            data: Raw OHLCV data
            symbol: Symbol name for logging

        Returns:
            Cleaned and validated DataFrame
        """
        original_len = len(data)

        # Remove rows with missing values
        data = data.dropna()

        # Validate price relationships
        invalid_mask = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close']) |
            (data['volume'] < 0) |
            (data['open'] <= 0) |
            (data['high'] <= 0) |
            (data['low'] <= 0) |
            (data['close'] <= 0) |
            (data['adj_close'] <= 0)
        )

        if invalid_mask.any():
            logger.warning(f"Removing {invalid_mask.sum()} invalid rows from {symbol}")
            data = data[~invalid_mask]

        # Check for duplicates
        duplicates = data.duplicated(subset=['symbol', 'date'])
        if duplicates.any():
            logger.warning(f"Removing {duplicates.sum()} duplicate rows from {symbol}")
            data = data[~duplicates]

        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)

        cleaned_len = len(data)
        if cleaned_len < original_len:
            logger.info(f"Data validation for {symbol}: {original_len} -> {cleaned_len} rows")

        return data

    def fetch_data(self, symbols: Optional[List[str]] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  source: Optional[str] = None) -> pd.DataFrame:
        """Fetch market data from specified source.

        Args:
            symbols: List of symbols to fetch (optional, uses config default)
            start_date: Start date (optional, uses config default)
            end_date: End date (optional, uses config default)
            source: Data source (optional, uses config default)

        Returns:
            DataFrame with OHLCV data
        """
        # Use parameters or fall back to config
        symbols = symbols or self.data_config.get('symbols', ['SPY'])
        start_date = start_date or self.data_config.get('start_date', '2010-01-01')
        end_date = end_date or self.data_config.get('end_date', '2024-12-31')
        source = source or self.source

        logger.info(f"Fetching data: symbols={symbols}, dates={start_date} to {end_date}, source={source}")

        # Handle special symbol lists
        if symbols == ['SP500'] or 'SP500' in symbols:
            logger.info("Fetching S&P 500 constituent symbols")
            sp500_symbols = self.get_sp500_symbols()
            if symbols == ['SP500']:
                symbols = sp500_symbols
            else:
                # Replace SP500 with actual symbols
                symbols = [sym for sym in symbols if sym != 'SP500'] + sp500_symbols

        # Remove duplicates while preserving order
        symbols = list(dict.fromkeys(symbols))

        # Fetch data based on source
        if source == 'yfinance':
            data = self.fetch_yfinance_data(symbols, start_date, end_date)
        elif source == 'csv':
            data = self.load_csv_data()
            # Filter by symbols and dates if specified
            if symbols:
                data = data[data['symbol'].isin(symbols)]
            data['date'] = pd.to_datetime(data['date'])
            if start_date:
                data = data[data['date'] >= start_date]
            if end_date:
                data = data[data['date'] <= end_date]
        else:
            raise ValueError(f"Unsupported data source: {source}")

        # Final validation
        if data.empty:
            raise ValueError("No data was fetched")

        logger.info(f"Final dataset: {len(data)} rows, {data['symbol'].nunique()} symbols")

        return data

    def store_data(self, data: pd.DataFrame, replace: bool = False) -> None:
        """Store OHLCV data in database.

        Args:
            data: OHLCV DataFrame to store
            replace: Whether to replace existing data
        """
        if replace:
            logger.info("Clearing existing bars data")
            self.db.clear_table('bars')

        # Check for existing data to avoid duplicates
        existing_symbols = []
        for symbol in data['symbol'].unique():
            existing_data = self.db.get_bars(symbols=[symbol])
            if not existing_data.empty:
                existing_symbols.append(symbol)

        if existing_symbols and not replace:
            logger.warning(f"Data already exists for {len(existing_symbols)} symbols: {existing_symbols}")
            logger.info("Use replace=True to overwrite existing data")
            # Filter out existing symbols
            data = data[~data['symbol'].isin(existing_symbols)]

        if not data.empty:
            self.db.insert_bars(data)
        else:
            logger.info("No new data to store")

    def run_ingestion(self, symbols: Optional[List[str]] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     source: Optional[str] = None,
                     replace: bool = False) -> Dict[str, Union[int, List[str]]]:
        """Run complete data ingestion pipeline.

        Args:
            symbols: List of symbols to fetch
            start_date: Start date
            end_date: End date
            source: Data source
            replace: Whether to replace existing data

        Returns:
            Dictionary with ingestion results
        """
        start_time = datetime.now()

        try:
            # Fetch data
            data = self.fetch_data(symbols, start_date, end_date, source)

            # Store data
            self.store_data(data, replace=replace)

            # Calculate results
            results = {
                'success': True,
                'rows_fetched': len(data),
                'symbols_count': data['symbol'].nunique(),
                'symbols_list': sorted(data['symbol'].unique().tolist()),
                'date_range': {
                    'start': data['date'].min().strftime('%Y-%m-%d'),
                    'end': data['date'].max().strftime('%Y-%m-%d')
                },
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            }

            logger.info(f"Data ingestion completed successfully in {results['duration_seconds']:.1f} seconds")
            logger.info(f"Fetched {results['rows_fetched']} rows for {results['symbols_count']} symbols")

            return results

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            }

    def get_data_summary(self) -> Dict:
        """Get summary of data in database.

        Returns:
            Dictionary with data summary statistics
        """
        bars_data = self.db.get_bars()

        if bars_data.empty:
            return {'message': 'No data found in database'}

        summary = {
            'total_rows': len(bars_data),
            'symbols_count': bars_data['symbol'].nunique(),
            'symbols_list': sorted(bars_data['symbol'].unique().tolist()),
            'date_range': {
                'start': bars_data['date'].min().strftime('%Y-%m-%d'),
                'end': bars_data['date'].max().strftime('%Y-%m-%d')
            },
            'days_count': bars_data['date'].nunique(),
            'symbols_summary': bars_data.groupby('symbol').agg({
                'date': ['count', 'min', 'max']
            }).round(2).to_dict()
        }

        return summary

    def create_sample_data(self) -> None:
        """Create sample CSV data for testing/demonstration."""
        sample_dir = Path("data/sample")
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Generate sample data for SPY
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='B')  # Business days only
        np.random.seed(42)

        # Generate realistic price data
        initial_price = 300.0
        returns = np.random.normal(0.0005, 0.015, len(dates))  # Daily returns
        prices = [initial_price]

        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices[1:])  # Remove initial price

        # Generate OHLC from close prices
        opens = np.roll(prices, 1)  # Open = previous close
        opens[0] = initial_price

        # Add some intraday movement
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices))))

        # Ensure high >= max(open, close) and low <= min(open, close)
        highs = np.maximum(highs, np.maximum(opens, prices))
        lows = np.minimum(lows, np.minimum(opens, prices))

        # Generate volume
        volumes = np.random.lognormal(15, 0.5, len(dates)).astype(int)

        # Create DataFrame
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': opens.round(2),
            'High': highs.round(2),
            'Low': lows.round(2),
            'Close': prices.round(2),
            'Adj Close': prices.round(2),  # Assume no splits/dividends
            'Volume': volumes
        })

        # Save to CSV
        sample_file = sample_dir / "SPY.csv"
        sample_data.to_csv(sample_file, index=False)

        logger.info(f"Created sample data: {sample_file} ({len(sample_data)} rows)")
