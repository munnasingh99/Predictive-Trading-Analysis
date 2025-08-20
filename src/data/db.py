"""
Database management module for SQLite operations.

This module provides a centralized interface for all database operations
including table creation, data insertion, querying, and schema management.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import TypeDecorator, VARCHAR

logger = logging.getLogger(__name__)

Base = declarative_base()


class DateType(TypeDecorator):
    """Custom date type for SQLite compatibility."""

    impl = VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            return value.strftime('%Y-%m-%d')
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return pd.to_datetime(value).date()
        return value


class Bars(Base):
    """OHLCV bars table."""

    __tablename__ = 'bars'

    symbol = Column(String(10), primary_key=True)
    date = Column(DateType, primary_key=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)


class Features(Base):
    """Technical features table."""

    __tablename__ = 'features'

    symbol = Column(String(10), primary_key=True)
    date = Column(DateType, primary_key=True)

    # Moving averages
    sma_5 = Column(Float)
    sma_10 = Column(Float)
    sma_20 = Column(Float)
    ema_5 = Column(Float)
    ema_10 = Column(Float)
    ema_20 = Column(Float)

    # Momentum indicators
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    stoch_k = Column(Float)

    # Volatility
    atr_14 = Column(Float)
    volatility_20 = Column(Float)

    # Returns
    return_1d = Column(Float)
    return_5d = Column(Float)

    # Statistical
    zscore_20 = Column(Float)
    momentum_10 = Column(Float)
    skew_60 = Column(Float)
    kurtosis_60 = Column(Float)


class Labels(Base):
    """Labels table."""

    __tablename__ = 'labels'

    symbol = Column(String(10), primary_key=True)
    date = Column(DateType, primary_key=True)  # Feature date (predicting t+1)
    y_next = Column(Integer, nullable=False)  # 1 if price goes up next day, 0 otherwise


class Predictions(Base):
    """Model predictions table."""

    __tablename__ = 'predictions'

    symbol = Column(String(10), primary_key=True)
    date = Column(DateType, primary_key=True)
    model_name = Column(String(50), primary_key=True)
    proba = Column(Float, nullable=False)  # Probability of positive class
    pred_label = Column(Integer, nullable=False)  # Predicted class
    timestamp = Column(String(50), nullable=False)  # When prediction was made


class Trades(Base):
    """Backtest trades table."""

    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    entry_date = Column(DateType, nullable=False)
    exit_date = Column(DateType, nullable=True)
    side = Column(String(5), nullable=False)  # 'long' or 'short'
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    model_name = Column(String(50), nullable=False)
    signal_proba = Column(Float, nullable=False)


class EquityCurves(Base):
    """Daily equity curves table."""

    __tablename__ = 'equity_curves'

    date = Column(DateType, primary_key=True)
    model_name = Column(String(50), primary_key=True)
    symbol = Column(String(10), primary_key=True)
    equity = Column(Float, nullable=False)
    daily_return = Column(Float, nullable=False)
    cumulative_return = Column(Float, nullable=False)
    drawdown = Column(Float, nullable=False)
    position = Column(Float, nullable=False)  # Position size (-1 to 1)


class DatabaseManager:
    """Centralized database management for the trading pipeline."""

    def __init__(self, db_path: str = "data/trading.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.Session = sessionmaker(bind=self.engine)

        # Create all tables
        self.create_tables()

        logger.info(f"Database initialized: {db_path}")

    def create_tables(self) -> None:
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")

    def drop_tables(self) -> None:
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(self.engine)
        logger.warning("All database tables dropped")

    def get_session(self):
        """Get a database session."""
        return self.Session()

    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            Query results as DataFrame
        """
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql_query(text(query), conn, params=params or {})
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def insert_bars(self, df: pd.DataFrame) -> None:
        """Insert OHLCV bars into database.

        Args:
            df: DataFrame with columns [symbol, date, open, high, low, close, adj_close, volume]
        """
        try:
            # Ensure date column is properly formatted
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.date

            df.to_sql('bars', self.engine, if_exists='append', index=False, method='multi')
            logger.info(f"Inserted {len(df)} bars into database")

        except Exception as e:
            logger.error(f"Failed to insert bars: {e}")
            raise

    def insert_features(self, df: pd.DataFrame) -> None:
        """Insert features into database.

        Args:
            df: DataFrame with features
        """
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.date

            df.to_sql('features', self.engine, if_exists='append', index=False, method='multi')
            logger.info(f"Inserted {len(df)} feature rows into database")

        except Exception as e:
            logger.error(f"Failed to insert features: {e}")
            raise

    def insert_labels(self, df: pd.DataFrame) -> None:
        """Insert labels into database.

        Args:
            df: DataFrame with labels
        """
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.date

            df.to_sql('labels', self.engine, if_exists='append', index=False, method='multi')
            logger.info(f"Inserted {len(df)} label rows into database")

        except Exception as e:
            logger.error(f"Failed to insert labels: {e}")
            raise

    def insert_predictions(self, df: pd.DataFrame) -> None:
        """Insert predictions into database.

        Args:
            df: DataFrame with predictions
        """
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.date

            df.to_sql('predictions', self.engine, if_exists='append', index=False, method='multi')
            logger.info(f"Inserted {len(df)} prediction rows into database")

        except Exception as e:
            logger.error(f"Failed to insert predictions: {e}")
            raise

    def insert_trades(self, df: pd.DataFrame) -> None:
        """Insert trades into database.

        Args:
            df: DataFrame with trade data
        """
        try:
            df = df.copy()
            df['entry_date'] = pd.to_datetime(df['entry_date']).dt.date
            if 'exit_date' in df.columns:
                df['exit_date'] = pd.to_datetime(df['exit_date']).dt.date

            df.to_sql('trades', self.engine, if_exists='append', index=False, method='multi')
            logger.info(f"Inserted {len(df)} trades into database")

        except Exception as e:
            logger.error(f"Failed to insert trades: {e}")
            raise

    def insert_equity_curves(self, df: pd.DataFrame) -> None:
        """Insert equity curve data into database.

        Args:
            df: DataFrame with equity curve data
        """
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.date

            df.to_sql('equity_curves', self.engine, if_exists='append', index=False, method='multi')
            logger.info(f"Inserted {len(df)} equity curve rows into database")

        except Exception as e:
            logger.error(f"Failed to insert equity curves: {e}")
            raise

    def get_bars(self, symbols: Optional[List[str]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve OHLCV bars from database.

        Args:
            symbols: Optional list of symbols to filter
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        query = "SELECT * FROM bars"
        conditions = []
        params = {}

        if symbols:
            placeholders = ','.join([f':symbol_{i}' for i in range(len(symbols))])
            conditions.append(f"symbol IN ({placeholders})")
            for i, symbol in enumerate(symbols):
                params[f'symbol_{i}'] = symbol

        if start_date:
            conditions.append("date >= :start_date")
            params['start_date'] = start_date

        if end_date:
            conditions.append("date <= :end_date")
            params['end_date'] = end_date

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY symbol, date"

        df = self.execute_query(query, params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def get_features_and_labels(self, symbols: Optional[List[str]] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve features and labels joined together.

        Args:
            symbols: Optional list of symbols to filter
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with features and labels
        """
        query = """
        SELECT f.*, l.y_next
        FROM features f
        LEFT JOIN labels l ON f.symbol = l.symbol AND f.date = l.date
        """

        conditions = []
        params = {}

        if symbols:
            placeholders = ','.join([f':symbol_{i}' for i in range(len(symbols))])
            conditions.append(f"f.symbol IN ({placeholders})")
            for i, symbol in enumerate(symbols):
                params[f'symbol_{i}'] = symbol

        if start_date:
            conditions.append("f.date >= :start_date")
            params['start_date'] = start_date

        if end_date:
            conditions.append("f.date <= :end_date")
            params['end_date'] = end_date

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY f.symbol, f.date"

        df = self.execute_query(query, params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def get_predictions(self, model_name: Optional[str] = None,
                       symbols: Optional[List[str]] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve model predictions.

        Args:
            model_name: Optional model name to filter
            symbols: Optional list of symbols to filter
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with predictions
        """
        query = "SELECT * FROM predictions"
        conditions = []
        params = {}

        if model_name:
            conditions.append("model_name = :model_name")
            params['model_name'] = model_name

        if symbols:
            placeholders = ','.join([f':symbol_{i}' for i in range(len(symbols))])
            conditions.append(f"symbol IN ({placeholders})")
            for i, symbol in enumerate(symbols):
                params[f'symbol_{i}'] = symbol

        if start_date:
            conditions.append("date >= :start_date")
            params['start_date'] = start_date

        if end_date:
            conditions.append("date <= :end_date")
            params['end_date'] = end_date

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY symbol, date"

        df = self.execute_query(query, params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def get_trades(self, model_name: Optional[str] = None,
                   symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Retrieve backtest trades.

        Args:
            model_name: Optional model name to filter
            symbols: Optional list of symbols to filter

        Returns:
            DataFrame with trades
        """
        query = "SELECT * FROM trades"
        conditions = []
        params = {}

        if model_name:
            conditions.append("model_name = :model_name")
            params['model_name'] = model_name

        if symbols:
            placeholders = ','.join([f':symbol_{i}' for i in range(len(symbols))])
            conditions.append(f"symbol IN ({placeholders})")
            for i, symbol in enumerate(symbols):
                params[f'symbol_{i}'] = symbol

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY entry_date"

        df = self.execute_query(query, params)
        if not df.empty:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'])

        return df

    def get_equity_curves(self, model_name: Optional[str] = None,
                         symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Retrieve equity curves.

        Args:
            model_name: Optional model name to filter
            symbols: Optional list of symbols to filter

        Returns:
            DataFrame with equity curve data
        """
        query = "SELECT * FROM equity_curves"
        conditions = []
        params = {}

        if model_name:
            conditions.append("model_name = :model_name")
            params['model_name'] = model_name

        if symbols:
            placeholders = ','.join([f':symbol_{i}' for i in range(len(symbols))])
            conditions.append(f"symbol IN ({placeholders})")
            for i, symbol in enumerate(symbols):
                params[f'symbol_{i}'] = symbol

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date"

        df = self.execute_query(query, params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])

        return df

    def clear_table(self, table_name: str, model_name: Optional[str] = None) -> None:
        """Clear data from a specific table.

        Args:
            table_name: Name of table to clear
            model_name: Optional model name to filter (for model-specific tables)
        """
        query = f"DELETE FROM {table_name}"
        params = {}

        if model_name and table_name in ['predictions', 'trades', 'equity_curves']:
            query += " WHERE model_name = :model_name"
            params['model_name'] = model_name

        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            conn.commit()

        logger.info(f"Cleared {result.rowcount} rows from {table_name}")

    def get_table_info(self) -> Dict[str, int]:
        """Get row counts for all tables.

        Returns:
            Dictionary mapping table names to row counts
        """
        tables = ['bars', 'features', 'labels', 'predictions', 'trades', 'equity_curves']
        info = {}

        for table in tables:
            try:
                result = self.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                info[table] = result.iloc[0]['count']
            except Exception:
                info[table] = 0

        return info

    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        with self.engine.connect() as conn:
            conn.execute(text("VACUUM"))
        logger.info("Database vacuumed")

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()
        logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
