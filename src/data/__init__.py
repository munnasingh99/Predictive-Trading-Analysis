"""
Data package for predictive trading signals.

This package handles all data-related operations including:
- Market data ingestion from yfinance and CSV sources
- Database management with SQLite
- Data validation and cleaning
- Storage and retrieval operations

Main components:
- DatabaseManager: Handles SQLite database operations
- DataIngestor: Fetches and processes market data
"""

from src.data.db import DatabaseManager
from src.data.ingest import DataIngestor

__all__ = [
    "DatabaseManager",
    "DataIngestor",
]
