"""
Utilities package for predictive trading signals.

This package provides various utility modules for the trading system including:
- Configuration management with YAML support
- Centralized logging configuration
- File I/O operations and data handling
- Progress tracking and performance monitoring
- Error handling and validation utilities

Key features:
- Centralized configuration with environment variable overrides
- Structured logging with component-specific loggers
- Safe file operations with error handling
- Data serialization and validation utilities
- Progress tracking for long-running operations

Main components:
- ConfigManager: Centralized configuration management
- LoggingManager: Logging setup and utilities
- FileManager: File operations and path management
- DataSerializer: Data serialization utilities
- CSVHandler: CSV file handling and validation
"""

from src.utils.config import ConfigManager, load_config, create_default_config
from src.utils.logging import LoggingManager, setup_logging, get_logger
from src.utils.io import (
    FileManager,
    DataSerializer,
    CSVHandler,
    ArchiveManager,
    ProgressTracker,
    ensure_path_exists,
    format_file_size,
    get_file_size_mb
)

__all__ = [
    # Configuration
    "ConfigManager",
    "load_config",
    "create_default_config",

    # Logging
    "LoggingManager",
    "setup_logging",
    "get_logger",

    # File I/O
    "FileManager",
    "DataSerializer",
    "CSVHandler",
    "ArchiveManager",
    "ProgressTracker",
    "ensure_path_exists",
    "format_file_size",
    "get_file_size_mb",
]
