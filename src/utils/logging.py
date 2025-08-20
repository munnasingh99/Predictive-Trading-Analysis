"""
Logging utility module for predictive trading signals.

This module provides centralized logging configuration and utilities for the
trading system. It sets up structured logging with appropriate formatters,
handlers, and log levels for different components of the system.

Key features:
- Centralized logging configuration
- File and console logging
- Component-specific loggers
- Performance logging utilities
- Error tracking and alerts
- Log rotation and management
"""

import logging
import logging.config
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class TradingFormatter(logging.Formatter):
    """Custom formatter for trading system logs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with additional context."""
        # Add timestamp if not present
        if not hasattr(record, 'timestamp'):
            record.timestamp = datetime.now().isoformat()

        # Add component info
        if hasattr(record, 'component'):
            record.component_info = f"[{record.component}]"
        else:
            record.component_info = ""

        return super().format(record)


class ComponentFilter(logging.Filter):
    """Filter logs by component name."""

    def __init__(self, component: str):
        super().__init__()
        self.component = component

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter records by component."""
        return getattr(record, 'component', '') == self.component


class LoggingManager:
    """Manages logging configuration for the trading system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize logging manager.

        Args:
            config: Logging configuration dictionary
        """
        self.config = config or self._get_default_config()
        self._setup_logging()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/trading_pipeline.log',
            'console': True,
            'max_bytes': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'components': {
                'data': 'DEBUG',
                'features': 'INFO',
                'modeling': 'INFO',
                'backtest': 'INFO',
                'reporting': 'INFO'
            }
        }

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create logs directory
        log_file = Path(self.config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config['level'].upper()))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Create formatter
        formatter = TradingFormatter(
            fmt=self.config['format'],
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler with rotation
        if self.config['file']:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.config['file'],
                maxBytes=self.config.get('max_bytes', 10 * 1024 * 1024),
                backupCount=self.config.get('backup_count', 5),
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Console handler
        if self.config.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Component-specific loggers
        components = self.config.get('components', {})
        for component, level in components.items():
            logger = logging.getLogger(f'trading.{component}')
            logger.setLevel(getattr(logging, level.upper()))

    def get_logger(self, name: str, component: Optional[str] = None) -> logging.Logger:
        """Get logger instance for component.

        Args:
            name: Logger name
            component: Component name for filtering

        Returns:
            Logger instance
        """
        logger = logging.getLogger(name)

        if component:
            # Add component info to all records
            old_factory = logging.getLogRecordFactory()

            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                record.component = component
                return record

            logging.setLogRecordFactory(record_factory)

        return logger

    def set_level(self, level: str, component: Optional[str] = None) -> None:
        """Set logging level.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            component: Optional component name
        """
        level_obj = getattr(logging, level.upper())

        if component:
            logger = logging.getLogger(f'trading.{component}')
            logger.setLevel(level_obj)
        else:
            logging.getLogger().setLevel(level_obj)

    def add_file_handler(self, filename: str, level: str = 'INFO',
                        component: Optional[str] = None) -> None:
        """Add additional file handler.

        Args:
            filename: Log file path
            level: Logging level
            component: Optional component filter
        """
        # Create handler
        handler = logging.handlers.RotatingFileHandler(
            filename=filename,
            maxBytes=self.config.get('max_bytes', 10 * 1024 * 1024),
            backupCount=self.config.get('backup_count', 5),
            encoding='utf-8'
        )

        # Set level and formatter
        handler.setLevel(getattr(logging, level.upper()))
        formatter = TradingFormatter(fmt=self.config['format'])
        handler.setFormatter(formatter)

        # Add filter if component specified
        if component:
            handler.addFilter(ComponentFilter(component))

        # Add to root logger
        logging.getLogger().addHandler(handler)

    def setup_component_logger(self, component: str, level: str = 'INFO',
                              separate_file: bool = False) -> logging.Logger:
        """Setup logger for specific component.

        Args:
            component: Component name
            level: Logging level
            separate_file: Whether to create separate log file

        Returns:
            Component logger
        """
        logger_name = f'trading.{component}'
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))

        if separate_file:
            log_file = Path(self.config['file']).parent / f'{component}.log'
            self.add_file_handler(str(log_file), level, component)

        return logger


class PerformanceLogger:
    """Performance logging utilities."""

    def __init__(self, logger: logging.Logger):
        """Initialize performance logger.

        Args:
            logger: Logger instance to use
        """
        self.logger = logger
        self.start_times = {}

    def start_timer(self, operation: str) -> None:
        """Start timing an operation.

        Args:
            operation: Operation name
        """
        self.start_times[operation] = datetime.now()
        self.logger.debug(f"Started {operation}")

    def end_timer(self, operation: str, log_level: str = 'INFO') -> float:
        """End timing an operation and log duration.

        Args:
            operation: Operation name
            log_level: Log level for duration message

        Returns:
            Duration in seconds
        """
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0

        start_time = self.start_times.pop(operation)
        duration = (datetime.now() - start_time).total_seconds()

        log_func = getattr(self.logger, log_level.lower())
        log_func(f"Completed {operation} in {duration:.2f} seconds")

        return duration

    def log_memory_usage(self, operation: str) -> None:
        """Log memory usage for operation.

        Args:
            operation: Operation name
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            self.logger.debug(f"{operation} - Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")


class ErrorTracker:
    """Tracks and logs errors with context."""

    def __init__(self, logger: logging.Logger):
        """Initialize error tracker.

        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.error_counts = {}

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                 component: Optional[str] = None) -> None:
        """Log error with context information.

        Args:
            error: Exception that occurred
            context: Additional context information
            component: Component where error occurred
        """
        error_type = type(error).__name__
        error_msg = str(error)

        # Track error count
        error_key = f"{component}.{error_type}" if component else error_type
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Build log message
        log_msg = f"Error in {component}: {error_type}: {error_msg}" if component else f"{error_type}: {error_msg}"

        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            log_msg += f" (Context: {context_str})"

        self.logger.error(log_msg, exc_info=True)

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts.

        Returns:
            Dictionary of error counts by type
        """
        return self.error_counts.copy()

    def clear_error_counts(self) -> None:
        """Clear error count tracking."""
        self.error_counts.clear()


def setup_logging(config: Optional[Dict[str, Any]] = None) -> LoggingManager:
    """Setup logging for the trading system.

    Args:
        config: Optional logging configuration

    Returns:
        LoggingManager instance
    """
    return LoggingManager(config)


def get_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """Get logger instance.

    Args:
        name: Logger name
        component: Optional component name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if component and not logger.handlers:
        # If no handlers and component specified, setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    return logger


def log_function_call(logger: logging.Logger, level: str = 'DEBUG'):
    """Decorator to log function calls.

    Args:
        logger: Logger instance
        level: Log level

    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            log_func = getattr(logger, level.lower())

            # Log function entry
            log_func(f"Entering {func_name}")

            try:
                result = func(*args, **kwargs)
                log_func(f"Exiting {func_name}")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", exc_info=True)
                raise

        return wrapper
    return decorator


def configure_matplotlib_logging():
    """Configure matplotlib logging to reduce noise."""
    import matplotlib
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)

    # Suppress specific matplotlib warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def configure_third_party_logging():
    """Configure third-party library logging."""
    # Matplotlib
    configure_matplotlib_logging()

    # Urllib3 (used by requests)
    urllib3_logger = logging.getLogger('urllib3')
    urllib3_logger.setLevel(logging.WARNING)

    # Requests
    requests_logger = logging.getLogger('requests')
    requests_logger.setLevel(logging.WARNING)

    # yfinance
    yfinance_logger = logging.getLogger('yfinance')
    yfinance_logger.setLevel(logging.WARNING)

    # SQLAlchemy
    sqlalchemy_logger = logging.getLogger('sqlalchemy')
    sqlalchemy_logger.setLevel(logging.WARNING)
