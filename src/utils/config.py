"""
Configuration management utility for predictive trading signals.

This module provides centralized configuration management with YAML file support,
environment variable overrides, and validation. It handles loading configuration
from files, merging with defaults, and providing easy access to configuration
parameters throughout the application.

Key features:
- YAML configuration file support
- Environment variable overrides
- Configuration validation
- Default value management
- Hierarchical configuration access
- Configuration persistence
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration management."""

    def __init__(self, config_path: Optional[str] = None, env_prefix: str = "TRADING_"):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file
            env_prefix: Prefix for environment variables
        """
        self.config_path = Path(config_path) if config_path else Path("config/default.yaml")
        self.env_prefix = env_prefix
        self.config = {}

        # Load configuration
        self.load_config()

        logger.info(f"ConfigManager initialized with config: {self.config_path}")

    def load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Start with default configuration
        self.config = self._get_default_config()

        # Load from YAML file
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        self.config = self._merge_configs(self.config, yaml_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration file {self.config_path}: {e}")
                logger.info("Using default configuration")

        # Override with environment variables
        self._load_env_overrides()

        # Validate configuration
        self._validate_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values.

        Returns:
            Dictionary with default configuration
        """
        return {
            'data': {
                'symbols': ['SPY'],
                'start_date': '2010-01-01',
                'end_date': '2024-12-31',
                'source': 'yfinance',
                'csv_dir': 'data/raw',
                'db_path': 'data/trading.db',
                'min_trading_days': 252
            },
            'features': {
                'sma_periods': [5, 10, 20],
                'ema_periods': [5, 10, 20],
                'rsi_period': 14,
                'macd_params': {'fast': 12, 'slow': 26, 'signal': 9},
                'stoch_k_period': 14,
                'atr_period': 14,
                'returns_periods': [1, 5],
                'volatility_period': 20,
                'zscore_period': 20,
                'momentum_period': 10,
                'stats_periods': {'skew': 60, 'kurtosis': 60},
                'drop_correlated': True,
                'correlation_threshold': 0.95
            },
            'labeling': {
                'target_type': 'binary',
                'lookahead_days': 1,
                'min_return_threshold': 0.0
            },
            'modeling': {
                'train_end': '2019-12-31',
                'validation_start': '2015-01-01',
                'test_start': '2020-01-01',
                'cv_method': 'TimeSeriesSplit',
                'cv_folds': 5,
                'cv_gap': 1,
                'models': {
                    'logreg': {
                        'class_weight': 'balanced',
                        'C': 1.0,
                        'max_iter': 1000,
                        'random_state': 42
                    },
                    'rf': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'min_samples_split': 10,
                        'min_samples_leaf': 5,
                        'class_weight': 'balanced',
                        'random_state': 42,
                        'n_jobs': -1
                    }
                },
                'scale_features': True,
                'handle_missing': 'drop',
                'model_dir': 'models',
                'save_predictions': True
            },
            'backtesting': {
                'threshold_long': 0.55,
                'threshold_short': 0.45,
                'long_only': True,
                'transaction_cost': 0.0005,
                'slippage': 0.0001,
                'position_size': 1.0,
                'max_positions': 1,
                'max_drawdown_stop': None,
                'max_leverage': 1.0,
                'signal_time': 'close',
                'execution_time': 'open',
                'benchmark_symbol': 'SPY'
            },
            'reporting': {
                'output_dir': 'reports',
                'report_name': 'trading_report.html',
                'metrics': [
                    'total_return', 'cagr', 'sharpe_ratio', 'max_drawdown',
                    'calmar_ratio', 'hit_rate', 'avg_win', 'avg_loss',
                    'win_loss_ratio', 'turnover', 'exposure'
                ],
                'risk_free_rate': 0.0,
                'plot_equity_curve': True,
                'plot_drawdown': True,
                'plot_returns_distribution': True,
                'plot_feature_importance': True,
                'plot_threshold_sweep': True,
                'round_decimals': 4
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/trading_pipeline.log',
                'console': True
            },
            'system': {
                'random_seed': 42,
                'n_jobs': -1,
                'memory_limit': None,
                'parallel_backend': 'threading',
                'cache_features': True,
                'cache_dir': 'data/cache'
            },
            'validation': {
                'check_missing_data': True,
                'check_future_leakage': True,
                'check_data_alignment': True,
                'min_test_accuracy': 0.51,
                'max_overfitting_ratio': 2.0,
                'min_sharpe_ratio': 0.0,
                'max_drawdown_threshold': 0.50
            },
            'development': {
                'debug_mode': False,
                'save_intermediate': False,
                'profile_performance': False,
                'verbose_logging': False
            }
        }

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        merged = base.copy()

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        env_overrides = {}

        for env_var, env_value in os.environ.items():
            if env_var.startswith(self.env_prefix):
                # Remove prefix and convert to lowercase
                config_key = env_var[len(self.env_prefix):].lower()

                # Convert environment variable to nested dict structure
                keys = config_key.split('_')
                current = env_overrides

                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                # Type conversion
                final_key = keys[-1]
                try:
                    # Try to convert to appropriate type
                    if env_value.lower() in ('true', 'false'):
                        current[final_key] = env_value.lower() == 'true'
                    elif env_value.replace('.', '').replace('-', '').isdigit():
                        current[final_key] = float(env_value) if '.' in env_value else int(env_value)
                    else:
                        current[final_key] = env_value
                except (ValueError, AttributeError):
                    current[final_key] = env_value

        if env_overrides:
            self.config = self._merge_configs(self.config, env_overrides)
            logger.info("Applied environment variable overrides")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        validation_errors = []

        # Data validation
        data_config = self.config.get('data', {})
        if not data_config.get('symbols'):
            validation_errors.append("No symbols specified in data.symbols")

        if not data_config.get('start_date') or not data_config.get('end_date'):
            validation_errors.append("start_date and end_date must be specified")

        # Model validation
        modeling_config = self.config.get('modeling', {})
        if not modeling_config.get('models'):
            validation_errors.append("No models specified in modeling.models")

        # Backtest validation
        backtest_config = self.config.get('backtesting', {})
        threshold_long = backtest_config.get('threshold_long', 0.5)
        threshold_short = backtest_config.get('threshold_short', 0.5)

        if threshold_long <= threshold_short:
            validation_errors.append("threshold_long must be greater than threshold_short")

        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Configuration validation passed")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation, e.g., 'data.symbols')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        logger.debug(f"Set configuration {key} = {value}")

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary of values.

        Args:
            updates: Dictionary of configuration updates
        """
        self.config = self._merge_configs(self.config, updates)
        logger.info("Configuration updated")

    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to YAML file.

        Args:
            output_path: Optional output path (defaults to original config path)
        """
        output_path = Path(output_path) if output_path else self.config_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2, sort_keys=True)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {output_path}: {e}")
            raise

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name

        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})

    def list_keys(self, section: Optional[str] = None) -> List[str]:
        """List all configuration keys in a section.

        Args:
            section: Optional section name (lists all top-level keys if None)

        Returns:
            List of configuration keys
        """
        if section:
            config_section = self.get_section(section)
            return list(config_section.keys())
        else:
            return list(self.config.keys())

    def validate_section(self, section: str, required_keys: List[str]) -> bool:
        """Validate that a section contains required keys.

        Args:
            section: Section name
            required_keys: List of required keys

        Returns:
            True if all required keys are present

        Raises:
            ValueError: If required keys are missing
        """
        section_config = self.get_section(section)
        missing_keys = [key for key in required_keys if key not in section_config]

        if missing_keys:
            error_msg = f"Missing required keys in section '{section}': {missing_keys}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return True

    def create_config_template(self, output_path: str) -> None:
        """Create a configuration template file.

        Args:
            output_path: Path for template file
        """
        template_config = self._get_default_config()

        # Add comments to template
        template_content = """# Predictive Trading Signals Configuration Template
#
# This file contains all configuration options for the trading system.
# Uncomment and modify values as needed.
#
# Environment variables can override any setting using the format:
# TRADING_<SECTION>_<KEY>=value
# Example: TRADING_DATA_SYMBOLS=SPY,AAPL,MSFT

"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
                yaml.dump(template_config, f, default_flow_style=False, indent=2, sort_keys=True)
            logger.info(f"Configuration template created: {output_path}")
        except Exception as e:
            logger.error(f"Failed to create configuration template: {e}")
            raise

    def print_config(self, section: Optional[str] = None) -> None:
        """Print configuration to console.

        Args:
            section: Optional section to print (prints all if None)
        """
        if section:
            config_to_print = {section: self.get_section(section)}
        else:
            config_to_print = self.config

        print("Current Configuration:")
        print("=" * 50)
        print(yaml.dump(config_to_print, default_flow_style=False, indent=2, sort_keys=True))
        print("=" * 50)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration dictionary

        Raises:
            ValueError: If model configuration not found
        """
        models_config = self.get('modeling.models', {})

        if model_name not in models_config:
            available_models = list(models_config.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")

        return models_config[model_name]

    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled.

        Returns:
            True if debug mode is enabled
        """
        return self.get('development.debug_mode', False)

    def get_log_level(self) -> str:
        """Get logging level.

        Returns:
            Logging level string
        """
        return self.get('logging.level', 'INFO').upper()

    def get_random_seed(self) -> int:
        """Get random seed for reproducibility.

        Returns:
            Random seed integer
        """
        return self.get('system.random_seed', 42)

    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        return f"ConfigManager(config_path='{self.config_path}', sections={list(self.config.keys())})"

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting of configuration."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        return self.get(key) is not None


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Convenience function to load configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path=config_path)


def create_default_config(output_path: str = "config/default.yaml") -> None:
    """Create default configuration file.

    Args:
        output_path: Path for default configuration file
    """
    config_manager = ConfigManager()
    config_manager.save_config(output_path)
    logger.info(f"Default configuration created: {output_path}")
