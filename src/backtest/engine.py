"""
Backtest engine for trading strategy simulation.

This module provides a comprehensive backtesting framework for evaluating
trading strategies based on machine learning predictions. It simulates
realistic trading conditions with transaction costs, position sizing,
and risk management.

Key features:
- Realistic trade execution (signal at close, execute at next open)
- Transaction costs and slippage modeling
- Position sizing and risk management
- Long-only and long/short strategies
- Comprehensive performance tracking
- Trade-level analysis and reporting
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.db import DatabaseManager
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Comprehensive backtesting engine for trading strategies."""

    def __init__(self, config: Union[Dict, ConfigManager], db: Optional[DatabaseManager] = None):
        """Initialize backtest engine.

        Args:
            config: Configuration dict or ConfigManager instance
            db: Optional DatabaseManager instance
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.config

        self.db = db or DatabaseManager(self.config.get('data', {}).get('db_path', 'data/trading.db'))

        # Backtest configuration
        self.backtest_config = self.config.get('backtesting', {})

        # Signal thresholds
        self.threshold_long = self.backtest_config.get('threshold_long', 0.55)
        self.threshold_short = self.backtest_config.get('threshold_short', 0.45)
        self.long_only = self.backtest_config.get('long_only', True)

        # Transaction costs
        self.transaction_cost = self.backtest_config.get('transaction_cost', 0.0005)  # 5 bps
        self.slippage = self.backtest_config.get('slippage', 0.0001)  # 1 bp

        # Position sizing
        self.position_size = self.backtest_config.get('position_size', 1.0)
        self.max_positions = self.backtest_config.get('max_positions', 1)

        # Risk management
        self.max_drawdown_stop = self.backtest_config.get('max_drawdown_stop', None)
        self.max_leverage = self.backtest_config.get('max_leverage', 1.0)

        # Execution timing
        self.signal_time = self.backtest_config.get('signal_time', 'close')
        self.execution_time = self.backtest_config.get('execution_time', 'open')

        # Benchmark
        self.benchmark_symbol = self.backtest_config.get('benchmark_symbol', 'SPY')

        logger.info("BacktestEngine initialized")

    def generate_signals(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from model predictions.

        Args:
            predictions_df: DataFrame with predictions (symbol, date, proba, pred_label, model_name)

        Returns:
            DataFrame with trading signals
        """
        signals_df = predictions_df.copy()

        # Generate position signals based on probability thresholds
        def determine_position(row):
            proba = row['proba']

            if proba >= self.threshold_long:
                return 1  # Long position
            elif not self.long_only and proba <= self.threshold_short:
                return -1  # Short position
            else:
                return 0  # Flat/no position

        signals_df['signal'] = signals_df.apply(determine_position, axis=1)

        # Add signal strength (distance from threshold)
        signals_df['signal_strength'] = np.where(
            signals_df['signal'] == 1,
            signals_df['proba'] - self.threshold_long,
            np.where(
                signals_df['signal'] == -1,
                self.threshold_short - signals_df['proba'],
                0
            )
        )

        logger.info(f"Generated signals: {len(signals_df)} total, "
                   f"{(signals_df['signal'] == 1).sum()} long, "
                   f"{(signals_df['signal'] == -1).sum()} short, "
                   f"{(signals_df['signal'] == 0).sum()} flat")

        return signals_df

    def simulate_strategy(self, signals_df: pd.DataFrame,
                         bars_df: Optional[pd.DataFrame] = None,
                         initial_capital: float = 100000.0) -> Dict[str, pd.DataFrame]:
        """Simulate trading strategy execution.

        Args:
            signals_df: DataFrame with trading signals
            bars_df: Optional OHLCV data (will fetch from DB if not provided)
            initial_capital: Initial capital for simulation

        Returns:
            Dictionary with simulation results (trades, equity_curve, metrics)
        """
        logger.info(f"Starting strategy simulation with ${initial_capital:,.0f} initial capital")

        # Get price data if not provided
        if bars_df is None:
            symbols = signals_df['symbol'].unique().tolist()
            start_date = signals_df['date'].min().strftime('%Y-%m-%d')
            end_date = signals_df['date'].max().strftime('%Y-%m-%d')
            bars_df = self.db.get_bars(symbols=symbols, start_date=start_date, end_date=end_date)

        if bars_df.empty:
            raise ValueError("No price data available for simulation")

        # Merge signals with price data
        simulation_data = signals_df.merge(
            bars_df[['symbol', 'date', 'open', 'close', 'adj_close', 'volume']],
            on=['symbol', 'date'],
            how='inner'
        )

        if simulation_data.empty:
            raise ValueError("No matching dates between signals and price data")

        # Sort by symbol and date
        simulation_data = simulation_data.sort_values(['symbol', 'date']).reset_index(drop=True)

        # Initialize tracking variables
        trades = []
        equity_curves = []
        current_positions = {}
        current_capital = initial_capital
        peak_equity = initial_capital

        # Process each symbol separately
        for symbol in simulation_data['symbol'].unique():
            symbol_data = simulation_data[simulation_data['symbol'] == symbol].copy()
            symbol_trades, symbol_equity = self._simulate_symbol(
                symbol, symbol_data, current_capital / len(simulation_data['symbol'].unique())
            )
            trades.extend(symbol_trades)
            equity_curves.extend(symbol_equity)

        # Convert to DataFrames
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        equity_df = pd.DataFrame(equity_curves) if equity_curves else pd.DataFrame()

        # Calculate portfolio-level equity curve
        if not equity_df.empty:
            portfolio_equity = self._calculate_portfolio_equity(equity_df, initial_capital)
        else:
            portfolio_equity = pd.DataFrame()

        results = {
            'trades': trades_df,
            'equity_curves': equity_df,
            'portfolio_equity': portfolio_equity,
            'initial_capital': initial_capital,
            'final_capital': portfolio_equity['equity'].iloc[-1] if not portfolio_equity.empty else initial_capital
        }

        logger.info(f"Simulation completed: {len(trades_df)} trades executed")

        return results

    def _simulate_symbol(self, symbol: str, data: pd.DataFrame, allocated_capital: float) -> Tuple[List[Dict], List[Dict]]:
        """Simulate trading for a single symbol.

        Args:
            symbol: Symbol to simulate
            data: Price and signal data for the symbol
            allocated_capital: Capital allocated to this symbol

        Returns:
            Tuple of (trades_list, equity_curve_list)
        """
        trades = []
        equity_curve = []

        current_position = 0  # Current position size
        entry_price = 0
        entry_date = None
        cash = allocated_capital
        equity = allocated_capital

        for idx, row in data.iterrows():
            date = row['date']
            signal = row['signal']
            open_price = row['open']
            close_price = row['close']

            # Get next day's open price for execution (if available)
            if idx + 1 < len(data):
                next_open = data.iloc[idx + 1]['open']
            else:
                next_open = close_price  # Use current close if no next day

            position_change = 0
            new_position = signal * self.position_size

            # Check if position change is needed
            if new_position != current_position:
                position_change = new_position - current_position

                # Close existing position if needed
                if current_position != 0:
                    exit_trade = self._execute_trade(
                        symbol, entry_date, date, 'exit',
                        entry_price, next_open, abs(current_position),
                        row['model_name'], row['proba']
                    )
                    trades.append(exit_trade)

                    # Update cash
                    pnl = exit_trade['pnl']
                    cash += pnl + (abs(current_position) * entry_price)

                # Open new position if needed
                if new_position != 0:
                    # Calculate position size based on available cash
                    max_shares = cash / next_open
                    actual_position = min(abs(new_position), max_shares) * np.sign(new_position)

                    if abs(actual_position) > 0:
                        entry_price = next_open
                        entry_date = date
                        current_position = actual_position

                        # Update cash (subtract invested amount)
                        invested_amount = abs(actual_position) * entry_price
                        cash -= invested_amount
                    else:
                        current_position = 0

                else:
                    current_position = 0
                    entry_price = 0
                    entry_date = None

            # Calculate current equity
            if current_position != 0:
                position_value = current_position * close_price
                equity = cash + position_value
            else:
                equity = cash

            # Record equity curve
            daily_return = (equity / allocated_capital - 1) if allocated_capital > 0 else 0
            equity_curve.append({
                'symbol': symbol,
                'date': date,
                'equity': equity,
                'cash': cash,
                'position': current_position,
                'position_value': current_position * close_price if current_position != 0 else 0,
                'daily_return': daily_return,
                'cumulative_return': equity / allocated_capital - 1
            })

        # Close final position if open
        if current_position != 0:
            final_row = data.iloc[-1]
            exit_trade = self._execute_trade(
                symbol, entry_date, final_row['date'], 'exit',
                entry_price, final_row['close'], abs(current_position),
                final_row['model_name'], final_row['proba']
            )
            trades.append(exit_trade)

        return trades, equity_curve

    def _execute_trade(self, symbol: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp,
                      trade_type: str, entry_price: float, exit_price: float,
                      quantity: float, model_name: str, signal_proba: float) -> Dict:
        """Execute a trade and calculate PnL.

        Args:
            symbol: Trading symbol
            entry_date: Trade entry date
            exit_date: Trade exit date
            trade_type: 'entry' or 'exit'
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            model_name: Model that generated the signal
            signal_proba: Signal probability

        Returns:
            Trade dictionary
        """
        # Apply transaction costs and slippage
        total_cost_rate = self.transaction_cost + self.slippage

        # Calculate gross PnL
        gross_pnl = (exit_price - entry_price) * quantity

        # Calculate transaction costs
        trade_value = quantity * entry_price
        transaction_costs = trade_value * total_cost_rate * 2  # Entry + exit

        # Net PnL
        net_pnl = gross_pnl - transaction_costs
        net_pnl_pct = net_pnl / trade_value if trade_value > 0 else 0

        trade = {
            'symbol': symbol,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'side': 'long',  # Assume long for now
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'gross_pnl': gross_pnl,
            'transaction_costs': transaction_costs,
            'pnl': net_pnl,
            'pnl_pct': net_pnl_pct,
            'model_name': model_name,
            'signal_proba': signal_proba,
            'trade_duration': (exit_date - entry_date).days,
            'trade_value': trade_value
        }

        return trade

    def _calculate_portfolio_equity(self, equity_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        """Calculate portfolio-level equity curve.

        Args:
            equity_df: Individual symbol equity curves
            initial_capital: Initial capital

        Returns:
            Portfolio equity curve DataFrame
        """
        # Group by date and sum equity across symbols
        portfolio_equity = equity_df.groupby('date').agg({
            'equity': 'sum',
            'daily_return': 'mean',  # Average return across symbols
            'position': 'sum'
        }).reset_index()

        # Calculate cumulative returns
        portfolio_equity['cumulative_return'] = portfolio_equity['equity'] / initial_capital - 1

        # Calculate drawdown
        portfolio_equity['peak'] = portfolio_equity['equity'].expanding().max()
        portfolio_equity['drawdown'] = (portfolio_equity['equity'] - portfolio_equity['peak']) / portfolio_equity['peak']

        return portfolio_equity.sort_values('date')

    def run_backtest(self, model_name: str,
                    symbols: Optional[List[str]] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    initial_capital: float = 100000.0) -> Dict[str, any]:
        """Run complete backtest for a model.

        Args:
            model_name: Name of model to backtest
            symbols: Optional list of symbols to include
            start_date: Optional start date for backtest
            end_date: Optional end date for backtest
            initial_capital: Initial capital

        Returns:
            Dictionary with complete backtest results
        """
        logger.info(f"Running backtest for {model_name}")

        # Get predictions from database
        predictions_df = self.db.get_predictions(
            model_name=model_name,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        if predictions_df.empty:
            raise ValueError(f"No predictions found for model {model_name}")

        # Generate signals
        signals_df = self.generate_signals(predictions_df)

        # Run simulation
        simulation_results = self.simulate_strategy(signals_df, initial_capital=initial_capital)

        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(simulation_results)

        # Calculate benchmark comparison
        benchmark_metrics = self.calculate_benchmark_comparison(
            simulation_results['portfolio_equity'],
            start_date or predictions_df['date'].min().strftime('%Y-%m-%d'),
            end_date or predictions_df['date'].max().strftime('%Y-%m-%d'),
            initial_capital
        )

        # Combine results
        backtest_results = {
            'model_name': model_name,
            'backtest_config': {
                'initial_capital': initial_capital,
                'threshold_long': self.threshold_long,
                'threshold_short': self.threshold_short,
                'long_only': self.long_only,
                'transaction_cost': self.transaction_cost,
                'slippage': self.slippage
            },
            'simulation_results': simulation_results,
            'performance_metrics': performance_metrics,
            'benchmark_comparison': benchmark_metrics,
            'backtest_date': datetime.now().isoformat()
        }

        # Store results in database
        self._store_backtest_results(backtest_results)

        logger.info(f"Backtest completed for {model_name}")
        return backtest_results

    def calculate_performance_metrics(self, simulation_results: Dict) -> Dict[str, float]:
        """Calculate comprehensive performance metrics.

        Args:
            simulation_results: Results from simulate_strategy

        Returns:
            Dictionary of performance metrics
        """
        trades_df = simulation_results['trades']
        portfolio_equity = simulation_results['portfolio_equity']
        initial_capital = simulation_results['initial_capital']
        final_capital = simulation_results['final_capital']

        metrics = {}

        # Basic return metrics
        metrics['total_return'] = (final_capital - initial_capital) / initial_capital
        metrics['final_capital'] = final_capital

        if not portfolio_equity.empty:
            # Time-based metrics
            start_date = portfolio_equity['date'].min()
            end_date = portfolio_equity['date'].max()
            trading_days = len(portfolio_equity)
            years = trading_days / 252  # Approximate trading days per year

            # CAGR
            if years > 0:
                metrics['cagr'] = (final_capital / initial_capital) ** (1 / years) - 1
            else:
                metrics['cagr'] = 0

            # Volatility and Sharpe ratio
            daily_returns = portfolio_equity['daily_return'].dropna()
            if len(daily_returns) > 1:
                metrics['volatility'] = daily_returns.std() * np.sqrt(252)
                metrics['sharpe_ratio'] = (metrics['cagr']) / metrics['volatility'] if metrics['volatility'] > 0 else 0
            else:
                metrics['volatility'] = 0
                metrics['sharpe_ratio'] = 0

            # Drawdown metrics
            metrics['max_drawdown'] = portfolio_equity['drawdown'].min()
            metrics['calmar_ratio'] = abs(metrics['cagr'] / metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else 0

        # Trade-based metrics
        if not trades_df.empty:
            metrics['total_trades'] = len(trades_df)
            metrics['winning_trades'] = (trades_df['pnl'] > 0).sum()
            metrics['losing_trades'] = (trades_df['pnl'] < 0).sum()
            metrics['hit_rate'] = metrics['winning_trades'] / metrics['total_trades']

            # Win/Loss analysis
            winning_trades = trades_df[trades_df['pnl'] > 0]['pnl']
            losing_trades = trades_df[trades_df['pnl'] < 0]['pnl']

            metrics['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
            metrics['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] < 0 else 0

            # Profit factor
            total_wins = winning_trades.sum()
            total_losses = abs(losing_trades.sum())
            metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')

            # Exposure and turnover
            avg_position = trades_df['trade_value'].mean()
            metrics['avg_position_size'] = avg_position / initial_capital
            metrics['turnover'] = trades_df['trade_value'].sum() / initial_capital / years if years > 0 else 0

        else:
            # No trades executed
            for key in ['total_trades', 'winning_trades', 'losing_trades', 'hit_rate',
                       'avg_win', 'avg_loss', 'win_loss_ratio', 'profit_factor',
                       'avg_position_size', 'turnover']:
                metrics[key] = 0

        return metrics

    def calculate_benchmark_comparison(self, portfolio_equity: pd.DataFrame,
                                    start_date: str, end_date: str,
                                    initial_capital: float) -> Dict[str, any]:
        """Calculate benchmark comparison metrics.

        Args:
            portfolio_equity: Portfolio equity curve
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital

        Returns:
            Dictionary with benchmark comparison
        """
        try:
            # Get benchmark data
            benchmark_data = self.db.get_bars(
                symbols=[self.benchmark_symbol],
                start_date=start_date,
                end_date=end_date
            )

            if benchmark_data.empty:
                return {'error': f'No benchmark data available for {self.benchmark_symbol}'}

            # Calculate benchmark returns
            benchmark_data = benchmark_data.sort_values('date')
            benchmark_start_price = benchmark_data.iloc[0]['adj_close']
            benchmark_end_price = benchmark_data.iloc[-1]['adj_close']
            benchmark_total_return = (benchmark_end_price - benchmark_start_price) / benchmark_start_price

            # Calculate benchmark CAGR
            trading_days = len(benchmark_data)
            years = trading_days / 252
            benchmark_cagr = (1 + benchmark_total_return) ** (1 / years) - 1 if years > 0 else 0

            # Calculate benchmark volatility
            benchmark_data['daily_return'] = benchmark_data['adj_close'].pct_change()
            benchmark_volatility = benchmark_data['daily_return'].std() * np.sqrt(252)
            benchmark_sharpe = benchmark_cagr / benchmark_volatility if benchmark_volatility > 0 else 0

            # Calculate benchmark drawdown
            benchmark_data['peak'] = benchmark_data['adj_close'].expanding().max()
            benchmark_data['drawdown'] = (benchmark_data['adj_close'] - benchmark_data['peak']) / benchmark_data['peak']
            benchmark_max_drawdown = benchmark_data['drawdown'].min()

            # Strategy metrics for comparison
            if not portfolio_equity.empty:
                strategy_total_return = portfolio_equity.iloc[-1]['cumulative_return']
                strategy_cagr = (1 + strategy_total_return) ** (1 / years) - 1 if years > 0 else 0
                strategy_volatility = portfolio_equity['daily_return'].std() * np.sqrt(252)
                strategy_sharpe = strategy_cagr / strategy_volatility if strategy_volatility > 0 else 0
                strategy_max_drawdown = portfolio_equity['drawdown'].min()
            else:
                strategy_total_return = 0
                strategy_cagr = 0
                strategy_volatility = 0
                strategy_sharpe = 0
                strategy_max_drawdown = 0

            comparison = {
                'benchmark_symbol': self.benchmark_symbol,
                'benchmark_metrics': {
                    'total_return': benchmark_total_return,
                    'cagr': benchmark_cagr,
                    'volatility': benchmark_volatility,
                    'sharpe_ratio': benchmark_sharpe,
                    'max_drawdown': benchmark_max_drawdown
                },
                'strategy_metrics': {
                    'total_return': strategy_total_return,
                    'cagr': strategy_cagr,
                    'volatility': strategy_volatility,
                    'sharpe_ratio': strategy_sharpe,
                    'max_drawdown': strategy_max_drawdown
                },
                'relative_metrics': {
                    'excess_return': strategy_total_return - benchmark_total_return,
                    'excess_cagr': strategy_cagr - benchmark_cagr,
                    'sharpe_improvement': strategy_sharpe - benchmark_sharpe,
                    'drawdown_improvement': strategy_max_drawdown - benchmark_max_drawdown  # Negative is better
                }
            }

            return comparison

        except Exception as e:
            logger.error(f"Failed to calculate benchmark comparison: {e}")
            return {'error': str(e)}

    def _store_backtest_results(self, backtest_results: Dict) -> None:
        """Store backtest results in database.

        Args:
            backtest_results: Complete backtest results
        """
        try:
            # Store trades
            trades_df = backtest_results['simulation_results']['trades']
            if not trades_df.empty:
                # Clear existing trades for this model
                self.db.clear_table('trades', model_name=backtest_results['model_name'])
                self.db.insert_trades(trades_df)

            # Store equity curves
            equity_df = backtest_results['simulation_results']['portfolio_equity']
            if not equity_df.empty:
                # Add model name
                equity_df['model_name'] = backtest_results['model_name']
                equity_df['symbol'] = 'PORTFOLIO'  # Portfolio-level equity

                # Clear existing equity curves for this model
                self.db.clear_table('equity_curves', model_name=backtest_results['model_name'])
                self.db.insert_equity_curves(equity_df)

            logger.info(f"Stored backtest results for {backtest_results['model_name']}")

        except Exception as e:
            logger.error(f"Failed to store backtest results: {e}")

    def threshold_sweep(self, model_name: str,
                       threshold_range: Tuple[float, float] = (0.4, 0.7),
                       step: float = 0.05,
                       metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """Perform threshold sweep analysis.

        Args:
            model_name: Model to analyze
            threshold_range: Range of thresholds to test
            step: Step size for threshold sweep
            metric: Metric to optimize

        Returns:
            DataFrame with threshold sweep results
        """
        logger.info(f"Performing threshold sweep for {model_name}")

        # Get predictions
        predictions_df = self.db.get_predictions(model_name=model_name)
        if predictions_df.empty:
            raise ValueError(f"No predictions found for model {model_name}")

        results = []
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)

        original_threshold = self.threshold_long

        for threshold in thresholds:
            try:
                # Update threshold
                self.threshold_long = threshold
                self.threshold_short = 1 - threshold

                # Run backtest
                signals_df = self.generate_signals(predictions_df)
                simulation_results = self.simulate_strategy(signals_df)
                performance_metrics = self.calculate_performance_metrics(simulation_results)

                # Record results
                result = {
                    'threshold': threshold,
                    'total_trades': performance_metrics.get('total_trades', 0),
                    'hit_rate': performance_metrics.get('hit_rate', 0),
                    'total_return': performance_metrics.get('total_return', 0),
                    'cagr': performance_metrics.get('cagr', 0),
                    'volatility': performance_metrics.get('volatility', 0),
                    'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                    'max_drawdown': performance_metrics.get('max_drawdown', 0),
                    'calmar_ratio': performance_metrics.get('calmar_ratio', 0)
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Threshold sweep failed at threshold {threshold}: {e}")

        # Restore original threshold
        self.threshold_long = original_threshold
        self.threshold_short = 1 - original_threshold

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Find optimal threshold
            if metric in results_df.columns:
                optimal_idx = results_df[metric].idxmax()
                optimal_threshold = results_df.loc[optimal_idx, 'threshold']
                optimal_value = results_df.loc[optimal_idx, metric]
                logger.info(f"Optimal threshold: {optimal_threshold:.3f} "
                           f"({metric}: {optimal_value:.4f})")

        return results_df
