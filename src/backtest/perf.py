"""
Performance analysis module for backtesting results.

This module provides comprehensive performance analysis and reporting for
trading strategy backtests. It calculates various risk-adjusted metrics,
generates performance visualizations, and provides detailed trade analysis.

Key features:
- Risk-adjusted performance metrics (Sharpe, Calmar, Sortino ratios)
- Drawdown analysis and risk metrics
- Trade analysis and statistics
- Rolling performance analysis
- Risk attribution and factor analysis
- Performance comparison utilities
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Comprehensive performance analysis for trading strategies."""

    def __init__(self, risk_free_rate: float = 0.0):
        """Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive return-based metrics.

        Args:
            equity_curve: DataFrame with equity curve data

        Returns:
            Dictionary of return metrics
        """
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}

        # Ensure equity curve is sorted by date
        equity_curve = equity_curve.sort_values('date').copy()

        # Calculate returns
        if 'daily_return' not in equity_curve.columns:
            equity_curve['daily_return'] = equity_curve['equity'].pct_change()

        returns = equity_curve['daily_return'].dropna()

        if len(returns) == 0:
            return {}

        # Basic return metrics
        metrics = {}

        # Total return
        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]
        metrics['total_return'] = (final_equity - initial_equity) / initial_equity

        # Time-based metrics
        start_date = equity_curve['date'].min()
        end_date = equity_curve['date'].max()
        days = (end_date - start_date).days
        years = days / 365.25

        # Annualized return (CAGR)
        if years > 0:
            metrics['cagr'] = (final_equity / initial_equity) ** (1 / years) - 1
        else:
            metrics['cagr'] = 0

        # Risk metrics
        metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
        metrics['downside_volatility'] = returns[returns < 0].std() * np.sqrt(252)

        # Risk-adjusted returns
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = (metrics['cagr'] - self.risk_free_rate) / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0

        if metrics['downside_volatility'] > 0:
            metrics['sortino_ratio'] = (metrics['cagr'] - self.risk_free_rate) / metrics['downside_volatility']
        else:
            metrics['sortino_ratio'] = 0

        # Distribution metrics
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()

        # VaR and CVaR
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['var_99'] = np.percentile(returns, 1)
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()

        return metrics

    def calculate_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate drawdown-based risk metrics.

        Args:
            equity_curve: DataFrame with equity curve data

        Returns:
            Dictionary of drawdown metrics
        """
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}

        equity_curve = equity_curve.sort_values('date').copy()
        equity = equity_curve['equity']

        # Calculate running maximum (peak)
        running_max = equity.expanding().max()

        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        drawdown_pct = drawdown * 100

        metrics = {}

        # Maximum drawdown
        metrics['max_drawdown'] = drawdown.min()
        metrics['max_drawdown_pct'] = drawdown_pct.min()

        # Calmar ratio
        cagr = self._calculate_cagr(equity_curve)
        if metrics['max_drawdown'] < 0:
            metrics['calmar_ratio'] = abs(cagr / metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0

        # Drawdown duration analysis
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            # Find drawdown periods
            drawdown_periods = self._find_drawdown_periods(drawdown, equity_curve['date'])

            if drawdown_periods:
                durations = [period['duration_days'] for period in drawdown_periods]
                depths = [period['max_drawdown'] for period in drawdown_periods]

                metrics['avg_drawdown_duration'] = np.mean(durations)
                metrics['max_drawdown_duration'] = max(durations)
                metrics['avg_drawdown_depth'] = np.mean(depths)
                metrics['num_drawdown_periods'] = len(drawdown_periods)

                # Recovery metrics
                recoveries = [period['recovery_days'] for period in drawdown_periods if period['recovery_days'] is not None]
                if recoveries:
                    metrics['avg_recovery_time'] = np.mean(recoveries)
                    metrics['max_recovery_time'] = max(recoveries)
                else:
                    metrics['avg_recovery_time'] = None
                    metrics['max_recovery_time'] = None
            else:
                metrics['avg_drawdown_duration'] = 0
                metrics['max_drawdown_duration'] = 0
                metrics['avg_drawdown_depth'] = 0
                metrics['num_drawdown_periods'] = 0
                metrics['avg_recovery_time'] = 0
                metrics['max_recovery_time'] = 0
        else:
            # No drawdowns
            metrics['avg_drawdown_duration'] = 0
            metrics['max_drawdown_duration'] = 0
            metrics['avg_drawdown_depth'] = 0
            metrics['num_drawdown_periods'] = 0
            metrics['avg_recovery_time'] = 0
            metrics['max_recovery_time'] = 0

        return metrics

    def analyze_trades(self, trades_df: pd.DataFrame) -> Dict[str, Union[float, int]]:
        """Analyze individual trades for detailed statistics.

        Args:
            trades_df: DataFrame with trade data

        Returns:
            Dictionary of trade analysis metrics
        """
        if trades_df.empty:
            return {'total_trades': 0}

        metrics = {}

        # Basic trade statistics
        metrics['total_trades'] = len(trades_df)
        metrics['winning_trades'] = (trades_df['pnl'] > 0).sum()
        metrics['losing_trades'] = (trades_df['pnl'] < 0).sum()
        metrics['breakeven_trades'] = (trades_df['pnl'] == 0).sum()

        # Win rate
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
        metrics['loss_rate'] = metrics['losing_trades'] / metrics['total_trades']

        # PnL statistics
        metrics['total_pnl'] = trades_df['pnl'].sum()
        metrics['avg_pnl_per_trade'] = trades_df['pnl'].mean()
        metrics['median_pnl_per_trade'] = trades_df['pnl'].median()

        # Win/Loss analysis
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        if not winning_trades.empty:
            metrics['avg_win'] = winning_trades['pnl'].mean()
            metrics['median_win'] = winning_trades['pnl'].median()
            metrics['max_win'] = winning_trades['pnl'].max()
            metrics['min_win'] = winning_trades['pnl'].min()
        else:
            metrics['avg_win'] = 0
            metrics['median_win'] = 0
            metrics['max_win'] = 0
            metrics['min_win'] = 0

        if not losing_trades.empty:
            metrics['avg_loss'] = losing_trades['pnl'].mean()
            metrics['median_loss'] = losing_trades['pnl'].median()
            metrics['max_loss'] = losing_trades['pnl'].min()  # Most negative
            metrics['min_loss'] = losing_trades['pnl'].max()  # Least negative
        else:
            metrics['avg_loss'] = 0
            metrics['median_loss'] = 0
            metrics['max_loss'] = 0
            metrics['min_loss'] = 0

        # Win/Loss ratio
        if metrics['avg_loss'] != 0:
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss'])
        else:
            metrics['win_loss_ratio'] = float('inf') if metrics['avg_win'] > 0 else 0

        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0

        if gross_loss > 0:
            metrics['profit_factor'] = gross_profit / gross_loss
        else:
            metrics['profit_factor'] = float('inf') if gross_profit > 0 else 0

        # Trade duration analysis
        if 'trade_duration' in trades_df.columns:
            durations = trades_df['trade_duration'].dropna()
            if not durations.empty:
                metrics['avg_trade_duration'] = durations.mean()
                metrics['median_trade_duration'] = durations.median()
                metrics['max_trade_duration'] = durations.max()
                metrics['min_trade_duration'] = durations.min()

        # Consecutive wins/losses
        metrics.update(self._analyze_consecutive_trades(trades_df))

        # Trade size analysis
        if 'trade_value' in trades_df.columns:
            trade_values = trades_df['trade_value'].dropna()
            if not trade_values.empty:
                metrics['avg_trade_size'] = trade_values.mean()
                metrics['median_trade_size'] = trade_values.median()
                metrics['max_trade_size'] = trade_values.max()
                metrics['min_trade_size'] = trade_values.min()

        return metrics

    def calculate_rolling_metrics(self, equity_curve: pd.DataFrame,
                                 window_days: int = 252) -> pd.DataFrame:
        """Calculate rolling performance metrics.

        Args:
            equity_curve: DataFrame with equity curve data
            window_days: Rolling window size in days

        Returns:
            DataFrame with rolling metrics
        """
        if equity_curve.empty or len(equity_curve) < window_days:
            return pd.DataFrame()

        equity_curve = equity_curve.sort_values('date').copy()

        # Calculate daily returns if not present
        if 'daily_return' not in equity_curve.columns:
            equity_curve['daily_return'] = equity_curve['equity'].pct_change()

        returns = equity_curve['daily_return'].fillna(0)

        rolling_metrics = pd.DataFrame(index=equity_curve.index)
        rolling_metrics['date'] = equity_curve['date']

        # Rolling return metrics
        rolling_metrics['rolling_return'] = returns.rolling(window_days).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )

        rolling_metrics['rolling_volatility'] = returns.rolling(window_days).std() * np.sqrt(252)

        # Rolling Sharpe ratio
        rolling_mean = returns.rolling(window_days).mean() * 252 - self.risk_free_rate
        rolling_vol = rolling_metrics['rolling_volatility']
        rolling_metrics['rolling_sharpe'] = rolling_mean / rolling_vol

        # Rolling maximum drawdown
        rolling_metrics['rolling_max_dd'] = equity_curve['equity'].rolling(
            window_days
        ).apply(self._rolling_max_drawdown, raw=False)

        # Rolling win rate (if trade data available)
        if 'position' in equity_curve.columns:
            position_changes = equity_curve['position'].diff() != 0
            rolling_metrics['rolling_trades'] = position_changes.rolling(window_days).sum()

        return rolling_metrics.dropna()

    def benchmark_comparison(self, strategy_equity: pd.DataFrame,
                           benchmark_equity: pd.DataFrame) -> Dict[str, float]:
        """Compare strategy performance against benchmark.

        Args:
            strategy_equity: Strategy equity curve
            benchmark_equity: Benchmark equity curve

        Returns:
            Dictionary with comparison metrics
        """
        if strategy_equity.empty or benchmark_equity.empty:
            return {}

        # Align dates
        merged = pd.merge(
            strategy_equity[['date', 'equity']].rename(columns={'equity': 'strategy_equity'}),
            benchmark_equity[['date', 'equity']].rename(columns={'equity': 'benchmark_equity'}),
            on='date', how='inner'
        )

        if merged.empty:
            return {}

        # Calculate returns
        merged['strategy_return'] = merged['strategy_equity'].pct_change()
        merged['benchmark_return'] = merged['benchmark_equity'].pct_change()
        merged = merged.dropna()

        if merged.empty:
            return {}

        comparison = {}

        # Basic comparison
        strategy_total_return = (merged['strategy_equity'].iloc[-1] / merged['strategy_equity'].iloc[0]) - 1
        benchmark_total_return = (merged['benchmark_equity'].iloc[-1] / merged['benchmark_equity'].iloc[0]) - 1

        comparison['strategy_total_return'] = strategy_total_return
        comparison['benchmark_total_return'] = benchmark_total_return
        comparison['excess_return'] = strategy_total_return - benchmark_total_return

        # Risk-adjusted metrics
        strategy_vol = merged['strategy_return'].std() * np.sqrt(252)
        benchmark_vol = merged['benchmark_return'].std() * np.sqrt(252)

        comparison['strategy_volatility'] = strategy_vol
        comparison['benchmark_volatility'] = benchmark_vol

        # Sharpe ratios
        days_per_year = 252
        years = len(merged) / days_per_year

        if years > 0:
            strategy_cagr = (1 + strategy_total_return) ** (1 / years) - 1
            benchmark_cagr = (1 + benchmark_total_return) ** (1 / years) - 1
        else:
            strategy_cagr = 0
            benchmark_cagr = 0

        comparison['strategy_sharpe'] = (strategy_cagr - self.risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
        comparison['benchmark_sharpe'] = (benchmark_cagr - self.risk_free_rate) / benchmark_vol if benchmark_vol > 0 else 0

        # Beta and Alpha
        if benchmark_vol > 0:
            covariance = merged['strategy_return'].cov(merged['benchmark_return'])
            comparison['beta'] = covariance / (benchmark_vol / np.sqrt(252)) ** 2
            comparison['alpha'] = strategy_cagr - (self.risk_free_rate + comparison['beta'] * (benchmark_cagr - self.risk_free_rate))
        else:
            comparison['beta'] = 0
            comparison['alpha'] = strategy_cagr - self.risk_free_rate

        # Correlation
        comparison['correlation'] = merged['strategy_return'].corr(merged['benchmark_return'])

        # Information ratio
        excess_returns = merged['strategy_return'] - merged['benchmark_return']
        tracking_error = excess_returns.std() * np.sqrt(252)

        if tracking_error > 0:
            comparison['information_ratio'] = excess_returns.mean() * 252 / tracking_error
        else:
            comparison['information_ratio'] = 0

        # Up/Down capture ratios
        up_market = merged['benchmark_return'] > 0
        down_market = merged['benchmark_return'] < 0

        if up_market.any():
            up_strategy = merged.loc[up_market, 'strategy_return'].mean()
            up_benchmark = merged.loc[up_market, 'benchmark_return'].mean()
            comparison['up_capture'] = up_strategy / up_benchmark if up_benchmark != 0 else 0
        else:
            comparison['up_capture'] = 0

        if down_market.any():
            down_strategy = merged.loc[down_market, 'strategy_return'].mean()
            down_benchmark = merged.loc[down_market, 'benchmark_return'].mean()
            comparison['down_capture'] = down_strategy / down_benchmark if down_benchmark != 0 else 0
        else:
            comparison['down_capture'] = 0

        return comparison

    def _calculate_cagr(self, equity_curve: pd.DataFrame) -> float:
        """Calculate Compound Annual Growth Rate."""
        if equity_curve.empty:
            return 0

        initial_equity = equity_curve['equity'].iloc[0]
        final_equity = equity_curve['equity'].iloc[-1]

        start_date = equity_curve['date'].min()
        end_date = equity_curve['date'].max()
        years = (end_date - start_date).days / 365.25

        if years > 0 and initial_equity > 0:
            return (final_equity / initial_equity) ** (1 / years) - 1
        else:
            return 0

    def _find_drawdown_periods(self, drawdown: pd.Series, dates: pd.Series) -> List[Dict]:
        """Find individual drawdown periods."""
        periods = []
        in_drawdown = False
        start_idx = None
        start_date = None
        peak_value = None
        max_dd = 0

        for i, (dd, date) in enumerate(zip(drawdown, dates)):
            if not in_drawdown and dd < 0:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                start_date = date
                peak_value = drawdown.iloc[i-1] if i > 0 else 0
                max_dd = dd
            elif in_drawdown and dd < max_dd:
                # Deeper drawdown
                max_dd = dd
            elif in_drawdown and dd >= 0:
                # End of drawdown
                end_date = date
                duration_days = (end_date - start_date).days

                # Find recovery (when equity exceeds previous peak)
                recovery_idx = None
                for j in range(i, len(drawdown)):
                    if drawdown.iloc[j] >= 0:
                        recovery_idx = j
                        break

                recovery_days = None
                if recovery_idx is not None:
                    recovery_date = dates.iloc[recovery_idx]
                    recovery_days = (recovery_date - start_date).days

                periods.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_days': duration_days,
                    'max_drawdown': max_dd,
                    'recovery_days': recovery_days
                })

                in_drawdown = False

        return periods

    def _rolling_max_drawdown(self, equity_window: pd.Series) -> float:
        """Calculate maximum drawdown for a rolling window."""
        if equity_window.empty:
            return 0

        peak = equity_window.expanding().max()
        drawdown = (equity_window - peak) / peak
        return drawdown.min()

    def _analyze_consecutive_trades(self, trades_df: pd.DataFrame) -> Dict[str, int]:
        """Analyze consecutive winning/losing streaks."""
        if trades_df.empty or 'pnl' not in trades_df.columns:
            return {}

        # Create win/loss indicator
        wins = (trades_df['pnl'] > 0).astype(int)
        losses = (trades_df['pnl'] < 0).astype(int)

        # Find consecutive runs
        def find_consecutive_runs(series):
            if series.empty:
                return []

            runs = []
            current_run = 1

            for i in range(1, len(series)):
                if series.iloc[i] == series.iloc[i-1] and series.iloc[i] == 1:
                    current_run += 1
                else:
                    if series.iloc[i-1] == 1:
                        runs.append(current_run)
                    current_run = 1

            # Don't forget the last run
            if series.iloc[-1] == 1:
                runs.append(current_run)

            return runs

        win_runs = find_consecutive_runs(wins)
        loss_runs = find_consecutive_runs(losses)

        metrics = {}

        if win_runs:
            metrics['max_consecutive_wins'] = max(win_runs)
            metrics['avg_consecutive_wins'] = np.mean(win_runs)
        else:
            metrics['max_consecutive_wins'] = 0
            metrics['avg_consecutive_wins'] = 0

        if loss_runs:
            metrics['max_consecutive_losses'] = max(loss_runs)
            metrics['avg_consecutive_losses'] = np.mean(loss_runs)
        else:
            metrics['max_consecutive_losses'] = 0
            metrics['avg_consecutive_losses'] = 0

        return metrics

    def generate_performance_summary(self, equity_curve: pd.DataFrame,
                                   trades_df: Optional[pd.DataFrame] = None,
                                   benchmark_equity: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """Generate comprehensive performance summary.

        Args:
            equity_curve: Strategy equity curve
            trades_df: Optional trades data
            benchmark_equity: Optional benchmark equity curve

        Returns:
            Comprehensive performance summary dictionary
        """
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'period_start': equity_curve['date'].min().isoformat() if not equity_curve.empty else None,
            'period_end': equity_curve['date'].max().isoformat() if not equity_curve.empty else None,
        }

        # Return metrics
        summary['returns'] = self.calculate_returns_metrics(equity_curve)

        # Drawdown metrics
        summary['drawdown'] = self.calculate_drawdown_metrics(equity_curve)

        # Trade analysis
        if trades_df is not None and not trades_df.empty:
            summary['trades'] = self.analyze_trades(trades_df)

        # Benchmark comparison
        if benchmark_equity is not None and not benchmark_equity.empty:
            summary['benchmark'] = self.benchmark_comparison(equity_curve, benchmark_equity)

        # Risk metrics summary
        summary['risk_summary'] = self._generate_risk_summary(summary)

        return summary

    def _generate_risk_summary(self, performance_summary: Dict) -> Dict[str, str]:
        """Generate risk assessment summary."""
        risk_summary = {}

        returns = performance_summary.get('returns', {})
        drawdown = performance_summary.get('drawdown', {})

        # Volatility assessment
        vol = returns.get('volatility', 0)
        if vol < 0.1:
            risk_summary['volatility_level'] = 'Low'
        elif vol < 0.2:
            risk_summary['volatility_level'] = 'Moderate'
        elif vol < 0.3:
            risk_summary['volatility_level'] = 'High'
        else:
            risk_summary['volatility_level'] = 'Very High'

        # Drawdown assessment
        max_dd = abs(drawdown.get('max_drawdown', 0))
        if max_dd < 0.05:
            risk_summary['drawdown_level'] = 'Low'
        elif max_dd < 0.15:
            risk_summary['drawdown_level'] = 'Moderate'
        elif max_dd < 0.30:
            risk_summary['drawdown_level'] = 'High'
        else:
            risk_summary['drawdown_level'] = 'Very High'

        # Sharpe ratio assessment
        sharpe = returns.get('sharpe_ratio', 0)
        if sharpe > 2.0:
            risk_summary['risk_adjusted_return'] = 'Excellent'
        elif sharpe > 1.0:
            risk_summary['risk_adjusted_return'] = 'Good'
        elif sharpe > 0.5:
            risk_summary['risk_adjusted_return'] = 'Moderate'
        else:
            risk_summary['risk_adjusted_return'] = 'Poor'

        return risk_summary
