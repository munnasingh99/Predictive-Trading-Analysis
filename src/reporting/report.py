"""
Report generation module for predictive trading signals.

This module generates comprehensive HTML reports for model performance,
backtesting results, and strategy analysis. It creates professional-looking
reports with interactive visualizations, performance metrics tables,
and detailed analysis sections.

Key features:
- Comprehensive HTML report generation
- Interactive plots using Plotly
- Performance metrics tables
- Feature importance analysis
- Threshold sweep visualizations
- Benchmark comparison charts
- Risk analysis and trade statistics
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from src.data.db import DatabaseManager
from src.backtest.perf import PerformanceAnalyzer
from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive HTML reports for trading strategy analysis."""

    def __init__(self, config: Union[Dict, ConfigManager], db: Optional[DatabaseManager] = None):
        """Initialize report generator.

        Args:
            config: Configuration dict or ConfigManager instance
            db: Optional DatabaseManager instance
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.config

        self.db = db or DatabaseManager(self.config.get('data', {}).get('db_path', 'data/trading.db'))

        # Reporting configuration
        self.reporting_config = self.config.get('reporting', {})
        self.output_dir = Path(self.reporting_config.get('output_dir', 'reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Performance analyzer
        risk_free_rate = self.reporting_config.get('risk_free_rate', 0.0)
        self.perf_analyzer = PerformanceAnalyzer(risk_free_rate=risk_free_rate)

        # Plotting configuration
        plt.style.use('seaborn-v0_8-whitegrid')
        self.plot_config = {
            'template': 'plotly_white',
            'height': 500,
            'width': 800,
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        }

        logger.info("ReportGenerator initialized")

    def create_model_performance_plot(self, model_results: Dict[str, Any]) -> str:
        """Create model performance visualization.

        Args:
            model_results: Model training results

        Returns:
            HTML string for the plot
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance by Set', 'ROC Curve', 'Precision-Recall Curve', 'Feature Importance'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # Performance by set
        sets = ['train', 'val', 'test']
        accuracy_scores = []
        auc_scores = []

        for set_name in sets:
            key = f'{set_name}_metrics'
            if key in model_results:
                accuracy_scores.append(model_results[key].get('accuracy', 0))
                auc_scores.append(model_results[key].get('roc_auc', 0))
            else:
                accuracy_scores.append(0)
                auc_scores.append(0)

        fig.add_trace(
            go.Bar(name='Accuracy', x=sets, y=accuracy_scores, marker_color='#1f77b4'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='ROC AUC', x=sets, y=auc_scores, marker_color='#ff7f0e'),
            row=1, col=1
        )

        # ROC Curve
        if 'curves' in model_results and 'roc' in model_results['curves']:
            roc_data = model_results['curves']['roc']
            fig.add_trace(
                go.Scatter(
                    x=roc_data['fpr'],
                    y=roc_data['tpr'],
                    mode='lines',
                    name='ROC Curve',
                    line=dict(color='#2ca02c', width=2)
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(dash='dash', color='gray')
                ),
                row=1, col=2
            )

        # Precision-Recall Curve
        if 'curves' in model_results and 'pr' in model_results['curves']:
            pr_data = model_results['curves']['pr']
            fig.add_trace(
                go.Scatter(
                    x=pr_data['recall'],
                    y=pr_data['precision'],
                    mode='lines',
                    name='PR Curve',
                    line=dict(color='#d62728', width=2)
                ),
                row=2, col=1
            )

        # Feature Importance
        if 'feature_importance' in model_results:
            feature_imp = model_results['feature_importance']
            top_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10]

            features, importance = zip(*top_features) if top_features else ([], [])

            fig.add_trace(
                go.Bar(
                    x=list(importance),
                    y=list(features),
                    orientation='h',
                    marker_color='#9467bd'
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            title_text=f"Model Performance Analysis: {model_results.get('model_name', 'Unknown')}",
            template=self.plot_config['template'],
            showlegend=True
        )

        return fig.to_html(include_plotlyjs='inline', div_id="model_performance_plot")

    def create_equity_curve_plot(self, equity_curve: pd.DataFrame,
                               benchmark_equity: Optional[pd.DataFrame] = None) -> str:
        """Create equity curve visualization.

        Args:
            equity_curve: Strategy equity curve
            benchmark_equity: Optional benchmark equity curve

        Returns:
            HTML string for the plot
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Equity Curve', 'Drawdown'),
            vertical_spacing=0.1
        )

        # Strategy equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['equity'],
                mode='lines',
                name='Strategy',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )

        # Benchmark equity curve
        if benchmark_equity is not None and not benchmark_equity.empty:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_equity['date'],
                    y=benchmark_equity['equity'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#ff7f0e', width=2)
                ),
                row=1, col=1
            )

        # Drawdown
        if 'drawdown' in equity_curve.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity_curve['date'],
                    y=equity_curve['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown %',
                    fill='tonexty',
                    line=dict(color='#d62728', width=1),
                    fillcolor='rgba(214, 39, 40, 0.3)'
                ),
                row=2, col=1
            )

        fig.update_layout(
            height=600,
            title_text="Strategy Performance vs Benchmark",
            template=self.plot_config['template'],
            xaxis=dict(title='Date'),
            yaxis=dict(title='Portfolio Value ($)'),
            xaxis2=dict(title='Date'),
            yaxis2=dict(title='Drawdown (%)')
        )

        return fig.to_html(include_plotlyjs='inline', div_id="equity_curve_plot")

    def create_returns_distribution_plot(self, equity_curve: pd.DataFrame) -> str:
        """Create returns distribution visualization.

        Args:
            equity_curve: Strategy equity curve

        Returns:
            HTML string for the plot
        """
        if 'daily_return' not in equity_curve.columns:
            equity_curve['daily_return'] = equity_curve['equity'].pct_change()

        returns = equity_curve['daily_return'].dropna() * 100

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Returns Distribution', 'QQ Plot vs Normal'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker_color='#1f77b4',
                opacity=0.7
            ),
            row=1, col=1
        )

        # Normal distribution overlay
        mean_return = returns.mean()
        std_return = returns.std()
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = len(returns) * (returns.max() - returns.min()) / 50 * \
                 (1 / (std_return * np.sqrt(2 * np.pi))) * \
                 np.exp(-0.5 * ((x_norm - mean_return) / std_return) ** 2)

        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Normal Dist',
                line=dict(color='#ff7f0e', width=2)
            ),
            row=1, col=1
        )

        # QQ Plot
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm", plot=None)

        fig.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode='markers',
                name='Actual',
                marker=dict(color='#2ca02c', size=4)
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=osm,
                y=slope * osm + intercept,
                mode='lines',
                name='Expected Normal',
                line=dict(color='#d62728', width=2)
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            title_text="Daily Returns Analysis",
            template=self.plot_config['template']
        )

        fig.update_xaxes(title_text="Daily Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

        return fig.to_html(include_plotlyjs='inline', div_id="returns_distribution_plot")

    def create_threshold_sweep_plot(self, threshold_data: pd.DataFrame) -> str:
        """Create threshold sweep analysis plot.

        Args:
            threshold_data: DataFrame with threshold sweep results

        Returns:
            HTML string for the plot
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sharpe Ratio vs Threshold', 'Total Return vs Threshold',
                           'Hit Rate vs Threshold', 'Total Trades vs Threshold')
        )

        # Sharpe Ratio
        fig.add_trace(
            go.Scatter(
                x=threshold_data['threshold'],
                y=threshold_data['sharpe_ratio'],
                mode='lines+markers',
                name='Sharpe Ratio',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )

        # Total Return
        fig.add_trace(
            go.Scatter(
                x=threshold_data['threshold'],
                y=threshold_data['total_return'] * 100,
                mode='lines+markers',
                name='Total Return %',
                line=dict(color='#ff7f0e', width=2)
            ),
            row=1, col=2
        )

        # Hit Rate
        fig.add_trace(
            go.Scatter(
                x=threshold_data['threshold'],
                y=threshold_data['hit_rate'] * 100,
                mode='lines+markers',
                name='Hit Rate %',
                line=dict(color='#2ca02c', width=2)
            ),
            row=2, col=1
        )

        # Total Trades
        fig.add_trace(
            go.Scatter(
                x=threshold_data['threshold'],
                y=threshold_data['total_trades'],
                mode='lines+markers',
                name='Total Trades',
                line=dict(color='#d62728', width=2)
            ),
            row=2, col=2
        )

        # Mark optimal threshold for Sharpe ratio
        if not threshold_data.empty:
            optimal_idx = threshold_data['sharpe_ratio'].idxmax()
            optimal_threshold = threshold_data.loc[optimal_idx, 'threshold']
            optimal_sharpe = threshold_data.loc[optimal_idx, 'sharpe_ratio']

            # Add vertical line at optimal threshold
            for row in [1, 2]:
                for col in [1, 2]:
                    fig.add_vline(
                        x=optimal_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Optimal: {optimal_threshold:.2f}",
                        row=row, col=col
                    )

        fig.update_layout(
            height=600,
            title_text="Threshold Sweep Analysis",
            template=self.plot_config['template'],
            showlegend=False
        )

        return fig.to_html(include_plotlyjs='inline', div_id="threshold_sweep_plot")

    def create_performance_metrics_table(self, performance_metrics: Dict[str, Any]) -> str:
        """Create performance metrics table.

        Args:
            performance_metrics: Performance metrics dictionary

        Returns:
            HTML table string
        """
        # Organize metrics into categories
        categories = {
            'Returns': {
                'Total Return': f"{performance_metrics.get('total_return', 0):.2%}",
                'CAGR': f"{performance_metrics.get('cagr', 0):.2%}",
                'Volatility': f"{performance_metrics.get('volatility', 0):.2%}",
                'Sharpe Ratio': f"{performance_metrics.get('sharpe_ratio', 0):.3f}",
                'Calmar Ratio': f"{performance_metrics.get('calmar_ratio', 0):.3f}"
            },
            'Risk': {
                'Max Drawdown': f"{performance_metrics.get('max_drawdown', 0):.2%}",
                'VaR (95%)': f"{performance_metrics.get('var_95', 0):.2%}",
                'CVaR (95%)': f"{performance_metrics.get('cvar_95', 0):.2%}"
            },
            'Trading': {
                'Total Trades': f"{performance_metrics.get('total_trades', 0):,}",
                'Hit Rate': f"{performance_metrics.get('hit_rate', 0):.2%}",
                'Win/Loss Ratio': f"{performance_metrics.get('win_loss_ratio', 0):.2f}",
                'Profit Factor': f"{performance_metrics.get('profit_factor', 0):.2f}",
                'Average Win': f"${performance_metrics.get('avg_win', 0):,.2f}",
                'Average Loss': f"${performance_metrics.get('avg_loss', 0):,.2f}"
            }
        }

        html = '<div class="metrics-table">'

        for category, metrics in categories.items():
            html += f'''
            <div class="metric-category">
                <h4>{category}</h4>
                <table class="table table-striped">
                    <tbody>
            '''

            for metric_name, metric_value in metrics.items():
                html += f'''
                        <tr>
                            <td><strong>{metric_name}</strong></td>
                            <td>{metric_value}</td>
                        </tr>
                '''

            html += '''
                    </tbody>
                </table>
            </div>
            '''

        html += '</div>'

        return html

    def create_benchmark_comparison_table(self, benchmark_comparison: Dict[str, Any]) -> str:
        """Create benchmark comparison table.

        Args:
            benchmark_comparison: Benchmark comparison metrics

        Returns:
            HTML table string
        """
        if not benchmark_comparison or 'strategy_metrics' not in benchmark_comparison:
            return '<p>No benchmark comparison data available.</p>'

        strategy = benchmark_comparison['strategy_metrics']
        benchmark = benchmark_comparison['benchmark_metrics']
        relative = benchmark_comparison.get('relative_metrics', {})

        html = '''
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Strategy</th>
                    <th>Benchmark</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody>
        '''

        comparisons = [
            ('Total Return', 'total_return', lambda x: f"{x:.2%}"),
            ('CAGR', 'cagr', lambda x: f"{x:.2%}"),
            ('Volatility', 'volatility', lambda x: f"{x:.2%}"),
            ('Sharpe Ratio', 'sharpe_ratio', lambda x: f"{x:.3f}"),
            ('Max Drawdown', 'max_drawdown', lambda x: f"{x:.2%}")
        ]

        for label, key, formatter in comparisons:
            strat_val = strategy.get(key, 0)
            bench_val = benchmark.get(key, 0)
            diff_val = strat_val - bench_val

            diff_class = "text-success" if diff_val > 0 else "text-danger" if diff_val < 0 else ""
            diff_sign = "+" if diff_val > 0 else ""

            html += f'''
                <tr>
                    <td><strong>{label}</strong></td>
                    <td>{formatter(strat_val)}</td>
                    <td>{formatter(bench_val)}</td>
                    <td class="{diff_class}">{diff_sign}{formatter(diff_val)}</td>
                </tr>
            '''

        html += '''
            </tbody>
        </table>
        '''

        return html

    def create_data_summary_table(self, data_summary: Dict[str, Any]) -> str:
        """Create data summary table.

        Args:
            data_summary: Data summary information

        Returns:
            HTML table string
        """
        html = '''
        <table class="table table-striped">
            <tbody>
        '''

        summary_items = [
            ('Total Records', data_summary.get('total_rows', 0)),
            ('Symbols', data_summary.get('symbols_count', 0)),
            ('Date Range', f"{data_summary.get('date_range', {}).get('start', 'N/A')} to {data_summary.get('date_range', {}).get('end', 'N/A')}"),
            ('Trading Days', data_summary.get('days_count', 0))
        ]

        for label, value in summary_items:
            html += f'''
                <tr>
                    <td><strong>{label}</strong></td>
                    <td>{value:,}</td>
                </tr>
            '''

        html += '''
            </tbody>
        </table>
        '''

        return html

    def generate_html_report(self, model_name: str,
                           model_results: Optional[Dict] = None,
                           backtest_results: Optional[Dict] = None,
                           threshold_data: Optional[pd.DataFrame] = None) -> str:
        """Generate comprehensive HTML report.

        Args:
            model_name: Name of the model
            model_results: Model training results
            backtest_results: Backtest results
            threshold_data: Threshold sweep data

        Returns:
            Path to generated HTML report
        """
        logger.info(f"Generating HTML report for {model_name}")

        # Get data from database if not provided
        if model_results is None:
            # Try to get model results from database
            model_results = {}

        if backtest_results is None:
            # Get backtest data from database
            trades_df = self.db.get_trades(model_name=model_name)
            equity_df = self.db.get_equity_curves(model_name=model_name)

            if not equity_df.empty:
                backtest_results = {
                    'simulation_results': {
                        'trades': trades_df,
                        'portfolio_equity': equity_df
                    }
                }
            else:
                backtest_results = {}

        # Get data summary
        data_summary = self.db.get_table_info()

        # Create plots
        plots_html = ""

        # Model performance plot
        if model_results:
            plots_html += f'<div class="plot-section">{self.create_model_performance_plot(model_results)}</div>'

        # Equity curve plot
        if backtest_results and 'simulation_results' in backtest_results:
            equity_curve = backtest_results['simulation_results'].get('portfolio_equity', pd.DataFrame())
            if not equity_curve.empty:
                # Get benchmark data for comparison
                benchmark_equity = None
                try:
                    benchmark_symbol = self.config.get('backtesting', {}).get('benchmark_symbol', 'SPY')
                    start_date = equity_curve['date'].min().strftime('%Y-%m-%d')
                    end_date = equity_curve['date'].max().strftime('%Y-%m-%d')
                    benchmark_bars = self.db.get_bars(symbols=[benchmark_symbol], start_date=start_date, end_date=end_date)

                    if not benchmark_bars.empty:
                        initial_value = equity_curve['equity'].iloc[0]
                        benchmark_equity = pd.DataFrame({
                            'date': benchmark_bars['date'],
                            'equity': initial_value * benchmark_bars['adj_close'] / benchmark_bars['adj_close'].iloc[0]
                        })
                except Exception as e:
                    logger.warning(f"Failed to get benchmark data: {e}")

                plots_html += f'<div class="plot-section">{self.create_equity_curve_plot(equity_curve, benchmark_equity)}</div>'
                plots_html += f'<div class="plot-section">{self.create_returns_distribution_plot(equity_curve)}</div>'

        # Threshold sweep plot
        if threshold_data is not None and not threshold_data.empty:
            plots_html += f'<div class="plot-section">{self.create_threshold_sweep_plot(threshold_data)}</div>'

        # Create tables
        tables_html = ""

        # Performance metrics table
        if backtest_results and 'performance_metrics' in backtest_results:
            performance_metrics = backtest_results['performance_metrics']
            tables_html += f'''
            <div class="table-section">
                <h3>Performance Metrics</h3>
                {self.create_performance_metrics_table(performance_metrics)}
            </div>
            '''

        # Benchmark comparison table
        if backtest_results and 'benchmark_comparison' in backtest_results:
            benchmark_comparison = backtest_results['benchmark_comparison']
            tables_html += f'''
            <div class="table-section">
                <h3>Benchmark Comparison</h3>
                {self.create_benchmark_comparison_table(benchmark_comparison)}
            </div>
            '''

        # Data summary table
        tables_html += f'''
        <div class="table-section">
            <h3>Data Summary</h3>
            {self.create_data_summary_table(data_summary)}
        </div>
        '''

        # Generate complete HTML
        report_html = self._create_html_template(
            title=f"Trading Strategy Report - {model_name}",
            model_name=model_name,
            plots_html=plots_html,
            tables_html=tables_html,
            model_results=model_results,
            backtest_results=backtest_results
        )

        # Save report
        report_filename = f"trading_report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)

        logger.info(f"HTML report saved: {report_path}")
        return str(report_path)

    def _create_html_template(self, title: str, model_name: str, plots_html: str,
                            tables_html: str, model_results: Dict, backtest_results: Dict) -> str:
        """Create complete HTML template for the report.

        Args:
            title: Report title
            model_name: Model name
            plots_html: HTML content for plots
            tables_html: HTML content for tables
            model_results: Model results data
            backtest_results: Backtest results data

        Returns:
            Complete HTML string
        """
        # Executive summary
        executive_summary = self._create_executive_summary(model_results, backtest_results)

        # CSS styles
        css_styles = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #007bff;
            }
            .header h1 {
                color: #007bff;
                margin-bottom: 10px;
            }
            .header p {
                color: #6c757d;
                font-size: 16px;
            }
            .section {
                margin: 30px 0;
            }
            .section h2 {
                color: #343a40;
                border-bottom: 1px solid #dee2e6;
                padding-bottom: 10px;
            }
            .section h3 {
                color: #495057;
                margin-top: 25px;
            }
            .plot-section {
                margin: 20px 0;
                text-align: center;
            }
            .table-section {
                margin: 20px 0;
            }
            .metrics-table {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }
            .metric-category {
                flex: 1;
                min-width: 300px;
            }
            .metric-category h4 {
                color: #007bff;
                margin-bottom: 15px;
                padding-bottom: 5px;
                border-bottom: 1px solid #007bff;
            }
            .table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            .table th, .table td {
                padding: 8px 12px;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
            }
            .table th {
                background-color: #f8f9fa;
                font-weight: 600;
                color: #495057;
            }
            .table-striped tbody tr:nth-of-type(odd) {
                background-color: rgba(0,123,255,.05);
            }
            .text-success { color: #28a745; }
            .text-danger { color: #dc3545; }
            .alert {
                padding: 15px;
                margin-bottom: 20px;
                border: 1px solid transparent;
                border-radius: 5px;
            }
            .alert-info {
                color: #0c5460;
                background-color: #d1ecf1;
                border-color: #bee5eb;
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #dee2e6;
                color: #6c757d;
                font-size: 14px;
            }
        </style>
        """

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            {css_styles}
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>

                <div class="section">
                    <h2>Executive Summary</h2>
                    {executive_summary}
                </div>

                <div class="section">
                    <h2>Performance Visualizations</h2>
                    {plots_html}
                </div>

                <div class="section">
                    <h2>Performance Metrics</h2>
                    {tables_html}
                </div>

                <div class="footer">
                    <p>Report generated by Predictive Trading Signals Framework</p>
                    <p>Model: {model_name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html_template

    def _create_executive_summary(self, model_results: Dict, backtest_results: Dict) -> str:
        """Create executive summary section.

        Args:
            model_results: Model training results
            backtest_results: Backtest results

        Returns:
            HTML string for executive summary
        """
        summary_points = []

        # Model performance summary
        if model_results and 'test_metrics' in model_results:
            test_metrics = model_results['test_metrics']
            accuracy = test_metrics.get('accuracy', 0)
            roc_auc = test_metrics.get('roc_auc', 0)

            summary_points.append(f"• Model achieved <strong>{accuracy:.1%}</strong> accuracy and <strong>{roc_auc:.3f}</strong> ROC AUC on the test set")

        # Strategy performance summary
        if backtest_results and 'performance_metrics' in backtest_results:
            perf = backtest_results['performance_metrics']
            total_return = perf.get('total_return', 0)
            sharpe_ratio = perf.get('sharpe_ratio', 0)
            max_drawdown = perf.get('max_drawdown', 0)
            hit_rate = perf.get('hit_rate', 0)

            summary_points.append(f"• Strategy generated <strong>{total_return:.1%}</strong> total return with a Sharpe ratio of <strong>{sharpe_ratio:.2f}</strong>")
            summary_points.append(f"• Maximum drawdown was <strong>{max_drawdown:.1%}</strong> with a hit rate of <strong>{hit_rate:.1%}</strong>")

        # Benchmark comparison summary
        if backtest_results and 'benchmark_comparison' in backtest_results:
            bench = backtest_results['benchmark_comparison']
            if 'relative_metrics' in bench:
                excess_return = bench['relative_metrics'].get('excess_return', 0)
                if excess_return > 0:
                    summary_points.append(f"• Strategy outperformed benchmark by <strong>{excess_return:.1%}</strong>")
                else:
                    summary_points.append(f"• Strategy underperformed benchmark by <strong>{abs(excess_return):.1%}</strong>")

        # Risk assessment
        if backtest_results and 'performance_metrics' in backtest_results:
            volatility = backtest_results['performance_metrics'].get('volatility', 0)
            if volatility < 0.15:
                risk_assessment = "low"
            elif volatility < 0.25:
                risk_assessment = "moderate"
            else:
                risk_assessment = "high"

            summary_points.append(f"• Strategy exhibits <strong>{risk_assessment}</strong> risk with {volatility:.1%} annualized volatility")

        if not summary_points:
            summary_points.append("• No performance data available for analysis")

        summary_html = f"""
        <div class="alert alert-info">
            <h4>Key Findings:</h4>
            <ul style="margin-bottom: 0;">
                {''.join(f'<li>{point}</li>' for point in summary_points)}
            </ul>
        </div>
        """

        return summary_html

    def create_daily_signal_report(self, symbol: str, model_name: str,
                                 current_date: str) -> Dict[str, Any]:
        """Create daily signal report for deployment.

        Args:
            symbol: Trading symbol
            model_name: Model name
            current_date: Current date (YYYY-MM-DD)

        Returns:
            Dictionary with daily signal information
        """
        try:
            # Get latest prediction
            predictions = self.db.get_predictions(
                model_name=model_name,
                symbols=[symbol],
                start_date=current_date,
                end_date=current_date
            )

            if predictions.empty:
                return {
                    'error': f'No predictions found for {symbol} on {current_date}',
                    'symbol': symbol,
                    'date': current_date,
                    'model_name': model_name
                }

            latest_prediction = predictions.iloc[-1]

            # Get latest price data
            bars = self.db.get_bars(
                symbols=[symbol],
                start_date=current_date,
                end_date=current_date
            )

            if bars.empty:
                return {
                    'error': f'No price data found for {symbol} on {current_date}',
                    'symbol': symbol,
                    'date': current_date,
                    'model_name': model_name
                }

            latest_bar = bars.iloc[-1]

            # Generate signal
            probability = latest_prediction['proba']
            threshold_long = self.config.get('backtesting', {}).get('threshold_long', 0.55)
            threshold_short = self.config.get('backtesting', {}).get('threshold_short', 0.45)

            if probability >= threshold_long:
                signal = 'LONG'
                signal_strength = probability - threshold_long
            elif probability <= threshold_short:
                signal = 'SHORT'
                signal_strength = threshold_short - probability
            else:
                signal = 'FLAT'
                signal_strength = 0

            return {
                'symbol': symbol,
                'date': current_date,
                'model_name': model_name,
                'signal': signal,
                'probability': probability,
                'signal_strength': signal_strength,
                'current_price': latest_bar['close'],
                'volume': latest_bar['volume'],
                'prediction_label': latest_prediction['pred_label'],
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to create daily signal report: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'date': current_date,
                'model_name': model_name
            }

    def generate_model_comparison_report(self, model_names: List[str]) -> str:
        """Generate model comparison report.

        Args:
            model_names: List of model names to compare

        Returns:
            Path to generated comparison report
        """
        logger.info(f"Generating model comparison report for {len(model_names)} models")

        comparison_data = []

        for model_name in model_names:
            # Get backtest results for each model
            trades_df = self.db.get_trades(model_name=model_name)
            equity_df = self.db.get_equity_curves(model_name=model_name)

            if not equity_df.empty:
                # Calculate performance metrics
                performance = self.perf_analyzer.calculate_returns_metrics(equity_df)
                drawdown = self.perf_analyzer.calculate_drawdown_metrics(equity_df)

                model_data = {
                    'model_name': model_name,
                    'total_return': performance.get('total_return', 0),
                    'cagr': performance.get('cagr', 0),
                    'volatility': performance.get('volatility', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0),
                    'max_drawdown': drawdown.get('max_drawdown', 0),
                    'calmar_ratio': drawdown.get('calmar_ratio', 0),
                    'total_trades': len(trades_df) if not trades_df.empty else 0
                }
                comparison_data.append(model_data)

        if not comparison_data:
            logger.warning("No model data found for comparison")
            return ""

        comparison_df = pd.DataFrame(comparison_data)

        # Create comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Return Comparison', 'Sharpe Ratio Comparison',
                           'Max Drawdown Comparison', 'Risk-Return Scatter')
        )

        # Total Return
        fig.add_trace(
            go.Bar(x=comparison_df['model_name'], y=comparison_df['total_return'],
                   name='Total Return', marker_color='#1f77b4'),
            row=1, col=1
        )

        # Sharpe Ratio
        fig.add_trace(
            go.Bar(x=comparison_df['model_name'], y=comparison_df['sharpe_ratio'],
                   name='Sharpe Ratio', marker_color='#ff7f0e'),
            row=1, col=2
        )

        # Max Drawdown
        fig.add_trace(
            go.Bar(x=comparison_df['model_name'], y=comparison_df['max_drawdown'],
                   name='Max Drawdown', marker_color='#d62728'),
            row=2, col=1
        )

        # Risk-Return Scatter
        fig.add_trace(
            go.Scatter(x=comparison_df['volatility'], y=comparison_df['total_return'],
                      mode='markers+text', text=comparison_df['model_name'],
                      textposition='top center', marker=dict(size=10),
                      name='Risk-Return'),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Model Performance Comparison",
            template=self.plot_config['template'],
            showlegend=False
        )

        plots_html = fig.to_html(include_plotlyjs='inline', div_id="model_comparison_plot")

        # Create comparison table
        comparison_table = comparison_df.round(4).to_html(
            classes='table table-striped',
            table_id='comparison_table',
            escape=False
        )

        # Generate HTML report
        report_html = self._create_html_template(
            title="Model Comparison Report",
            model_name="Multiple Models",
            plots_html=plots_html,
            tables_html=f'<div class="table-section"><h3>Model Comparison</h3>{comparison_table}</div>',
            model_results={},
            backtest_results={}
        )

        # Save report
        report_filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path = self.output_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)

        logger.info(f"Model comparison report saved: {report_path}")
        return str(report_path)
