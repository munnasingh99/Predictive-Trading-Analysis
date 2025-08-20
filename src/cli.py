"""
Command-line interface for predictive trading signals.

This module provides a comprehensive CLI for the trading system using Typer.
It includes commands for data ingestion, feature engineering, model training,
backtesting, and report generation.

Key features:
- Interactive CLI with help messages and validation
- Configuration file support with overrides
- Progress tracking for long operations
- Error handling and logging
- Modular command structure
- Support for both single commands and full pipelines
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingest import DataIngestor
from src.features.engineer import FeatureEngineer
from src.labeling.labels import LabelGenerator
from src.modeling.models import ModelTrainer
from src.backtest.engine import BacktestEngine
from src.reporting.report import ReportGenerator
from src.utils.config import ConfigManager
from src.utils.logging import setup_logging
from src.data.db import DatabaseManager

# Initialize Typer app
app = typer.Typer(
    name="trading-signals",
    help="Predictive Trading Signals - ML-based trading strategy development",
    add_completion=False
)

# Initialize Rich console
console = Console()

# Global variables
config_manager = None
logger = None


def setup_globals(config_path: Optional[str] = None, verbose: bool = False):
    """Setup global configuration and logging."""
    global config_manager, logger

    try:
        # Load configuration
        config_manager = ConfigManager(config_path=config_path)

        # Setup logging
        log_level = 'DEBUG' if verbose else config_manager.get('logging.level', 'INFO')
        logging_config = config_manager.get_section('logging')
        logging_config['level'] = log_level

        logging_manager = setup_logging(logging_config)
        logger = logging_manager.get_logger(__name__)

        # Configure third-party logging
        from src.utils.logging import configure_third_party_logging
        configure_third_party_logging()

        logger.info("CLI initialized successfully")

    except Exception as e:
        console.print(f"[red]Failed to initialize CLI: {e}[/red]")
        raise typer.Exit(1)


def handle_error(operation: str, error: Exception):
    """Handle and log errors consistently."""
    error_msg = f"Failed to {operation}: {error}"
    if logger:
        logger.error(error_msg, exc_info=True)
    console.print(f"[red]{error_msg}[/red]")
    raise typer.Exit(1)


@app.command()
def fetch_data(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Comma-separated list of symbols (e.g., 'SPY,AAPL,MSFT')"
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD)"
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD)"
    ),
    source: Optional[str] = typer.Option(
        None,
        "--source",
        help="Data source ('yfinance' or 'csv')"
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace existing data"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Fetch market data from various sources."""
    setup_globals(config, verbose)

    with console.status("[bold green]Fetching market data..."):
        try:
            # Parse symbols if provided
            symbol_list = None
            if symbols:
                symbol_list = [s.strip().upper() for s in symbols.split(',')]

            # Initialize data ingestor
            db = DatabaseManager(config_manager.get('data.db_path'))
            data_ingestor = DataIngestor(config_manager, db)

            # Run data ingestion
            results = data_ingestor.run_ingestion(
                symbols=symbol_list,
                start_date=start,
                end_date=end,
                source=source,
                replace=replace
            )

            if results['success']:
                console.print("[green]✓[/green] Data fetch completed successfully")

                # Display results
                table = Table(title="Data Ingestion Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Rows Fetched", f"{results['rows_fetched']:,}")
                table.add_row("Symbols", f"{results['symbols_count']}")
                table.add_row("Date Range", f"{results['date_range']['start']} to {results['date_range']['end']}")
                table.add_row("Duration", f"{results['duration_seconds']:.1f}s")

                console.print(table)

                if verbose:
                    rprint(f"Symbols: {', '.join(results['symbols_list'])}")
            else:
                handle_error("fetch data", Exception(results['error']))

        except Exception as e:
            handle_error("fetch data", e)


@app.command()
def build_features(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Comma-separated list of symbols"
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD)"
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD)"
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace existing features"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Build features and labels from market data."""
    setup_globals(config, verbose)

    with console.status("[bold green]Building features and labels..."):
        try:
            # Parse symbols if provided
            symbol_list = None
            if symbols:
                symbol_list = [s.strip().upper() for s in symbols.split(',')]

            # Initialize components
            db = DatabaseManager(config_manager.get('data.db_path'))
            feature_engineer = FeatureEngineer(config_manager, db)
            label_generator = LabelGenerator(config_manager, db)

            # Build features
            console.print("Building technical features...")
            features_df = feature_engineer.build_features_from_db(
                symbols=symbol_list,
                start_date=start,
                end_date=end,
                replace=replace
            )

            # Generate labels
            console.print("Generating labels...")
            labels_df = label_generator.generate_labels_from_db(
                symbols=symbol_list,
                start_date=start,
                end_date=end,
                replace=replace
            )

            console.print("[green]✓[/green] Features and labels built successfully")

            # Display results
            table = Table(title="Feature Engineering Results")
            table.add_column("Component", style="cyan")
            table.add_column("Rows Created", style="green")
            table.add_column("Symbols", style="blue")

            table.add_row("Features", f"{len(features_df):,}", f"{features_df['symbol'].nunique()}")
            table.add_row("Labels", f"{len(labels_df):,}", f"{labels_df['symbol'].nunique()}")

            console.print(table)

        except Exception as e:
            handle_error("build features", e)


@app.command()
def train(
    model: str = typer.Option(
        "rf",
        "--model",
        "-m",
        help="Model to train ('logreg' or 'rf')"
    ),
    hyperparameter_search: bool = typer.Option(
        False,
        "--search",
        help="Perform hyperparameter search"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Train machine learning models."""
    setup_globals(config, verbose)

    with console.status(f"[bold green]Training {model} model..."):
        try:
            # Initialize model trainer
            db = DatabaseManager(config_manager.get('data.db_path'))
            model_trainer = ModelTrainer(config_manager, db)

            # Train model
            results = model_trainer.train_model(
                model_name=model,
                hyperparameter_search=hyperparameter_search
            )

            console.print(f"[green]✓[/green] Model {model} trained successfully")

            # Display results
            table = Table(title=f"Training Results - {model.upper()}")
            table.add_column("Set", style="cyan")
            table.add_column("Accuracy", style="green")
            table.add_column("ROC AUC", style="blue")
            table.add_column("Samples", style="yellow")

            # Training metrics
            if 'train_metrics' in results:
                train = results['train_metrics']
                table.add_row(
                    "Training",
                    f"{train['accuracy']:.3f}",
                    f"{train.get('roc_auc', 0):.3f}",
                    f"{train['samples']:,}"
                )

            # Validation metrics
            if 'val_metrics' in results:
                val = results['val_metrics']
                table.add_row(
                    "Validation",
                    f"{val['accuracy']:.3f}",
                    f"{val.get('roc_auc', 0):.3f}",
                    f"{val['samples']:,}"
                )

            # Test metrics
            if 'test_metrics' in results:
                test = results['test_metrics']
                table.add_row(
                    "Test",
                    f"{test['accuracy']:.3f}",
                    f"{test.get('roc_auc', 0):.3f}",
                    f"{test['samples']:,}"
                )

            console.print(table)

            # Cross-validation results
            if 'cv_metrics' in results:
                cv = results['cv_metrics']
                rprint(f"\n[bold]Cross-Validation:[/bold]")
                rprint(f"Accuracy: {cv['accuracy_scores_mean']:.3f} ± {cv['accuracy_scores_std']:.3f}")
                rprint(f"ROC AUC: {cv['roc_auc_scores_mean']:.3f} ± {cv['roc_auc_scores_std']:.3f}")

            # Model path
            if 'model_path' in results:
                rprint(f"\n[bold]Model saved:[/bold] {results['model_path']}")

        except Exception as e:
            handle_error("train model", e)


@app.command()
def backtest(
    model: str = typer.Option(
        "rf",
        "--model",
        "-m",
        help="Model to backtest"
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        help="Long threshold for signals"
    ),
    long_only: bool = typer.Option(
        True,
        "--long-only/--long-short",
        help="Long-only strategy"
    ),
    initial_capital: float = typer.Option(
        100000.0,
        "--capital",
        help="Initial capital for backtest"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Run strategy backtest."""
    setup_globals(config, verbose)

    with console.status(f"[bold green]Running backtest for {model}..."):
        try:
            # Initialize backtest engine
            db = DatabaseManager(config_manager.get('data.db_path'))

            # Override configuration if specified
            if threshold is not None:
                config_manager.set('backtesting.threshold_long', threshold)
                config_manager.set('backtesting.threshold_short', 1 - threshold)

            config_manager.set('backtesting.long_only', long_only)

            backtest_engine = BacktestEngine(config_manager, db)

            # Run backtest
            results = backtest_engine.run_backtest(
                model_name=model,
                initial_capital=initial_capital
            )

            console.print(f"[green]✓[/green] Backtest completed for {model}")

            # Display performance metrics
            perf = results['performance_metrics']

            table = Table(title=f"Backtest Results - {model.upper()}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Return", f"{perf['total_return']:.2%}")
            table.add_row("CAGR", f"{perf['cagr']:.2%}")
            table.add_row("Sharpe Ratio", f"{perf['sharpe_ratio']:.3f}")
            table.add_row("Max Drawdown", f"{perf['max_drawdown']:.2%}")
            table.add_row("Calmar Ratio", f"{perf['calmar_ratio']:.3f}")
            table.add_row("Total Trades", f"{perf['total_trades']:,}")
            table.add_row("Hit Rate", f"{perf['hit_rate']:.2%}")
            table.add_row("Win/Loss Ratio", f"{perf['win_loss_ratio']:.2f}")

            console.print(table)

            # Benchmark comparison
            if 'benchmark_comparison' in results:
                bench = results['benchmark_comparison']
                if 'relative_metrics' in bench:
                    rel = bench['relative_metrics']
                    rprint(f"\n[bold]vs Benchmark:[/bold]")
                    rprint(f"Excess Return: {rel['excess_return']:.2%}")
                    rprint(f"Excess CAGR: {rel['excess_cagr']:.2%}")
                    rprint(f"Sharpe Improvement: {rel['sharpe_improvement']:.3f}")

        except Exception as e:
            handle_error("run backtest", e)


@app.command()
def report(
    model: str = typer.Option(
        "rf",
        "--model",
        "-m",
        help="Model for report generation"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for report"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Generate comprehensive HTML report."""
    setup_globals(config, verbose)

    with console.status(f"[bold green]Generating report for {model}..."):
        try:
            # Initialize report generator
            db = DatabaseManager(config_manager.get('data.db_path'))

            if output:
                config_manager.set('reporting.output_dir', output)

            report_generator = ReportGenerator(config_manager, db)

            # Generate report
            report_path = report_generator.generate_html_report(model_name=model)

            console.print(f"[green]✓[/green] Report generated successfully")
            rprint(f"\n[bold]Report saved:[/bold] {report_path}")

        except Exception as e:
            handle_error("generate report", e)


@app.command()
def threshold_sweep(
    model: str = typer.Option(
        "rf",
        "--model",
        "-m",
        help="Model for threshold analysis"
    ),
    min_threshold: float = typer.Option(
        0.4,
        "--min",
        help="Minimum threshold"
    ),
    max_threshold: float = typer.Option(
        0.7,
        "--max",
        help="Maximum threshold"
    ),
    step: float = typer.Option(
        0.05,
        "--step",
        help="Threshold step size"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Perform threshold sweep analysis."""
    setup_globals(config, verbose)

    with console.status(f"[bold green]Running threshold sweep for {model}..."):
        try:
            # Initialize backtest engine
            db = DatabaseManager(config_manager.get('data.db_path'))
            backtest_engine = BacktestEngine(config_manager, db)

            # Run threshold sweep
            results_df = backtest_engine.threshold_sweep(
                model_name=model,
                threshold_range=(min_threshold, max_threshold),
                step=step,
                metric='sharpe_ratio'
            )

            if not results_df.empty:
                console.print(f"[green]✓[/green] Threshold sweep completed")

                # Find optimal threshold
                optimal_idx = results_df['sharpe_ratio'].idxmax()
                optimal_threshold = results_df.loc[optimal_idx, 'threshold']
                optimal_sharpe = results_df.loc[optimal_idx, 'sharpe_ratio']

                # Display top results
                top_results = results_df.nlargest(5, 'sharpe_ratio')

                table = Table(title="Top 5 Threshold Results")
                table.add_column("Threshold", style="cyan")
                table.add_column("Sharpe Ratio", style="green")
                table.add_column("Total Return", style="blue")
                table.add_column("Hit Rate", style="yellow")
                table.add_column("Total Trades", style="magenta")

                for _, row in top_results.iterrows():
                    table.add_row(
                        f"{row['threshold']:.2f}",
                        f"{row['sharpe_ratio']:.3f}",
                        f"{row['total_return']:.2%}",
                        f"{row['hit_rate']:.2%}",
                        f"{int(row['total_trades'])}"
                    )

                console.print(table)

                rprint(f"\n[bold]Optimal Threshold:[/bold] {optimal_threshold:.2f} (Sharpe: {optimal_sharpe:.3f})")
            else:
                console.print("[yellow]No threshold sweep results generated[/yellow]")

        except Exception as e:
            handle_error("run threshold sweep", e)


@app.command()
def status(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Show system status and data summary."""
    setup_globals(config, verbose)

    try:
        # Initialize database
        db = DatabaseManager(config_manager.get('data.db_path'))

        # Get table info
        table_info = db.get_table_info()

        console.print("[bold]System Status[/bold]")

        # Data status
        data_table = Table(title="Database Status")
        data_table.add_column("Table", style="cyan")
        data_table.add_column("Rows", style="green")

        for table_name, row_count in table_info.items():
            data_table.add_row(table_name.title(), f"{row_count:,}")

        console.print(data_table)

        # Configuration status
        rprint(f"\n[bold]Configuration:[/bold]")
        rprint(f"Database Path: {config_manager.get('data.db_path')}")
        rprint(f"Symbols: {', '.join(config_manager.get('data.symbols', []))}")
        rprint(f"Date Range: {config_manager.get('data.start_date')} to {config_manager.get('data.end_date')}")

        # Model status
        model_dir = Path(config_manager.get('modeling.model_dir', 'models'))
        if model_dir.exists():
            model_files = list(model_dir.glob("*.joblib"))
            if model_files:
                rprint(f"\n[bold]Trained Models:[/bold] {len(model_files)}")
                for model_file in model_files[-3:]:  # Show last 3
                    rprint(f"  • {model_file.name}")
            else:
                rprint(f"\n[yellow]No trained models found[/yellow]")

        # Report status
        report_dir = Path(config_manager.get('reporting.output_dir', 'reports'))
        if report_dir.exists():
            report_files = list(report_dir.glob("*.html"))
            if report_files:
                rprint(f"\n[bold]Generated Reports:[/bold] {len(report_files)}")
                for report_file in sorted(report_files, key=lambda x: x.stat().st_mtime)[-3:]:
                    rprint(f"  • {report_file.name}")

    except Exception as e:
        handle_error("get status", e)


@app.command()
def pipeline(
    symbols: Optional[str] = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Comma-separated list of symbols"
    ),
    model: str = typer.Option(
        "rf",
        "--model",
        "-m",
        help="Model to use ('logreg' or 'rf')"
    ),
    skip_data: bool = typer.Option(
        False,
        "--skip-data",
        help="Skip data fetching"
    ),
    skip_features: bool = typer.Option(
        False,
        "--skip-features",
        help="Skip feature building"
    ),
    hyperparameter_search: bool = typer.Option(
        False,
        "--search",
        help="Perform hyperparameter search"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """Run complete pipeline from data to report."""
    setup_globals(config, verbose)

    console.print("[bold]Starting complete pipeline...[/bold]")

    try:
        # Parse symbols if provided
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]

        start_time = datetime.now()

        # Step 1: Fetch data
        if not skip_data:
            console.print("\n[bold blue]Step 1: Fetching data...[/bold blue]")
            with console.status("[bold green]Fetching market data..."):
                db = DatabaseManager(config_manager.get('data.db_path'))
                data_ingestor = DataIngestor(config_manager, db)
                results = data_ingestor.run_ingestion(symbols=symbol_list)

                if not results['success']:
                    handle_error("fetch data", Exception(results['error']))

                console.print(f"[green]✓[/green] Fetched {results['rows_fetched']:,} rows")

        # Step 2: Build features
        if not skip_features:
            console.print("\n[bold blue]Step 2: Building features and labels...[/bold blue]")
            with console.status("[bold green]Engineering features..."):
                db = DatabaseManager(config_manager.get('data.db_path'))
                feature_engineer = FeatureEngineer(config_manager, db)
                label_generator = LabelGenerator(config_manager, db)

                features_df = feature_engineer.build_features_from_db(symbols=symbol_list)
                labels_df = label_generator.generate_labels_from_db(symbols=symbol_list)

                console.print(f"[green]✓[/green] Built {len(features_df):,} features and {len(labels_df):,} labels")

        # Step 3: Train model
        console.print(f"\n[bold blue]Step 3: Training {model} model...[/bold blue]")
        with console.status(f"[bold green]Training {model}..."):
            db = DatabaseManager(config_manager.get('data.db_path'))
            model_trainer = ModelTrainer(config_manager, db)

            train_results = model_trainer.train_model(
                model_name=model,
                hyperparameter_search=hyperparameter_search
            )

            test_acc = train_results.get('test_metrics', {}).get('accuracy', 0)
            console.print(f"[green]✓[/green] Model trained (Test Accuracy: {test_acc:.3f})")

        # Step 4: Run backtest
        console.print(f"\n[bold blue]Step 4: Running backtest...[/bold blue]")
        with console.status("[bold green]Backtesting strategy..."):
            backtest_engine = BacktestEngine(config_manager, db)

            backtest_results = backtest_engine.run_backtest(model_name=model)

            perf = backtest_results['performance_metrics']
            console.print(f"[green]✓[/green] Backtest completed (Sharpe: {perf['sharpe_ratio']:.3f}, Return: {perf['total_return']:.2%})")

        # Step 5: Generate report
        console.print(f"\n[bold blue]Step 5: Generating report...[/bold blue]")
        with console.status("[bold green]Creating report..."):
            report_generator = ReportGenerator(config_manager, db)
            report_path = report_generator.generate_html_report(model_name=model)

            console.print(f"[green]✓[/green] Report generated: {Path(report_path).name}")

        # Pipeline complete
        duration = (datetime.now() - start_time).total_seconds()

        console.print(f"\n[bold green]🎉 Pipeline completed successfully in {duration:.1f}s![/bold green]")
        console.print(f"\n[bold]Results Summary:[/bold]")
        console.print(f"• Model: {model.upper()}")
        console.print(f"• Test Accuracy: {test_acc:.1%}")
        console.print(f"• Strategy Return: {perf['total_return']:.2%}")
        console.print(f"• Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        console.print(f"• Max Drawdown: {perf['max_drawdown']:.2%}")
        console.print(f"• Report: {report_path}")

    except Exception as e:
        handle_error("run pipeline", e)


@app.command()
def config_info(
    section: Optional[str] = typer.Option(
        None,
        "--section",
        "-s",
        help="Show specific configuration section"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file"
    )
):
    """Display configuration information."""
    try:
        config_manager = ConfigManager(config_path=config)

        if section:
            section_config = config_manager.get_section(section)
            if section_config:
                console.print(f"[bold]Configuration - {section.title()}:[/bold]")
                import yaml
                console.print(yaml.dump({section: section_config}, default_flow_style=False, indent=2))
            else:
                console.print(f"[red]Section '{section}' not found[/red]")
                console.print(f"Available sections: {', '.join(config_manager.list_keys())}")
        else:
            console.print("[bold]Configuration Overview:[/bold]")
            config_manager.print_config()

    except Exception as e:
        handle_error("display configuration", e)


if __name__ == "__main__":
    app()
