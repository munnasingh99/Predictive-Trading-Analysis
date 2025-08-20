"""
Reporting package for predictive trading signals.

This package handles comprehensive report generation including:
- HTML report creation with interactive visualizations
- Performance metrics tables and charts
- Model comparison reports
- Daily signal reports for deployment
- Benchmark analysis and risk assessment
- Feature importance analysis

Key features:
- Interactive Plotly visualizations
- Professional HTML report templates
- Comprehensive performance analysis
- Model comparison utilities
- Daily signal generation reports

Main components:
- ReportGenerator: Creates comprehensive HTML reports with plots and tables
"""

from src.reporting.report import ReportGenerator

__all__ = [
    "ReportGenerator",
]
