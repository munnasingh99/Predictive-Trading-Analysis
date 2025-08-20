.PHONY: help setup data features train backtest report test clean lint format check-deps

# Default target
help:
	@echo "Predictive Trading Signals - Build Automation"
	@echo ""
	@echo "Available targets:"
	@echo "  setup      - Install dependencies and setup environment"
	@echo "  data       - Fetch market data"
	@echo "  features   - Build features and labels"
	@echo "  train      - Train ML models"
	@echo "  backtest   - Run strategy backtest"
	@echo "  report     - Generate performance report"
	@echo "  test       - Run test suite"
	@echo "  clean      - Clean generated files"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code with black and isort"
	@echo "  check-deps - Check if dependencies are installed"
	@echo ""
	@echo "Quick start: make setup && make data && make features && make train && make backtest && make report"

# Setup environment and install dependencies
setup:
	@echo "Setting up environment..."
	pip install -r requirements.txt
	@echo "Creating necessary directories..."
	mkdir -p data/raw data/processed models reports logs
	@echo "Setup complete!"

# Check if key dependencies are available
check-deps:
	@echo "Checking dependencies..."
	@python -c "import pandas, numpy, sklearn, yfinance, ta; print('✓ Core dependencies available')"
	@python -c "import matplotlib, plotly; print('✓ Visualization libraries available')"
	@python -c "import sqlalchemy, yaml, typer; print('✓ Additional dependencies available')"

# Fetch market data
data: check-deps
	@echo "Fetching market data..."
	python src/cli.py fetch-data
	@echo "Data fetch complete!"

# Build features and labels
features: check-deps
	@echo "Building features and labels..."
	python src/cli.py build-features
	@echo "Feature engineering complete!"

# Train models
train: check-deps
	@echo "Training models..."
	python src/cli.py train --model logreg
	python src/cli.py train --model rf
	@echo "Model training complete!"

# Run backtest
backtest: check-deps
	@echo "Running backtest..."
	python src/cli.py backtest --model rf
	@echo "Backtest complete!"

# Generate report
report: check-deps
	@echo "Generating performance report..."
	python src/cli.py report
	@echo "Report generated in reports/"

# Run all steps in sequence
all: setup data features train backtest report
	@echo "Full pipeline complete!"

# Run tests
test:
	@echo "Running test suite..."
	pytest tests/ -v --tb=short
	@echo "Tests complete!"

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# Lint code
lint:
	@echo "Running linting..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "Linting complete!"

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "Code formatting complete!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/trading.db
	rm -rf models/*.joblib
	rm -rf reports/*.html
	rm -rf logs/*.log
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "Clean complete!"

# Development setup (includes pre-commit hooks)
dev-setup: setup
	@echo "Setting up development environment..."
	pip install pre-commit
	pre-commit install
	@echo "Development setup complete!"

# Quick validation (small dataset test)
validate:
	@echo "Running validation with sample data..."
	python src/cli.py fetch-data --symbols SPY --start 2020-01-01 --end 2022-12-31
	python src/cli.py build-features
	python src/cli.py train --model logreg
	python src/cli.py backtest --model logreg
	@echo "Validation complete!"

# CI/CD pipeline simulation
ci: format lint test
	@echo "CI pipeline complete!"
