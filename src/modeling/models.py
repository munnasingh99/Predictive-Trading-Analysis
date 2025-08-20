"""
Machine learning models module for predictive trading signals.

This module provides model training, evaluation, and prediction functionality
with time-series cross-validation to prevent lookahead bias.

Key features:
- Time-series aware cross-validation
- Multiple model architectures (Logistic Regression, Random Forest)
- Hyperparameter optimization
- Feature preprocessing and scaling
- Model persistence and versioning
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.data.db import DatabaseManager
from src.modeling.splits import TimeSeriesSplitter
from src.utils.config import ConfigManager

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, validation, and prediction."""

    def __init__(self, config: Union[Dict, ConfigManager], db: Optional[DatabaseManager] = None):
        """Initialize model trainer.

        Args:
            config: Configuration dict or ConfigManager instance
            db: Optional DatabaseManager instance
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.config

        self.db = db or DatabaseManager(self.config.get('data', {}).get('db_path', 'data/trading.db'))

        # Model configuration
        self.modeling_config = self.config.get('modeling', {})
        self.models_config = self.modeling_config.get('models', {})

        # Dates for train/validation/test split
        self.train_end = self.modeling_config.get('train_end', '2019-12-31')
        self.validation_start = self.modeling_config.get('validation_start', '2015-01-01')
        self.test_start = self.modeling_config.get('test_start', '2020-01-01')

        # Cross-validation settings
        self.cv_method = self.modeling_config.get('cv_method', 'TimeSeriesSplit')
        self.cv_folds = self.modeling_config.get('cv_folds', 5)
        self.cv_gap = self.modeling_config.get('cv_gap', 1)

        # Feature preprocessing
        self.scale_features = self.modeling_config.get('scale_features', True)
        self.handle_missing = self.modeling_config.get('handle_missing', 'drop')

        # Model persistence
        self.model_dir = Path(self.modeling_config.get('model_dir', 'models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize time series splitter
        self.splitter = TimeSeriesSplitter(
            train_end=self.train_end,
            validation_start=self.validation_start,
            test_start=self.test_start,
            cv_folds=self.cv_folds,
            gap_days=self.cv_gap
        )

        logger.info("ModelTrainer initialized")

    def get_model_pipeline(self, model_name: str) -> Pipeline:
        """Create a scikit-learn pipeline for the specified model.

        Args:
            model_name: Name of the model ('logreg', 'rf')

        Returns:
            Sklearn Pipeline with preprocessor and model
        """
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not found in configuration")

        model_params = self.models_config[model_name].copy()

        # Create preprocessing steps
        steps = []

        if self.scale_features:
            steps.append(('scaler', StandardScaler()))

        # Create model
        if model_name == 'logreg':
            model = LogisticRegression(**model_params)
        elif model_name == 'rf':
            model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        steps.append(('model', model))

        pipeline = Pipeline(steps)

        logger.debug(f"Created {model_name} pipeline with {len(steps)} steps")
        return pipeline

    def prepare_features_and_labels(self, symbols: Optional[List[str]] = None,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features and labels from database.

        Args:
            symbols: Optional list of symbols to filter
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Tuple of (DataFrame with features and labels, list of feature columns)
        """
        # Get features and labels from database
        data = self.db.get_features_and_labels(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        if data.empty:
            raise ValueError("No features and labels found in database")

        # Sort by symbol and date
        data = data.sort_values(['symbol', 'date']).reset_index(drop=True)

        # Handle missing labels
        if data['y_next'].isna().any():
            logger.warning(f"Removing {data['y_next'].isna().sum()} rows with missing labels")
            data = data.dropna(subset=['y_next'])

        # Get feature columns (exclude symbol, date, y_next)
        feature_cols = [col for col in data.columns if col not in ['symbol', 'date', 'y_next']]

        if not feature_cols:
            raise ValueError("No feature columns found")

        # Handle missing features
        if self.handle_missing == 'drop':
            original_len = len(data)
            data = data.dropna(subset=feature_cols)
            if len(data) < original_len:
                logger.info(f"Dropped {original_len - len(data)} rows with missing features")

        elif self.handle_missing == 'fill':
            # Forward fill then backward fill
            for col in feature_cols:
                data[col] = data.groupby('symbol')[col].fillna(method='ffill').fillna(method='bfill')

        elif self.handle_missing == 'interpolate':
            # Linear interpolation within each symbol
            for col in feature_cols:
                data[col] = data.groupby('symbol')[col].interpolate(method='linear')

        logger.info(f"Prepared dataset: {len(data)} rows, {len(feature_cols)} features, "
                   f"{data['symbol'].nunique()} symbols")

        return data, feature_cols

    def train_model(self, model_name: str,
                   data: Optional[pd.DataFrame] = None,
                   feature_cols: Optional[List[str]] = None,
                   hyperparameter_search: bool = False) -> Dict[str, Any]:
        """Train a model with time-series cross-validation.

        Args:
            model_name: Name of model to train
            data: Optional DataFrame with features and labels
            feature_cols: Optional list of feature column names
            hyperparameter_search: Whether to perform hyperparameter search

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_name} model")

        # Prepare data if not provided
        if data is None or feature_cols is None:
            data, feature_cols = self.prepare_features_and_labels()

        # Get train/validation/test splits
        train_data, val_data, test_data = self.splitter.split_data(data)

        if train_data.empty:
            raise ValueError("No training data available")

        # Prepare features and labels
        X_train = train_data[feature_cols]
        y_train = train_data['y_next']
        X_val = val_data[feature_cols] if not val_data.empty else None
        y_val = val_data['y_next'] if not val_data.empty else None
        X_test = test_data[feature_cols] if not test_data.empty else None
        y_test = test_data['y_next'] if not test_data.empty else None

        logger.info(f"Training set: {len(X_train)} samples")
        if X_val is not None:
            logger.info(f"Validation set: {len(X_val)} samples")
        if X_test is not None:
            logger.info(f"Test set: {len(X_test)} samples")

        # Get base pipeline
        pipeline = self.get_model_pipeline(model_name)

        # Perform hyperparameter search if requested
        if hyperparameter_search:
            pipeline = self._hyperparameter_search(pipeline, model_name, X_train, y_train)

        # Train final model
        pipeline.fit(X_train, y_train)

        # Evaluate model
        results = self._evaluate_model(
            pipeline, model_name,
            X_train, y_train, X_val, y_val, X_test, y_test,
            train_data, val_data, test_data
        )

        # Save model
        model_path = self._save_model(pipeline, model_name, results)
        results['model_path'] = model_path

        # Save predictions to database
        self._save_predictions(pipeline, model_name, data, feature_cols)

        logger.info(f"Model training completed: {model_name}")
        return results

    def _hyperparameter_search(self, pipeline: Pipeline, model_name: str,
                              X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Perform hyperparameter search using time-series cross-validation.

        Args:
            pipeline: Base pipeline
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels

        Returns:
            Pipeline with best hyperparameters
        """
        logger.info(f"Performing hyperparameter search for {model_name}")

        # Define parameter grids
        param_grids = {
            'logreg': {
                'model__C': [0.1, 1.0, 10.0],
                'model__class_weight': ['balanced', None],
                'model__max_iter': [1000, 2000]
            },
            'rf': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, None],
                'model__min_samples_split': [5, 10, 20],
                'model__min_samples_leaf': [2, 5, 10],
                'model__class_weight': ['balanced', None]
            }
        }

        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}, skipping search")
            return pipeline

        # Create time series CV splits for hyperparameter search
        cv_splits = self.splitter.get_cv_splits(X_train, y_train)

        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=cv_splits,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def _evaluate_model(self, pipeline: Pipeline, model_name: str,
                       X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: Optional[pd.DataFrame], y_val: Optional[pd.Series],
                       X_test: Optional[pd.DataFrame], y_test: Optional[pd.Series],
                       train_data: pd.DataFrame, val_data: pd.DataFrame,
                       test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance on train/validation/test sets.

        Args:
            pipeline: Trained pipeline
            model_name: Name of the model
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            X_test, y_test: Test data (optional)
            train_data, val_data, test_data: Full datasets with metadata

        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(X_train.columns),
            'feature_names': list(X_train.columns)
        }

        # Training metrics
        y_train_pred = pipeline.predict(X_train)
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]

        results['train_metrics'] = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_proba),
            'samples': len(y_train),
            'positive_rate': y_train.mean()
        }

        # Validation metrics
        if X_val is not None and not X_val.empty:
            y_val_pred = pipeline.predict(X_val)
            y_val_proba = pipeline.predict_proba(X_val)[:, 1]

            results['val_metrics'] = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'roc_auc': roc_auc_score(y_val, y_val_proba),
                'samples': len(y_val),
                'positive_rate': y_val.mean()
            }

        # Test metrics
        if X_test is not None and not X_test.empty:
            y_test_pred = pipeline.predict(X_test)
            y_test_proba = pipeline.predict_proba(X_test)[:, 1]

            results['test_metrics'] = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'roc_auc': roc_auc_score(y_test, y_test_proba),
                'samples': len(y_test),
                'positive_rate': y_test.mean(),
                'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
            }

        # Cross-validation metrics
        cv_scores = self._cross_validate_model(pipeline, model_name, X_train, y_train)
        results['cv_metrics'] = cv_scores

        # Feature importance (if available)
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importances = pipeline.named_steps['model'].feature_importances_
            feature_importance = dict(zip(X_train.columns, importances))
            results['feature_importance'] = feature_importance

        elif hasattr(pipeline.named_steps['model'], 'coef_'):
            coefficients = pipeline.named_steps['model'].coef_[0]
            feature_importance = dict(zip(X_train.columns, np.abs(coefficients)))
            results['feature_importance'] = feature_importance

        self._log_results(results)
        return results

    def _cross_validate_model(self, pipeline: Pipeline, model_name: str,
                             X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform time-series cross-validation.

        Args:
            pipeline: Model pipeline
            model_name: Name of the model
            X: Features
            y: Labels

        Returns:
            Dictionary with CV metrics
        """
        cv_splits = self.splitter.get_cv_splits(X, y)

        cv_scores = {
            'accuracy_scores': [],
            'roc_auc_scores': []
        }

        for train_idx, val_idx in cv_splits:
            X_train_cv = X.iloc[train_idx]
            y_train_cv = y.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_val_cv = y.iloc[val_idx]

            # Clone and train pipeline
            cv_pipeline = Pipeline(pipeline.steps)
            cv_pipeline.fit(X_train_cv, y_train_cv)

            # Predict and score
            y_pred_cv = cv_pipeline.predict(X_val_cv)
            y_proba_cv = cv_pipeline.predict_proba(X_val_cv)[:, 1]

            cv_scores['accuracy_scores'].append(accuracy_score(y_val_cv, y_pred_cv))
            cv_scores['roc_auc_scores'].append(roc_auc_score(y_val_cv, y_proba_cv))

        # Calculate summary statistics
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)

        logger.info(f"CV Results - Accuracy: {cv_results['accuracy_scores_mean']:.4f} ± {cv_results['accuracy_scores_std']:.4f}")
        logger.info(f"CV Results - ROC AUC: {cv_results['roc_auc_scores_mean']:.4f} ± {cv_results['roc_auc_scores_std']:.4f}")

        return cv_results

    def _save_model(self, pipeline: Pipeline, model_name: str, results: Dict[str, Any]) -> str:
        """Save trained model to disk.

        Args:
            pipeline: Trained pipeline
            model_name: Name of the model
            results: Training results

        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = self.model_dir / model_filename

        # Save model and metadata
        model_data = {
            'pipeline': pipeline,
            'results': results,
            'config': self.config
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved: {model_path}")

        return str(model_path)

    def _save_predictions(self, pipeline: Pipeline, model_name: str,
                         data: pd.DataFrame, feature_cols: List[str]) -> None:
        """Save model predictions to database.

        Args:
            pipeline: Trained pipeline
            model_name: Name of the model
            data: Full dataset
            feature_cols: List of feature columns
        """
        logger.info("Saving predictions to database")

        # Make predictions
        X = data[feature_cols]
        y_proba = pipeline.predict_proba(X)[:, 1]
        y_pred = pipeline.predict(X)

        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'symbol': data['symbol'],
            'date': data['date'],
            'model_name': model_name,
            'proba': y_proba,
            'pred_label': y_pred,
            'timestamp': datetime.now().isoformat()
        })

        # Clear existing predictions for this model
        self.db.clear_table('predictions', model_name=model_name)

        # Store new predictions
        self.db.insert_predictions(predictions_df)

        logger.info(f"Saved {len(predictions_df)} predictions for {model_name}")

    def _log_results(self, results: Dict[str, Any]) -> None:
        """Log training results.

        Args:
            results: Training results dictionary
        """
        model_name = results['model_name']

        logger.info(f"=== {model_name.upper()} TRAINING RESULTS ===")

        # Training metrics
        train_acc = results['train_metrics']['accuracy']
        train_auc = results['train_metrics']['roc_auc']
        logger.info(f"Training - Accuracy: {train_acc:.4f}, ROC AUC: {train_auc:.4f}")

        # Validation metrics
        if 'val_metrics' in results:
            val_acc = results['val_metrics']['accuracy']
            val_auc = results['val_metrics']['roc_auc']
            logger.info(f"Validation - Accuracy: {val_acc:.4f}, ROC AUC: {val_auc:.4f}")

        # Test metrics
        if 'test_metrics' in results:
            test_acc = results['test_metrics']['accuracy']
            test_auc = results['test_metrics']['roc_auc']
            logger.info(f"Test - Accuracy: {test_acc:.4f}, ROC AUC: {test_auc:.4f}")

        # Feature importance (top 10)
        if 'feature_importance' in results:
            importance = results['feature_importance']
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info("Top 10 Features:")
            for feature, score in top_features:
                logger.info(f"  {feature}: {score:.4f}")

    def load_model(self, model_path: str) -> Tuple[Pipeline, Dict[str, Any]]:
        """Load a saved model from disk.

        Args:
            model_path: Path to saved model

        Returns:
            Tuple of (pipeline, results)
        """
        model_data = joblib.load(model_path)
        return model_data['pipeline'], model_data['results']

    def predict(self, model_path: str, data: pd.DataFrame,
               feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Make predictions using a saved model.

        Args:
            model_path: Path to saved model
            data: DataFrame with features
            feature_cols: Optional list of feature columns

        Returns:
            DataFrame with predictions
        """
        pipeline, results = self.load_model(model_path)

        if feature_cols is None:
            feature_cols = results['feature_names']

        X = data[feature_cols]
        y_proba = pipeline.predict_proba(X)[:, 1]
        y_pred = pipeline.predict(X)

        predictions_df = data[['symbol', 'date']].copy()
        predictions_df['proba'] = y_proba
        predictions_df['pred_label'] = y_pred

        return predictions_df

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained models.

        Returns:
            Dictionary with model summary
        """
        model_files = list(self.model_dir.glob("*.joblib"))

        summary = {
            'total_models': len(model_files),
            'models': []
        }

        for model_file in model_files:
            try:
                model_data = joblib.load(model_file)
                results = model_data['results']

                model_info = {
                    'filename': model_file.name,
                    'model_name': results['model_name'],
                    'training_date': results['training_date'],
                    'feature_count': results['feature_count'],
                }

                if 'test_metrics' in results:
                    model_info['test_accuracy'] = results['test_metrics']['accuracy']
                    model_info['test_roc_auc'] = results['test_metrics']['roc_auc']

                summary['models'].append(model_info)

            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")

        return summary

    def train_all_models(self, hyperparameter_search: bool = False) -> Dict[str, Dict[str, Any]]:
        """Train all configured models.

        Args:
            hyperparameter_search: Whether to perform hyperparameter search

        Returns:
            Dictionary with results for all models
        """
        logger.info("Training all models")

        # Prepare data once
        data, feature_cols = self.prepare_features_and_labels()

        all_results = {}

        for model_name in self.models_config.keys():
            try:
                logger.info(f"Training {model_name}")
                results = self.train_model(
                    model_name=model_name,
                    data=data,
                    feature_cols=feature_cols,
                    hyperparameter_search=hyperparameter_search
                )
                all_results[model_name] = results

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}

        logger.info(f"Completed training {len(all_results)} models")
        return all_results
