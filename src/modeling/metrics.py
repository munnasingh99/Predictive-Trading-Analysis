"""
Model evaluation metrics module for predictive trading signals.

This module provides comprehensive evaluation metrics for classification models
used in trading signal generation, including standard ML metrics and
trading-specific performance measures.

Key features:
- Standard classification metrics (accuracy, precision, recall, F1, AUC)
- Trading-specific metrics (hit rate, win/loss ratio, etc.)
- Time-series aware metric calculations
- Confidence intervals and statistical significance tests
- Metric visualization utilities
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Comprehensive model evaluation metrics."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics = {}

    def calculate_classification_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None,
                                       average: str = 'binary') -> Dict[str, float]:
        """Calculate standard classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            average: Averaging strategy for multi-class

        Returns:
            Dictionary of classification metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average)
        metrics['recall'] = recall_score(y_true, y_pred, average=average)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        if len(cm) == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)

            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Balanced accuracy
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['balanced_accuracy'] = (sensitivity + metrics['specificity']) / 2

        # Probability-based metrics
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['log_loss'] = log_loss(y_true, y_proba)

            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            metrics['pr_auc'] = auc(recall_curve, precision_curve)

        return metrics

    def calculate_trading_metrics(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate trading-specific metrics.

        Args:
            y_true: True labels (1 for up, 0 for down)
            y_pred: Predicted labels
            returns: Actual returns (optional)

        Returns:
            Dictionary of trading metrics
        """
        metrics = {}

        # Hit rate (same as accuracy for binary)
        metrics['hit_rate'] = accuracy_score(y_true, y_pred)

        # True/False positive/negative counts
        cm = confusion_matrix(y_true, y_pred)
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()

            # Win/Loss statistics
            total_predictions = len(y_pred)
            correct_predictions = tp + tn
            incorrect_predictions = fp + fn

            metrics['total_trades'] = total_predictions
            metrics['winning_trades'] = correct_predictions
            metrics['losing_trades'] = incorrect_predictions

            # Win/Loss ratio
            if incorrect_predictions > 0:
                metrics['win_loss_ratio'] = correct_predictions / incorrect_predictions
            else:
                metrics['win_loss_ratio'] = float('inf')

        # Return-based metrics if returns are provided
        if returns is not None:
            metrics.update(self._calculate_return_metrics(y_true, y_pred, returns))

        return metrics

    def _calculate_return_metrics(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                returns: np.ndarray) -> Dict[str, float]:
        """Calculate return-based trading metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            returns: Actual returns

        Returns:
            Dictionary of return-based metrics
        """
        metrics = {}

        # Strategy returns (assuming we trade based on predictions)
        # If we predict up (1), we go long; if we predict down (0), we go short
        strategy_returns = np.where(y_pred == 1, returns, -returns)

        # Average returns for different prediction outcomes
        correct_mask = (y_true == y_pred)
        incorrect_mask = ~correct_mask

        if np.sum(correct_mask) > 0:
            metrics['avg_return_correct'] = np.mean(strategy_returns[correct_mask])
        else:
            metrics['avg_return_correct'] = 0.0

        if np.sum(incorrect_mask) > 0:
            metrics['avg_return_incorrect'] = np.mean(strategy_returns[incorrect_mask])
        else:
            metrics['avg_return_incorrect'] = 0.0

        # Separate win/loss analysis
        win_mask = strategy_returns > 0
        loss_mask = strategy_returns <= 0

        if np.sum(win_mask) > 0:
            metrics['avg_win'] = np.mean(strategy_returns[win_mask])
            metrics['win_rate'] = np.sum(win_mask) / len(strategy_returns)
        else:
            metrics['avg_win'] = 0.0
            metrics['win_rate'] = 0.0

        if np.sum(loss_mask) > 0:
            metrics['avg_loss'] = np.mean(strategy_returns[loss_mask])
            metrics['loss_rate'] = np.sum(loss_mask) / len(strategy_returns)
        else:
            metrics['avg_loss'] = 0.0
            metrics['loss_rate'] = 0.0

        # Profit factor
        total_wins = np.sum(strategy_returns[win_mask]) if np.sum(win_mask) > 0 else 0
        total_losses = abs(np.sum(strategy_returns[loss_mask])) if np.sum(loss_mask) > 0 else 0

        if total_losses > 0:
            metrics['profit_factor'] = total_wins / total_losses
        else:
            metrics['profit_factor'] = float('inf') if total_wins > 0 else 0.0

        # Total strategy return
        metrics['total_return'] = np.sum(strategy_returns)
        metrics['mean_return'] = np.mean(strategy_returns)
        metrics['return_std'] = np.std(strategy_returns)

        # Sharpe ratio (assuming daily returns, annualized)
        if metrics['return_std'] > 0:
            metrics['sharpe_ratio'] = (metrics['mean_return'] * np.sqrt(252)) / (metrics['return_std'] * np.sqrt(252))
        else:
            metrics['sharpe_ratio'] = 0.0

        return metrics

    def calculate_threshold_metrics(self,
                                  y_true: np.ndarray,
                                  y_proba: np.ndarray,
                                  thresholds: Optional[List[float]] = None) -> pd.DataFrame:
        """Calculate metrics for different probability thresholds.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            thresholds: List of thresholds to evaluate

        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)

        results = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Calculate metrics
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }

            # Add confusion matrix elements
            cm = confusion_matrix(y_true, y_pred)
            if len(cm) == 2:
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                    'positive_predictions': int(tp + fp),
                    'negative_predictions': int(tn + fn)
                })

            results.append(metrics)

        return pd.DataFrame(results)

    def calculate_confidence_intervals(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     metric: str = 'accuracy',
                                     confidence: float = 0.95,
                                     n_bootstrap: int = 1000) -> Tuple[float, float, float]:
        """Calculate confidence intervals for metrics using bootstrap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric: Metric to calculate CI for
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (point_estimate, lower_ci, upper_ci)
        """
        np.random.seed(42)  # For reproducibility

        # Define metric function
        metric_funcs = {
            'accuracy': accuracy_score,
            'precision': lambda yt, yp: precision_score(yt, yp, zero_division=0),
            'recall': lambda yt, yp: recall_score(yt, yp, zero_division=0),
            'f1_score': lambda yt, yp: f1_score(yt, yp, zero_division=0)
        }

        if metric not in metric_funcs:
            raise ValueError(f"Unsupported metric: {metric}")

        metric_func = metric_funcs[metric]

        # Point estimate
        point_estimate = metric_func(y_true, y_pred)

        # Bootstrap samples
        n_samples = len(y_true)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Calculate metric
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)

        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_ci = np.percentile(bootstrap_scores, lower_percentile)
        upper_ci = np.percentile(bootstrap_scores, upper_percentile)

        return point_estimate, lower_ci, upper_ci

    def statistical_significance_test(self,
                                    y_true: np.ndarray,
                                    y_pred1: np.ndarray,
                                    y_pred2: np.ndarray,
                                    metric: str = 'accuracy') -> Dict[str, Union[float, bool]]:
        """Test statistical significance between two models.

        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            metric: Metric to compare

        Returns:
            Dictionary with test results
        """
        # Calculate individual sample scores for both models
        if metric == 'accuracy':
            scores1 = (y_true == y_pred1).astype(float)
            scores2 = (y_true == y_pred2).astype(float)
        else:
            raise ValueError(f"Statistical test not implemented for metric: {metric}")

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        # Effect size (Cohen's d)
        mean_diff = np.mean(scores1) - np.mean(scores2)
        pooled_std = np.sqrt(((np.var(scores1) + np.var(scores2)) / 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        return {
            'metric': metric,
            'model1_mean': np.mean(scores1),
            'model2_mean': np.mean(scores2),
            'mean_difference': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_cohens_d(abs(cohens_d))
        }

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            d: Cohen's d value (absolute)

        Returns:
            Effect size interpretation
        """
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def calculate_learning_curve_metrics(self,
                                       train_sizes: List[int],
                                       train_scores: List[float],
                                       val_scores: List[float]) -> Dict[str, Union[float, bool]]:
        """Analyze learning curve metrics.

        Args:
            train_sizes: Training set sizes
            val_scores: Validation scores for each size
            train_scores: Training scores for each size

        Returns:
            Dictionary with learning curve analysis
        """
        metrics = {}

        # Final performance
        metrics['final_train_score'] = train_scores[-1]
        metrics['final_val_score'] = val_scores[-1]

        # Overfitting detection
        score_gap = train_scores[-1] - val_scores[-1]
        metrics['train_val_gap'] = score_gap
        metrics['is_overfitting'] = score_gap > 0.1  # 10% threshold

        # Learning efficiency
        if len(val_scores) >= 2:
            early_score = val_scores[len(val_scores)//2]  # Mid-point score
            final_score = val_scores[-1]
            metrics['score_improvement'] = final_score - early_score
            metrics['is_still_learning'] = metrics['score_improvement'] > 0.01  # 1% threshold

        # Stability
        if len(val_scores) >= 3:
            last_three_scores = val_scores[-3:]
            metrics['score_stability'] = np.std(last_three_scores)
            metrics['is_stable'] = metrics['score_stability'] < 0.02  # 2% threshold

        return metrics

    def comprehensive_evaluation(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None,
                               returns: Optional[np.ndarray] = None,
                               model_name: str = "Model") -> Dict[str, any]:
        """Perform comprehensive model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            returns: Actual returns (optional)
            model_name: Name of the model

        Returns:
            Dictionary with all evaluation metrics
        """
        evaluation = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'n_positive': int(np.sum(y_true)),
            'n_negative': int(len(y_true) - np.sum(y_true)),
            'positive_rate': np.mean(y_true)
        }

        # Classification metrics
        classification_metrics = self.calculate_classification_metrics(y_true, y_pred, y_proba)
        evaluation['classification'] = classification_metrics

        # Trading metrics
        trading_metrics = self.calculate_trading_metrics(y_true, y_pred, returns)
        evaluation['trading'] = trading_metrics

        # Confidence intervals for key metrics
        try:
            acc_est, acc_lower, acc_upper = self.calculate_confidence_intervals(y_true, y_pred, 'accuracy')
            evaluation['confidence_intervals'] = {
                'accuracy': {'estimate': acc_est, 'lower': acc_lower, 'upper': acc_upper}
            }
        except Exception as e:
            logger.warning(f"Failed to calculate confidence intervals: {e}")

        # ROC and PR curves data (if probabilities available)
        if y_proba is not None:
            try:
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)

                evaluation['curves'] = {
                    'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': roc_thresholds.tolist()},
                    'pr': {'precision': precision.tolist(), 'recall': recall.tolist(), 'thresholds': pr_thresholds.tolist()}
                }
            except Exception as e:
                logger.warning(f"Failed to calculate curve data: {e}")

        return evaluation


def compare_models(evaluations: List[Dict[str, any]]) -> pd.DataFrame:
    """Compare multiple model evaluations.

    Args:
        evaluations: List of evaluation dictionaries from comprehensive_evaluation

    Returns:
        DataFrame comparing models across key metrics
    """
    comparison_data = []

    for eval_dict in evaluations:
        model_data = {
            'model_name': eval_dict['model_name'],
            'n_samples': eval_dict['n_samples'],
            'accuracy': eval_dict['classification']['accuracy'],
            'precision': eval_dict['classification']['precision'],
            'recall': eval_dict['classification']['recall'],
            'f1_score': eval_dict['classification']['f1_score'],
            'hit_rate': eval_dict['trading']['hit_rate']
        }

        # Add optional metrics if available
        if 'roc_auc' in eval_dict['classification']:
            model_data['roc_auc'] = eval_dict['classification']['roc_auc']

        if 'avg_return_correct' in eval_dict['trading']:
            model_data['avg_return_correct'] = eval_dict['trading']['avg_return_correct']

        if 'sharpe_ratio' in eval_dict['trading']:
            model_data['sharpe_ratio'] = eval_dict['trading']['sharpe_ratio']

        comparison_data.append(model_data)

    comparison_df = pd.DataFrame(comparison_data)

    # Rank models by key metrics
    comparison_df['accuracy_rank'] = comparison_df['accuracy'].rank(ascending=False)
    comparison_df['f1_rank'] = comparison_df['f1_score'].rank(ascending=False)

    if 'roc_auc' in comparison_df.columns:
        comparison_df['auc_rank'] = comparison_df['roc_auc'].rank(ascending=False)

    return comparison_df


def calculate_class_imbalance_metrics(y_true: np.ndarray) -> Dict[str, float]:
    """Calculate metrics related to class imbalance.

    Args:
        y_true: True labels

    Returns:
        Dictionary with imbalance metrics
    """
    unique_labels, counts = np.unique(y_true, return_counts=True)

    metrics = {
        'n_classes': len(unique_labels),
        'class_counts': dict(zip(unique_labels.astype(int), counts.astype(int))),
        'total_samples': len(y_true)
    }

    if len(unique_labels) == 2:
        majority_count = max(counts)
        minority_count = min(counts)

        metrics['majority_class_size'] = int(majority_count)
        metrics['minority_class_size'] = int(minority_count)
        metrics['imbalance_ratio'] = majority_count / minority_count
        metrics['minority_class_percentage'] = (minority_count / len(y_true)) * 100

        # Classify imbalance severity
        if metrics['imbalance_ratio'] <= 1.5:
            metrics['imbalance_severity'] = 'balanced'
        elif metrics['imbalance_ratio'] <= 4:
            metrics['imbalance_severity'] = 'moderate'
        elif metrics['imbalance_ratio'] <= 9:
            metrics['imbalance_severity'] = 'severe'
        else:
            metrics['imbalance_severity'] = 'extreme'

    return metrics
