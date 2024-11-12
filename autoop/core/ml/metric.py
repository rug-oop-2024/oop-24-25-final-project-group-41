from abc import ABC, abstractmethod
# from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "root_mean_squared_error",
    "mean_absolute_error",
    "categorical_accuracy",
    "balanced_accuracy"
]


def get_metric(name: str):
    """
    Factory function to get a metric by name.

    Args:
        name: Name of the metric to instantiate

    Returns:
        Metric instance
    """
    metrics = {
        "mean_squared_error": MeanSquaredError(),
        "accuracy": Accuracy(),
        "root_mean_squared_error": RootMeanSquaredError(),
        "mean_absolute_error": MeanAbsoluteError(),
        "categorical_accuracy": CategoricalAccuracy(),
        "balanced_accuracy": BalancedAccuracy()
    }
    return metrics.get(name)


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the metric value.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            float: Metric value
        """
        pass

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Wrapper around __call__ to ensure numpy arrays."""
        return self.__call__(np.array(y_true), np.array(y_pred))


# Classification Metrics
class Accuracy(Metric):
    """
    Accuracy metric for classification.
    Calculates the fraction of predictions that match the ground truth.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


class CategoricalAccuracy(Metric):
    """
    Categorical Accuracy for multi-class classification.
    Similar to regular accuracy but ensures inputs are properly shaped.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return np.sum(y_true == y_pred) / len(y_true)


class BalancedAccuracy(Metric):
    """
    Balanced Accuracy for imbalanced classification problems.
    Calculates the average of recall obtained on each class.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        classes = np.unique(y_true)
        per_class_accuracy = []

        for cls in classes:
            # Get indices for this class
            idx = (y_true == cls)
            if np.any(idx):
                # Calculate accuracy for this class
                class_accuracy = np.mean(y_pred[idx] == y_true[idx])
                per_class_accuracy.append(class_accuracy)

        return np.mean(per_class_accuracy)


# Regression Metrics
class MeanSquaredError(Metric):
    """
    Mean Squared Error for regression.
    Calculates the average squared difference
    between predictions and ground truth.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class RootMeanSquaredError(Metric):
    """
    Root Mean Squared Error for regression.
    Square root of the average squared difference
    between predictions and ground truth.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error for regression.
    Calculates the average absolute difference
    between predictions and ground truth.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
