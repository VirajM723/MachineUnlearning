"""Regression metrics reported in dollars of unpaid principal balance."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # R^2 replaces the original "accuracy = 1 - RMSE / range" score, which is
    # not an accuracy and stays near 100% even for a useless model.
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def format_metrics(m: dict[str, float]) -> str:
    return f"RMSE ${m['rmse']:,.0f} | MAE ${m['mae']:,.0f} | R² {m['r2']:.3f}"
