"""
Time-series utilities for helicity_dynamics.

Data loading, preprocessing, and windowing helpers.
"""

import numpy as np
from typing import Tuple, Optional, Generator
import csv
import os


def load_csv(
    path: str,
    skip_header: bool = True,
    delimiter: str = ",",
    use_cols: Optional[slice] = None,
) -> np.ndarray:
    """Load a CSV file as a (n_steps, n_features) array.

    Parameters
    ----------
    path : str
        Path to CSV file.
    skip_header : bool
        Skip the first row (header). Default: True.
    delimiter : str
        Column delimiter. Default: ",".
    use_cols : slice, optional
        Column slice to use. Default: all numeric columns.

    Returns
    -------
    np.ndarray, shape (n_rows, n_cols)
    """
    data = []
    with open(path, "r") as f:
        if skip_header:
            next(f)
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            try:
                vals = [float(x.strip()) for x in row if x.strip()]
                if use_cols is not None:
                    vals = vals[use_cols]
                if vals:
                    data.append(vals)
            except (ValueError, IndexError):
                continue

    return np.array(data)


def load_npy(path: str) -> np.ndarray:
    """Load a .npy or .npz file.

    For .npz, returns the first array.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        # Return the first array
        for key in data.files:
            return data[key]
        raise ValueError(f"No arrays found in {path}")
    return np.load(path)


def normalize(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score normalise: (data - mean) / std, per feature.

    Parameters
    ----------
    data : np.ndarray
    axis : int
        Axis along which to compute mean/std.

    Returns
    -------
    np.ndarray
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    std = np.where(std < 1e-10, 1.0, std)
    return (data - mean) / std


def detrend(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Remove linear trend (subtract best-fit line).

    Parameters
    ----------
    data : np.ndarray
    axis : int
        Time axis.

    Returns
    -------
    np.ndarray
    """
    n = data.shape[axis]
    x = np.arange(n, dtype=float)
    x_mean = x.mean()

    # Move time axis to position 0 for broadcasting
    d = np.moveaxis(data, axis, 0)
    trend = np.zeros_like(d)

    for idx in np.ndindex(d.shape[1:]):
        y = d[(slice(None),) + idx]
        slope = np.sum((x - x_mean) * y) / np.sum((x - x_mean) ** 2)
        intercept = y.mean() - slope * x_mean
        trend[(slice(None),) + idx] = slope * x + intercept

    return data - np.moveaxis(trend, 0, axis)


def sliding_window(
    data: np.ndarray,
    window_size: int,
    step: int = 1,
) -> Generator[np.ndarray, None, None]:
    """Generate sliding windows over a time series.

    Parameters
    ----------
    data : np.ndarray, shape (n_steps, n_features)
    window_size : int
        Number of steps per window.
    step : int
        Step size between windows.

    Yields
    ------
    np.ndarray, shape (window_size, n_features)
    """
    n = data.shape[0]
    for i in range(0, n - window_size + 1, step):
        yield data[i : i + window_size]
