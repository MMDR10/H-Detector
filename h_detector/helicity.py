"""
Helicity (H) metric computation for LLM hidden states.

H measures the rotational-to-forward tension ratio in PCA-transformed
hidden state trajectories through the model's layers.

H ≈ 1   → Straightforward processing (factual)
H > 2   → Semantic tension (abstraction, conflict)
H spike → Potential jailbreak / role confusion
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional


def compute_helicity(
    hidden_states: np.ndarray,
    pca_components: int = 10,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute helicity (H) from layer-wise hidden state activations.

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (n_layers, hidden_dim). Hidden states from each layer.
    pca_components : int
        Number of PCA components to reduce to (default: 10).
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Helicity value H. Higher values indicate more rotational tension.
    """
    n_layers = hidden_states.shape[0]

    # Edge case: need at least 2 layers for gradient computation
    if n_layers < 2:
        return 1.0

    # Edge case: avoid PCA with too few samples
    n_components = min(pca_components, n_layers - 1, hidden_states.shape[1])
    if n_components < 2:
        # Fall back to raw norm ratio
        diffs = np.diff(hidden_states, axis=0)
        return float(np.linalg.norm(diffs) / (np.linalg.norm(hidden_states) + epsilon))

    # PCA reduction
    pca = PCA(n_components=n_components)
    traj = pca.fit_transform(hidden_states)

    # Compute tangent vectors (forward progression)
    tangents = np.diff(traj, axis=0)

    # Compute radial vectors (distance from origin / center)
    # Use centroid as reference
    centroid = np.mean(traj, axis=0)
    radials = traj[:-1] - centroid  # align with tangents dimension

    # Normalize vectors
    tan_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    rad_norm = np.linalg.norm(radials, axis=1, keepdims=True)

    tan_unit = tangents / (tan_norm + epsilon)
    rad_unit = radials / (rad_norm + epsilon)

    # Compute perpendicular component (rotational) and forward component
    # H = ||rotational|| / ||forward||
    forward = np.sum(tan_unit * rad_unit, axis=1, keepdims=True) * rad_unit
    rotational = tan_unit - forward

    forward_mag = np.linalg.norm(forward, axis=1)
    rotational_mag = np.linalg.norm(rotational, axis=1)

    # Layer-wise H values
    h_per_layer = rotational_mag / (forward_mag + epsilon)

    # Aggregate: mean H across all layers
    h = float(np.mean(h_per_layer))

    return h


def compute_layer_h(
    hidden_states: np.ndarray,
    pca_components: int = 10,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute per-layer helicity for trajectory analysis.
    Useful for identifying storm zones (layers where H peaks).

    Returns
    -------
    np.ndarray
        Shape (n_layers-1,). H value between each consecutive layer pair.
    """
    n_layers = hidden_states.shape[0]
    if n_layers < 2:
        return np.array([1.0])

    n_components = min(pca_components, n_layers - 1, hidden_states.shape[1])
    if n_components < 2:
        diffs = np.diff(hidden_states, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        return norms / (np.linalg.norm(hidden_states[:-1], axis=1) + epsilon)

    pca = PCA(n_components=n_components)
    traj = pca.fit_transform(hidden_states)

    tangents = np.diff(traj, axis=0)
    centroid = np.mean(traj, axis=0)
    radials = traj[:-1] - centroid

    tan_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    rad_norm = np.linalg.norm(radials, axis=1, keepdims=True)

    tan_unit = tangents / (tan_norm + epsilon)
    rad_unit = radials / (rad_norm + epsilon)

    forward = np.sum(tan_unit * rad_unit, axis=1, keepdims=True) * rad_unit
    rotational = tan_unit - forward

    forward_mag = np.linalg.norm(forward, axis=1)
    rotational_mag = np.linalg.norm(rotational, axis=1)

    return rotational_mag / (forward_mag + epsilon)


def compute_storm_zone_intensity(
    layer_h: np.ndarray,
    storm_start: int = 15,
    storm_end: int = 21,
    epsilon: float = 1e-10,
) -> tuple[float, float, float]:
    """
    Analyze the storm zone (typically L15-L21) intensity.

    Parameters
    ----------
    layer_h : np.ndarray
        Per-layer H values from compute_layer_h().
    storm_start, storm_end : int
        Layer range for storm zone (default: L15-L21).

    Returns
    -------
    tuple[float, float, float]
        (storm_mean, storm_peak, storm_baseline_ratio)
    """
    n = len(layer_h)

    # Clamp indices to valid range
    s_start = min(storm_start, n - 1)
    s_end = min(storm_end, n)

    storm_values = layer_h[s_start:s_end]
    pre_storm = layer_h[:s_start] if s_start > 0 else np.array([1.0])
    post_storm = layer_h[s_end:] if s_end < n else np.array([1.0])

    storm_mean = float(np.mean(storm_values))
    storm_peak = float(np.max(storm_values))
    baseline = float(np.mean(np.concatenate([pre_storm, post_storm])))

    ratio = storm_mean / (baseline + epsilon)

    return storm_mean, storm_peak, ratio
