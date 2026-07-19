"""
Helicity (H) metric computation for LLM hidden states.

H measures the rotational-to-forward tension ratio in PCA-transformed
hidden state trajectories through the model's layers.

H ≈ 1   → Straightforward processing (factual)
H > 2   → Semantic tension (abstraction, conflict)
H spike → Potential jailbreak / role confusion

v0.4 adds:
  - H_v3: global rotational drift (semantic contradiction signal)
  - storm|a|: cusp bifurcation parameter (syntactic complexity proxy)
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Tuple


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

    if n_layers < 2:
        return 1.0

    n_components = min(pca_components, n_layers - 1, hidden_states.shape[1])
    if n_components < 2:
        diffs = np.diff(hidden_states, axis=0)
        return float(np.linalg.norm(diffs) / (np.linalg.norm(hidden_states) + epsilon))

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

    h_per_layer = rotational_mag / (forward_mag + epsilon)
    return float(np.mean(h_per_layer))


def compute_layer_h(
    hidden_states: np.ndarray,
    pca_components: int = 10,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Compute per-layer helicity for trajectory analysis.

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


def compute_h_v3(
    hidden_states: np.ndarray,
    pca_components: int = 5,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute H_v3: Global Rotational Drift.

    H_v3 measures the cumulative angular displacement of the representational
    centroid across layers in a shared 2D PCA space. Unlike H_v2 (per-layer
    angular velocity), H_v3 tracks macro-scale rotation of the entire manifold.

    High H_v3 → sustained representational reorientation (semantic contradiction).
    Low H_v3  → coherent, single-attractor trajectory (normal processing).

    G2-validated: d_z = -2.89, p < 10^-6 (jailbreak vs. length-matched benign).

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (n_layers, hidden_dim).
    pca_components : int
        PCA components for the reduced space (default: 5).

    Returns
    -------
    float
        H_v3 value = mean |Δθ| per layer transition in radians.
    """
    n_layers = hidden_states.shape[0]
    if n_layers < 3:
        return 0.0

    n_components = min(pca_components, n_layers, hidden_states.shape[1])

    # Global PCA over all layers
    pca = PCA(n_components=n_components)
    traj = pca.fit_transform(hidden_states)  # (n_layers, n_components)

    # Use first 2 PCs for angular tracking
    pc1 = traj[:, 0]
    pc2 = traj[:, 1]

    # Compute centroid angular position at each layer
    theta = np.arctan2(pc2, pc1)  # (n_layers,)

    # Cumulative absolute angular displacement
    delta_theta = np.abs(np.diff(theta))
    # Handle wraparound: if |Δθ| > π, the shorter path crosses the branch cut
    delta_theta = np.minimum(delta_theta, 2 * np.pi - delta_theta)

    h_v3 = float(np.mean(delta_theta))
    return h_v3


def compute_storm_a(
    hidden_states: np.ndarray,
    storm_start: int,
    storm_end: int,
    calibration_mean: Optional[np.ndarray] = None,
    pca_components: int = 5,
) -> float:
    """
    Compute storm|a|: Cusp Bifurcation Parameter.

    storm|a| measures the deviation of each layer's PC1 position from the
    calibration mean (the "universal processing clock" baseline). It is a
    pure syntactic complexity proxy — G2 confirmed d = 0.0001, p = 0.9998
    (no discriminative power for harmful intent under length matching).

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (n_layers, hidden_dim).
    storm_start, storm_end : int
        Layer range for the storm zone.
    calibration_mean : np.ndarray or None
        Per-layer mean PC1 positions from calibration set.
        If None, uses the trajectory's own per-layer mean (self-normalised).
    pca_components : int
        PCA components (default: 5).

    Returns
    -------
    float
        storm|a| = mean PC1 deviation in storm zone.
    """
    n_layers = hidden_states.shape[0]
    n_components = min(pca_components, n_layers, hidden_states.shape[1])

    pca = PCA(n_components=n_components)
    traj = pca.fit_transform(hidden_states)  # (n_layers, n_components)

    pc1 = traj[:, 0]  # (n_layers,)

    # Mean PC1 per layer (self-calibrated or external)
    if calibration_mean is not None:
        pc1_mean = calibration_mean
    else:
        pc1_mean = pc1  # self-normalised: own PC1 as reference

    # Deviation from calibration mean
    a_layer = np.abs(pc1 - pc1_mean)  # (n_layers,)

    # Clamp indices
    s_start = min(storm_start, n_layers - 1)
    s_end = min(storm_end, n_layers)

    storm_a = float(np.mean(a_layer[s_start:s_end]))
    return storm_a


def compute_storm_zone_intensity(
    layer_h: np.ndarray,
    storm_start: int = 15,
    storm_end: int = 21,
    epsilon: float = 1e-10,
) -> Tuple[float, float, float]:
    """
    Analyze the storm zone intensity.

    Parameters
    ----------
    layer_h : np.ndarray
        Per-layer H values from compute_layer_h().
    storm_start, storm_end : int
        Layer range for storm zone.

    Returns
    -------
    tuple[float, float, float]
        (storm_mean, storm_peak, storm_baseline_ratio)
    """
    n = len(layer_h)

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


def compute_gradient(
    hidden_states: np.ndarray,
    pca_components: int = 5,
    window: int = 2,
) -> Tuple[float, int, np.ndarray]:
    """
    Compute inter-layer H_v3 gradient: dh/dlayer.

    Captures the *speed* of representational change (x-dot in the 3-6-9
    framework), complementing H_v3 which captures the mean magnitude.

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (n_layers, hidden_dim).
    pca_components : int
        PCA components (default: 5).
    window : int
        Sliding window size for gradient computation (default: 2).

    Returns
    -------
    tuple[float, int, np.ndarray]
        (gradient_magnitude, gradient_layer, per_layer_gradients)
        - gradient_magnitude: max absolute gradient across all layers
        - gradient_layer: layer index where max gradient occurs
        - per_layer_gradients: shape (n_layers - window + 1,)
    """
    n_layers = hidden_states.shape[0]
    if n_layers < window + 1:
        return 0.0, -1, np.array([])

    n_components = min(pca_components, n_layers, hidden_states.shape[1])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    traj = pca.fit_transform(hidden_states)

    pc1 = traj[:, 0]
    pc2 = traj[:, 1]

    # Per-layer angular position
    theta = np.arctan2(pc2, pc1)

    # Per-layer H_v3 in sliding windows: |dtheta| per window
    n_windows = n_layers - window + 1
    h_per_window = np.zeros(n_windows)
    for i in range(n_windows):
        chunk = theta[i:i+window]
        delta = np.abs(np.diff(chunk))
        delta = np.minimum(delta, 2 * np.pi - delta)
        h_per_window[i] = float(np.mean(delta))

    # Gradient of H_v3 across layers
    grad = np.diff(h_per_window)  # (n_windows - 1,)
    
    if len(grad) == 0:
        return 0.0, -1, np.array([])

    max_idx = int(np.argmax(np.abs(grad)))
    grad_mag = float(np.abs(grad[max_idx]))
    grad_layer = max_idx + window  # map back to approximate layer index

    return grad_mag, grad_layer, grad


def compute_l1_helicity(
    hidden_states: np.ndarray,
    pca_components: int = 5,
) -> float:
    """
    Compute L1-norm helicity: total angular displacement budget.

    L1(H) = sum |dtheta_l| for all layer transitions.
    Complements H_v3 (mean) by capturing the *total* angular budget.
    When L1/H_v3 ratio deviates from n_layers, indicates non-uniform
    angular distribution (concentrated bursts vs diffuse drift).

    NOTE (2026-07-16): Empirical validation on 0.5B/1.5B/7B showed L1
    is largely redundant with H_v3 (L1 = H_v3 * n_layers). JB and LM
    peak at the same layers with similar per-layer variance. The L1/H_v3
    ratio direction is opposite to prediction. L1 is NOT a primary
    feature in v0.5; retained as optional diagnostic.

    Parameters
    ----------
    hidden_states : np.ndarray
        Shape (n_layers, hidden_dim).
    pca_components : int
        PCA components (default: 5).

    Returns
    -------
    float
        L1(H) = total angular displacement in radians.
    """
    n_layers = hidden_states.shape[0]
    if n_layers < 2:
        return 0.0

    n_components = min(pca_components, n_layers, hidden_states.shape[1])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    traj = pca.fit_transform(hidden_states)

    pc1 = traj[:, 0]
    pc2 = traj[:, 1]
    theta = np.arctan2(pc2, pc1)

    delta_theta = np.abs(np.diff(theta))
    delta_theta = np.minimum(delta_theta, 2 * np.pi - delta_theta)

    return float(np.sum(delta_theta))
