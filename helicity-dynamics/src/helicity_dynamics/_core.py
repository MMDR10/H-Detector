"""
Core Ô computation engine: curl, helicity, balance.

Given a multivariate time series X(t) of shape (n_steps, n_features),
the Ô operator maps it onto three scalar coordinates in phase space.

    Ô : X(t) → (curl, helicity, balance)

Algorithm
---------
1. PCA reduction (or use pre-reduced trajectory)
2. Tangential vs. radial decomposition at each step
3. Curl = mean |tangential|; Helicity = mean |tangential_rot|/|tangential_forward|
4. Balance = distance from critical helicity in normalised units

Classification
--------------
5 types + null baseline, based on helicity-curl topology:

    Type I    : ORDERED / FERRO     — low helicity, high curl consistency
    Type II   : QUASI-ORDERED       — moderate helicity, moderate curl
    Type III  : CRITICAL            — peak helicity, peak curl fluctuation
    Type IV   : QUASI-DISORDERED    — declining helicity, scattered curl
    Type V    : DISORDERED / PARA   — minimal helicity, minimal curl
    Null      : Noise / baseline    — no structural signal

References
----------
DR & MKP. Ô — Cross-Domain Phase Transition Detector. RN#15.
Zenodo: https://zenodo.org/record/21442523
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict, List, Union

# ---------------------------------------------------------------------------
# Phase type definitions
# ---------------------------------------------------------------------------

PHASE_TYPES: Dict[str, Dict[str, str]] = {
    "I": {
        "name": "ORDERED / FERRO",
        "description": "Strong cooperative order. Low helicity, high curl consistency.",
        "curl_range": "high, stable",
        "helicity_range": "low",
        "balance_range": "low (< 0.3)",
    },
    "II": {
        "name": "QUASI-ORDERED",
        "description": "Approaching criticality. Moderate helicity, moderate curl.",
        "curl_range": "moderate",
        "helicity_range": "moderate",
        "balance_range": "moderate (0.3–0.45)",
    },
    "III": {
        "name": "CRITICAL FLUCTUATION",
        "description": "Maximum fluctuation at the critical point. Peak helicity, peak curl fluctuation, infinite correlation length.",
        "curl_range": "high, fluctuating",
        "helicity_range": "peak",
        "balance_range": "critical (~0.45–0.55)",
    },
    "IV": {
        "name": "QUASI-DISORDERED",
        "description": "Declining from criticality. Declining helicity, scattered curl.",
        "curl_range": "scattered",
        "helicity_range": "declining",
        "balance_range": "moderate-high (0.55–0.7)",
    },
    "V": {
        "name": "DISORDERED / PARA",
        "description": "No cooperative order. Minimal helicity, minimal curl.",
        "curl_range": "low",
        "helicity_range": "minimal",
        "balance_range": "high (> 0.7)",
    },
    "Null": {
        "name": "NOISE / BASELINE",
        "description": "No structural signal. Pure noise or insufficient data.",
        "curl_range": "any (random)",
        "helicity_range": "near-zero",
        "balance_range": "N/A",
    },
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def reduce(
    data: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """Reduce multivariate time series to a low-dimensional trajectory via PCA.

    Parameters
    ----------
    data : np.ndarray, shape (n_steps, n_features)
        Input time series.
    n_components : int
        Number of PCA components (default: 3).

    Returns
    -------
    traj : np.ndarray, shape (n_steps, n_components)
        Reduced trajectory in PCA space.
    pca : PCA
        Fitted PCA object (for inverse transform / explained variance).
    """
    n_steps, n_features = data.shape
    n_comp = min(n_components, n_steps, n_features)

    pca = PCA(n_components=n_comp, random_state=random_state)
    traj = pca.fit_transform(data)
    return traj, pca


def curl_timeseries(
    traj: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute per-step curl (directional change rate) along a trajectory.

    Curl measures how sharply the trajectory bends at each step.
    High curl → rapid directional shift (characteristic of turbulence,
    phase transitions, regime changes).

    Parameters
    ----------
    traj : np.ndarray, shape (n_steps, n_dims)
        Trajectory in reduced space.

    Returns
    -------
    curl : np.ndarray, shape (n_steps - 2,)
        Per-step curl values. curl[i] is the angle between step (i→i+1)
        and step (i+1→i+2).
    """
    n_steps = traj.shape[0]
    if n_steps < 3:
        return np.array([0.0])

    # Step vectors
    steps = np.diff(traj, axis=0)  # (n_steps-1, n_dims)

    # Angle between consecutive steps
    v1 = steps[:-1]   # (n_steps-2, n_dims)
    v2 = steps[1:]    # (n_steps-2, n_dims)

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)

    cos_angle = np.sum(v1 * v2, axis=1) / (n1 * n2 + epsilon)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    curl = np.arccos(cos_angle)  # radians, [0, π]

    return curl


def helicity_timeseries(
    traj: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute per-step helicity (rotational-to-forward tension ratio).

    Decomposes each trajectory step into a *forward* component (radial,
    toward/away from centroid) and a *rotational* component (tangential,
    orbiting the centroid). Helicity = |rotational| / |forward|.

    High helicity → the system is winding around its attractor (structural
    complexity, entanglement). Low helicity → direct, straight-line motion.

    Parameters
    ----------
    traj : np.ndarray, shape (n_steps, n_dims)
        Trajectory in reduced space.

    Returns
    -------
    helicity : np.ndarray, shape (n_steps - 1,)
        Per-step helicity values.
    """
    n_steps = traj.shape[0]
    if n_steps < 2:
        return np.array([0.0])

    tangents = np.diff(traj, axis=0)  # (n_steps-1, n_dims)
    centroid = np.mean(traj, axis=0)
    radials = traj[:-1] - centroid    # (n_steps-1, n_dims)

    tan_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    rad_norm = np.linalg.norm(radials, axis=1, keepdims=True)

    tan_unit = tangents / (tan_norm + epsilon)
    rad_unit = radials / (rad_norm + epsilon)

    # Forward = projection of tangent onto radial direction
    forward_proj = np.sum(tan_unit * rad_unit, axis=1, keepdims=True)
    forward = forward_proj * rad_unit
    rotational = tan_unit - forward

    fwd_mag = np.linalg.norm(forward, axis=1)
    rot_mag = np.linalg.norm(rotational, axis=1)

    return rot_mag / (fwd_mag + epsilon)


def balance_timeseries(
    traj: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Compute per-step balance (proximity to the centroid / attractor).

    Balance measures how far the trajectory is from its attractor centre.
    Normalised so that 0 = at the centroid, 1 = at the farthest point.

    At criticality (Type III), the system oscillates around balance ≈ 0.5
    with maximum variance. In ordered phases, balance is low and stable.
    In disordered phases, balance wanders without structure.

    Parameters
    ----------
    traj : np.ndarray, shape (n_steps, n_dims)
        Trajectory in reduced space.

    Returns
    -------
    balance : np.ndarray, shape (n_steps,)
        Per-step balance values in [0, 1].
    """
    centroid = np.mean(traj, axis=0)
    distances = np.linalg.norm(traj - centroid, axis=1)
    d_max = distances.max()
    if d_max < epsilon:
        return np.zeros(traj.shape[0])
    return distances / d_max


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------

def compute(
    data: np.ndarray,
    n_components: int = 3,
    epsilon: float = 1e-10,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Compute Ô = (curl, helicity, balance) for a time series.

    This is the main entry point. Drop any multivariate time series and
    get three numbers back.

    Parameters
    ----------
    data : np.ndarray, shape (n_steps, n_features)
        Input time series. Each row is a time step; each column is a feature.
        Must have at least 3 time steps and 2 features.
    n_components : int
        PCA components for trajectory reduction (default: 3).

    Returns
    -------
    curl : float
        Mean directional change rate (radians). Higher → more turbulent.
    helicity : float
        Mean rotational-to-forward tension ratio. Peaks at criticality.
    balance : float
        Mean proximity to centroid, in [0, 1]. ~0.5 at criticality.

    Examples
    --------
    >>> import numpy as np
    >>> import helicity_dynamics as hd
    >>> # Random noise → Null type
    >>> noise = np.random.randn(200, 5)
    >>> c, h, b = hd.compute(noise)
    >>> hd.classify(h, b)
    'Type Null: NOISE / BASELINE'

    >>> # Sine wave → Type I (ordered)
    >>> t = np.linspace(0, 10*np.pi, 200)
    >>> signal = np.column_stack([np.sin(t), np.cos(t), np.sin(2*t)])
    >>> c, h, b = hd.compute(signal)
    >>> hd.classify(h, b)
    'Type I: ORDERED / FERRO'
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (n_steps, n_features), got shape {data.shape}")
    if data.shape[0] < 3:
        raise ValueError(f"Need at least 3 time steps, got {data.shape[0]}")
    if data.shape[1] < 2:
        raise ValueError(f"Need at least 2 features, got {data.shape[1]}")

    traj, pca = reduce(data, n_components, random_state)

    curl_ts = curl_timeseries(traj, epsilon)
    helicity_ts = helicity_timeseries(traj, epsilon)
    balance_ts = balance_timeseries(traj, epsilon)

    return (
        float(np.mean(curl_ts)),
        float(np.mean(helicity_ts)),
        float(np.mean(balance_ts)),
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(
    helicity: float,
    balance: float,
) -> str:
    """Classify a system into one of six phase types.

    Parameters
    ----------
    helicity : float
        Mean helicity value from ``compute()``.
    balance : float
        Mean balance value from ``compute()``.

    Returns
    -------
    str
        Phase type label, e.g. "Type III: CRITICAL FLUCTUATION".

    Notes
    -----
    Classification thresholds are calibrated on 2D Ising model (L=32–128)
    and North Anatolian Fault creepmeter data. They generalise across domains
    but may need recalibration for highly non-stationary systems.
    """
    # Null: effectively no structural signal
    if helicity < 0.01:
        return f"Type Null: {PHASE_TYPES['Null']['name']}"

    # Type III: Peak helicity at balance ~ 0.5
    if 0.45 <= balance <= 0.55 and helicity > 0.4:
        return f"Type III: {PHASE_TYPES['III']['name']}"

    # Type I: Low helicity, low balance (ordered, far from criticality)
    if balance < 0.3 and helicity < 0.3:
        return f"Type I: {PHASE_TYPES['I']['name']}"

    # Type II: Moderate (approaching critical)
    if 0.3 <= balance < 0.45:
        return f"Type II: {PHASE_TYPES['II']['name']}"

    # Type IV: Declining from critical
    if 0.55 < balance <= 0.7:
        return f"Type IV: {PHASE_TYPES['IV']['name']}"

    # Type V: High balance, low helicity (disordered)
    if balance > 0.7:
        return f"Type V: {PHASE_TYPES['V']['name']}"

    # Default: best-fit
    return f"Type III: {PHASE_TYPES['III']['name']}"


def spectrum_classify(
    helicity: float,
    balance: float,
    curl: float,
) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """Full spectrum classification with confidence scores.

    Returns a detailed classification with per-type likelihood scores
    based on distance from each type's centroid in (helicity, balance) space.

    Parameters
    ----------
    helicity : float
        Mean helicity.
    balance : float
        Mean balance.
    curl : float
        Mean curl.

    Returns
    -------
    dict
        Keys: primary_type, confidence, scores (per-type), curl_rad.
    """
    # Type centroids in (helicity, balance) space
    centroids = {
        "I": (0.15, 0.15),
        "II": (0.35, 0.38),
        "III": (0.55, 0.50),
        "IV": (0.40, 0.63),
        "V": (0.10, 0.85),
        "Null": (0.01, 0.50),
    }

    point = np.array([helicity, balance])
    scores = {}
    for t, c in centroids.items():
        dist = np.linalg.norm(point - np.array(c))
        scores[t] = float(1.0 / (1.0 + dist))  # higher = closer

    total = sum(scores.values())
    scores_norm = {k: round(v / total, 3) for k, v in scores.items()}

    best = max(scores_norm, key=scores_norm.get)
    confidence = scores_norm[best]

    return {
        "primary_type": f"Type {best}: {PHASE_TYPES[best]['name']}",
        "confidence": confidence,
        "scores": scores_norm,
        "curl_rad": float(curl),
        "helicity": float(helicity),
        "balance": float(balance),
    }


# ---------------------------------------------------------------------------
# Cross-domain comparison
# ---------------------------------------------------------------------------

def compute_cross_domain(
    datasets: Dict[str, np.ndarray],
    n_components: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Compute Ô for multiple datasets and return a comparison table.

    Useful for cross-domain analysis: compare an Ising simulation, a
    creepmeter signal, a stock index, etc. in the same phase space.

    Parameters
    ----------
    datasets : dict
        Mapping of label → data array. Each data array is (n_steps, n_features).

    Returns
    -------
    dict
        label → {"curl": ..., "helicity": ..., "balance": ..., "type": ...}
    """
    results = {}
    for label, data in datasets.items():
        c, h, b = compute(data, n_components=n_components)
        results[label] = {
            "curl": round(c, 4),
            "helicity": round(h, 4),
            "balance": round(b, 4),
            "type": classify(h, b),
        }
    return results
