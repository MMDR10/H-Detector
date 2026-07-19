"""
helicity_dynamics — Ô (O-hat): A Cross-Domain Phase Transition Detector.

Drop a time series. Get three numbers. Know if your system is stable,
heading for trouble, or already broken.

``curl``
    Directional change rate of the system trajectory.
    High curl → rapid directional shifts (turbulence / phase transition).
``helicity``
    Internal winding complexity / structural entanglement.
    Peaks at criticality; dips in ordered/disordered phases.
``balance``
    Proximity to criticality. balance ~ 0.5 at the critical point
    (maximum fluctuation); balance → 0 or 1 in ordered/disordered phases.

Quickstart::

    >>> import helicity_dynamics as hd
    >>> import numpy as np
    >>> data = np.random.randn(200, 10)  # (time_steps, features)
    >>> curl, helicity, balance = hd.compute(data)
    >>> hd.classify(helicity, balance)
    'Type III: CRITICAL FLUCTUATION'

For raw trajectory analysis::

    >>> traj = hd.reduce(data)           # PCA reduction
    >>> curl_ts = hd.curl_timeseries(traj)
    >>> helicity_ts = hd.helicity_timeseries(traj)

Validated across: 2D Ising (L=32–128), North Anatolian Fault creepmeter,
ENSO, COVID-19, stock markets, LLM hidden states, EEG, ECG, and more.
"""

from ._core import (
    compute,
    reduce,
    curl_timeseries,
    helicity_timeseries,
    balance_timeseries,
    classify,
    spectrum_classify,
    compute_cross_domain,
    PHASE_TYPES,
)
from ._ising import (
    Ising2D,
    ising_helicity_scan,
    binder_cumulant,
    critical_temperature,
)
from ._timeseries import (
    load_csv,
    load_npy,
    sliding_window,
    normalize,
    detrend,
)

__version__ = "1.0.0"
__all__ = [
    # Core API
    "compute",
    "reduce",
    "curl_timeseries",
    "helicity_timeseries",
    "balance_timeseries",
    "classify",
    "spectrum_classify",
    "compute_cross_domain",
    "PHASE_TYPES",
    # Ising
    "Ising2D",
    "ising_helicity_scan",
    "binder_cumulant",
    "critical_temperature",
    # Utilities
    "load_csv",
    "load_npy",
    "sliding_window",
    "normalize",
    "detrend",
]
