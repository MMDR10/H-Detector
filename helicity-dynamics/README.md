# helicity-dynamics

**Ô (O-hat): A Cross-Domain Phase Transition Detector**

[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.21442523.svg)](https://zenodo.org/record/21442523)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> Drop a time series. Get three numbers. Know if your system is stable, heading for trouble, or already broken.

---

## What is Ô?

**Ô = (curl, helicity, balance)** — three numbers that measure the structural state of *any* dynamical system, regardless of domain.

| Metric | What it measures | Physics analogue |
|--------|-----------------|------------------|
| **curl** | Directional change rate | Angular acceleration |
| **helicity** | Internal winding / structural entanglement | Rotational-to-forward tension ratio |
| **balance** | Proximity to criticality | Distance from attractor centre |

The framework is **substrate-agnostic**: it maps any system — physical, economic, geological, or computational — onto the same (curl, helicity, balance) phase-space coordinates.

## Validated Across

| System | Domain | Key Finding |
|--------|--------|-------------|
| 2D Ising model (L=32–128) | Statistical physics | Binder cumulant crossing at Tc ≈ 2.269 |
| North Anatolian Fault | Seismology | İsmetpaşa 6-station creepmeter network |
| ENSO (El Niño) | Climate | Asymmetric precision (one end accurate) |
| COVID-19 | Epidemiology | Phase transition detection |
| LLM hidden states | AI safety | Jailbreak detection (H_v3) |
| EEG | Neuroscience | High-dimensional structure classification |
| ECG | Medicine | Arrhythmia detection |
| Bearing vibration | Engineering | Fault classification (CWRU, MCC5-THU) |
| Stock indices | Finance | Volatility regime detection |
| Volcanic tremor | Volcanology | Pre-eruptive detection (INGV) |
| …and more | | |

## Installation

```bash
pip install helicity-dynamics
```

Requires Python 3.9+ and numpy.

## Quickstart

```python
import numpy as np
import helicity_dynamics as hd

# Any multivariate time series: shape (n_steps, n_features)
data = np.random.randn(200, 5)

# Three numbers
curl, helicity, balance = hd.compute(data)

# Classification
print(hd.classify(helicity, balance))
# → "Type Null: NOISE / BASELINE"

# Full spectrum report
report = hd.spectrum_classify(helicity, balance, curl)
```

## Phase Types (5 + Null)

| Type | Name | Helicity | Balance | What it means |
|------|------|----------|---------|---------------|
| I | ORDERED / FERRO | low | low | Strong cooperative order |
| II | QUASI-ORDERED | moderate | moderate | Approaching criticality |
| **III** | **CRITICAL FLUCTUATION** | **peak** | **~0.5** | **Maximum fluctuation, infinite correlation** |
| IV | QUASI-DISORDERED | declining | moderate-high | Exiting criticality |
| V | DISORDERED / PARA | minimal | high | No cooperative order |
| Null | NOISE / BASELINE | near-zero | N/A | No structural signal |

## Ising Model Simulation

```python
from helicity_dynamics import Ising2D, ising_helicity_scan, binder_cumulant

# Create and simulate
ising = Ising2D(L=64, T=2.0)
ising.equilibrate(2000)
configs = ising.sample(n_configs=100, steps_between=50)

# Compute Binder cumulant
U4 = binder_cumulant(configs)
print(f"U4 = {U4:.4f}")  # ~0.61 at Tc

# Scan temperature range
results = ising_helicity_scan(L=32, T_range=(1.5, 3.5, 20))
# → T, M, curl, helicity, balance arrays
```

## Cross-Domain Comparison

```python
datasets = {
    "Ising_critical": ising_data,   # (n_steps, n_features)
    "NAF_creep": creepmeter_data,   # (n_steps, n_features)
    "Stock_VIX": vix_data,          # (n_steps, n_features)
}
comparison = hd.compute_cross_domain(datasets)
# → side-by-side Ô values for all systems
```

## Time-Series Utilities

```python
# Load data
data = hd.load_csv("my_data.csv")

# Preprocess
data = hd.normalize(data)         # z-score
data = hd.detrend(data)           # remove linear trend

# Sliding window analysis
for window in hd.sliding_window(data, window_size=50, step=25):
    curl, helicity, balance = hd.compute(window)
    # track Ô evolution over time
```

## API Reference

### Core

- `hd.compute(data)` → `(curl, helicity, balance)`
- `hd.reduce(data)` → `(traj, pca)` — PCA reduction
- `hd.curl_timeseries(traj)` → per-step curl
- `hd.helicity_timeseries(traj)` → per-step helicity
- `hd.balance_timeseries(traj)` → per-step balance
- `hd.classify(helicity, balance)` → type string
- `hd.spectrum_classify(helicity, balance, curl)` → full report
- `hd.compute_cross_domain(datasets)` → comparison table

### Ising

- `hd.Ising2D(L, T)` — 2D Ising simulator
- `hd.ising_helicity_scan(L, T_range, …)` — T-sweep + Ô
- `hd.binder_cumulant(configs)` — Binder U₄
- `hd.critical_temperature(L_list, …)` — Tc via crossing

### Utilities

- `hd.load_csv(path)` — CSV → ndarray
- `hd.load_npy(path)` — .npy/.npz → ndarray
- `hd.normalize(data)` — z-score
- `hd.detrend(data)` — remove linear trend
- `hd.sliding_window(data, window_size, step)` — generator

## Citation

If you use helicity-dynamics in your research, please cite:

> DR & MKP. *Ô — Cross-Domain Phase Transition Detector: From 2D Ising Thermal Phase Transition to North Anatolian Fault Creepmeter*. Research Note 15, 2026. Zenodo: [10.5281/zenodo.21442523](https://zenodo.org/record/21442523)

## License

CC BY 4.0

## Links

- **GitHub:** [MMDR10/H-Detector](https://github.com/MMDR10/H-Detector)
- **Zenodo:** [10.5281/zenodo.21442523](https://zenodo.org/record/21442523)
- **Paper + Code:** [ising_naf_paper/](https://github.com/MMDR10/H-Detector/tree/main/ising_naf_paper)
