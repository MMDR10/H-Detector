# Ô — Cross-Domain Phase Transition Detector

**Ô (read "O-hat"): A General Cross-Domain Measurement Operator for Phase Transitions**

## Paper

- `manuscript.pdf` — PRL-format, 6 pages
- `manuscript.tex` — LaTeX source

## Abstract

We propose Ô, a general-purpose measurement operator that quantifies three orthogonal axes of any dynamical system: curl (directional change rate of the trajectory), helicity (internal winding complexity / structural entanglement), and balance (proximity to criticality). The framework is substrate-agnostic: it maps any system—physical, economic, geological, or computational—onto the same (curl, helicity, balance) phase-space coordinates, enabling cross-domain phase transition comparison.

We validate Ô on two radically different systems: a 2D Ising model (thermal phase transition, L=32–128, Binder cumulant crossing at Tc ≈ 2.269) and the North Anatolian Fault creepmeter network (slow earthquake cycle at İsmetpaşa, 2001–present). We establish a five-class taxonomy (Type I–V) from the helicity-curl plane, plus a Null baseline. Positive and Negative Controls confirm the framework's discriminability: L128 paramagnetic (Null), L64 at Tc (Type III), L128 at Tc (Type III), İsmetpaşa East Wall (Type IV/V transition zone), and pure noise (Null). Key cross-domain finding: the helicity-curl topology reveals universal phase-space structures independent of substrate.

## Directory Layout

```
release/
├── manuscript.pdf           # PRL-format paper (6 pages)
├── manuscript.tex           # LaTeX source
├── figures/                 # 3 figures (PDF + PNG + generation scripts)
├── ising/                   # Ising model: simulation + analysis
│   ├── ising_vectorized.py       # Core Ising simulation engine
│   ├── ising_generate_L128.py    # L=128 generation script
│   ├── o_ising_L128_analysis.py  # L=128 helicity/Binder analysis
│   ├── o_ising_cross_lattice.py  # Cross-lattice (L=32,64,128) analysis
│   └── ising_2d_L*.npz          # Precomputed simulation data
├── helicity/
│   └── helicity.py           # Core Ô computation engine
├── naf_ismetpasa/            # NAF creepmeter analysis
│   ├── *.txt                 # Raw creepmeter data (6 stations)
│   ├── o_hat_ismetpasa.py    # Station-level Ô analysis
│   ├── naf_multi_station_o.py # Multi-station cross-analysis
│   └── o_hat_creepmeter.py   # General creepmeter Ô framework
└── research_notes/           # Detailed research notes (Markdown)
    ├── research-note-15-ising-l128-20260719.md
    ├── o-positive-negative-control-case-study-20260719.md
    └── o-phase-transition-spectrum-v5-final-20260719.md
```

## Key Results

| System | L/Station | Ô Type | Tc ~ | Binder Crossing |
|--------|-----------|--------|------|-----------------|
| 2D Ising | L=32 | Type II → III | 2.269 | ✓ |
| 2D Ising | L=64 | Type III | 2.269 | ✓ |
| 2D Ising | L=128 | Type III | 2.269 | ✓ |
| NAF Ismetpasa | East Wall | Type IV/V | (geological) | N/A |
| NAF Ismetpasa | West Wall | Type III | (geological) | N/A |
| NAF Ismetpasa | Hamamli | Type II | (geological) | N/A |
| NAF Ismetpasa | Petrol | Type IV | (geological) | N/A |
| NAF Ismetpasa | Sazlık East | Type III | (geological) | N/A |
| NAF Ismetpasa | Sazlık West | Type III | (geological) | N/A |

## 16-System Cross-Domain Classification

Full classification across 16 systems × 10 domains, using 5-type (I–V) + Null taxonomy. See `research_notes/o-phase-transition-spectrum-v5-final-20260719.md` for the complete spectrum analysis.

## License

CC BY 4.0 — see manuscript for full details.

## Citation

If you use this code or framework in your research, please cite:

> DR & MKP. _Ô — Cross-Domain Phase Transition Detector: From 2D Ising Thermal Phase Transition to North Anatolian Fault Creepmeter_. Research Note 15, 2026.
