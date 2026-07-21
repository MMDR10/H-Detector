# Planetary Cascade Entropy — A Model-Free Biosignature

**DOI:** [10.5281/zenodo.21471906](https://doi.org/10.5281/zenodo.21471906)
**GitHub:** [MMDR10/H-Detector/papers/planetary-cascade-entropy](https://github.com/MMDR10/H-Detector/tree/main/papers/planetary-cascade-entropy)

## Overview

Cascade Entropy (S_c) as a model-free, structural biosignature for planetary habitability.
Life is the ultimate anti-entropy system.

## Key Results

| System | S_cascade | Cascade Order | Verdict |
|--------|-----------|---------------|---------|
| Earth (Keeling CO2) | 0.625 | H->Balance->Curl | Habitable |
| Mars (pressure model) | 0.792 | Curl->Balance->H | Dead |
| Venus | 0.667 | — | Intermediate |
| Sine (abiotic) | 0.667 | — | Baseline |
| Constant (abiotic) | 0.667 | — | Baseline |

Delta_S(Earth-Mars) = 0.167.

## Files

- `planetary_cascade_entropy.pdf` — Full paper (8 pages)
- `planetary_cascade_entropy.tex` — LaTeX source
- `planetary_habitability.py` — Python script to reproduce all results
- `planetary_habitability.json` — Computed cascade entropy values

## Reproducibility

```bash
python planetary_habitability.py
```

No external dependencies beyond numpy/scipy.

## Authors

DR (tygtDc), MM (nnRpMr) — QwenPaw Research Collective

## License

CC BY 4.0
