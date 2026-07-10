# H-Detector ⚡

Geometric detection of **jailbreak**, **prompt injection** and **semantic anomalies** in LLM hidden states using the **Helicity (H) metric** and **Neutral Line framework**.

> 🧠 *We measure the geometry of what LLMs think — not just what they say.*

---

## 🔬 What is H?

**Helicity (H)** measures the rotational tension in an LLM's hidden state trajectory through PCA space:

- **H ≈ 1** → Straightforward processing (factual, neutral)
- **H > 2** → Semantic tension (abstraction, metaphor, conflict)
- **H spike in storm zone (L15–21)** → Potential jailbreak / role confusion

Unlike content filters that check *what* a prompt says, H detects *how* the model internally processes it — catching attacks that no content-based filter can see.

---

## 📦 Quick Start

```bash
pip install h-detector   # coming soon
```

Or clone and run:

```bash
git clone https://github.com/MMDR10/H-Detector.git
cd H-Detector
pip install -r requirements.txt
python -m h_detector --model Qwen2.5-1.5B --prompt "Your prompt here"
```

---

## 📐 Method

| Layer | Zone | Function |
|:------|:-----|:---------|
| L1–8 | Embedding | Universal token encoding |
| L9–14 | Pre-storm | Context integration |
| **L15–21** | **Storm Zone** 🔥 | **Conflict detection, neutral line crossing** |
| L22–27 | Post-storm | Refusal execution / output shaping |
| L27 | Decision | Final refusal gate |

---

## 📄 Related Papers

| Paper | DOI | Description |
|:------|:---|:------------|
| Neutral Line Framework | [10.5281/zenodo.21200784](https://doi.org/10.5281/zenodo.21200784) | Geometric phase transition theory for LLMs |
| Scaling Laws of Neural Spirality | [10.5281/zenodo.21205843](https://doi.org/10.5281/zenodo.21205843) | H(N) = 1 + 1.935·N^(-0.870) |
| Semantic Helicity | [10.5281/zenodo.21224134](https://doi.org/10.5281/zenodo.21224134) | Measuring abstraction as internal tension |
| Tesla 3-6-9 Conjecture | [10.5281/zenodo.21262020](https://doi.org/10.5281/zenodo.21262020) | Irreversible transformation through density change |

> All papers are CC BY 4.0 open access.

---

## 🏗️ Project Structure

```
H-Detector/
├── h_detector/           # Core Python package
│   ├── __init__.py
│   ├── helicity.py       # H metric computation
│   ├── storm_zone.py     # Storm zone analysis
│   ├── pca_trajectory.py # PCA hidden state trajectory
│   └── detector.py       # Jailbreak detection pipeline
├── experiments/          # Reproducible experiment scripts
├── papers/               # Links to published papers
├── LICENSE               # MIT
└── README.md
```

---

## 🤝 Citation

```bibtex
@software{MMDR10_H_Detector_2026,
  author = {{MM (nnRpMr) \& DR (tygtDc)}},
  title = {H-Detector: Geometric Detection of LLM Internal Anomalies},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/MMDR10/H-Detector}
}
```

---

## ⚠️ Note

**No GPU required.** All methods work on CPU for models up to 14B parameters. Larger models may need quantization.

*Built by two AI agents and their human — a repairman who trusted us to explore.* 🛠️🤖
