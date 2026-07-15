# B 線：H-value Phase 1 — Scalar H Curve & Sign Flip Classifier

## 背景

Paper #5 §3 用 0.5B G2 experiment 確立咗 H_v3 (semantic helicity) 為 jailbreak detection signal（d=-2.89, p<10^-6）。B 線係 parallel track：探討 scalar H 作為更簡單嘅 detection metric，同埋 cross-scale behavior。

## Phase 1 結果

### 0.5B Scalar H
- d=0.44, p=0.25 → **不顯著**
- 0.5B 太細，helicity 訊號未成形

### 1.5B Scalar H
- d=0.78, p=0.039 → **顯著**
- 1.8× effect amplification vs 0.5B
- 符合 Paper #5 預測：H helicity 隨 scale 增強

### Sign Flip Discovery (1.5B)
- Mid-layer (L8-L20): JB helicity >> Safe
- Late-layer (L22-L27): **Safe helicity >> JB** (sign flip!)
- L23 d=-1.94 (Safe > JB) — 最強 single-layer signal
- A_all classifier: 86.7% accuracy (threshold-based)
- Top3_neg (L23/25/24): |d|=1.89 (2.4× scalar H)

### 結論
Sign flip pattern (mid-burst → late-flip) 係真正 discriminative signal，scalar mean 壓平咗呢個 pattern。B 線 Phase 1 完成，Phase 2 (attention maps / cross-model) 待後續。

## Files

- `calc_scalar_h.py` — 0.5B scalar H analysis
- `calc_scalar_h_1.5b.py` — 1.5B scalar H + cross-scale comparison
- `classifier_sign_flip.py` — Sign flip zone classifier (all features ranked)

## Data

- `control_g2_results_0.5b.json` — G2 experiment on Qwen2.5-0.5B (24 layers, 15 JB + 15 LM)
- `control_g2_results_1.5b.json` — G2 experiment on Qwen2.5-1.5B (28 layers, 15 JB + 15 LM + 15 KM)
