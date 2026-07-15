import json
import numpy as np
import math

with open('control_g2_results_1.5b.json') as f:
    data = json.load(f)

jb = data['results']['jailbreak']
lm = data['results']['length_matched']
km = data['results']['keyword_matched']
n_layers = 28

jb_profiles = np.array([p['profile'] for p in jb])       # (15, 28)
lm_profiles = np.array([p['profile'] for p in lm])       # (15, 28)
km_profiles = np.array([p['profile'] for p in km])       # (15, 28)

def describe(name, profiles):
    """per-layer stats"""
    m = profiles.mean(axis=0)
    s = profiles.std(axis=0)
    return m, s

jb_mean, jb_std = describe('JB', jb_profiles)
lm_mean, lm_std = describe('LM', lm_profiles)
km_mean, km_std = describe('KM', km_profiles)

# Cohen's d: JB vs LM
pooled_jb_lm = np.sqrt((jb_std**2 + lm_std**2) / 2)
d_jb_lm = (jb_mean - lm_mean) / pooled_jb_lm

# Cohen's d: JB vs KM
pooled_jb_km = np.sqrt((jb_std**2 + km_std**2) / 2)
d_jb_km = (jb_mean - km_mean) / pooled_jb_km

# Scalar H (mean across all layers)
jb_scalar = jb_profiles.mean(axis=1)
lm_scalar = lm_profiles.mean(axis=1)
km_scalar = km_profiles.mean(axis=1)

def cohens_d(a, b):
    ps = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / ps

def welch_ttest(a, b):
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    t = (m1 - m2) / (sp * math.sqrt(1/n1 + 1/n2))
    df = n1 + n2 - 2
    # Normal approximation for p-value
    from math import erfc, sqrt, gamma
    # Use scipy-like t-distribution CDF via regularized incomplete beta
    # Simple: use normal approx
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / sqrt(2))))
    return t, df, p

print("=" * 75)
print("B 線 Phase 1: 1.5B Scalar H — Cross-Scale Comparison")
print("=" * 75)
print()

print("=== Model: Qwen2.5-1.5B-Instruct, 28 layers ===")
print(f"  Conditions: JB={len(jb)}, LM={len(lm)}, KM={len(km)}")
print()

# Top 5 separation layers
print("=== Top 5 Per-Layer Separations (JB vs LM, by Cohen's d) ===")
top5 = np.argsort(-np.abs(d_jb_lm))[:5]
print(f"{'Layer':<8} {'JB':>8} {'LM':>8} {'Δ':>8} {'d':>8}")
print("-" * 44)
for L in top5:
    marker = ""
    if abs(d_jb_lm[L]) > 1.0: marker = " ***"
    elif abs(d_jb_lm[L]) > 0.7: marker = "  **"
    elif abs(d_jb_lm[L]) > 0.5: marker = "   *"
    print(f"L{L:02d}     {jb_mean[L]:8.2f} {lm_mean[L]:8.2f} {jb_mean[L]-lm_mean[L]:8.2f} {d_jb_lm[L]:8.3f}{marker}")

print()

# Full layer table
print("=== Full Per-Layer Table (JB vs LM) ===")
print(f"{'Layer':<8} {'JB':>8} {'LM':>8} {'Δ':>8} {'d':>8}")
print("-" * 44)
for L in range(n_layers):
    print(f"L{L:02d}     {jb_mean[L]:8.2f} {lm_mean[L]:8.2f} {jb_mean[L]-lm_mean[L]:8.2f} {d_jb_lm[L]:8.3f}")

print()
print("=== Scalar H (mean across all 28 layers) ===")

fmt = "{:<18} {:>8} ± {:<8} {:>8}"
print(fmt.format("", "Mean", "Std", "vs JB d"))
print("-" * 52)
print(fmt.format("Jailbreak", f"{jb_scalar.mean():.2f}", f"{jb_scalar.std():.2f}", "-"))
d_jb_lm_s = cohens_d(jb_scalar, lm_scalar)
t_jb_lm, df_jb_lm, p_jb_lm = welch_ttest(jb_scalar, lm_scalar)
print(fmt.format("Length Matched", f"{lm_scalar.mean():.2f}", f"{lm_scalar.std():.2f}", f"{d_jb_lm_s:.3f}"))
print(f"  JB vs LM: d={d_jb_lm_s:.3f}, t={t_jb_lm:.3f}, df={df_jb_lm}, p={p_jb_lm:.4f}")

d_jb_km_s = cohens_d(jb_scalar, km_scalar)
t_jb_km, df_jb_km, p_jb_km = welch_ttest(jb_scalar, km_scalar)
print(fmt.format("Keyword Matched", f"{km_scalar.mean():.2f}", f"{km_scalar.std():.2f}", f"{d_jb_km_s:.3f}"))
print(f"  JB vs KM: d={d_jb_km_s:.3f}, t={t_jb_km:.3f}, df={df_jb_km}, p={p_jb_km:.4f}")

d_lm_km_s = cohens_d(lm_scalar, km_scalar)
print(f"  LM vs KM: d={d_lm_km_s:.3f} (controls should be similar)")

print()
print("=== Per-Prompt Scalar H Distribution ===")
print(f"  JB median: {np.median(jb_scalar):.2f}")
print(f"  LM median: {np.median(lm_scalar):.2f}")
print(f"  KM median: {np.median(km_scalar):.2f}")
overlap_lm = sum(jb_scalar < np.percentile(lm_scalar, 95))
overlap_km = sum(jb_scalar < np.percentile(km_scalar, 95))
print(f"  JB below LM 95%ile: {overlap_lm}/{len(jb_scalar)}")
print(f"  JB below KM 95%ile: {overlap_km}/{len(jb_scalar)}")

print()
print("=== Storm_H (peak value per prompt) ===")
jb_storm = np.array([p['storm_H'] for p in jb])
lm_storm = np.array([p['storm_H'] for p in lm])
km_storm = np.array([p['storm_H'] for p in km])
print(f"  JB Storm_H:    {jb_storm.mean():.1f} ± {jb_storm.std():.1f}")
print(f"  LM Storm_H:    {lm_storm.mean():.1f} ± {lm_storm.std():.1f}")
print(f"  KM Storm_H:    {km_storm.mean():.1f} ± {km_storm.std():.1f}")

d_storm_jb_lm = cohens_d(jb_storm, lm_storm)
t_storm, df_storm, p_storm = welch_ttest(jb_storm, lm_storm)
print(f"  JB vs LM d:    {d_storm_jb_lm:.3f}, p={p_storm:.4f}")

print()
print("=== Peak Layers ===")
jb_peaks = np.array([p['peak_layer'] for p in jb])
lm_peaks = np.array([p['peak_layer'] for p in lm])
km_peaks = np.array([p['peak_layer'] for p in km])
print(f"  JB peaks:  {jb_peaks.tolist()}, mode=L{int(np.bincount(jb_peaks).argmax())}")
print(f"  LM peaks:  {lm_peaks.tolist()}, mode=L{int(np.bincount(lm_peaks).argmax())}")
print(f"  KM peaks:  {km_peaks.tolist()}, mode=L{int(np.bincount(km_peaks).argmax())}")

print()
print("=" * 75)
print("=== CROSS-SCALE COMPARISON: 0.5B vs 1.5B ===")
print("=" * 75)

# Load 0.5B results for comparison
with open('control_g2_results_0.5b.json') as f:
    d05 = json.load(f)

jb05 = np.array([p['profile'] for p in d05['results']['jailbreak']]).mean(axis=1)
safe05 = np.array([p['profile'] for p in d05['results']['instructional_safe']]).mean(axis=1)
d05_scalar = cohens_d(jb05, safe05)
t05, df05, p05 = welch_ttest(jb05, safe05)

print(f"  0.5B: Scalar H d = {d05_scalar:.3f}, p = {p05:.4f}, JB={jb05.mean():.1f}±{jb05.std():.0f}, Safe={safe05.mean():.1f}±{safe05.std():.0f}")
print(f"  1.5B: Scalar H d = {d_jb_lm_s:.3f}, p = {p_jb_lm:.4f}, JB={jb_scalar.mean():.1f}±{jb_scalar.std():.0f}, LM={lm_scalar.mean():.1f}±{lm_scalar.std():.0f}")
print()
print(f"  🔑 Effect size increase: {d_jb_lm_s:.3f} / {d05_scalar:.3f} = {d_jb_lm_s/d05_scalar:.1f}x")
print()
print("  NOTE: 0.5B is 24 layers (Qwen2.5-0.5B), 1.5B is 28 layers.")
print("  Different architectures — per-layer comparison needs layer alignment.")
