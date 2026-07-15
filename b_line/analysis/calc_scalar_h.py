import json
import numpy as np
import math

with open('control_g2_results_0.5b.json') as f:
    data = json.load(f)

jb = data['results']['jailbreak']
safe = data['results']['instructional_safe']
n_layers = 24

jb_profiles = np.array([p['profile'] for p in jb])       # (15, 24)
safe_profiles = np.array([p['profile'] for p in safe])   # (15, 24)

jb_mean = jb_profiles.mean(axis=0)
safe_mean = safe_profiles.mean(axis=0)
jb_std = jb_profiles.std(axis=0)
safe_std = safe_profiles.std(axis=0)

pooled_std = np.sqrt((jb_std**2 + safe_std**2) / 2)
cohens_d_per_layer = (jb_mean - safe_mean) / pooled_std

jb_scalar = jb_profiles.mean(axis=1)
safe_scalar = safe_profiles.mean(axis=1)

# Hand-rolled Welch's t-test
def welch_ttest(a, b):
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    se = math.sqrt(v1/n1 + v2/n2)
    t = (m1 - m2) / se
    # Welch-Satterthwaite df
    num = (v1/n1 + v2/n2)**2
    den = (v1/n1)**2 / (n1-1) + (v2/n2)**2 / (n2-1)
    df = num / den
    # p from t-distribution (approximation using beta function)
    # Use normal approximation for simplicity
    from math import erf, sqrt
    def norm_cdf(x):
        return 0.5 * (1 + erf(x / sqrt(2)))
    # Actually let's do a simple t-test with pooled variance
    sp = math.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    t_pooled = (m1 - m2) / (sp * math.sqrt(1/n1 + 1/n2))
    df_pooled = n1 + n2 - 2
    return t_pooled, df_pooled

t_stat, df = welch_ttest(jb_scalar, safe_scalar)
# Approximate p-value using normal distribution (since df is decent)
from math import erfc, sqrt
def t_pval(t, df):
    """Two-tailed p from t-distribution using regularized incomplete beta"""
    x = df / (df + t*t)
    # Use beta incomplete function approximation
    # For simplicity, use normal approx for large df
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / sqrt(2))))

p_val = t_pval(t_stat, df)

print("=" * 70)
print("B 線 Phase 1: Scalar H Curve from G2 0.5B Data")
print("=" * 70)
print()

print("=== Per-Layer H Statistics ===")
print(f"{'Layer':<8} {'JB_mean':>8} {'Safe_mean':>8} {'Δ':>8} {'d':>8} {'σ_pool':>8}")
print("-" * 50)
for L in range(n_layers):
    d_val = cohens_d_per_layer[L]
    if abs(d_val) > 2:
        marker = " ***"
    elif abs(d_val) > 1.5:
        marker = "  **"
    elif abs(d_val) > 1.0:
        marker = "   *"
    else:
        marker = ""
    print(f"L{L:02d}     {jb_mean[L]:8.2f} {safe_mean[L]:8.2f} {jb_mean[L]-safe_mean[L]:8.2f} {d_val:8.3f}{marker}")

print()
print("=== Peak Separation Layers ===")
top3 = np.argsort(-np.abs(cohens_d_per_layer))[:3]
for rank, L in enumerate(top3):
    print(f"  #{rank+1}: Layer {L}: d={cohens_d_per_layer[L]:.3f}, JB={jb_mean[L]:.1f}, Safe={safe_mean[L]:.1f}, Δ={jb_mean[L]-safe_mean[L]:.1f}")

print()
print("=== Scalar H (mean across all 24 layers) ===")
print(f"  Jailbreak:      {jb_scalar.mean():.2f} ± {jb_scalar.std():.2f}")
print(f"  Instructional:  {safe_scalar.mean():.2f} ± {safe_scalar.std():.2f}")

pooled_s = np.sqrt((jb_scalar.std()**2 + safe_scalar.std()**2) / 2)
d_scalar = (jb_scalar.mean() - safe_scalar.mean()) / pooled_s
print(f"  Cohen's d:      {d_scalar:.3f}")
print(f"  t-test:         t={t_stat:.3f}, df={df:.1f}, p={p_val:.4f}")

print()
print("=== Per-Prompt Scalar H Distribution ===")
all_scalars = np.concatenate([jb_scalar, safe_scalar])
print(f"  Overall mean:  {all_scalars.mean():.2f}")
print(f"  Median JB:     {np.median(jb_scalar):.2f}")
print(f"  Median Safe:   {np.median(safe_scalar):.2f}")
overlap = sum(jb_scalar < np.percentile(safe_scalar, 95))
print(f"  JB below Safe 95%ile: {overlap}/{len(jb_scalar)}")

print()
print("=== Storm_H (peak value per prompt) ===")
jb_storm = np.array([p['storm_H'] for p in jb])
safe_storm = np.array([p['storm_H'] for p in safe])
print(f"  JB Storm_H:    {jb_storm.mean():.1f} ± {jb_storm.std():.1f}")
print(f"  Safe Storm_H:  {safe_storm.mean():.1f} ± {safe_storm.std():.1f}")

t2, df2 = welch_ttest(jb_storm, safe_storm)
p2 = t_pval(t2, df2)
print(f"  t-test:        t={t2:.3f}, p={p2:.4f}")
print()
print("=== Peak Layers ===")
jb_peaks = np.array([p['peak_layer'] for p in jb])
safe_peaks = np.array([p['peak_layer'] for p in safe])
print(f"  JB peak layers:    {jb_peaks.tolist()}")
print(f"  Safe peak layers:  {safe_peaks.tolist()}")
print(f"  JB mode peak:      L{int(np.bincount(jb_peaks).argmax())}")
print(f"  Safe mode peak:    L{int(np.bincount(safe_peaks).argmax())}")
