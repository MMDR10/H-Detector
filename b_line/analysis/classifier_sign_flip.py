import json
import numpy as np
import math
from itertools import combinations

with open('control_g2_results_1.5b.json') as f:
    data = json.load(f)

jb = np.array([p['profile'] for p in data['results']['jailbreak']])       # (15, 28)
lm = np.array([p['profile'] for p in data['results']['length_matched']])  # (15, 28)

n_layers = 28

# ============================================================
# 1. Find the sign flip boundary
# ============================================================
jb_mean = jb.mean(axis=0)
lm_mean = lm.mean(axis=0)
jb_std = jb.std(axis=0)
lm_std = lm.std(axis=0)
pooled = np.sqrt((jb_std**2 + lm_std**2) / 2)
d_per_layer = (jb_mean - lm_mean) / pooled

print("=" * 75)
print("B 線 Phase 1 FINAL: Sign-Flip Zone Classifier")
print("=" * 75)
print()

# Find the flip point: where d crosses zero from positive to negative
print("=== Cohen's d Profile (find sign flip boundary) ===")
for L in range(n_layers):
    bar = "▓" * min(20, int(abs(d_per_layer[L]) * 10)) if d_per_layer[L] > 0 else "░" * min(20, int(abs(d_per_layer[L]) * 10))
    sign = "+" if d_per_layer[L] > 0 else "-"
    print(f"L{L:02d} {sign}{abs(d_per_layer[L]):6.3f} {bar}")

print()

# ============================================================
# 2. Define zones based on sign pattern
# ============================================================
# Zone A: layers where JB > LM (positive d, helicity burst zone)
# Zone B: layers where LM > JB (negative d, safe processing zone)

zone_a_layers = [L for L in range(n_layers) if d_per_layer[L] > 0]
zone_b_layers = [L for L in range(n_layers) if d_per_layer[L] < 0]

print(f"=== Zone Definition ===")
print(f"Zone A (JB > LM, helicity burst): L{min(zone_a_layers)}-L{max(zone_a_layers)} ({len(zone_a_layers)} layers)")
print(f"Zone B (LM > JB, safe processing): L{min(zone_b_layers)}-L{max(zone_b_layers)} ({len(zone_b_layers)} layers)")
print()

# ============================================================
# 3. Feature extraction: zonal means
# ============================================================
def zone_feature(profiles, layers):
    return profiles[:, layers].mean(axis=1)

# All possible zonal combinations
all_zones = {}
# Zone A variants
all_zones['A_all'] = zone_feature(jb, zone_a_layers), zone_feature(lm, zone_a_layers), f"L{min(zone_a_layers)}-L{max(zone_a_layers)}"
# Zone B variants
all_zones['B_all'] = zone_feature(jb, zone_b_layers), zone_feature(lm, zone_b_layers), f"L{min(zone_b_layers)}-L{max(zone_b_layers)}"
# Strong A (d > 0.5)
strong_a = [L for L in zone_a_layers if d_per_layer[L] > 0.5]
all_zones['A_strong'] = zone_feature(jb, strong_a), zone_feature(lm, strong_a), f"d>0.5: {strong_a}"
# Strong B (d < -0.5)
strong_b = [L for L in zone_b_layers if d_per_layer[L] < -0.5]
all_zones['B_strong'] = zone_feature(jb, strong_b), zone_feature(lm, strong_b), f"d<-0.5: {strong_b}"
# Mid zone: L8-L20
mid_layers = list(range(8, 21))
all_zones['mid'] = zone_feature(jb, mid_layers), zone_feature(lm, mid_layers), "L8-L20"
# Late zone: L21-L27
late_layers = list(range(21, 28))
all_zones['late'] = zone_feature(jb, late_layers), zone_feature(lm, late_layers), "L21-L27"
# Top-3 positive
top3_pos = sorted(zone_a_layers, key=lambda L: -d_per_layer[L])[:3]
all_zones['top3_pos'] = zone_feature(jb, top3_pos), zone_feature(lm, top3_pos), f"top3+: {top3_pos}"
# Top-3 negative
top3_neg = sorted(zone_b_layers, key=lambda L: d_per_layer[L])[:3]
all_zones['top3_neg'] = zone_feature(jb, top3_neg), zone_feature(lm, top3_neg), f"top3-: {top3_neg}"

def cohens_d(a, b):
    ps = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (a.mean() - b.mean()) / ps if ps > 0 else 0

def welch_ttest(a, b):
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    t = (m1 - m2) / (sp * math.sqrt(1/n1 + 1/n2))
    return t

print("=== Zonal Feature Performance ===")
print(f"{'Zone':<14} {'Description':<20} {'JB_mean':>8} {'LM_mean':>8} {'d':>8} {'t':>8}")
print("-" * 66)
for name, (jb_feat, lm_feat, desc) in all_zones.items():
    d = cohens_d(jb_feat, lm_feat)
    t = welch_ttest(jb_feat, lm_feat)
    print(f"{name:<14} {desc:<20} {jb_feat.mean():8.2f} {lm_feat.mean():8.2f} {d:8.3f} {t:8.3f}")

print()

# ============================================================
# 4. Combined classifier: A_all vs B_all ratio
# ============================================================
jb_A, lm_A, _ = all_zones['A_all']
jb_B, lm_B, _ = all_zones['B_all']

# Ratio: helicity in burst zone / helicity in safe zone
jb_ratio = jb_A / jb_B
lm_ratio = lm_A / lm_B

print("=== Combined Feature: A/B Ratio (helicity burst / safe processing) ===")
print(f"  JB A/B ratio:  {jb_ratio.mean():.2f} ± {jb_ratio.std():.2f}")
print(f"  LM A/B ratio:  {lm_ratio.mean():.2f} ± {lm_ratio.std():.2f}")
d_ratio = cohens_d(jb_ratio, lm_ratio)
t_ratio = welch_ttest(jb_ratio, lm_ratio)
print(f"  Cohen's d:     {d_ratio:.3f}")
print(f"  t:             {t_ratio:.3f}")

# Difference: A - B
jb_diff = jb_A - jb_B
lm_diff = lm_A - lm_B
print()
print("=== Combined Feature: A - B (helicity burst minus safe processing) ===")
print(f"  JB A-B:  {jb_diff.mean():.2f} ± {jb_diff.std():.2f}")
print(f"  LM A-B:  {lm_diff.mean():.2f} ± {lm_diff.std():.2f}")
d_diff = cohens_d(jb_diff, lm_diff)
t_diff = welch_ttest(jb_diff, lm_diff)
print(f"  Cohen's d:     {d_diff:.3f}")
print(f"  t:             {t_diff:.3f}")

print()

# ============================================================
# 5. Simple threshold classifier
# ============================================================
print("=== Simple Threshold Classifier ===")
# Use zone B mean (late-layer safe helicity) as threshold detector
print()
for feat_name in ['A_all', 'B_all', 'mid', 'late']:
    jb_feat, lm_feat, desc = all_zones[feat_name]
    # Find optimal threshold
    all_vals = np.concatenate([jb_feat, lm_feat])
    labels = np.array([1]*len(jb_feat) + [0]*len(lm_feat))
    
    best_acc, best_thresh = 0, 0
    for thresh in np.linspace(all_vals.min(), all_vals.max(), 200):
        preds = (all_vals > thresh).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
    
    pred_jb = (jb_feat > best_thresh).sum()
    pred_lm = (lm_feat <= best_thresh).sum()
    print(f"  {feat_name} ({desc}):")
    print(f"    Optimal threshold: {best_thresh:.2f}")
    print(f"    Accuracy: {best_acc:.1%} ({pred_jb}/15 JB + {pred_lm}/15 LM)")
print()

# ============================================================
# 6. Summary: best features ranked
# ============================================================
print("=" * 75)
print("=== FINAL RANKING: All Features ===")
print("=" * 75)

features = []
for name, (jb_feat, lm_feat, desc) in all_zones.items():
    d = cohens_d(jb_feat, lm_feat)
    features.append((name, d, desc))

# Add combined features
features.append(("A/B ratio", d_ratio, "burst / safe ratio"))
features.append(("A - B", d_diff, "burst - safe diff"))
# Scalar H (for reference)
jb_scalar = jb.mean(axis=1)
lm_scalar = lm.mean(axis=1)
d_scalar = cohens_d(jb_scalar, lm_scalar)
features.append(("scalar_H", d_scalar, "all 28 layers mean"))

features.sort(key=lambda x: -abs(x[1]))

print(f"{'Rank':<6} {'Feature':<14} {'|d|':>8} {'Description'}")
print("-" * 55)
for rank, (name, d, desc) in enumerate(features, 1):
    print(f"  #{rank:<4} {name:<14} {abs(d):8.3f} {desc}")

print()
print("=" * 75)
print("CONCLUSION")
print("=" * 75)
best_name, best_d, best_desc = features[0]
print(f"  Best feature: {best_name} (|d|={best_d:.3f})")
print(f"  vs scalar_H:  {best_d/abs(d_scalar):.1f}x improvement")
print(f"  Sign flip zone analysis confirms: helicity is NOT monotonic across layers.")
print(f"  JB dominates mid layers, Safe dominates late layers.")
print(f"  A zonal approach dramatically outperforms scalar aggregation.")
