#!/usr/bin/env python3
"""
Cascade Entropy 行星宜居性測試 (Planetary Habitability Test)
=============================================================
核心假設 (MKP): 生命 = 終極抗熵系統 → S_cascade 可作 model-free biosignature

測試對象:
  1. Earth (Keeling CO2, 1958-2026) — 已知有生命
  2. Mars (Gale Crater pressure model) — 已知冇生命，但有強 abiotic 季節性
  3. Venus (surface pressure model) — 已知冇生命，極端穩定
  4. Pure Sine Wave — abiotic 對照組
  5. Random Walk — 純熵對照組
"""

import numpy as np
from collections import Counter

# ============================================================
# CORE OPERATORS
# ============================================================

def compute_o_components(data, window=24, stride=3):
    n = len(data)
    tau = max(1, window // 3)
    m = window - 2 * tau
    H = np.zeros(n)
    Curl = np.zeros(n)
    Balance = np.zeros(n)
    prev_theta = None
    for i in range(window, n, stride):
        chunk = data[i - window:i]
        if m < 4: continue
        X = np.column_stack([chunk[:m], chunk[tau:tau+m], chunk[2*tau:2*tau+m]])
        X_c = X - X.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            pc1 = U[:, 0]
            pc2 = U[:, 1] if U.shape[1] >= 2 else np.zeros(m)
            theta = np.arctan2(pc2, pc1)
            delta = np.abs(np.diff(theta))
            delta = np.minimum(delta, 2*np.pi - delta)
            h_val = float(np.mean(delta))
            angle = np.arctan2(np.mean(pc2), np.mean(pc1))
            c_val = 0.0
            if prev_theta is not None:
                delta_a = abs(angle - prev_theta)
                delta_a = min(delta_a, 2*np.pi - delta_a)
                c_val = delta_a
            prev_theta = angle
            b_val = float(np.var(theta))
            end = min(i + stride, n)
            H[i:end] = h_val
            Curl[i:end] = c_val
            Balance[i:end] = b_val
        except: pass
    return H, Curl, Balance

def noise_power(x, smooth_window=2):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if smooth_window > 1 and n > smooth_window:
        dX = np.zeros(n)
        half = smooth_window // 2
        for i in range(half, n - half):
            dX[i] = (x[i + half] - x[i - half]) / (2 * half)
        dX[:half] = dX[half]
        dX[-half:] = dX[-half-1]
    else:
        dX = np.gradient(x)
    P = np.abs(dX)
    nonzero = np.where(x > 1e-10)[0]
    if len(nonzero) > 0:
        P[:nonzero[0] + smooth_window + 2] = 0.0
    return P

def find_peak(P, min_prominence_ratio=1.5):
    n = len(P)
    if n == 0: return -1
    pk = int(np.argmax(P))
    peak_val = P[pk]
    lo = max(0, pk - 5)
    hi = min(n, pk + 6)
    left_min = np.min(P[lo:pk+1]) if pk > lo else peak_val
    right_min = np.min(P[pk:hi]) if hi > pk+1 else peak_val
    prominence = peak_val - max(left_min, right_min)
    prom_ratio = prominence / max(np.std(P), 1e-10)
    if prom_ratio < min_prominence_ratio:
        for idx in np.argsort(P)[::-1][:5]:
            lo2 = max(0, idx - 5)
            hi2 = min(n, idx + 6)
            left_min2 = np.min(P[lo2:idx+1]) if idx > lo2 else P[idx]
            right_min2 = np.min(P[idx:hi2]) if hi2 > idx+1 else P[idx]
            prom2 = P[idx] - max(left_min2, right_min2)
            if prom2 / max(np.std(P), 1e-10) >= min_prominence_ratio:
                return int(idx)
    return pk

def cascade_order(data, window, stride):
    H, Curl, Balance = compute_o_components(data, window=window, stride=stride)
    P_H = noise_power(H, smooth_window=2)
    P_C = noise_power(Curl, smooth_window=2)
    P_B = noise_power(Balance, smooth_window=2)
    pk_H = find_peak(P_H)
    pk_C = find_peak(P_C)
    pk_B = find_peak(P_B)
    if pk_H < 0 or pk_C < 0 or pk_B < 0: return None
    peaks = [('H', pk_H), ('Curl', pk_C), ('Balance', pk_B)]
    peaks.sort(key=lambda x: x[1])
    return ' → '.join([p[0] for p in peaks])

def cascade_entropy(data, label, window_sizes=[12,16,20,24,30,36], strides=[2,3,4,6]):
    orders = []
    for w in window_sizes:
        for s in strides:
            if w > len(data) // 2: continue
            order = cascade_order(data, w, s)
            if order is not None:
                orders.append(order)
    if len(orders) < 3:
        return None, None, None
    counts = Counter(orders)
    dominant = counts.most_common(1)[0]
    dominant_order = dominant[0]
    dominant_freq = dominant[1] / len(orders)
    entropy = 1.0 - dominant_freq
    return entropy, dominant_order, dict(counts)

# ============================================================
# DATA GENERATORS
# ============================================================

def generate_mars_pressure(n_points=800):
    """
    Mars Gale Crater pressure model based on published parameters:
    - Mean: ~750 Pa (varies with elevation, ~730-780 Pa at Gale)
    - Annual cycle: peak at Ls~250° (~870 Pa), trough at Ls~50° (~690 Pa)
    - Amplitude: ~90 Pa (semi-annual, from CO2 sublimation/condensation)
    - High-frequency noise: ~10 Pa (diurnal tide + weather)
    - Long-term trend: slight decrease as rover climbs Mt. Sharp (~-0.02 Pa/sol)
    - No biological modulation
    """
    t = np.arange(n_points)
    # Mars year = 668.6 sols, but we use Earth-equivalent months for comparison
    # Semi-annual cycle (CO2 sublimation at poles)
    annual = 90 * np.sin(2 * np.pi * t / 668.6 * 2)  # 2 cycles per Mars year
    # Diurnal tide (strong on Mars, ~10 Pa)
    diurnal = 10 * np.sin(2 * np.pi * t / 1.0)
    # Longer-term variations
    seasonal = 15 * np.sin(2 * np.pi * t / 334.3)
    # Random weather noise
    weather = np.random.normal(0, 8, n_points)
    # Dust storm events (rare, large pressure drops)
    dust_storms = np.zeros(n_points)
    for storm_start in [200, 450, 700]:
        if storm_start < n_points:
            duration = np.random.randint(5, 20)
            end = min(storm_start + duration, n_points)
            dust_storms[storm_start:end] = np.random.normal(-20, 10, end - storm_start)
    
    pressure = 750 + annual + diurnal + seasonal + weather + dust_storms
    return pressure

def generate_venus_pressure(n_points=800):
    """
    Venus surface pressure model:
    - Surface: ~92 bar (9,200,000 Pa), essentially constant
    - Temperature: ~737 K, constant
    - No seasonal cycle (axial tilt ~2.6°)
    - No weather (super-rotating atmosphere at cloud level, but surface is stagnant)
    - Only tiny variations from atmospheric tides
    This is a "dead" planet in terms of observable variability
    """
    t = np.arange(n_points)
    # Mean pressure at surface (normalized to show structure)
    pressure = 9200000 + 100 * np.sin(2 * np.pi * t / 117)  # Tiny atmospheric tide
    pressure += np.random.normal(0, 200, n_points)  # Measurement noise
    return pressure / 10000  # Scale down for numerical stability

def generate_sine_wave(n_points=800):
    """Pure abiotic seasonal cycle (single frequency + noise)."""
    t = np.arange(n_points)
    return 100 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 5, n_points)

def generate_random_walk(n_points=800):
    """Pure entropy - random walk."""
    return np.cumsum(np.random.normal(0, 1, n_points))

# ============================================================
# MAIN
# ============================================================

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  CASCADE ENTROPY — Planetary Habitability Test                 ║")
    print("║  Hypothesis: Life = Anti-Entropy → Lower S_cascade            ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    # Load Earth CO2 data
    import urllib.request
    url = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv'
    co2_data = []
    with urllib.request.urlopen(url) as f:
        lines = f.read().decode().split('\n')
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split(',')
            if len(parts) >= 5:
                try:
                    co2_data.append(float(parts[4]))
                except: pass

    np.random.seed(42)
    n_points = min(820, len(co2_data))  # Match Earth data length
    
    datasets = [
        ('🌍 Earth (Keeling CO2)',       np.array(co2_data[:n_points]), 'habitable'),
        ('🪨 Mars (Gale pressure model)', generate_mars_pressure(n_points), 'dead'),
        ('☁️  Venus (surface model)',      generate_venus_pressure(n_points), 'dead'),
        ('〰️  Pure Sine Wave',            generate_sine_wave(n_points), 'abiotic'),
        ('🎲 Random Walk',                generate_random_walk(n_points), 'entropy'),
    ]

    results = []

    for name, data, category in datasets:
        print("=" * 70)
        print(f"  {name} ({category}) — {len(data)} points")
        print(f"  Range: {data.min():.1f} - {data.max():.1f}, σ={data.std():.1f}")
        print("=" * 70)

        S, dom_order, counts = cascade_entropy(data, name)

        if S is None:
            print("  ⚠️  Insufficient data")
            continue

        # Classification
        if S < 0.55:
            nl = '🟢 STRONG ANTI-ENTROPY'
        elif S < 0.65:
            nl = '🟡 MODERATE'
        elif S < 0.75:
            nl = '🟠 WEAK'
        else:
            nl = '🔴 ENTROPY-DOMINATED'

        print(f"  S_cascade = {S:.3f} → {nl}")
        print(f"  Dominant Order: {dom_order}")
        print(f"  Configs tested: {sum(counts.values())}")
        for o, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            pct = c / sum(counts.values()) * 100
            print(f"    {o}: {c} ({pct:.1f}%)")

        results.append({
            'name': name, 'category': category,
            'S_cascade': float(S),
            'dominant_order': dom_order,
            'n_configs': sum(counts.values()),
            'distribution': {k: v for k, v in counts.items()},
        })

    # === CROSS-PLANET SPECTRUM ===
    print("\n" + "=" * 70)
    print("  PLANETARY CASCADE ENTROPY SPECTRUM")
    print("=" * 70)
    print(f"\n  {'Planet':<35} {'Category':<12} {'S_cascade':>10} {'Anti-Entropy':>18} {'Dominant Order'}")
    print("  " + "-" * 95)

    results.sort(key=lambda x: x['S_cascade'])

    for r in results:
        s = r['S_cascade']
        if s < 0.55:
            nl = '🟢 STRONG'
        elif s < 0.65:
            nl = '🟡 MODERATE'
        elif s < 0.75:
            nl = '🟠 WEAK'
        else:
            nl = '🔴 BROKEN'

        print(f"  {r['name']:<35} {r['category']:<12} {s:>10.3f} {nl:<18} {r['dominant_order']}")

    # === HABITABILITY THRESHOLD ===
    print("\n" + "=" * 70)
    print("  HABITABILITY ANALYSIS")
    print("=" * 70)
    
    earth_S = next(r['S_cascade'] for r in results if 'Earth' in r['name'])
    mars_S = next(r['S_cascade'] for r in results if 'Mars' in r['name'])
    
    print(f"""
    Earth (biotic)  S_cascade = {earth_S:.3f}
    Mars  (abiotic) S_cascade = {mars_S:.3f}
    
    ΔS = {abs(earth_S - mars_S):.3f}
    
    Interpretation:
    - Earth CO2 has a complex multi-scale structure (seasonal + anthropogenic + biological)
    - Mars pressure has a simpler structure (abiotic seasonal cycle)
    - S_cascade {("CAN distinguish" if abs(earth_S - mars_S) > 0.05 else "CANNOT distinguish")} biotic from abiotic
    
    This is a {("PROMISING" if abs(earth_S - mars_S) > 0.05 else "WEAK")} biosignature candidate.
    """)

    # Save results
    import json
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj
    
    out = '/app/working/workspaces/tygtDc/output/transition_cost/planetary_habitability.json'
    with open(out, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == '__main__':
    main()