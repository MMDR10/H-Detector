"""
Ising L=128 Ô + Binder Cumulant Analysis
Head-to-head: helicity H vs Binder cumulant U₄ vs magnetization M
With L=64 comparison for finite-size scaling.
"""
import numpy as np
import json, os

# Load L=128 data
d128 = np.load('output/o_quantum/ising_2d_L128.npz')
configs = d128['configs']  # (340, 128, 128)
T_arr = d128['T']          # (340,)
M_arr = d128['M']          # (340,)
E_arr = d128['E']          # (340,)
L = int(d128['L'])
Tc = float(d128['Tc'])

print(f"L=128: {len(configs)} configs, {len(np.unique(T_arr))} T points")
print(f"Tc = {Tc}")

# ── 1. Binder Cumulant ──
def binder_cumulant(M_samples):
    """U₄ = 1 - <M⁴>/(3<M²>²)"""
    M2 = np.mean(np.array(M_samples)**2)
    M4 = np.mean(np.array(M_samples)**4)
    if M2 < 1e-20:
        return np.nan
    return 1.0 - M4 / (3.0 * M2**2)

# Group by temperature
unique_T = np.unique(T_arr)
results = {}

for T in unique_T:
    mask = np.abs(T_arr - T) < 1e-6
    idx = np.where(mask)[0]
    Ms = [float(M_arr[i]) for i in idx]
    Es = [float(E_arr[i]) for i in idx]
    Ms_abs = [abs(m) for m in Ms]
    
    U4 = binder_cumulant(Ms)
    M_mean = float(np.mean(Ms_abs))
    M_std = float(np.std(Ms_abs))
    E_mean = float(np.mean(Es))
    
    # ── 2. Ô Helicity ──
    # For each config, flatten spins into (x,y) embedding:
    # x = row index (0..127), y = column-averaged spin
    helicity_vals = []
    for i in idx:
        spins = configs[i]
        # Helicity: consistency of angular displacement in magnetization space
        # Use two projections of the lattice for helicity
        row_mean = np.mean(spins, axis=1) * L  # (L,) total magnetization per row
        col_mean = np.mean(spins, axis=0) * L  # (L,) total magnetization per col
        
        # Embed in 2D: (row_mean, col_mean) trajectory
        v1 = np.diff(row_mean)
        v2 = np.diff(col_mean)
        
        # Cross products of consecutive vectors
        cross = v1[:-1] * v2[1:] - v2[:-1] * v1[1:]
        n_pos = np.sum(cross > 0)
        n_neg = np.sum(cross < 0)
        total = n_pos + n_neg
        if total == 0:
            continue
        h = max(n_pos, n_neg) / total
        helicity_vals.append(h)
    
    H_mean = float(np.mean(helicity_vals)) if helicity_vals else np.nan
    H_std = float(np.std(helicity_vals)) if helicity_vals else np.nan
    
    # Phase classification
    if M_mean > 0.5:
        phase = 'FERRO'
    elif T < 10:  # below Tc vicinity
        if abs(M_mean) < 0.03:
            phase = 'CRITICAL'
        else:
            phase = 'TRANSITION'
    else:
        phase = 'PARA'
    
    results[str(T)] = {
        'T': float(T),
        'n_configs': int(len(idx)),
        'M_mean': M_mean,
        'M_std': M_std,
        'E_mean': E_mean,
        'U4': float(U4) if not np.isnan(U4) else None,
        'H_mean': H_mean,
        'H_std': H_std,
        'phase': phase
    }

# ── 3. Load L=64 for comparison ──
d64 = np.load('output/o_quantum/ising_2d_L64.npz')
T64 = d64['T']
M64 = d64['M']

# Compute U4 for L=64
unique_T64 = np.unique(T64)
l64_results = {}
for T in unique_T64:
    mask = np.abs(T64 - T) < 1e-6
    Ms = [float(M64[i]) for i in np.where(mask)[0]]
    U4 = binder_cumulant(Ms)
    l64_results[float(T)] = {'T': float(T), 'U4': float(U4) if not np.isnan(U4) else None}

# ── 4. Output ──
out = {
    'L128': results,
    'L64_binder': l64_results,
    'Tc': Tc,
    'L': L,
    'summary': {
        'n_T_points': len(unique_T),
        'n_configs_total': len(configs),
        'L128_binder_crossing_region': 'TBD',
        'L128_helicity_ferro': None,
        'L128_helicity_critical': None,
        'L128_helicity_para': None,
    }
}

# Compute summary helicity by phase
ferro_H = [r['H_mean'] for r in results.values() if r['phase']=='FERRO' and r['H_mean'] is not None]
crit_H = [r['H_mean'] for r in results.values() if r['phase']=='CRITICAL' and r['H_mean'] is not None]
para_H = [r['H_mean'] for r in results.values() if r['phase']=='PARA' and r['H_mean'] is not None]

out['summary']['L128_helicity_ferro'] = float(np.mean(ferro_H)) if ferro_H else None
out['summary']['L128_helicity_critical'] = float(np.mean(crit_H)) if crit_H else None
out['summary']['L128_helicity_para'] = float(np.mean(para_H)) if para_H else None

# Print summary table
print(f"\n{'='*80}")
print(f"  Ising L=128: Ô Helicity vs Binder Cumulant")
print(f"{'='*80}")
print(f"{'T':>7s}  {'|M|':>8s}  {'U4':>8s}  {'H_mean':>8s}  {'H_std':>8s}  {'Phase':>12s}")
print(f"{'-'*80}")

for T in sorted(unique_T):
    r = results[str(T)]
    u4_str = f"{r['U4']:.4f}" if r['U4'] is not None else "N/A"
    h_str = f"{r['H_mean']:.4f}" if r['H_mean'] is not None else "N/A"
    hs_str = f"{r['H_std']:.4f}" if r['H_std'] is not None else "N/A"
    print(f"{T:7.3f}  {r['M_mean']:8.4f}  {u4_str:>8s}  {h_str:>8s}  {hs_str:>8s}  {r['phase']:>12s}")

print(f"\n  Helicity by phase:")
print(f"    FERRO:     H = {out['summary']['L128_helicity_ferro']}")
print(f"    CRITICAL:  H = {out['summary']['L128_helicity_critical']}")
print(f"    PARA:      H = {out['summary']['L128_helicity_para']}")

# Save
outpath = 'output/o_quantum/ising_L128_binder_o.json'
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n✅ Saved → {outpath}")
