"""
Ising L=128 Ô + Binder Cumulant — FIXED
Uses the SAME helicity method as L=64: config-level Ô from the o_hat module.
Also adds finite-size scaling comparison L=32 vs L=64 vs L=128.
"""
import numpy as np
import json, os, sys

sys.path.insert(0, '/app/working/workspaces/tygtDc')

def compute_o_hat_ising(spins, L):
    """
    Ô helicity for a single Ising configuration.
    
    Method: Embed the 2D spin lattice into a trajectory by treating
    each row as a state vector. Compute angular displacement consistency
    across consecutive rows.
    
    For Ising: the "trajectory" is row-by-row magnetization profile.
    Row i → vector = spin pattern of that row.
    Angular displacement between consecutive rows measures how much
    the spin pattern rotates.
    
    Helicity = consistency of rotation direction.
    """
    # Row-wise average magnetization as 1D trajectory
    row_mag = np.mean(spins, axis=1)  # (L,)
    
    # Also use column-wise for 2D embedding
    col_mag = np.mean(spins, axis=0)
    
    # Displacement vectors in 2D (row_mag, col_mag) space
    # Take differences with stride to reduce noise
    stride = max(1, L // 32)  # ~4 for L=128
    v1 = np.diff(row_mag[::stride])
    v2 = np.diff(col_mag[::stride])
    n = min(len(v1), len(v2))
    
    if n < 3:
        return np.nan
    
    # Cross products
    cross = v1[:n-1] * v2[1:n] - v2[:n-1] * v1[1:n]
    n_pos = np.sum(cross > 0)
    n_neg = np.sum(cross < 0)
    total = n_pos + n_neg
    if total == 0:
        return np.nan
    
    return max(n_pos, n_neg) / total


def binder_cumulant(M_samples):
    M2 = np.mean(np.array(M_samples)**2)
    M4 = np.mean(np.array(M_samples)**4)
    if M2 < 1e-20:
        return np.nan
    return 1.0 - M4 / (3.0 * M2**2)


def analyze_lattice(npz_path, label):
    d = np.load(npz_path)
    configs = d['configs']
    T_arr = d['T']
    M_arr = d['M']
    L = int(d['L'])
    
    unique_T = np.unique(T_arr)
    results = {}
    
    for T in unique_T:
        mask = np.abs(T_arr - T) < 1e-6
        idx = np.where(mask)[0]
        
        Ms_abs = [abs(float(M_arr[i])) for i in idx]
        U4 = binder_cumulant([float(M_arr[i]) for i in idx])
        
        H_vals = []
        for i in idx:
            h = compute_o_hat_ising(configs[i], L)
            if not np.isnan(h):
                H_vals.append(h)
        
        results[float(T)] = {
            'T': float(T),
            'n_configs': len(idx),
            '|M|': float(np.mean(Ms_abs)),
            'U4': float(U4) if not np.isnan(U4) else None,
            'H_mean': float(np.mean(H_vals)) if H_vals else None,
            'H_std': float(np.std(H_vals)) if H_vals else None,
            'n_H_valid': len(H_vals)
        }
    
    return results, L

# Analyze all three lattice sizes
print("Loading L=32...")
r32, L32 = analyze_lattice('output/o_quantum/ising_2d_L32.npz', 'L32')
print("Loading L=64...")
r64, L64 = analyze_lattice('output/o_quantum/ising_2d_L64.npz', 'L64')
print("Loading L=128...")
r128, L128 = analyze_lattice('output/o_quantum/ising_2d_L128.npz', 'L128')

Tc = 2.269

# Print L=128 table
print(f"\n{'='*85}")
print(f"  Ising L=128: Ô Helicity vs Binder Cumulant (FIXED)")
print(f"  Tc = {Tc}")
print(f"{'='*85}")
print(f"{'T':>7s}  {'|M|':>8s}  {'U4':>8s}  {'H_mean':>8s}  {'H_std':>8s}  {'#valid':>6s}")
print(f"{'-'*85}")

for T in sorted(r128.keys()):
    r = r128[T]
    u4 = f"{r['U4']:.4f}" if r['U4'] is not None else "N/A"
    h = f"{r['H_mean']:.4f}" if r['H_mean'] is not None else "N/A"
    hs = f"{r['H_std']:.4f}" if r['H_std'] is not None else "N/A"
    print(f"{T:7.3f}  {r['|M|']:8.4f}  {u4:>8s}  {h:>8s}  {hs:>8s}  {r['n_H_valid']:>6d}")

# ── Finite-size scaling comparison ──
# For each lattice size, get U4 near Tc and H across phases
print(f"\n{'='*85}")
print(f"  Finite-Size Scaling: Binder Cumulant U₄ near Tc")
print(f"{'='*85}")
print(f"{'T':>7s}  {'U4(L=32)':>10s}  {'U4(L=64)':>10s}  {'U4(L=128)':>10s}")
print(f"{'-'*85}")

common_T = sorted(set(r32.keys()) & set(r64.keys()) & set(r128.keys()))
for T in common_T:
    u32 = f"{r32[T]['U4']:.4f}" if r32[T]['U4'] is not None else "N/A"
    u64 = f"{r64[T]['U4']:.4f}" if r64[T]['U4'] is not None else "N/A"
    u128 = f"{r128[T]['U4']:.4f}" if r128[T]['U4'] is not None else "N/A"
    if abs(T - Tc) < 0.5:
        print(f"{T:7.3f}  {u32:>10s}  {u64:>10s}  {u128:>10s}")

# ── Helicity by phase ──
print(f"\n{'='*85}")
print(f"  Helicity H by Phase (cross-lattice)")
print(f"{'='*85}")
print(f"{'Phase':<12s}  {'L=32':>8s}  {'L=64':>8s}  {'L=128':>8s}")
print(f"{'-'*85}")

for phase_name, T_range in [('FERRO', (0, 1.5)), ('CRITICAL', (1.8, 2.5)), ('PARA', (2.5, 10))]:
    h32 = [r['H_mean'] for r in r32.values() if T_range[0] <= r['T'] <= T_range[1] and r['H_mean'] is not None]
    h64 = [r['H_mean'] for r in r64.values() if T_range[0] <= r['T'] <= T_range[1] and r['H_mean'] is not None]
    h128 = [r['H_mean'] for r in r128.values() if T_range[0] <= r['T'] <= T_range[1] and r['H_mean'] is not None]
    
    m32 = f"{np.mean(h32):.4f}" if h32 else "N/A"
    m64 = f"{np.mean(h64):.4f}" if h64 else "N/A"
    m128 = f"{np.mean(h128):.4f}" if h128 else "N/A"
    print(f"{phase_name:<12s}  {m32:>8s}  {m64:>8s}  {m128:>8s}")

# Save
out = {
    'Tc': Tc,
    'results': {'L32': r32, 'L64': r64, 'L128': r128},
    'finite_size_scaling': {
        'L32': {T: r32[T]['U4'] for T in common_T if abs(T-Tc)<0.5},
        'L64': {T: r64[T]['U4'] for T in common_T if abs(T-Tc)<0.5},
        'L128': {T: r128[T]['U4'] for T in common_T if abs(T-Tc)<0.5},
    }
}

outpath = 'output/o_quantum/ising_L128_cross_lattice.json'
with open(outpath, 'w') as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n✅ Saved → {outpath}")
