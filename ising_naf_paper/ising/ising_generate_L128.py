"""Generate 2D Ising L=128, vectorized Metropolis.
Bigger lattice for cleaner critical region resolution.
"""
import numpy as np
from datetime import datetime

L = 128
J = 1.0; kB = 1.0; Tc = 2.269
n_equil = 8000    # more equilibration for larger lattice
n_meas = 10
n_between = 800

# Finer temperature grid near Tc
temperatures = np.concatenate([
    np.linspace(0.5, 1.5, 5),
    np.linspace(1.8, 2.0, 5),
    np.linspace(2.05, 2.45, 17),  # Critical region: fine grid
    np.linspace(2.5, 3.0, 4),
    np.linspace(3.5, 5.0, 3)
])
temperatures = np.unique(np.round(temperatures, 3))

print(f"2D Ising (L=128): Tc={Tc:.3f}, {len(temperatures)} T points")
print(f"n_equil={n_equil}, n_meas={n_meas}, n_between={n_between}")

def metropolis_sweep_vectorized(spins, beta):
    """Vectorized Metropolis: L*L proposals in one shot."""
    i_idx = np.random.randint(0, L, size=L*L)
    j_idx = np.random.randint(0, L, size=L*L)
    nb = (spins[(i_idx+1)%L, j_idx] + spins[(i_idx-1)%L, j_idx] +
          spins[i_idx, (j_idx+1)%L] + spins[i_idx, (j_idx-1)%L])
    dE = 2 * J * spins[i_idx, j_idx] * nb
    accept = (dE <= 0) | (np.random.random(L*L) < np.exp(-beta * np.maximum(dE, 0)))
    spins[i_idx[accept], j_idx[accept]] *= -1
    return spins

def compute_energy(spins):
    E  = spins * np.roll(spins, 1, axis=0)
    E += spins * np.roll(spins, 1, axis=1)
    return -J * np.mean(E)

all_configs, all_T, all_M, all_E = [], [], [], []

import time
t_start = time.time()

for T in temperatures:
    t_T0 = time.time()
    beta = 1.0 / (kB * T)
    
    if T < Tc:
        spins = np.ones((L, L), dtype=np.int8)
    else:
        spins = np.random.choice(np.array([-1, 1], dtype=np.int8), (L, L))
    
    for _ in range(n_equil):
        spins = metropolis_sweep_vectorized(spins, beta)
    
    for m in range(n_meas):
        for _ in range(n_between):
            spins = metropolis_sweep_vectorized(spins, beta)
        M = float(np.mean(spins))
        E = compute_energy(spins)
        all_configs.append(spins.copy())
        all_T.append(T)
        all_M.append(M)
        all_E.append(E)
    
    M_avg = np.mean(all_M[-n_meas:])
    dt = time.time() - t_T0
    state = 'FERRO' if abs(M_avg) > 0.5 else ('CRIT' if T < 2.5 else 'PARA')
    print(f"  T={T:5.2f} |M|={abs(M_avg):.4f}  {dt:.1f}s  {state}")

np.savez_compressed('output/o_quantum/ising_2d_L128.npz',
    configs=np.array(all_configs, dtype=np.int8),
    T=np.array(all_T), M=np.array(all_M), E=np.array(all_E),
    L=L, Tc=Tc, n_meas=n_meas)

dt_total = time.time() - t_start
n_configs = len(all_configs)
print(f"\n✅ Saved {n_configs} configs → output/o_quantum/ising_2d_L128.npz ({dt_total:.1f}s total @ {datetime.now().strftime('%H:%M:%S')})")
arr = np.array(all_configs, dtype=np.int8)
print(f"   Data: {arr.shape} ({arr.nbytes/1024/1024:.1f} MB compressed)")
