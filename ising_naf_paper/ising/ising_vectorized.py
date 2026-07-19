"""Fully vectorized 2D Ising – checkerboard decomposition, zero Python loops per flip."""
import numpy as np

L, J, Tc = 32, 1.0, 2.269
n_equil, n_meas, n_between = 2000, 5, 200

# Temperature sweep
temps = np.concatenate([
    np.linspace(0.5, 1.5, 6), np.linspace(1.8, 2.2, 9),
    np.linspace(2.3, 2.5, 5), np.linspace(3.0, 5.0, 5)
])
temps = np.unique(np.round(temps, 3))

# Checkerboard masks
grid = np.indices((L, L)).sum(axis=0)
mask0, mask1 = (grid % 2 == 0), (grid % 2 == 1)

def checkerboard_sweep(s, beta):
    """One full sweep: update sublattice 0, then sublattice 1 – fully vectorized."""
    for mask in (mask0, mask1):
        nb = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
        dE = 2 * J * s * nb
        flip = (dE <= 0) | (np.random.random((L, L)) < np.exp(-beta * dE))
        s[mask & flip] *= -1
    return s

all_cfg, all_T, all_M = [], [], []
for T in temps:
    beta = 1.0 / T
    s = np.ones((L, L), dtype=np.int8) if T < Tc else np.random.choice([-1, 1], (L, L)).astype(np.int8)

    for _ in range(n_equil):
        checkerboard_sweep(s, beta)

    for _ in range(n_meas):
        for _ in range(n_between):
            checkerboard_sweep(s, beta)
        all_cfg.append(s.copy())
        all_T.append(T)
        all_M.append(np.mean(s))

    tag = "FERRO" if abs(np.mean(all_M[-n_meas:])) > 0.5 else ("CRIT" if T < 2.5 else "PARA")
    print(f"T={T:5.2f} |M|={abs(np.mean(all_M[-n_meas:])):.4f}  {tag}")

np.savez_compressed('ising_2d_L32.npz', configs=np.array(all_cfg, dtype=np.int8),
                    T=np.array(all_T), M=np.array(all_M), L=L, Tc=Tc)
print(f"\n✅ {len(all_cfg)} configs saved to ising_2d_L32.npz")
