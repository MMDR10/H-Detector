"""
2D Ising model simulation and analysis.

Provides a fast numpy-vectorised Ising model and tools for Ô-based
phase transition analysis, including Binder cumulant computation.
"""

import numpy as np
from typing import Tuple, Optional, List


class Ising2D:
    """2D Ising model with vectorised Metropolis Monte Carlo.

    Parameters
    ----------
    L : int
        Linear system size (L×L lattice).
    T : float
        Temperature in units of J/kB.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> ising = Ising2D(L=32, T=2.0)
    >>> ising.equilibrate(1000)
    >>> configs = ising.sample(n_configs=10, steps_between=50)
    >>> M = np.abs(np.mean(configs, axis=(1, 2)))  # magnetisation
    """

    def __init__(self, L: int, T: float, seed: int = 42):
        self.L = L
        self.T = T
        self.N = L * L
        self.beta = 1.0 / T if T > 0 else float("inf")
        rng = np.random.RandomState(seed)
        # Random initial configuration (±1)
        self.lattice = np.where(rng.rand(L, L) > 0.5, 1, -1).astype(np.int8)
        self.rng = rng
        self._n_accepted = 0
        self._n_attempts = 0

    def _energy_single(self, i: int, j: int) -> float:
        """Energy of a single spin (nearest-neighbour)."""
        L = self.L
        neighbours = (
            self.lattice[(i + 1) % L, j]
            + self.lattice[(i - 1) % L, j]
            + self.lattice[i, (j + 1) % L]
            + self.lattice[i, (j - 1) % L]
        )
        return -self.lattice[i, j] * neighbours

    def step(self) -> float:
        """One full Monte Carlo sweep (N spin flip attempts).

        Returns
        -------
        float
            Acceptance ratio for this sweep.
        """
        L = self.L
        accepted = 0
        for _ in range(self.N):
            i = self.rng.randint(0, L)
            j = self.rng.randint(0, L)

            dE = 2 * self._energy_single(i, j)  # flip changes E by -2*E_old

            if dE <= 0 or self.rng.rand() < np.exp(-self.beta * dE):
                self.lattice[i, j] *= -1
                accepted += 1

        self._n_accepted += accepted
        self._n_attempts += 1
        return accepted / self.N

    def equilibrate(self, n_sweeps: int = 2000) -> None:
        """Run equilibration sweeps (discard samples)."""
        for _ in range(n_sweeps):
            self.step()

    def sample(self, n_configs: int = 100, steps_between: int = 50) -> np.ndarray:
        """Sample configurations after equilibration.

        Parameters
        ----------
        n_configs : int
            Number of configurations to sample.
        steps_between : int
            Number of MC sweeps between samples (for decorrelation).

        Returns
        -------
        np.ndarray, shape (n_configs, L, L)
            Spin configurations (±1).
        """
        configs = []
        for _ in range(n_configs):
            for _ in range(steps_between):
                self.step()
            configs.append(self.lattice.copy())
        return np.array(configs, dtype=np.int8)

    @property
    def magnetisation(self) -> float:
        """Current magnetisation per spin, m = ⟨s⟩."""
        return float(np.mean(self.lattice))

    @property
    def energy(self) -> float:
        """Current energy per spin."""
        E = 0.0
        L = self.L
        for i in range(L):
            for j in range(L):
                E -= self.lattice[i, j] * (
                    self.lattice[(i + 1) % L, j] + self.lattice[i, (j + 1) % L]
                )
        return E / self.N


def ising_helicity_scan(
    L: int = 32,
    T_range: Tuple[float, float, int] = (1.0, 4.0, 25),
    n_equil: int = 2000,
    n_configs: int = 100,
    steps_between: int = 50,
    seed: int = 42,
) -> dict:
    """Scan temperature range and compute Ô for each T.

    Parameters
    ----------
    L : int
        System size.
    T_range : tuple (T_min, T_max, n_T)
        Temperature sweep parameters.
    n_equil : int
        Equilibration sweeps.
    n_configs : int
        Configurations to sample per T.

    Returns
    -------
    dict
        Keys: T (list), M (list), curl, helicity, balance (each a list).
    """
    T_min, T_max, n_T = T_range
    Ts = np.linspace(T_min, T_max, n_T)

    results = {"T": [], "M": [], "curl": [], "helicity": [], "balance": []}

    for T in Ts:
        ising = Ising2D(L=L, T=T, seed=seed)
        ising.equilibrate(n_equil)
        configs = ising.sample(n_configs, steps_between)

        # Flatten configs → (n_configs, L*L)
        data = configs.reshape(n_configs, -1).astype(float)

        # Compute Ô
        from ._core import compute as _compute
        c, h, b = _compute(data)

        results["T"].append(round(T, 4))
        results["M"].append(round(float(np.abs(np.mean(configs))), 6))
        results["curl"].append(round(c, 6))
        results["helicity"].append(round(h, 6))
        results["balance"].append(round(b, 6))

    return results


def binder_cumulant(
    configs: np.ndarray,
) -> float:
    """Compute the 4th-order Binder cumulant U₄.

    U₄ = 1 - ⟨m⁴⟩ / (3 ⟨m²⟩²)

    At the critical point (T = Tc), U₄ is independent of system size L,
    enabling finite-size scaling to locate the phase transition.

    For 2D Ising: U₄(T≪Tc) → 2/3, U₄(T≫Tc) → 0, U₄(Tc) ≈ 0.61.

    Parameters
    ----------
    configs : np.ndarray, shape (n_configs, L, L) or (n_configs, N)
        Spin configurations.

    Returns
    -------
    float
        Binder cumulant U₄.
    """
    # Per-config magnetisation
    m = np.mean(configs, axis=tuple(range(1, configs.ndim)))

    m2 = np.mean(m ** 2)
    m4 = np.mean(m ** 4)

    if m2 < 1e-15:
        return 0.0

    return float(1.0 - m4 / (3.0 * m2 ** 2))


def critical_temperature(
    L_list: List[int] = [32, 64, 128],
    T_range: Tuple[float, float, int] = (2.0, 2.5, 20),
    n_equil: int = 5000,
    n_configs: int = 200,
    steps_between: int = 100,
) -> dict:
    """Estimate Tc via Binder cumulant crossing for multiple system sizes.

    Parameters
    ----------
    L_list : list of int
        System sizes to simulate.
    T_range : tuple
        Temperature range around expected Tc.

    Returns
    -------
    dict
        L → {"T": [...], "U4": [...]}.
    """
    T_min, T_max, n_T = T_range
    Ts = np.linspace(T_min, T_max, n_T)
    results = {}

    for L in L_list:
        U4_list = []
        for T in Ts:
            ising = Ising2D(L=L, T=T, seed=42 + L)
            ising.equilibrate(n_equil)
            configs = ising.sample(n_configs, steps_between)
            U4_list.append(binder_cumulant(configs))

        results[f"L={L}"] = {"T": list(Ts), "U4": U4_list}

    return results
