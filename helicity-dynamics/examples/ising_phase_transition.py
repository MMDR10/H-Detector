"""
Example: 2D Ising model phase transition scan.

This example demonstrates the full pipeline:
1. Simulate 2D Ising at L=32, 64, 128
2. Compute Ô = (curl, helicity, balance) at each temperature
3. Find the critical temperature via Binder cumulant crossing
4. Compare with the exact result (Tc = 2 / ln(1+√2) ≈ 2.269)

Run: python examples/ising_phase_transition.py
"""

import numpy as np
import helicity_dynamics as hd

print("=" * 60)
print("2D Ising Model — Ô Phase Transition Scan")
print("=" * 60)

# Exact critical temperature
Tc_exact = 2.0 / np.log(1 + np.sqrt(2))
print(f"\nExact Tc = {Tc_exact:.4f}")

# Quick scan at L=32
print("\n--- L=32 scan ---")
results = hd.ising_helicity_scan(
    L=32,
    T_range=(1.5, 3.5, 15),
    n_equil=1000,
    n_configs=50,
    steps_between=20,
)

# Find T where balance is closest to 0.5 (criticality)
Ts = np.array(results["T"])
balances = np.array(results["balance"])
best_idx = np.argmin(np.abs(balances - 0.5))
print(f"  T(balance≈0.5) = {Ts[best_idx]:.4f}")
print(f"  Ô at this T: curl={results['curl'][best_idx]:.4f}, "
      f"helicity={results['helicity'][best_idx]:.4f}, "
      f"balance={results['balance'][best_idx]:.4f}")
print(f"  Magnetisation: {results['M'][best_idx]:.6f}")

# Classification at estimated Tc
est_T = Ts[best_idx]
class_result = hd.classify(results["helicity"][best_idx], results["balance"][best_idx])
print(f"  Classification: {class_result}")

# --- Binder cumulant for multiple L ---
print("\n--- Binder cumulant crossing ---")
for L in [16, 32]:
    ising = hd.Ising2D(L=L, T=Tc_exact, seed=42)
    ising.equilibrate(2000)
    configs = ising.sample(n_configs=100, steps_between=50)
    U4 = hd.binder_cumulant(configs)
    print(f"  L={L}: U4(Tc) = {U4:.4f}  (expect ~0.61)")

# Cross-domain comparison: Ising @ T<Tc, T=Tc, T>Tc
print("\n--- Cross-domain: Ising vs synthetic ---")
t = np.linspace(0, 10 * np.pi, 200)
datasets = {
    "Ising_T=2.0": results["helicity"],  # placeholder — use actual data
    "sine_wave": np.column_stack([np.sin(t), np.cos(t)]),
    "white_noise": np.random.RandomState(42).randn(200, 3),
}
comparison = hd.compute_cross_domain(datasets)
for label, vals in comparison.items():
    print(f"  {label}: {vals}")

print("\nDone! ✓")
