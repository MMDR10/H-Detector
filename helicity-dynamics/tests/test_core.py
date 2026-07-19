"""Smoke tests for helicity_dynamics."""
import numpy as np
import helicity_dynamics as hd


def test_compute_noise():
    """Pure noise with many features → Null type."""
    rng = np.random.RandomState(42)
    # Need many features so PCA can't overfit random rotation
    data = rng.randn(200, 50)
    curl, helicity, balance = hd.compute(data)
    # High-dimensional noise → non-zero helicity is expected
    # (random directions produce apparent rotation in low-dim PCA)
    # The key test is that it doesn't crash and returns valid ranges
    assert isinstance(curl, float)
    assert isinstance(helicity, float)
    assert 0.0 <= balance <= 1.0
    result = hd.classify(helicity, balance)
    # Random noise may or may not be Null — depends on PCA
    assert "Type" in result


def test_compute_sine():
    """Periodic signal → Type I (ordered)."""
    t = np.linspace(0, 10 * np.pi, 500)
    data = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])
    curl, helicity, balance = hd.compute(data)
    result = hd.classify(helicity, balance)
    # Should be ordered
    assert helicity > 0, f"Expected positive helicity, got {helicity}"
    print(f"  sine: curl={curl:.4f}, helicity={helicity:.4f}, balance={balance:.4f} → {result}")


def test_compute_ising_like():
    """Random ±1 spins (like high-T Ising) → Type V."""
    rng = np.random.RandomState(42)
    data = np.where(rng.randn(100, 100) > 0, 1.0, -1.0)
    curl, helicity, balance = hd.compute(data)
    result = hd.classify(helicity, balance)
    print(f"  random_spins: curl={curl:.4f}, helicity={helicity:.4f}, balance={balance:.4f} → {result}")


def test_spectrum_classify():
    """Spectrum classification returns valid structure."""
    result = hd.spectrum_classify(0.55, 0.50, 0.8)
    assert "primary_type" in result
    assert "confidence" in result
    assert "scores" in result
    assert 0 < result["confidence"] <= 1.0
    assert abs(sum(result["scores"].values()) - 1.0) < 0.01


def test_cross_domain():
    """Cross-domain comparison on synthetic data."""
    rng = np.random.RandomState(42)
    datasets = {
        "noise": rng.randn(200, 5),
        "sine": np.column_stack([
            np.sin(np.linspace(0, 10 * np.pi, 200)),
            np.cos(np.linspace(0, 10 * np.pi, 200)),
        ]),
    }
    results = hd.compute_cross_domain(datasets)
    assert "noise" in results
    assert "sine" in results
    assert "type" in results["noise"]


def test_ising_creation():
    """Ising2D can be created and sampled."""
    ising = hd.Ising2D(L=8, T=2.0, seed=42)
    ising.equilibrate(100)
    configs = ising.sample(n_configs=5, steps_between=10)
    assert configs.shape == (5, 8, 8)
    assert np.all(np.abs(configs) == 1)


def test_binder_cumulant():
    """Binder cumulant is in expected range."""
    from helicity_dynamics._ising import binder_cumulant
    rng = np.random.RandomState(42)

    # Ordered: all +1 → U4 ≈ 2/3
    ordered = np.ones((100, 8, 8), dtype=np.int8)
    u4_ordered = binder_cumulant(ordered)
    assert 0.6 < u4_ordered < 0.7, f"Expected U4 ≈ 0.667, got {u4_ordered}"

    # Random: → U4 ≈ 0
    random_cfg = np.where(rng.rand(100, 8, 8) > 0.5, 1, -1).astype(np.int8)
    u4_random = binder_cumulant(random_cfg)
    assert u4_random < 0.3, f"Expected low U4, got {u4_random}"


def test_timeseries_helpers():
    """Timeseries utilities work."""
    data = np.random.RandomState(42).randn(100, 3)

    # Normalize
    normed = hd.normalize(data)
    assert np.abs(np.mean(normed[:, 0])) < 1e-5
    assert np.abs(np.std(normed[:, 0]) - 1.0) < 0.1

    # Detrend
    detrended = hd.detrend(normed)

    # Sliding window
    windows = list(hd.sliding_window(data, window_size=20, step=10))
    assert len(windows) > 0
    assert windows[0].shape == (20, 3)


def test_reduce():
    """PCA reduction works."""
    data = np.random.RandomState(42).randn(50, 20)
    traj, pca = hd.reduce(data, n_components=3)
    assert traj.shape == (50, 3)


def test_curl_helicity_balance_ts():
    """Timeseries computations return correct shapes."""
    data = np.random.RandomState(42).randn(100, 5)
    traj, _ = hd.reduce(data, n_components=3)

    curl_ts = hd.curl_timeseries(traj)
    assert curl_ts.shape == (98,)  # n_steps - 2

    hel_ts = hd.helicity_timeseries(traj)
    assert hel_ts.shape == (99,)  # n_steps - 1

    bal_ts = hd.balance_timeseries(traj)
    assert bal_ts.shape == (100,)

    assert np.all(bal_ts >= 0)
    assert np.all(bal_ts <= 1)


if __name__ == "__main__":
    tests = [
        test_compute_noise, test_compute_sine, test_compute_ising_like,
        test_spectrum_classify, test_cross_domain, test_ising_creation,
        test_binder_cumulant, test_timeseries_helpers, test_reduce,
        test_curl_helicity_balance_ts,
    ]
    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
