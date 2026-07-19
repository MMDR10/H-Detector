#!/usr/bin/env python3
"""Ô framework applied to USGS creepmeter data — San Andreas Fault.
Tests MKP's hypothesis: H_v3 ↑ (structural emergence) as pre-seismic precursor.

Data format: year, day-of-year, slip(mm)
Target: XTA1 (Taylor Ranch) 10-min sampling, 1985-2013
Key event: 2004 Parkfield M6.0 earthquake (day 272 = Sept 28)
"""
import sys, json, gzip
import numpy as np
from pathlib import Path

# ── config ──────────────────────────────────────────────────
WINDOW_DAYS = 30          # sliding window in days
STEP_DAYS = 3             # step size
EMBED_DIM = 5             # time-delay embedding dimension
TAU = 1                   # embedding lag (in window steps)
OUTBREAK_PERCENTILE = 95  # for balance ratio
BASELINE_FRAC = 0.3       # first fraction as baseline
MIN_POINTS_PER_WINDOW = 10

# ── load data ────────────────────────────────────────────────
def load_creepmeter(path):
    """Load USGS creepmeter ASCII: year day-of-year slip(mm)."""
    data = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            yr, doy, slip = int(parts[0]), float(parts[1]), float(parts[2])
            data.append((yr, doy, slip))
    return np.array(data)

def to_datetime_index(data):
    """Convert (year, doy) to decimal year for plotting."""
    from datetime import datetime, timedelta
    dates = []
    for yr, doy, _ in data:
        d = datetime(int(yr), 1, 1) + timedelta(days=int(doy) - 1)
        dates.append(d)
    return dates

# ── time-delay embedding ────────────────────────────────────
def embed_phase_space(signal, embed_dim=EMBED_DIM, tau=TAU):
    """Embed 1D signal into phase space via time-delay."""
    n = len(signal) - (embed_dim - 1) * tau
    if n <= 0:
        return None
    X = np.zeros((n, embed_dim))
    for i in range(embed_dim):
        X[:, i] = signal[i * tau:n + i * tau]
    return X

# ── helicity (angular velocity in PCA space) ─────────────────
def compute_helicity(X, n_components=3):
    """Compute helicity as angular velocity of trajectory in PCA space."""
    from numpy.linalg import svd
    X_centered = X - X.mean(axis=0)
    U, S, Vt = svd(X_centered, full_matrices=False)
    proj = X_centered @ Vt[:n_components].T

    angles = []
    for i in range(1, len(proj)):
        v1, v2 = proj[i-1], proj[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            angles.append(0.0)
        else:
            cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            angles.append(np.arccos(cos_ang))
    return np.mean(angles) if angles else 0.0

# ── curl (directional change rate) ───────────────────────────
def compute_curl(signal):
    """Compute curl as mean absolute second difference."""
    if len(signal) < 3:
        return 0.0
    d2 = np.diff(signal, n=2)
    return np.mean(np.abs(d2))

# ── balance (skewness ratio) ─────────────────────────────────
def compute_balance(signal, outbreak_pct=OUTBREAK_PERCENTILE):
    """Balance = skewness(outbreak region) / skewness(baseline region)."""
    n = len(signal)
    if n < 20:
        return 0.0
    cutoff = int(n * BASELINE_FRAC)
    baseline = signal[:cutoff]
    threshold = np.percentile(np.abs(signal), outbreak_pct)
    outbreak = signal[np.abs(signal) >= threshold]

    skew_base = abs(np.mean((baseline - np.mean(baseline))**3)) / (np.std(baseline)**3 + 1e-12)
    if len(outbreak) < 3:
        return 0.0
    skew_out = abs(np.mean((outbreak - np.mean(outbreak))**3)) / (np.std(outbreak)**3 + 1e-12)

    return skew_out / (skew_base + 1e-12)

# ── H_v3 (3-component helicity) ──────────────────────────────
def compute_hv3(signal, embed_dim=EMBED_DIM, tau=TAU):
    """Compute H_v3 = mean angular velocity via 3D PCA embedding."""
    X = embed_phase_space(signal, embed_dim, tau)
    if X is None or len(X) < 10:
        return 0.0
    return compute_helicity(X, n_components=3)

# ── sliding window scan ──────────────────────────────────────
def sliding_window_scan(signal, window_size, step_size):
    """Scan signal with sliding windows, compute Ô triple."""
    results = []
    n = len(signal)
    i = 0
    while i + window_size <= n:
        chunk = signal[i:i + window_size]
        if len(chunk) < MIN_POINTS_PER_WINDOW:
            i += step_size
            continue
        hv3 = compute_hv3(chunk)
        curl_val = compute_curl(chunk)
        bal = compute_balance(chunk)
        results.append({
            'idx': int(i),
            'idx_end': int(i + window_size),
            'window_center': int(i + window_size / 2),
            'h_v3': round(hv3, 6),
            'curl': round(curl_val, 6),
            'balance': round(bal, 4),
            'mean_slip': round(float(np.mean(chunk)), 4),
            'std_slip': round(float(np.std(chunk)), 4),
            'n_points': len(chunk)
        })
        i += step_size
    return results

# ── main ─────────────────────────────────────────────────────
def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "xta1.10min"
    outpath = sys.argv[2] if len(sys.argv) > 2 else "xta1_ohat_results.json"

    # Load
    raw = load_creepmeter(filepath)
    print(f"Loaded {len(raw):,} data points from {filepath}")
    print(f"  Range: {int(raw[0,0])}/{int(raw[0,1])} -> {int(raw[-1,0])}/{int(raw[-1,1])}")
    print(f"  Slip: {raw[0,2]:.2f} -> {raw[-1,2]:.2f} mm")

    # Daily resample for sliding window (avoid OOM with 10-min data)
    slip = raw[:, 2]
    dates = to_datetime_index(raw)

    # Downsample to daily for window scan
    daily_slip = []
    daily_dates = []
    current_day = None
    day_vals = []
    for i, (yr, doy, s) in enumerate(raw):
        day_id = (int(yr), int(doy))
        if day_id != current_day:
            if day_vals:
                daily_slip.append(np.mean(day_vals))
                daily_dates.append(dates[i-1] if i > 0 else dates[0])
            current_day = day_id
            day_vals = [s]
        else:
            day_vals.append(s)
    if day_vals:
        daily_slip.append(np.mean(day_vals))
        daily_dates.append(dates[-1])

    daily_slip = np.array(daily_slip)
    print(f"  Daily resampled: {len(daily_slip)} days")

    # Sliding window scan
    window_days = WINDOW_DAYS
    step_days = STEP_DAYS

    print(f"  Window: {window_days} days, step: {step_days} days")
    results = sliding_window_scan(daily_slip, window_days, step_days)
    print(f"  Computed {len(results)} windows")

    # Add date labels
    for r in results:
        r['date'] = str(daily_dates[min(r['window_center'], len(daily_dates)-1)])[:10]

    # Save
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} windows to {outpath}")

    # ── Quick stats ──────────────────────────────────────────
    hv3_vals = [r['h_v3'] for r in results]
    curl_vals = [r['curl'] for r in results]
    bal_vals = [r['balance'] for r in results]

    print(f"\n── Ô Results Summary ──")
    print(f"  H_v3:   mean={np.mean(hv3_vals):.4f}, max={np.max(hv3_vals):.4f}, std={np.std(hv3_vals):.4f}")
    print(f"  Curl:   mean={np.mean(curl_vals):.4f}, max={np.max(curl_vals):.4f}, std={np.std(curl_vals):.4f}")
    print(f"  Balance: mean={np.mean(bal_vals):.2f}, max={np.max(bal_vals):.1f}, std={np.std(bal_vals):.2f}")

    # Top 5 by balance
    sorted_by_bal = sorted(results, key=lambda x: x['balance'], reverse=True)
    print(f"\n── Top 10 Balance Windows ──")
    for r in sorted_by_bal[:10]:
        print(f"  {r['date']} | balance={r['balance']:8.1f} | H_v3={r['h_v3']:.4f} | curl={r['curl']:.4f} | slip={r['mean_slip']:.2f}mm")

    # Top 5 by H_v3
    sorted_by_hv3 = sorted(results, key=lambda x: x['h_v3'], reverse=True)
    print(f"\n── Top 10 H_v3 Windows ──")
    for r in sorted_by_hv3[:10]:
        print(f"  {r['date']} | H_v3={r['h_v3']:.4f} | balance={r['balance']:.1f} | curl={r['curl']:.4f} | slip={r['mean_slip']:.2f}mm")

    # Parkfield M6.0 was 2004-09-28
    print(f"\n── Windows near 2004 Parkfield M6.0 (2004-09-28) ──")
    parkfield_windows = [r for r in results if '2004' in r['date']]
    for r in sorted(parkfield_windows, key=lambda x: x['balance'], reverse=True)[:10]:
        print(f"  {r['date']} | balance={r['balance']:.1f} | H_v3={r['h_v3']:.4f} | curl={r['curl']:.4f} | slip={r['mean_slip']:.2f}mm")

if __name__ == '__main__':
    main()
