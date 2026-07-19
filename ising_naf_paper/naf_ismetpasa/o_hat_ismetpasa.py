"""
Ô-hat analysis for Ismetpasa (North Anatolian Fault) creepmeter data.
Compares rate-dependence to San Andreas (Parkfield) results.

Key question: Is post-seismic Ô emergence universal across fault types,
or specific to San Andreas (strike-slip, creeping)?
NAF Ismetpasa is also strike-slip creeping → direct comparison.

Data: 30-min sampling, MM/DD/YYYY HH:MM:SS, displacement(mm)
Stations: WN (Wall North, 33°), WS (Wall South, 33°)
Obliquity: 33° → raw / cos(33°) = fault-parallel slip
"""
import numpy as np
import pandas as pd
from datetime import datetime
import json, os, sys

# Use existing Ô-hat if available
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
try:
    from o_hat.o_hat import compute_o_hat
except ImportError:
    # Fallback: embedded Ô computation
    def compute_o_hat(data, window=100, stride=50, k=10):
        """Sliding-window helicity Ô for 1D time series.
        Ô = fraction of k-nearest neighbor pairs where angular 
        displacement vector rotates consistently.
        """
        n = len(data)
        n_windows = max(1, (n - window) // stride + 1)
        o_values = []
        positions = []
        
        for i in range(n_windows):
            start = i * stride
            end = min(start + window, n)
            if end - start < 3:
                continue
            segment = np.array(data[start:end], dtype=float)
            # Compute pairwise angular displacements in (t, y) embedding
            t = np.arange(len(segment))
            dx = np.diff(t)
            dy = np.diff(segment)
            angles = np.arctan2(dy, dx)
            # Helicity: consistency of angular displacement
            if len(angles) < 3:
                continue
            # Cross product of consecutive displacement vectors
            cross_products = []
            for j in range(len(angles) - 2):
                v1 = np.array([dx[j], dy[j]])
                v2 = np.array([dx[j+1], dy[j+1]])
                cross = np.cross(v1, v2)
                cross_products.append(cross)
            cross_products = np.array(cross_products)
            if len(cross_products) == 0:
                continue
            # Ô: fraction of consistent-sign cross products
            n_pos = np.sum(cross_products > 0)
            n_neg = np.sum(cross_products < 0)
            total = n_pos + n_neg
            if total == 0:
                continue
            o_val = max(n_pos, n_neg) / total
            o_values.append(float(o_val))
            positions.append(start + window // 2)
        
        return np.array(o_values), np.array(positions)


def parse_ismerpasa(path, name):
    """Parse Ismetpasa creepmeter file."""
    df = pd.read_csv(path, names=['datetime', 'displacement'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M:%S')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Fault-parallel slip (obliquity = 33°)
    obliquity = np.radians(33)
    df['slip_mm'] = df['displacement'] / np.cos(obliquity)
    
    return df

def compute_daily_rate(df):
    """Compute daily slip rate from cumulative displacement."""
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg(
        disp_start=('slip_mm', 'first'),
        disp_end=('slip_mm', 'last'),
        n_points=('slip_mm', 'count')
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily['rate_mm_day'] = (daily['disp_end'] - daily['disp_start'])
    # Filter days with sufficient coverage
    daily = daily[daily['n_points'] >= 24]  # at least 12h coverage
    return daily

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    stations = {
        'WN': os.path.join(data_dir, 'IsmetpasaWall%20North%209Oct2019.txt'),
        'WS': os.path.join(data_dir, 'IsmetpasaWall%20South%209Oct2019.txt'),
    }
    
    results = {}
    
    for code, path in stations.items():
        print(f"\n{'='*60}")
        print(f"  Station {code} (Ismetpasa, {33}° obliquity)")
        print(f"{'='*60}")
        
        df = parse_ismerpasa(path, code)
        print(f"  Raw: {len(df):,} records, {df['datetime'].min()} → {df['datetime'].max()}")
        
        # Daily rate
        daily = compute_daily_rate(df)
        print(f"  Daily: {len(daily)} days, rate {daily['rate_mm_day'].mean():.4f} ± {daily['rate_mm_day'].std():.4f} mm/day")
        
        # Ô on daily rate (30-day window, matching Parkfield analysis)
        o_vals, positions = compute_o_hat(daily['rate_mm_day'].values, window=30, stride=7)
        print(f"  Ô: {len(o_vals)} windows, mean = {np.mean(o_vals):.4f}")
        
        # Save
        out = {
            'code': code,
            'fault': 'North Anatolian Fault',
            'segment': 'Ismetpasa',
            'obliquity_deg': 33,
            'n_records': int(len(df)),
            'n_daily': int(len(daily)),
            't_start': str(df['datetime'].min()),
            't_end': str(df['datetime'].max()),
            'daily_rate_mean': float(daily['rate_mm_day'].mean()),
            'daily_rate_std': float(daily['rate_mm_day'].std()),
            'o_daily_mean': float(np.mean(o_vals)) if len(o_vals) > 0 else None,
            'o_daily_std': float(np.std(o_vals)) if len(o_vals) > 0 else None,
            'n_o_windows': int(len(o_vals)),
            'o_values': o_vals.tolist() if len(o_vals) > 0 else [],
            'dates': [str(d) for d in daily['date'].values],
            'rates': daily['rate_mm_day'].tolist(),
        }
        
        results[code] = out
        
        # Compare to San Andreas
        print(f"\n  --- Comparison with San Andreas (Parkfield) ---")
        print(f"  SA creep rate: ~15-25 mm/yr (Parkfield)")
        print(f"  NAF creep rate: {daily['rate_mm_day'].mean() * 365:.1f} mm/yr")
        print(f"  SA Ô post-seismic: 0.85-0.95 (EMERGENCE, p<0.001)")
        print(f"  NAF Ô mean:        {np.mean(o_vals):.4f}")
    
    # Combined output
    outpath = os.path.join(data_dir, 'ismetpasa_o_results.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {outpath}")

if __name__ == '__main__':
    main()
