"""
Multi-station Ô-hat analysis for all Ismetpasa/NAF creepmeter stations.
"""
import numpy as np
import pandas as pd
import json, os, sys

sys.path.insert(0, '/app/working/workspaces/tygtDc')
from output.creepmeter.ismetpasa.o_hat_ismetpasa import parse_ismerpasa, compute_daily_rate, compute_o_hat

def analyze_station(path, code, obliquity=30):
    """Run full Ô analysis on one station."""
    # Override obliquity per station
    df = pd.read_csv(path, names=['datetime', 'displacement'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M:%S')
    df = df.sort_values('datetime').reset_index(drop=True)
    df['slip_mm'] = df['displacement'] / np.cos(np.radians(obliquity))
    df['date'] = df['datetime'].dt.date
    
    daily = df.groupby('date').agg(
        disp_start=('slip_mm', 'first'),
        disp_end=('slip_mm', 'last'),
        n_points=('slip_mm', 'count')
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily['rate_mm_day'] = daily['disp_end'] - daily['disp_start']
    daily = daily[daily['n_points'] >= 24]
    
    if len(daily) < 30:
        o_mean = o_std = None
        n_o = 0
    else:
        window = min(30, len(daily) // 3)
        o_vals, _ = compute_o_hat(daily['rate_mm_day'].values, window=window, stride=max(7, window//4))
        o_mean = float(np.mean(o_vals)) if len(o_vals) > 0 else None
        o_std = float(np.std(o_vals)) if len(o_vals) > 0 else None
        n_o = len(o_vals)
    
    return {
        'code': code,
        'obliquity': obliquity,
        'n_records': len(df),
        'n_daily': len(daily),
        't_start': str(df['datetime'].min()),
        't_end': str(df['datetime'].max()),
        'rate_mm_day_mean': float(daily['rate_mm_day'].mean()),
        'rate_mm_day_std': float(daily['rate_mm_day'].std()),
        'rate_mm_yr': float(daily['rate_mm_day'].mean() * 365),
        'o_mean': o_mean,
        'o_std': o_std,
        'n_o_windows': n_o,
    }

def main():
    d = '/app/working/workspaces/tygtDc/output/creepmeter/ismetpasa'
    stations = [
        (f'{d}/IsmetpasaWall%20North%209Oct2019.txt', 'WN', 33),
        (f'{d}/IsmetpasaWall%20South%209Oct2019.txt', 'WS', 33),
        (f'{d}/Hamamli_9Oct2019.txt', 'HA', 30),
        (f'{d}/Petrol_9Oct2019.txt', 'PE', 30),
        (f'{d}/Sazlik_East_9Oct2019.txt', 'SW', 28),  # Sazlik E = SW in UNAVCO table
        (f'{d}/Sazlik_West_9Oct2019.txt', 'SE', 28),  # Sazlik W = SE in UNAVCO table
    ]
    
    results = {}
    for path, code, obl in stations:
        print(f"  {code} ({obl}°)...", end=' ', flush=True)
        r = analyze_station(path, code, obl)
        results[code] = r
        print(f"{r['n_daily']}d, Ô={r['o_mean']}")
    
    out = f'{d}/naf_multi_station_o.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  NAF Multi-Station Ô Summary")
    print(f"{'='*60}")
    print(f"{'Code':<6} {'Days':<8} {'Rate mm/d':<12} {'Rate mm/yr':<12} {'Ô mean':<10} {'#windows'}")
    print(f"{'-'*60}")
    for code in ['WN','WS','HA','PE','SW','SE']:
        r = results[code]
        o_str = f"{r['o_mean']:.4f}" if r['o_mean'] else "N/A"
        print(f"{code:<6} {r['n_daily']:<8} {r['rate_mm_day_mean']:<12.4f} {r['rate_mm_yr']:<12.1f} {o_str:<10} {r['n_o_windows']}")
    
    print(f"\n✅ Saved → {out}")

if __name__ == '__main__':
    main()
