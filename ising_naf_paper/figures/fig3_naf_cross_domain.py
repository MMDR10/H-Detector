"""Figure 3: NAF Ismetpasa — 6-station helicity baseline vs San Andreas emergence."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

# NAF data
stations = ['WN\n(Wall N)', 'HA\n(Hamamli)', 'PE\n(Petrol)', 'SW\n(Sazlik E)', 'SE\n(Sazlik W)', 'WS\n(Wall S)']
o_naf = [0.579, 0.557, 0.558, 0.557, 0.548, 0.533]
rates = [5.6, 9.0, 0.8, -1.2, 1.6, 1.0]
days = [1964, 598, 744, 675, 330, 132]

# Ising critical baseline
ising_crit = [0.579, 0.575, 0.587]  # L=32, 64, 128

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel (a): NAF station helicity
colors_naf = plt.cm.viridis(np.linspace(0.2, 0.8, 6))
for i, (sta, val, rate, c) in enumerate(zip(stations, o_naf, rates, colors_naf)):
    ax1.bar(i, val, color=c, alpha=0.85, edgecolor='k', linewidth=0.8, width=0.65)
    ax1.text(i, val + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax1.text(i, 0.445, f'{rate} mm/yr', ha='center', va='bottom', fontsize=7, color='gray', fontstyle='italic')

# Ising critical range
ising_mean = np.mean(ising_crit)
ising_std = np.std(ising_crit)
ax1.axhline(y=ising_mean, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.fill_between([-0.5, 5.5], ising_mean - ising_std, ising_mean + ising_std,
                 alpha=0.12, color='#1f77b4')
ax1.text(4.5, ising_mean + 0.005, f'Ising $T_c$ ({ising_mean:.3f})', fontsize=9,
         color='#1f77b4', fontweight='bold')

# Random baseline
ax1.axhline(y=0.5, color='k', linestyle=':', alpha=0.4, linewidth=1)

ax1.set_xticks(range(6))
ax1.set_xticklabels(stations, fontsize=8.5)
ax1.set_ylabel('Ô Helicity', fontsize=12)
ax1.set_xlim(-0.5, 5.5)
ax1.set_ylim(0.44, 0.65)
ax1.set_title('(a) NAF Ismetpasa (2014–2019): 6 Stations', fontsize=13, fontweight='bold')

# NAF ensemble annotation
naf_mean = np.mean(o_naf)
naf_sigma = np.std(o_naf)
ax1.text(2.5, 0.62, f'NAF ensemble: Ô = {naf_mean:.3f} ± {naf_sigma:.3f}\n(6 stations, σ = {naf_sigma:.3f})',
         ha='center', fontsize=9.5,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85))

# Panel (b): Cross-domain comparison
domains = ['Random\nBaseline', 'NAF\nIsmetpasa', 'Ising\n$T_c$', 'Ising\nPARA', 'Ising\nFERRO', 'San Andreas\nPost-seismic']
h_vals = [0.500, naf_mean, ising_mean, 0.599, 0.758, 0.90]
h_errs = [0, naf_sigma, ising_std, 0.012, 0.026, 0.05]
domain_colors = ['gray', '#2196F3', '#1f77b4', '#2ca02c', '#d62728', '#FF5722']

y_pos = range(len(domains))
bars = ax2.barh(y_pos, h_vals, xerr=h_errs, color=domain_colors, alpha=0.85,
                edgecolor='k', linewidth=0.8, height=0.6, capsize=3)

for i, (bar, val, err) in enumerate(zip(bars, h_vals, h_errs)):
    ax2.text(val + err + 0.01, i, f'{val:.3f} ± {err:.3f}', va='center', fontsize=10, fontweight='bold')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(domains, fontsize=9.5)
ax2.set_xlabel('Ô Helicity', fontsize=12)
ax2.set_xlim(0.4, 1.10)
ax2.axvline(x=0.5, color='k', linestyle=':', alpha=0.5, linewidth=1)
ax2.set_title('(b) Cross-Domain Ô Spectrum', fontsize=13, fontweight='bold')

# Annotations
ax2.annotate('NULL BASELINE', xy=(0.55, 1.5), fontsize=8, color='blue',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
ax2.annotate('CRITICAL\nFLUCTUATIONS', xy=(0.59, 2.5), fontsize=8, color='#1f77b4',
            bbox=dict(boxstyle='round', facecolor='lightskyblue', alpha=0.4))
ax2.annotate('STRUCTURED\nEMERGENCE', xy=(0.92, 5.5), fontsize=8, color='#FF5722',
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.4))

plt.tight_layout()
plt.savefig('output/arxiv/ising_naf_paper/figures/fig3_naf_cross_domain.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('output/arxiv/ising_naf_paper/figures/fig3_naf_cross_domain.png',
            dpi=300, bbox_inches='tight')
print("Figure 3 saved.")
