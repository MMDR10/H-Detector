"""Figure 2: Finite-size scaling — Critical helicity vs L."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

L_vals = [32, 64, 128]
H_ferro = [0.762, 0.805, 0.758]
H_crit = [0.579, 0.575, 0.587]
H_para = [0.575, 0.590, 0.599]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# Panel (a): H vs L for all phases
for phase, vals, label, color, marker in [
    ('FERRO', H_ferro, 'Ferromagnetic', '#d62728', '^'),
    ('PARA', H_para, 'Paramagnetic', '#2ca02c', 'v'),
    ('CRITICAL', H_crit, 'Critical ($T_c$)', '#1f77b4', 's'),
]:
    ax1.plot(L_vals, vals, '-o', color=color, marker=marker, markersize=10,
             linewidth=1.8, label=label, zorder=3)

ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, linewidth=1)
ax1.text(130, 0.505, 'Random', fontsize=9, color='gray')

ax1.set_xlabel('Lattice Size $L$', fontsize=12)
ax1.set_ylabel('Helicity $H$', fontsize=12)
ax1.set_title('(a) Finite-Size Scaling', fontsize=13, fontweight='bold')
ax1.legend(loc='best', framealpha=0.9, fontsize=10)
ax1.set_xlim(25, 135)
ax1.set_ylim(0.47, 0.87)
ax1.set_xticks(L_vals)

# Annotate σ(L)
for vals, color, phase in [(H_ferro, '#d62728', 'FERRO'),
                              (H_crit, '#1f77b4', 'CRITICAL'),
                              (H_para, '#2ca02c', 'PARA')]:
    sigma = np.std(vals)
    ax1.annotate(f'$\\sigma = {sigma:.3f}$', xy=(128, vals[-1]),
                xytext=(133, vals[-1]+0.015), fontsize=8, color=color)

# Panel (b): Cross-lattice comparison
x_labels = ['L=32', 'L=64', 'L=128']
x = np.arange(len(x_labels))
width = 0.25

for i, (phase, vals, color) in enumerate([
    ('FERRO', H_ferro, '#d62728'),
    ('CRITICAL', H_crit, '#1f77b4'),
    ('PARA', H_para, '#2ca02c'),
]):
    bars = ax2.bar(x + i*width, vals, width, color=color, alpha=0.85, edgecolor='k', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)

ax2.set_xticks(x + width)
ax2.set_xticklabels(x_labels)
ax2.set_ylabel('Helicity $H$', fontsize=12)
ax2.set_title('(b) Cross-Lattice Phase Consistency', fontsize=13, fontweight='bold')
ax2.set_ylim(0.50, 0.88)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', alpha=0.85, label='Ferromagnetic'),
    Patch(facecolor='#1f77b4', alpha=0.85, label='Critical'),
    Patch(facecolor='#2ca02c', alpha=0.85, label='Paramagnetic'),
]
ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)

plt.tight_layout()
plt.savefig('output/arxiv/ising_naf_paper/figures/fig2_finite_size_scaling.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('output/arxiv/ising_naf_paper/figures/fig2_finite_size_scaling.png',
            dpi=300, bbox_inches='tight')
print("Figure 2 saved.")
