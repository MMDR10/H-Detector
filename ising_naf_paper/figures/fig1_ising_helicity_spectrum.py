"""Figure 1: Cross-Lattice Helicity V-spectrum for Ising L=32/64/128."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Helicity data
L = [32, 64, 128]
phases = ['FERRO', 'CRITICAL', 'PARA']
H_data = {
    32:  [0.762, 0.579, 0.575],
    64:  [0.805, 0.575, 0.590],
    128: [0.758, 0.587, 0.599],
}
colors = {'FERRO': '#d62728', 'CRITICAL': '#1f77b4', 'PARA': '#2ca02c'}
markers = {32: 'o', 64: 's', 128: 'D'}

plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times New Roman']})
fig, ax = plt.subplots(figsize=(6, 4.5))

for L_i in L:
    for i, phase in enumerate(phases):
        ax.scatter(i, H_data[L_i][i], marker=markers[L_i], s=120,
                   c=colors[phase], edgecolors='k', linewidths=0.8,
                   zorder=3, label=f'L={L_i}' if i == 0 else None)

# Connect points for each L
for L_i in L:
    xs = [0, 1, 2]
    ys = H_data[L_i]
    ax.plot(xs, ys, '-', color='gray', alpha=0.3, zorder=1)

# Phase mean
for i, phase in enumerate(phases):
    vals = [H_data[l][i] for l in L]
    mean_val = np.mean(vals)
    ax.axhline(y=mean_val, xmin=0.08 + i*0.32, xmax=0.22 + i*0.32,
               color=colors[phase], linewidth=3, alpha=0.4, zorder=2)
    # Annotate mean
    ax.annotate(f'{mean_val:.3f}', xy=(i, mean_val), xytext=(i+0.2, mean_val+0.01),
                fontsize=9, color=colors[phase], fontweight='bold')

# Random baseline
ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.4, linewidth=1)
ax.text(2.3, 0.505, 'Random (0.5)', fontsize=9, color='gray', fontstyle='italic')

# Styling
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Ferromagnetic\n(Ordered)', 'Critical\n($T = T_c$)', 'Paramagnetic\n(Disordered)'])
ax.set_ylabel('Cross-Lattice Helicity $H$', fontsize=13)
ax.set_ylim(0.48, 0.85)
ax.set_xlim(-0.4, 2.4)

# Title
ax.set_title('Ising 2D Square Lattice — Cross-Lattice Helicity Spectrum', fontsize=14, fontweight='bold', pad=15)

# Legend
handles = []
for L_i in L:
    handles.append(plt.Line2D([0], [0], marker=markers[L_i], color='k', linestyle='None',
                               markersize=8, label=f'$L = {L_i}$'))
ax.legend(handles=handles, loc='upper right', framealpha=0.9, fontsize=10)

# Annotation
ax.annotate('$H_\\mathrm{FERRO} > H_\\mathrm{PARA} > H_\\mathrm{CRITICAL}$',
            xy=(0.5, 0.82), fontsize=11, color='#7f0000',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Phase arrows
for i, (phase, color) in enumerate(colors.items()):
    val_mean = np.mean([H_data[l][i] for l in L])
    label = f'{phase}\n({val_mean:.3f})'
    ax.annotate(label, xy=(i, val_mean), xytext=(i, val_mean-0.12),
                fontsize=8, ha='center', va='top', color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

plt.tight_layout()
plt.savefig('output/arxiv/ising_naf_paper/figures/fig1_ising_helicity_spectrum.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('output/arxiv/ising_naf_paper/figures/fig1_ising_helicity_spectrum.png',
            dpi=300, bbox_inches='tight')
print("Figure 1 saved.")
