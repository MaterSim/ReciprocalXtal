from pathlib import Path
import sys

# Allow importing sibling reciprocal.py when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyxtal import pyxtal
from reciprocal import RECP
import matplotlib.pyplot as plt
import numpy as np
import random


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


set_global_seed(42)

# Use a clean white background style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'axes.grid': False,
    'grid.alpha': 0.2,
    'grid.color': 'gray',
    'grid.linestyle': '-',
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Define colors matching template
color_reference = '#1f77b4'  # Blue
color_perturbed = '#ff7f0e'  # Orange

data = [(227, ['8a'], [3.567]),
        (216, ['4a', '4d'], [3.567]),
        (210, ['8b'], [3.567]),
        (203, ['8a'], [3.567]),
        (166, ['6c'], [2.522, 6.178, 0.125]),
        (141, ['4a'], [2.522, 3.567]),
        (122, ['4b'], [2.522, 3.567]),
        (119, ['2b', '2d'], [2.522, 3.567]),
        (109, ['4a'], [2.522, 3.567, 0.875]),
        (98, ['4b'], [2.522, 3.567]),
        (88, ['4a'], [2.522, 3.567], ),
        #(74, ['4e'], [2.522, 2.522, 3.567, 0.875]),
        (70, ['8a'], [3.567, 3.567, 3.567])]

data0 = []
recp = RECP(dmax=10.0, nmax=10, lmax=10, rbasis='Bessel')


xtal = pyxtal()
xtal.from_prototype('diamond')
xtal_sub = xtal.subgroup_once(H=141, eps=0)

fig, axs = plt.subplots(3, 2, figsize=(12, 7.5))
for row, eps in enumerate([0, 0.02, 0.05]):
    data1 = []
    for i, d in enumerate(data):
        if i == 0:
            xtal0 = xtal.copy()
        else:
            xtal0 = xtal.subgroup_once(H=d[0], eps=eps)
        if xtal0 is not None:
            p, rdf = recp.compute(xtal0.to_ase(), norm=True)
        else:
            xtal0 = xtal_sub.subgroup_once(H=d[0], eps=eps)
            if xtal0 is None:
                print('problem', d[0]); import sys; sys.exit()
            p, rdf = recp.compute(xtal0.to_ase(), norm=True)
        data1.append((p, rdf, f'{xtal0.group.number}'))

    # Plot p values
    axs[row, 0].set_title(f'$P$ (noise={eps})', y=0.80, x=0.01, loc='left')
    for p, _, label in data1:
        axs[row, 0].plot(p, label=label, alpha=0.5, lw=1.0)


    axs[row, 0].set_ylabel('$P_{nl}$')
    if row == 2:
        axs[row, 0].set_xlabel('Power Spectrum Index')

    # Plot rdf values
    axs[row, 1].set_title(f'$G$ (noise={eps})', y=0.80, x=0.01, loc='left')
    for _, rdf, label in data1:
        axs[row, 1].plot(rdf, label=label, alpha=0.5, lw=1.0)
    if row == 2:
        axs[row, 1].set_xlabel(r'$d$ ($\mathrm{\AA}^{-1}$)')
    else:
        axs[row, 0].set_xticklabels([])
        axs[row, 1].set_xticklabels([])

    axs[row, 1].set_ylabel('$G(d)$')
    if row == 0: axs[row, 1].legend(ncol=2, loc='upper right')

plt.tight_layout()
plt.savefig('diamond_noise_robustness.png')
plt.close()



