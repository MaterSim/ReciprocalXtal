from pathlib import Path
import sys

# Allow importing sibling reciprocal.py when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from reciprocal import RECP
from pyxtal import pyxtal
import matplotlib.pyplot as plt

# Use a clean white background style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'axes.grid': False,
    'grid.alpha': 0.2,
    'grid.color': 'gray',
    'grid.linestyle': '-',
    'lines.linewidth': 1.5,
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

recp = RECP(dmax=10.0, nmax=10, lmax=10, rbasis='Bessel')
prototypes = ['diamond',
              'h-diamond',
              #'graphite',
              #'$\alpha$-boron',
              'a-quartz',
              'b-quartz']

fig, axs = plt.subplots(len(prototypes), 1, figsize=(6, 1.6*len(prototypes)))
data = []
for row, prototype in enumerate(prototypes):
    xtal = pyxtal()
    xtal.from_prototype(prototype)
    p, rdf = recp.compute(xtal.to_ase(), norm=True)
    # Plot p values
    if prototype == 'a-quartz':
        label = r'$\alpha$-quartz'
    elif prototype == 'b-quartz':
        label = r'$\beta$-quartz'
    else:
        label = prototype
    label += f" ({xtal.group.number}, {xtal.group.symbol})"
    axs[row].plot(p, label=label)
    axs[row].legend(loc=1)
    axs[row].set_ylabel('$P_{nl}$')

    if row == len(prototypes) - 1:
        axs[row].set_xlabel('Power Spectrum Index')
    else:
        axs[row].set_xticklabels([])
    axs[row].set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig('Fig4.png', dpi=300)
plt.close()
