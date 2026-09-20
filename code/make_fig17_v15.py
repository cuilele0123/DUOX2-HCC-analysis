"""Redraw Figure 17 with a neutral title (ranking is descriptive, not 'ranks first')."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'figures')
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(BASE, 'data', 'ml_model_comparison_final.csv'))
NAME = {
    '三基因模型 (SLC16A3+SPP2+MMP7)': '3-gene LASSO-Cox',
    'Random Survival Forest': 'Random Survival Forest',
    'Gradient Boosting': 'Gradient Boosting',
    'Ridge-Cox (Coxnet, alpha tuned)': 'Ridge-Cox',
    'Survival SVM': 'Survival SVM',
    'Lasso-Cox (Coxnet, alpha tuned)': 'Lasso-Cox',
}
d['label'] = d['Model'].map(NAME)
d = d.sort_values('C_index')  # ascending so best ends on top

C_BAR = '#4C82B6'
C_TOP = '#C0504D'
colors = [C_TOP if l == '3-gene LASSO-Cox' else C_BAR for l in d['label']]

fig, ax = plt.subplots(figsize=(7.6, 4.4))
y = np.arange(len(d))
ax.barh(y, d['C_index'], xerr=d['SD'], color=colors, height=0.62,
        error_kw=dict(ecolor='#444444', lw=1.1, capsize=3))
for yi, (c, s) in enumerate(zip(d['C_index'], d['SD'])):
    ax.text(0.79, yi, '%.3f +/- %.3f' % (c, s), va='center', fontsize=9)
ax.axvline(0.5, color='#999999', ls='--', lw=1.0, zorder=0)
ax.set_yticks(y)
ax.set_yticklabels(d['label'], fontsize=10)
ax.set_xlabel('Cross-validated C-index (5 repeats x 5 folds)', fontsize=11)
ax.set_xlim(0.45, 0.80)
ax.set_title('Prognostic model comparison', fontsize=12, fontweight='bold', pad=10)
ax.text(0.502, -0.62, 'random (0.5)', color='#777777', fontsize=8)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'Fig17_model_comparison_v15.png'), dpi=300)
fig.savefig(os.path.join(OUT, 'Fig17_model_comparison_v15.pdf'), dpi=300)
print('saved Fig17_model_comparison_v15.png')
