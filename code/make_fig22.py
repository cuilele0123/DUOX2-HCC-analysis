#!/usr/bin/env python
"""Figure 22 for v12: head-to-head comparison against clinical staging and an
honest map of cohort independence."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    # Scientific Reports 图片规范：clear sans-serif（Helvetica / Arial）
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic',
    'mathtext.bf': 'Helvetica:bold',
    'mathtext.default': 'regular',
    'font.size': 5.9,
    'axes.labelsize': 6.6,
    'axes.titlesize': 6.9,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 5.6,
    'ytick.labelsize': 5.6,
    'legend.fontsize': 5.3,
    'axes.linewidth': 0.53,
    'pdf.fonttype': 42,
})

C = {'red': '#C0392B', 'blue': '#2E5C8A', 'green': '#3B7A57',
     'orange': '#D68910', 'purple': '#7D3C98', 'gray': '#7F7F7F',
     'light': '#B9C6D6'}

H = json.load(open(os.path.join(D, 'v12_headtohead.json')))
G = json.load(open(os.path.join(D, 'v12_independent_validation.json')))


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, '%s.%s' % (name, ext)), dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', name)


# ------------------------------------------------------------------ Figure 22
fig, axes = plt.subplots(2, 2, figsize=(7.66, 5.71))

# ---- A: TCGA head-to-head C-index
ax = axes[0, 0]
labs = ['AJCC pathologic\nstage', '3-gene signature\n(out-of-fold)',
        '3-gene + AJCC\n(combined)']
vals = [H['tcga_h2h_cindex']['AJCC pathologic stage'],
        H['tcga_h2h_cindex']['3-gene signature (out-of-fold)'],
        H['tcga_h2h_cindex']['3-gene + AJCC (combined)']]
cols = [C['gray'], C['red'], C['blue']]
b = ax.bar(range(3), vals, color=cols, width=0.58, edgecolor='black', linewidth=0.46)
ax.set_xticks(range(3))
ax.set_xticklabels(labs)
ax.set_ylim(0.45, 0.72)
ax.set_ylabel('C-index')
ax.axhline(0.5, color='black', ls=':', lw=0.53)
for i, v in enumerate(vals):
    ax.text(i, v + 0.008, '%.3f' % v, ha='center', fontsize=5.9, fontweight='bold')
d = H['tcga_dC_signature_vs_AJCC']
ax.annotate('', xy=(1, 0.692), xytext=(0, 0.692),
            arrowprops=dict(arrowstyle='-', lw=0.53, color='black'))
ax.text(0.5, 0.697, r'$\Delta$C = +%.3f (95%% CI %.3f to %.3f)'
        % (d['delta'], d['ci'][0], d['ci'][1]), ha='center', fontsize=5.3)
ax.set_title('Head-to-head in TCGA-LIHC (n = %d, %d events)'
             % (H['tcga_h2h_n'], H['tcga_h2h_events']), loc='left')
ax.spines[['top', 'right']].set_visible(False)

# ---- B: multivariable Cox forest (TCGA)
ax = axes[0, 1]
mv = H['tcga_multivariable_cox']
rows = [('3-gene signature\n(per SD)', mv['HR_risk_per_SD'], mv['CI_risk_per_SD'],
         mv['p_risk']),
        ('AJCC stage\n(per stage)', mv['HR_ajcc'], mv['CI_ajcc'], mv['p_ajcc'])]
y = [1, 0]
for i, (lab, hr, ci_, p) in enumerate(rows):
    ax.plot([ci_[0], ci_[1]], [y[i], y[i]], color=C['red'] if i == 0 else C['gray'],
            lw=1.45, solid_capstyle='round')
    ax.plot(hr, y[i], 'D' if i == 0 else 's', ms=5.28,
            color=C['red'] if i == 0 else C['gray'], mec='black', mew=0.6)
ax.axvline(1.0, color='black', ls='--', lw=0.59)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
ax.set_xlabel('Hazard ratio (95% CI)')
ax.set_xlim(0.85, 1.95)
ax.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8])
for i, (lab, hr, ci_, p) in enumerate(rows):
    ptxt = 'P = %.3g' % p if p >= 0.001 else 'P = %.1g' % p
    ax.text(1.92, y[i], '%.2f (%.2f-%.2f)\n%s' % (hr, ci_[0], ci_[1], ptxt),
            va='center', ha='right', fontsize=5.3)
ax.set_title('Multivariable Cox, TCGA-LIHC (n = %d)' % mv['n'], loc='left')
ax.spines[['top', 'right']].set_visible(False)

# ---- C: GSE76427 head-to-head
ax = axes[1, 0]
h = G['gse76427_h2h']
labs = ['3-gene signature\n(fixed coefficients)', 'BCLC stage',
        'Clinical TNM stage', '3-gene + BCLC\n(combined)']
vals = [h['3-gene signature']['c_index'], h['BCLC stage']['c_index'],
        h['Clinical TNM stage']['c_index'], h['3-gene + BCLC (combined)']['c_index']]
cols = [C['red'], C['gray'], C['green'], C['blue']]
ax.bar(range(4), vals, color=cols, width=0.58, edgecolor='black', linewidth=0.46)
ax.set_xticks(range(4))
ax.set_xticklabels(labs, fontsize=5.3)
ax.set_ylim(0.40, 0.72)
ax.set_ylabel('C-index')
ax.axhline(0.5, color='black', ls=':', lw=0.53)
for i, v in enumerate(vals):
    ax.text(i, v + 0.008, '%.3f' % v, ha='center', fontsize=5.9, fontweight='bold')
d = G['gse76427_dC_3gene_vs_BCLC']
ax.annotate('', xy=(1, 0.692), xytext=(0, 0.692),
            arrowprops=dict(arrowstyle='-', lw=0.53, color='black'))
ax.text(0.5, 0.697, r'$\Delta$C = %.3f (95%% CI %.3f to %.3f)'
        % (d['delta'], d['ci'][0], d['ci'][1]), ha='center', fontsize=5.3)
ax.set_title('Head-to-head in GSE76427 (n = %d, %d events)'
             % (G['gse76427_n_survival'], G['gse76427_events']), loc='left')
ax.spines[['top', 'right']].set_visible(False)

# ---- D: cohort independence map
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Cohort independence and signature performance', loc='left',
             x=-0.03)


def box(x, y, w, h, fc, ec, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.10',
                                fc=fc, ec=ec, lw=0.66))
    ax.text(x + w / 2, y + h - 0.42, title, ha='center', fontsize=5.7,
            fontweight='bold')
    for k, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.95 - 0.47 * k, ln, ha='center', fontsize=5.1)


box(0.25, 7.05, 4.46, 2.5, '#FDF3F2', C['red'],
    'TCGA-LIHC (training)',
    ['n = 258, 80 events', 'cross-validated C-index 0.690',
     'out-of-fold C-index 0.662'])

box(5.29, 7.05, 4.46, 2.5, '#F2F5F9', C['blue'],
    'Firehose Legacy (cross-pipeline)',
    ['n = 343 analysed', '244/346 (71%) shared with TCGA-LIHC',
     'C-index 0.642', '0.572 in 103 non-overlapping (P = 0.13)'])

box(0.25, 3.75, 4.46, 2.5, '#F1F8F3', C['green'],
    'GSE14520 (independent, platform 1)',
    ['n = 221, no patient overlap', 'fixed coefficients: log-rank P = 5.5e-4',
     'refitted: C-index 0.634'])

box(5.29, 3.75, 4.46, 2.5, '#FBF6EC', C['orange'],
    'GSE76427 (independent, platform 2)',
    ['n = 115, no patient overlap', 'fixed coefficients: C-index 0.482',
     'refitted: C-index 0.550 (log-rank P = 0.43)', 'BCLC / TNM stage: 0.639'])

box(0.25, 0.35, 9.5, 2.6, '#F7F7F7', C['gray'],
    'Consequence for interpretation',
    ['Only GSE14520 and GSE76427 are fully independent of TCGA-LIHC.',
     'Fixed coefficients reversed direction in GSE14520 and were uninformative',
     'in GSE76427, where clinical staging outperformed the signature.',
     'Transportability across platforms therefore requires recalibration.'])

# 两框之间的箭头：跨满空隙并居中于 x=5.0（两框绘出边界约 4.81 / 5.19）。
# 旧版 xytext=4.8 -> xy=5.0 尾巴埋在左框内、箭头贴左，故目视"偏左且不显眼"。
def _gap_arrow(y):
    # 实心箭头尖端很细，抗锯齿后视觉长度比代码端点短约 4 px（0.04 数据单位），
    # 故把头端外扩到 5.21 做补偿，使**渲染后**的箭头正好居中于两框空隙（x=5.0）。
    ax.annotate('', xy=(5.21, y), xytext=(4.83, y),
                arrowprops=dict(arrowstyle='-|>', lw=1.45, color='#333333',
                                mutation_scale=8.0, shrinkA=0, shrinkB=0))


_gap_arrow(8.3)   # 上排：TCGA-LIHC  <->  Firehose Legacy
_gap_arrow(5.0)   # 下排：GSE14520    <->  GSE76427

fig.tight_layout(rect=[0, 0, 1, 0.985])
save(fig, 'Fig22_headtohead_v12')
