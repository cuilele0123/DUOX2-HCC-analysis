#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线 B 第三步 · 附图：DUOX2 关联程序与免疫治疗应答（GSE140901, n=24）
输出 Figure_ICI_GSE140901.png（300 dpi）
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'axes.linewidth': 0.8, 'font.size': 7,
                     'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

D = pd.read_csv(os.path.join(HERE, 'GSE140901_scores.tsv'), sep='\t', index_col=0)
B = pd.read_csv(os.path.join(HERE, 'GSE140901_ICI_benefit.tsv'), sep='\t')
B = B.set_index('score')

ORDER = ['T效应/IFN-$\\gamma$', 'T细胞耗竭', '髓系抑制/M2', 'I型干扰素',
         '上皮/管状(近似)', 'P_DUOX2', '间质/CAF', '内皮', 'Treg']
NICE = {'P_DUOX2': 'DUOX2-associated programme',
        'T效应/IFN-$\\gamma$': 'T effector / IFN-$\\gamma$',
        'T细胞耗竭': 'T-cell exhaustion',
        '髓系抑制/M2': 'Myeloid suppression / M2',
        '上皮/管状(近似)': 'Epithelial / ductular',
        '间质/CAF': 'Stromal / CAF',
        '内皮': 'Endothelial',
        'Treg': 'Treg',
        'I型干扰素': 'Type I interferon'}

fig = plt.figure(figsize=(7.2, 2.55))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.85, 0.95],
                      wspace=0.62, left=0.218, right=0.985, top=0.86, bottom=0.20)

# ---------------- a: AUC forest ----------------
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(ORDER))[::-1]
for i, k in enumerate(ORDER):
    a, lo, hi = B.loc[k, 'AUC'], B.loc[k, 'AUC_lo'], B.loc[k, 'AUC_hi']
    isduox = (k == 'P_DUOX2')
    col = '#B2182B' if isduox else '#4A6FA5'
    ax.plot([lo, hi], [y[i], y[i]], color=col, lw=1.1, solid_capstyle='round', zorder=2)
    ax.plot([a], [y[i]], marker='o', ms=4.2, color=col, zorder=3)
ax.axvline(0.5, color='0.35', lw=0.8, ls='--', zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([NICE[k] for k in ORDER], fontsize=6.5)
ax.set_xlim(0.12, 1.0)
ax.set_xlabel('AUC for clinical benefit (95% CI)', fontsize=6.8)
ax.set_title('a', loc='left', fontsize=10, fontweight='bold', pad=3)
ax.tick_params(axis='x', labelsize=6.3)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.text(0.5, len(ORDER) - 0.35, 'no discrimination', fontsize=5.6, color='0.35',
        ha='center', va='bottom')

# ---------------- b: P_DUOX2 by benefit ----------------
ax2 = fig.add_subplot(gs[0, 1])
grp = [D.loc[D['cbr'] == 0, 'P_DUOX2'].values, D.loc[D['cbr'] == 1, 'P_DUOX2'].values]
bp = ax2.boxplot(grp, widths=0.5, patch_artist=True, showfliers=False,
                 medianprops=dict(color='black', lw=1.0),
                 boxprops=dict(facecolor='#EAEAEA', edgecolor='0.3', lw=0.7),
                 whiskerprops=dict(color='0.3', lw=0.7),
                 capprops=dict(color='0.3', lw=0.7))
rng = np.random.default_rng(7)
for j, v in enumerate(grp):
    ax2.scatter(j + 1 + rng.uniform(-0.13, 0.13, len(v)), v, s=9,
                color=('#4A6FA5' if j == 0 else '#B2182B'), alpha=0.85,
                edgecolor='none', zorder=4)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['No\n(n=%d)' % len(grp[0]), 'Yes\n(n=%d)' % len(grp[1])], fontsize=6.3)
ax2.set_ylabel('DUOX2-associated programme (z)', fontsize=6.8)
ax2.set_title('b', loc='left', fontsize=10, fontweight='bold', pad=3)
ax2.tick_params(axis='y', labelsize=6.3)
for s in ('top', 'right'):
    ax2.spines[s].set_visible(False)
ax2.text(0.5, 0.97, 'AUC 0.64\nP = 0.27', transform=ax2.transAxes, ha='center',
         va='top', fontsize=6.0)

# ---------------- c: KM for programme vs exhaustion ----------------
ax3 = fig.add_subplot(gs[0, 2])


def km(ax3, x, t, e, color, label, ls='-'):
    med = np.median(x)
    g = x > med
    for mask, col, lab, l in ((g, color, label, ls),
                              (~g, color, '_nolegend_', '--')):
        tt = np.sort(np.unique(t[mask & (e == 1)]))
        surv, at = [], []
        s = 1.0
        for k in tt:
            nrisk = (t[mask] >= k).sum()
            d = ((t[mask] == k) & (e[mask] == 1)).sum()
            s *= (1 - d / nrisk) if nrisk else 1
            at.append(k); surv.append(s)
        at = np.concatenate([[0], at, [t[mask].max()]])
        surv = np.concatenate([[1], surv, [surv[-1] if surv else 1]])
        ax3.step(at, surv, where='post', color=col, lw=1.1, ls=l, label=lab)
    return g


t_ = D['pfs_time'].values.astype(float)
e_ = D['pfs_event'].values.astype(int)
km(ax3, D['P_DUOX2'].values.astype(float), t_, e_, '#B2182B', 'DUOX2 programme')
km(ax3, D['T细胞耗竭'].values.astype(float), t_, e_, '#4A6FA5', 'exhaustion')
ax3.set_xlabel('Progression-free survival (months)', fontsize=6.8)
ax3.set_ylabel('PFS probability', fontsize=6.8)
ax3.set_ylim(0, 1.02)
ax3.set_xlim(0, 145)
ax3.set_title('c', loc='left', fontsize=10, fontweight='bold', pad=3)
ax3.tick_params(labelsize=6.3)
for s in ('top', 'right'):
    ax3.spines[s].set_visible(False)
ax3.legend(fontsize=5.4, frameon=False, loc='upper right', handlelength=1.5,
           labelspacing=0.28, borderaxespad=0.35)

out = os.path.join(HERE, 'Figure_ICI_GSE140901.png')
fig.savefig(out, dpi=300, facecolor='white')
fig.savefig(out.replace('.png', '.pdf'), facecolor='white')
print('已输出:', out)
