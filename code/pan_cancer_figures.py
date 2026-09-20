# -*- coding: utf-8 -*-
"""生成泛癌 DUOX2 分析图表"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'
FIG = f'{BASE}/figures'
os.makedirs(FIG, exist_ok=True)

C1, C2, C3 = '#C0392B', '#2980B9', '#27AE60'

df = pd.read_csv(f'{DATA}/pan_cancer_DUOX2.csv')
df['DUOX2_log2'] = np.log2(df['DUOX2_expr'] + 1)
surv = pd.read_csv(f'{DATA}/pan_cancer_DUOX2_survival.csv')

# ===== 图9：泛癌生存森林图 =====
fig, ax = plt.subplots(figsize=(8, 7))
d = surv.sort_values('HR')
colors = [C1 if (r < 0.05 and hr > 1) else (C3 if (r < 0.05 and hr < 1) else '#BDC3C7')
          for hr, r in zip(d['HR'], d['Cox_p'])]
y = np.arange(len(d))
ax.errorbar(d['HR'], y, xerr=[d['HR']-0.85, 1.15-d['HR']], fmt='none', ecolor='grey', elinewidth=1, capsize=3)
ax.scatter(d['HR'], y, c=colors, s=45, zorder=3)
ax.axvline(1, color='black', ls='--', lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(d['cancer'], fontsize=9)
ax.set_xlabel('Hazard ratio of DUOX2 (per log2 unit, OS)')
ax.set_title('Pan-cancer prognostic value of DUOX2 (TCGA PanCancer Atlas)')
for i, (hr, p) in enumerate(zip(d['HR'], d['Cox_p'])):
    star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    ax.text(hr + 0.06 if hr >= 1 else hr - 0.12, i, star, va='center', fontsize=10, color='black')
ax.set_xlim(0.7, 1.25)
plt.tight_layout()
plt.savefig(f'{FIG}/Fig9_pancancer_forest.png', dpi=300)
plt.savefig(f'{FIG}/Fig9_pancancer_forest.pdf')
plt.close()
print('图9 泛癌森林图 done')

# ===== 图10：泛癌表达分布（箱线图）=====
fig, ax = plt.subplots(figsize=(12, 4.5))
order = df.groupby('cancer')['DUOX2_log2'].median().sort_values().index
data = [df[df['cancer'] == c]['DUOX2_log2'].values for c in order]
bp = ax.boxplot(data, positions=range(len(order)), widths=0.6, patch_artist=True,
                medianprops=dict(color='black', lw=1.2), showfliers=False)
for i, c in enumerate(order):
    bp['boxes'][i].set_facecolor(C2 if c == 'LIHC' else '#ECF0F1')
    bp['boxes'][i].set_alpha(0.8)
ax.axhline(df[df['cancer']=='LIHC']['DUOX2_log2'].median(), color=C1, ls='--', lw=1, label='LIHC median')
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, fontsize=8)
ax.set_ylabel('DUOX2 log2(RSEM+1)')
ax.set_title('DUOX2 expression across 20 TCGA cancer types')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'{FIG}/Fig10_pancancer_expr.png', dpi=300)
plt.savefig(f'{FIG}/Fig10_pancancer_expr.pdf')
plt.close()
print('图10 泛癌表达 done')

# ===== 图11：KIRC 和 PAAD 的 DUOX2 KM =====
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, cancer in zip(axes, ['KIRC', 'PAAD']):
    g = df[df['cancer'] == cancer].copy()
    g = g[g['OS_months'].notna() & g['OS_event'].notna()]
    med = g['DUOX2_log2'].median()
    g['group'] = np.where(g['DUOX2_log2'] >= med, 'High', 'Low')
    kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
    hi = g[g['group'] == 'High']; lo = g[g['group'] == 'Low']
    kmf_h.fit(hi['OS_months'], hi['OS_event']); kmf_l.fit(lo['OS_months'], lo['OS_event'])
    ax.step(kmf_h.survival_function_.index, kmf_h.survival_function_.iloc[:, 0], where='post', color=C1, lw=2, label=f'DUOX2 High (n={len(hi)})')
    ax.step(kmf_l.survival_function_.index, kmf_l.survival_function_.iloc[:, 0], where='post', color=C2, lw=2, label=f'DUOX2 Low (n={len(lo)})')
    lr = logrank_test(hi['OS_months'], lo['OS_months'], hi['OS_event'], lo['OS_event'])
    ax.set_xlabel('Time (months)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
    ax.set_title(f'{cancer}: log-rank p={lr.p_value:.3f}')
    ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIG}/Fig11_pancancer_KM.png', dpi=300)
plt.savefig(f'{FIG}/Fig11_pancancer_KM.pdf')
plt.close()
print('图11 泛癌KM done')

print('\n泛癌图表完成')
