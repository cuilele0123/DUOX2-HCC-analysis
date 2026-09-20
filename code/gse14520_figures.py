# -*- coding: utf-8 -*-
"""生成 GSE14520 外部验证图表"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 字体：统一 clear sans-serif（Scientific Reports 图片规范：Helvetica / Arial）
from matplotlib import font_manager as fm
for _p in ['/System/Library/Fonts/Helvetica.ttc',
           '/System/Library/Fonts/Supplemental/Arial.ttf',
           '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
           '/System/Library/Fonts/Supplemental/Arial Italic.ttf',
           '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf']:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Helvetica'
rcParams['mathtext.it'] = 'Helvetica:italic'
rcParams['mathtext.bf'] = 'Helvetica:bold'
rcParams['mathtext.default'] = 'regular'
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 6.6
rcParams['axes.titlesize'] = 7.0
rcParams['axes.labelsize'] = 7.0
rcParams['xtick.labelsize'] = 6.3
rcParams['ytick.labelsize'] = 6.3
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'
FIG = f'{BASE}/figures'
os.makedirs(FIG, exist_ok=True)
C1, C2, C3 = '#C0392B', '#2980B9', '#27AE60'

df = pd.read_csv(f'{DATA}/GSE14520_validation_final.tsv', sep='\t')
print(f'n={len(df)}')

# ===== 图14：DUOX2 与 TNM 分期箱线图 =====
from scipy import stats
fig, axes = plt.subplots(1, 2, figsize=(7.22, 3.25))
ax = axes[0]
data_by_t = [df[df['tnm_s']==t]['DUOX2'].values for t in ['I','II','III']]
bp = ax.boxplot(data_by_t, positions=[1,2,3], widths=0.5, patch_artist=True, medianprops=dict(color='black', lw=0.79))
for i, box in enumerate(bp['boxes']):
    box.set_facecolor([C3, '#F39C12', C1][i])
for i, d in enumerate(data_by_t):
    ax.scatter(np.random.normal(i+1, 0.06, len(d)), d, s=1.74, alpha=0.4)
ax.set_xticks([1,2,3]); ax.set_xticklabels(['TNM I (n=93)','TNM II (n=77)','TNM III (n=49)'], fontsize=5.3)
ax.set_ylabel('DUOX2 expression (RMA log2)')
ax.set_title('A. GSE14520: DUOX2 by TNM stage\nKW p=0.45 (NS)')
ax = axes[1]
# 固定系数风险评分 KM
coef = {'SLC16A3': 0.092, 'SPP2': -0.038, 'MMP7': 0.024}
df['risk_fixed'] = sum(coef[g]*df[g] for g in coef)
med_r = df['risk_fixed'].median()
df['grp_f'] = np.where(df['risk_fixed']>=med_r, 'high', 'low')
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
hi = df[df['grp_f']=='high']; lo = df[df['grp_f']=='low']
kmf_h.fit(hi['OS_months'], hi['event']); kmf_l.fit(lo['OS_months'], lo['event'])
ax.step(kmf_h.survival_function_.index, kmf_h.survival_function_.iloc[:,0], where='post', color=C1, lw=1.32, label=f'High risk (n={len(hi)})')
ax.step(kmf_l.survival_function_.index, kmf_l.survival_function_.iloc[:,0], where='post', color=C2, lw=1.32, label=f'Low risk (n={len(lo)})')
lr = logrank_test(hi['OS_months'], lo['OS_months'], hi['event'], lo['event'])
ax.set_xlabel('Time (months)'); ax.set_ylabel('OS probability'); ax.set_ylim(0,1.05)
ax.set_title(f'B. GSE14520: LASSO risk score (fixed TCGA coef)\nlog-rank p={lr.p_value:.4f}')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'{FIG}/Fig14_GSE14520_validation.png', dpi=300)
plt.savefig(f'{FIG}/Fig14_GSE14520_validation.pdf')
plt.close()
print('图14 done')

# ===== 图15：三队列验证汇总图 =====
fig, ax = plt.subplots(figsize=(3.60, 2.16))
cohorts = ['TCGA-LIHC\n(5-fold CV)', 'TCGA-LIHC\n(70:30 split)', 'Firehose Legacy\n(cross-processing)', 'GSE14520\n(re-fit, cross-platform)']
cindex = [0.690, 0.602, 0.642, 0.634]
colors = [C1, '#E67E22', C3, '#8E44AD']
bars = ax.bar(range(len(cohorts)), cindex, 0.55, color=colors, alpha=0.85)
ax.axhline(0.7, color='grey', ls='--', lw=0.66)
# 纵轴为 C-index（生存数据下与时间依赖 AUC 等价），标注统一用 C-index 措辞
ax.text(3.42, 0.706, 'C-index = 0.7', fontsize=5.3, color='grey',
        ha='right', va='bottom')
for i, v in enumerate(cindex):
    # 白色底衬：避免 0.7 虚线穿过数值标签（如 0.690）
    ax.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=6.6, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.2))
ax.set_xticks(range(len(cohorts))); ax.set_xticklabels(cohorts, fontsize=5.9)
ax.set_ylabel('C-index')
ax.set_ylim(0.4, 0.85)
ax.set_title('LASSO-Cox model performance across cohorts')
plt.tight_layout()
plt.savefig(f'{FIG}/Fig15_cohort_summary.png', dpi=300)
plt.savefig(f'{FIG}/Fig15_cohort_summary.pdf')
plt.close()
print('图15 done')

print('\n完成')
