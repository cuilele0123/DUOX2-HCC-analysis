# -*- coding: utf-8 -*-
"""生成外部验证图表（Firehose Legacy vs TCGA PanCancer Atlas）"""
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

data2 = pd.read_csv(f'{DATA}/firehose_validation_combined.tsv', sep='\t')
data2['event'] = data2['OS_status'].apply(lambda x: 1 if 'DECEASED' in str(x).upper() else 0)
for g in ['DUOX2','SLC16A3','SPP2','MMP7']:
    data2[g+'_log2'] = np.log2(data2[g] + 1)

# ===== 图12：LASSO 风险评分 KM（外部验证）=====
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
coef = {'SLC16A3_log2': 0.092, 'SPP2_log2': -0.038, 'MMP7_log2': 0.024}
df = data2[['OS_months','event','SLC16A3_log2','SPP2_log2','MMP7_log2']].dropna().copy()
df['risk'] = sum(coef[g] * df[g] for g in coef)
med = df['risk'].median()
df['risk_group'] = np.where(df['risk']>=med, 'high', 'low')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
# A: 外部验证 Firehose
ax = axes[0]
hi = df[df['risk_group']=='high']; lo = df[df['risk_group']=='low']
kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
kmf_h.fit(hi['OS_months'], hi['event']); kmf_l.fit(lo['OS_months'], lo['event'])
ax.step(kmf_h.survival_function_.index, kmf_h.survival_function_.iloc[:, 0], where='post', color=C1, lw=2, label=f'High (n={len(hi)})')
ax.step(kmf_l.survival_function_.index, kmf_l.survival_function_.iloc[:, 0], where='post', color=C2, lw=2, label=f'Low (n={len(lo)})')
lr = logrank_test(hi['OS_months'], lo['OS_months'], hi['event'], lo['event'])
ax.set_xlabel('Time (months)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
ax.set_title(f'A. External validation (Firehose Legacy, n={len(df)})\nlog-rank p={lr.p_value:.2e}')
ax.legend(frameon=False)

# B: 训练集（TCGA pan_can_atlas）
from sksurv.metrics import cumulative_dynamic_auc
lasso_risk = pd.read_csv(f'{DATA}/lasso_risk_analysis.csv')
ax = axes[1]
hi2 = lasso_risk[lasso_risk['risk_group']=='high']; lo2 = lasso_risk[lasso_risk['risk_group']=='low']
kmf_h2 = KaplanMeierFitter(); kmf_l2 = KaplanMeierFitter()
kmf_h2.fit(hi2['OS_time_days']/30.44, hi2['OS_event']); kmf_l2.fit(lo2['OS_time_days']/30.44, lo2['OS_event'])
ax.step(kmf_h2.survival_function_.index, kmf_h2.survival_function_.iloc[:, 0], where='post', color=C1, lw=2, label=f'High (n={len(hi2)})')
ax.step(kmf_l2.survival_function_.index, kmf_l2.survival_function_.iloc[:, 0], where='post', color=C2, lw=2, label=f'Low (n={len(lo2)})')
lr2 = logrank_test(hi2['OS_time_days']/30.44, lo2['OS_time_days']/30.44, hi2['OS_event'], lo2['OS_event'])
ax.set_xlabel('Time (months)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
ax.set_title(f'B. Training cohort (TCGA PanCancer Atlas, n={len(lasso_risk)})\nlog-rank p={lr2.p_value:.2e}')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'{FIG}/Fig12_external_validation_KM.png', dpi=300)
plt.savefig(f'{FIG}/Fig12_external_validation_KM.pdf')
plt.close()
print('图12 外部验证KM done')

# ===== 图13：跨数据集 Cox 系数对比（训练 vs 外部验证）=====
fig, ax = plt.subplots(figsize=(7, 4.8))
from lifelines import CoxPHFitter
cph_train = CoxPHFitter()
cph_train.fit(data2[['OS_months','event','DUOX2_log2','SLC16A3_log2','SPP2_log2','MMP7_log2']], duration_col='OS_months', event_col='event')
train = cph_train.summary

# 训练集（从已有 LASSO 模型）
# 直接用之前 LASSO 的 coef: SLC16A3=0.092, SPP2=-0.038, MMP7=0.024
ext_coef = {'SLC16A3_log2': train.loc['SLC16A3_log2', 'coef'], 'SPP2_log2': train.loc['SPP2_log2', 'coef'], 'MMP7_log2': train.loc['MMP7_log2', 'coef']}
train_coef = {'SLC16A3_log2': 0.092, 'SPP2_log2': -0.038, 'MMP7_log2': 0.024}
genes = ['SLC16A3_log2','SPP2_log2','MMP7_log2']
labels = ['SLC16A3\n(positive risk)', 'SPP2\n(protective)', 'MMP7\n(positive risk)']
x = np.arange(len(genes))
w = 0.35
train_vals = [train_coef[g] for g in genes]
ext_vals = [ext_coef[g] for g in genes]
bars1 = ax.bar(x - w/2, train_vals, w, color=C2, alpha=0.8, label='TCGA PanCancer Atlas (training)')
bars2 = ax.bar(x + w/2, ext_vals, w, color=C3, alpha=0.8, label='Firehose Legacy (external)')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Cox coefficient')
ax.set_title('LASSO-Cox coefficients:\ntraining vs external validation (consistent direction)')
ax.legend(frameon=False)
for i, (t, e) in enumerate(zip(train_vals, ext_vals)):
    ax.text(i - w/2, t + (0.005 if t>=0 else -0.008), f'{t:.3f}', ha='center', fontsize=8)
    ax.text(i + w/2, e + (0.005 if e>=0 else -0.008), f'{e:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIG}/Fig13_coef_comparison.png', dpi=300)
plt.savefig(f'{FIG}/Fig13_coef_comparison.pdf')
plt.close()
print('图13 系数对比 done')

print('\n外部验证图表生成完成')
