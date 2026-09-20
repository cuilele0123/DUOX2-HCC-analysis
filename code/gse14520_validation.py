# -*- coding: utf-8 -*-
"""GSE14520 外部验证 v4（修正 TNM + 重新拟合 + 固定系数双模式）"""
import pandas as pd
import numpy as np
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

gene_df = pd.read_csv(f'{DATA}/GSE14520_gene_expr.csv', index_col=0)
clin = pd.read_csv(f'{DATA}/GSE14520_Extra_Supplement.txt.gz', sep='\t', compression='gzip')
clin['gsm'] = clin['Affy_GSM'].astype(str)
matched = clin[clin['gsm'].isin(gene_df.index)].copy()
matched = matched[matched['Tissue Type'] == 'Tumor']
for g in ['DUOX2', 'SLC16A3', 'SPP2', 'MMP7']:
    matched[g] = matched['gsm'].map(gene_df[g])
matched['OS_months'] = pd.to_numeric(matched['Survival months'], errors='coerce')
matched['event'] = matched['Survival status'].apply(lambda x: 1 if x == 1 else (0 if x == 0 else np.nan))
matched = matched[matched['OS_months'].notna() & matched['event'].notna()].copy()
print(f'肿瘤样本(有生存): {len(matched)}')

# ===== 1. TNM 分期（修正版：先检查 III/II/I 顺序）=====
def tnm_simple(s):
    if pd.isna(s) or str(s).strip() == '.' or str(s).strip() == '': return np.nan
    s = str(s).strip().upper()
    if s.startswith('III'): return 'III'
    if s.startswith('II'): return 'II'
    if s.startswith('I'): return 'I'
    if s.startswith('IV'): return 'IV'
    return np.nan
matched['tnm_s'] = matched['TNM staging'].apply(tnm_simple)
print('\n=== 1. DUOX2 与 TNM 分期 ===')
print('TNM 分布:', matched['tnm_s'].value_counts().sort_index().to_dict())
grps_t = [matched[matched['tnm_s']==t]['DUOX2'] for t in ['I','II','III'] if len(matched[matched['tnm_s']==t])>5]
if len(grps_t) >= 2:
    h, p = stats.kruskal(*grps_t)
    print(f'KW p={p:.4f}')
    for t in ['I','II','III']:
        d = matched[matched['tnm_s']==t]['DUOX2']
        if len(d): print(f'  TNM {t}: n={len(d)}, 中位 {np.median(d):.2f} (IQR {np.percentile(d,25):.2f}-{np.percentile(d,75):.2f})')
    e = matched[matched['tnm_s']=='I']['DUOX2']; l3 = matched[matched['tnm_s']=='III']['DUOX2']
    if len(e)>5 and len(l3)>5:
        _, pea = stats.mannwhitneyu(e, l3)
        print(f'  I vs III: Mann-Whitney p={pea:.4f}')

# ===== 2. BCLC 分期 =====
print('\n=== 2. DUOX2 与 BCLC 分期 ===')
matched['bclc'] = matched['BCLC staging'].astype(str)
print('BCLC 分布:', matched['bclc'].value_counts().to_dict())
grps_b = [matched[matched['bclc']==b]['DUOX2'] for b in ['0','A','B','C'] if len(matched[matched['bclc']==b])>5]
if len(grps_b) >= 2:
    h, p = stats.kruskal(*grps_b)
    print(f'KW p={p:.4f}')

# ===== 3. DUOX2 高低组 KM =====
print('\n=== 3. DUOX2 高低表达组 KM ===')
med = matched['DUOX2'].median()
matched['duox2_g'] = np.where(matched['DUOX2']>=med, 'high', 'low')
kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
hi = matched[matched['duox2_g']=='high']; lo = matched[matched['duox2_g']=='low']
kmf_h.fit(hi['OS_months'], hi['event']); kmf_l.fit(lo['OS_months'], lo['event'])
print(f'high: n={len(hi)}, 死亡 {hi["event"].sum()}, 中位 {kmf_h.median_survival_time_:.1f}月')
print(f'low : n={len(lo)}, 死亡 {lo["event"].sum()}, 中位 {kmf_l.median_survival_time_:.1f}月')
lr = logrank_test(hi['OS_months'], lo['OS_months'], hi['event'], lo['event'])
print(f'log-rank p={lr.p_value:.4f}')

# ===== 4A. LASSO 风险评分（固定 TCGA 系数）=====
print('\n=== 4A. 固定 TCGA 系数应用到 GSE14520 ===')
coef = {'SLC16A3': 0.092, 'SPP2': -0.038, 'MMP7': 0.024}
matched['risk_fixed'] = sum(coef[g] * matched[g] for g in coef)
med_r = matched['risk_fixed'].median()
matched['risk_g_f'] = np.where(matched['risk_fixed']>=med_r, 'high', 'low')
hi_r = matched[matched['risk_g_f']=='high']; lo_r = matched[matched['risk_g_f']=='low']
kmf_h.fit(hi_r['OS_months'], hi_r['event']); kmf_l.fit(lo_r['OS_months'], lo_r['event'])
print(f'high: n={len(hi_r)}, 死亡 {hi_r["event"].sum()}, 中位 {kmf_h.median_survival_time_:.1f}月')
print(f'low : n={len(lo_r)}, 死亡 {lo_r["event"].sum()}, 中位 {kmf_l.median_survival_time_:.1f}月')
lr_r = logrank_test(hi_r['OS_months'], lo_r['OS_months'], hi_r['event'], lo_r['event'])
print(f'固定系数 log-rank p={lr_r.p_value:.4f}')

# ===== 4B. GSE14520 重新拟合（检验方法可复现性）=====
print('\n=== 4B. GSE14520 内重新拟合 LASSO-Cox ===')
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.model_selection import KFold

X = matched[['DUOX2','SLC16A3','SPP2','MMP7']].values.astype(float)
y = Surv.from_arrays(matched['event'].astype(bool), matched['OS_months'])

# 5折CV重新拟合
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cindices = []
for tr, te in kf.split(X):
    m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.05, fit_baseline_model=True, max_iter=2000)
    m.fit(X[tr], y[tr])
    risk = m.predict(X[te])
    c = concordance_index_censored(y[te]['event'].astype(bool), y[te]['time'], risk)[0]
    cindices.append(c)
print(f'GSE14520 重新拟合 5折CV C-index: {np.mean(cindices):.3f} ± {np.std(cindices):.3f}')

# 全数据拟合看系数
m_full = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.05, fit_baseline_model=True, max_iter=2000)
m_full.fit(X, y)
coef_full = m_full.coef_.ravel()
for g, c in zip(['DUOX2','SLC16A3','SPP2','MMP7'], coef_full):
    print(f'  {g}: {c:.4f}')
c_full = concordance_index_censored(y['event'].astype(bool), y['time'], m_full.predict(X))[0]
print(f'全数据 C-index: {c_full:.3f}')

# ===== 5. 时间依赖 AUC（重新拟合模型）=====
print('\n=== 5. 时间依赖 AUC（重新拟合）===')
from sksurv.metrics import cumulative_dynamic_auc
risk_full = m_full.predict(X)
for t in [12, 36, 60]:
    try:
        auc, _ = cumulative_dynamic_auc(y, y, risk_full, times=[t])
        print(f'{t}月({t//12}年): AUC={auc[0]:.3f}')
    except Exception as e:
        print(f'{t}月 err: {e}')

matched.to_csv(f'{DATA}/GSE14520_validation_final.tsv', sep='\t', index=False)
print('\n保存完成')
