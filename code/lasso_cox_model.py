# -*- coding: utf-8 -*-
"""LASSO-Cox 预后模型（简化可靠版）
- CoxnetSurvivalAnalysis + 5折CV选择alpha
- 输出：模型基因、风险评分、时间依赖AUC、高低危组KM
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

# 数据
expr = pd.read_csv(f'{DATA}/expr_counts.tsv.gz', index_col=0, compression='gzip')
sample_meta = pd.read_csv(f'{DATA}/sample_meta.tsv', sep='\t')
gene_map = pd.read_csv(f'{DATA}/gene_id_name_map.tsv', sep='\t')
gene_map = gene_map[gene_map['gene_id'].str.startswith('ENSG')].dropna()
id2name = dict(zip(gene_map['gene_id'], gene_map['gene_name']))
expr.columns = [id2name.get(c, c) for c in expr.columns]
meta = pd.read_csv(f'{DATA}/meta_clinical_cleaned.tsv', sep='\t')

tumor_meta = sample_meta[sample_meta['sample_type'] == 'Primary Tumor']
analy = tumor_meta.merge(meta[['case_id', 'stage_group', 'OS_time_days', 'OS_event']], on='case_id', how='inner')
analy = analy.drop_duplicates(subset='case_id', keep='first')
analy = analy[analy['stage_group'].notna() & analy['OS_time_days'].notna() & analy['OS_event'].notna()]
analy = analy.reset_index(drop=True)
print(f'模型样本: {len(analy)}')

cpm = expr.div(expr.sum(axis=1), axis=0) * 1e6
logcpm = np.log2(cpm + 1)

deg = pd.read_csv(f'{DATA}/deseq2_early_vs_advanced_all.csv', index_col=0)
deg['gene_name'] = deg.index.map(id2name)
deg_sig = deg[deg['DEG']].sort_values('padj').head(500)
genes = [g for g in deg_sig['gene_name'].dropna().unique() if g in logcpm.columns]
print(f'候选基因数: {len(genes)}')

X = logcpm.loc[analy['sample_id'], genes].values.astype(float)
y = Surv.from_arrays(analy['OS_event'].astype(bool), analy['OS_time_days'])

# 5折CV：每折选最优alpha
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_cindices = []
best_alphas = []
for fold, (tr, te) in enumerate(kf.split(X)):
    c = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, fit_baseline_model=True, max_iter=2000)
    c.fit(X[tr], y[tr])
    alphas = c.alphas_
    c_indices = []
    for i, a in enumerate(alphas):
        # 用predict在该alpha下的线性预测值
        try:
            c_i = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[a], fit_baseline_model=True, max_iter=2000)
            c_i.fit(X[tr], y[tr])
            risk = c_i.predict(X[te])
            cidx = concordance_index_censored(y[te]['event'].astype(bool), y[te]['time'], risk)[0]
            c_indices.append(cidx)
        except Exception:
            c_indices.append(0.5)
    c_indices = np.array(c_indices)
    best = np.argmax(c_indices)
    best_alphas.append(alphas[best])
    fold_cindices.append(c_indices[best])
    print(f'fold {fold}: 最优alpha={alphas[best]:.4f}, C-index={c_indices[best]:.3f}')

print(f'\n平均最优 C-index: {np.mean(fold_cindices):.3f} (±{np.std(fold_cindices):.3f})')
best_alpha = float(np.median(best_alphas))
print(f'最终 alpha={best_alpha:.4f}')

# 全数据拟合
model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[best_alpha], fit_baseline_model=True, max_iter=2000)
model.fit(X, y)
coef = model.coef_.ravel()
sel = [(g, c) for g, c in zip(genes, coef) if abs(c) > 1e-6]
sel = sorted(sel, key=lambda x: -abs(x[1]))
print(f'\n模型纳入基因: {len(sel)}')
for g, c in sel:
    print(f'  {g}: {c:.4f}')

# 风险评分 + 分组
risk_score = X @ coef
analy['risk_score'] = risk_score
med = np.median(risk_score)
analy['risk_group'] = np.where(risk_score >= med, 'high', 'low')

kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
h = analy[analy['risk_group'] == 'high']; l = analy[analy['risk_group'] == 'low']
kmf_h.fit(h['OS_time_days'], h['OS_event']); kmf_l.fit(l['OS_time_days'], l['OS_event'])
print(f'\nhigh: n={len(h)}, 死亡 {h["OS_event"].sum()}, 中位生存 {kmf_h.median_survival_time_:.0f}天')
print(f'low : n={len(l)}, 死亡 {l["OS_event"].sum()}, 中位生存 {kmf_l.median_survival_time_:.0f}天')
res = logrank_test(h['OS_time_days'], l['OS_time_days'], h['OS_event'], l['OS_event'])
print(f'log-rank p={res.p_value:.2e}')

# 时间依赖AUC
print('\n时间依赖 AUC:')
for t in [365, 1095, 1825]:
    try:
        auc, _ = cumulative_dynamic_auc(y, y, risk_score, times=[t])
        print(f'  {t}天({t//365}年): AUC={auc[0]:.3f}')
    except Exception as e:
        print(f'  {t}天 err: {e}')

pd.DataFrame(sel, columns=['gene', 'coef']).to_csv(f'{DATA}/lasso_cox_model_genes.csv', index=False)
analy.to_csv(f'{DATA}/lasso_risk_analysis.csv', index=False)
print('\n保存完成')
