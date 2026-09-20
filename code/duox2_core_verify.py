# -*- coding: utf-8 -*-
"""DUOX2 核心验证（轻量，不跑 GSEA）"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

expr = pd.read_csv(f'{DATA}/expr_counts.tsv.gz', index_col=0, compression='gzip')
expr.index.name = 'sample_id'
sample_meta = pd.read_csv(f'{DATA}/sample_meta.tsv', sep='\t')
gene_map = pd.read_csv(f'{DATA}/gene_id_name_map.tsv', sep='\t')
gene_map = gene_map[gene_map['gene_id'].str.startswith('ENSG')].dropna()
id2name = dict(zip(gene_map['gene_id'], gene_map['gene_name']))
expr.columns = [id2name.get(c, c) for c in expr.columns]

cpm = expr.div(expr.sum(axis=1), axis=0) * 1e6
from scipy import stats

tumor_ids = sample_meta[sample_meta['sample_type'] == 'Primary Tumor']['sample_id']
normal_ids = sample_meta[sample_meta['sample_type'] == 'Solid Tissue Normal']['sample_id']

# 1. 肿瘤 vs 癌旁
t_cpm = cpm.loc[tumor_ids, 'DUOX2']; n_cpm = cpm.loc[normal_ids, 'DUOX2']
stat, pval = stats.mannwhitneyu(t_cpm, n_cpm)
print('=== 1. DUOX2 肿瘤 vs 癌旁 ===')
print(f'肿瘤 n={len(t_cpm)}: 中位数 {np.median(t_cpm):.1f} CPM (IQR {np.percentile(t_cpm,25):.1f}-{np.percentile(t_cpm,75):.1f})')
print(f'癌旁 n={len(n_cpm)}: 中位数 {np.median(n_cpm):.1f} CPM (IQR {np.percentile(n_cpm,25):.1f}-{np.percentile(n_cpm,75):.1f})')
print(f'Mann-Whitney p={pval:.2e}')

# 配对
paired = []
for tid in tumor_ids:
    case = tid[:12]
    nid = sample_meta[(sample_meta['case_id'] == case) & (sample_meta['sample_type'] == 'Solid Tissue Normal')]['sample_id']
    if len(nid) and nid.iloc[0] in cpm.index and tid in cpm.index:
        paired.append((tid, nid.iloc[0]))
if paired:
    diff = [np.log2(cpm.loc[t, 'DUOX2'] + 1) - np.log2(cpm.loc[n, 'DUOX2'] + 1) for t, n in paired]
    wp = stats.wilcoxon(diff)
    print(f'配对 {len(paired)} 对: log2FC 中位数 {np.median(diff):.2f}, Wilcoxon p={wp.pvalue:.2e}')

# 2. 分期关联
print('\n=== 2. DUOX2 与分期（肿瘤内）===')
meta = pd.read_csv(f'{DATA}/meta_clinical_cleaned.tsv', sep='\t')
tumor_meta = sample_meta[sample_meta['sample_type'] == 'Primary Tumor']
analy = tumor_meta.merge(meta[['case_id', 'stage_group', 'OS_time_days', 'OS_event']], on='case_id', how='inner')
analy = analy.drop_duplicates(subset='case_id', keep='first')
analy = analy[analy['stage_group'].notna()]
analy['DUOX2_cpm'] = [cpm.loc[sid, 'DUOX2'] for sid in analy['sample_id']]
for g in ['early', 'intermediate', 'advanced']:
    vals = analy[analy['stage_group'] == g]['DUOX2_cpm']
    print(f'{g}: n={len(vals)}, 中位数 {np.median(vals):.1f} (IQR {np.percentile(vals,25):.1f}-{np.percentile(vals,75):.1f})')
from scipy.stats import kruskal
groups = [analy[analy['stage_group'] == g]['DUOX2_cpm'].values for g in ['early', 'intermediate', 'advanced']]
h, p_kw = kruskal(*groups)
print(f'Kruskal-Wallis p={p_kw:.2e}')
e = analy[analy['stage_group'] == 'early']['DUOX2_cpm']
a = analy[analy['stage_group'] == 'advanced']['DUOX2_cpm']
_, p_ea = stats.mannwhitneyu(e, a)
print(f'早期 vs 晚期: p={p_ea:.2e}')

# 3. DUOX2 高低组 KM + Cox
print('\n=== 3. DUOX2 高低表达组（中位数）===')
med = analy['DUOX2_cpm'].median()
analy['DUOX2_group'] = np.where(analy['DUOX2_cpm'] >= med, 'high', 'low')
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
h = analy[analy['DUOX2_group'] == 'high']; l = analy[analy['DUOX2_group'] == 'low']
kmf_h.fit(h['OS_time_days'], h['OS_event']); kmf_l.fit(l['OS_time_days'], l['OS_event'])
print(f'high: n={len(h)}, 死亡 {h["OS_event"].sum()}, 中位生存 {kmf_h.median_survival_time_:.0f}天')
print(f'low : n={len(l)}, 死亡 {l["OS_event"].sum()}, 中位生存 {kmf_l.median_survival_time_:.0f}天')
res = logrank_test(h['OS_time_days'], l['OS_time_days'], h['OS_event'], l['OS_event'])
print(f'log-rank p={res.p_value:.2e}')
# 1/3/5年
for name, g in [('high', h), ('low', l)]:
    s = kmf_h.survival_function_ if name == 'high' else kmf_l.survival_function_
    vals = []
    for t in [365, 1095, 1825]:
        sub = s[s.index <= t]
        vals.append(f'{sub.iloc[-1].values[0]:.1%}' if len(sub) else 'NA')
    print(f'{name}: 1年 {vals[0]}, 3年 {vals[1]}, 5年 {vals[2]}')

df = analy[['OS_time_days', 'OS_event', 'DUOX2_cpm']].copy()
df['DUOX2_log2'] = np.log2(df['DUOX2_cpm'] + 1)
cph = CoxPHFitter()
cph.fit(df[['OS_time_days', 'OS_event', 'DUOX2_log2']], duration_col='OS_time_days', event_col='OS_event')
s = cph.summary.iloc[0]
print(f'Cox(连续): HR={s["exp(coef)"]:.3f} (95%CI {s["exp(coef) lower 95%"]:.3f}-{s["exp(coef) upper 95%"]:.3f}), p={s["p"]:.2e}')

# 4. DUOX2 共表达 Top 结果(前已存)
print('\n=== 4. 共表达 Top 基因 ===')
corrs = pd.read_csv(f'{DATA}/DUOX2_coexpression_all.csv', index_col=0)['pearson_r']
print('Top 10 正相关:', corrs.head(10).round(3).to_dict())
print('Top 10 负相关:', corrs.tail(10).round(3).to_dict())
print('\nDUOXA2 相关:', corrs.get('DUOXA2'))
print('TMC5 相关:', corrs.get('TMC5'))
print('KLK11 相关:', corrs.get('KLK11'))
