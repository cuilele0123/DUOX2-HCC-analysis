# -*- coding: utf-8 -*-
"""DUOX2 深度分析（路径A核心）：
1. DUOX2 在肿瘤 vs 癌旁表达
2. DUOX2 与分期/临床关联
3. DUOX2 高低表达组 KM
4. DUOX2 共表达网络（Top 基因）
5. DUOX2 相关 GSEA
6. ssGSEA 免疫浸润
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import gseapy as gp

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

# 读取数据
expr = pd.read_csv(f'{DATA}/expr_counts.tsv.gz', index_col=0, compression='gzip')
expr.index.name = 'sample_id'
sample_meta = pd.read_csv(f'{DATA}/sample_meta.tsv', sep='\t')
gene_map = pd.read_csv(f'{DATA}/gene_id_name_map.tsv', sep='\t')
gene_map = gene_map[gene_map['gene_id'].str.startswith('ENSG')].dropna()
id2name = dict(zip(gene_map['gene_id'], gene_map['gene_name']))
expr.columns = [id2name.get(c, c) for c in expr.columns]  # 转基因名

DUOX2_ID = 'ENSG00000140279.13'
print('DUOX2 在矩阵中:', 'DUOX2' in expr.columns)

# ===== 1. 肿瘤 vs 癌旁 =====
print('\n=== 1. DUOX2 肿瘤 vs 癌旁 ===')
tumor_ids = sample_meta[sample_meta['sample_type'] == 'Primary Tumor']['sample_id']
normal_ids = sample_meta[sample_meta['sample_type'] == 'Solid Tissue Normal']['sample_id']
print(f'肿瘤 {len(tumor_ids)}, 癌旁 {len(normal_ids)}')

from scipy import stats
t_expr = expr.loc[tumor_ids, 'DUOX2']
n_expr = expr.loc[normal_ids, 'DUOX2']
print(f'肿瘤: 中位数 {np.median(t_expr):.0f} (IQR {np.percentile(t_expr,25):.0f}-{np.percentile(t_expr,75):.0f})')
print(f'癌旁: 中位数 {np.median(n_expr):.0f} (IQR {np.percentile(n_expr,25):.0f}-{np.percentile(n_expr,75):.0f})')
# 用 counts 直接比不客观，用 CPM 归一化
cpm = expr.div(expr.sum(axis=1), axis=0) * 1e6
t_cpm = cpm.loc[tumor_ids, 'DUOX2']; n_cpm = cpm.loc[normal_ids, 'DUOX2']
stat, pval = stats.mannwhitneyu(t_cpm, n_cpm, alternative='two-sided')
print(f'CPM: 肿瘤 {np.median(t_cpm):.1f} vs 癌旁 {np.median(n_cpm):.1f}, Mann-Whitney p={pval:.2e}')

# 配对比较
tumor_meta = sample_meta[sample_meta['sample_type'] == 'Primary Tumor']
paired = []
for tid in tumor_meta['sample_id']:
    case = tid[:12]
    nid = sample_meta[(sample_meta['case_id'] == case) & (sample_meta['sample_type'] == 'Solid Tissue Normal')]['sample_id']
    if len(nid) and nid.iloc[0] in cpm.index and tid in cpm.index:
        paired.append((tid, nid.iloc[0]))
if paired:
    diff = [np.log2(cpm.loc[t, 'DUOX2'] + 1) - np.log2(cpm.loc[n, 'DUOX2'] + 1) for t, n in paired]
    wp = stats.wilcoxon(diff)
    print(f'配对样本 {len(paired)} 对, log2FC(肿瘤/癌旁) 中位数 {np.median(diff):.2f}, Wilcoxon p={wp.pvalue:.2e}')

# ===== 2. 肿瘤内部分期关联 =====
print('\n=== 2. DUOX2 与分期 ===')
meta = pd.read_csv(f'{DATA}/meta_clinical_cleaned.tsv', sep='\t')
analy = tumor_meta.merge(meta[['case_id', 'stage_group', 'stage_derived', 'OS_time_days', 'OS_event']],
                         on='case_id', how='inner')
analy = analy.drop_duplicates(subset='case_id', keep='first')
analy = analy[analy['stage_group'].notna()]
analy['DUOX2_cpm'] = [cpm.loc[sid, 'DUOX2'] for sid in analy['sample_id']]

for g in ['early', 'intermediate', 'advanced']:
    vals = analy[analy['stage_group'] == g]['DUOX2_cpm']
    print(f'{g}: n={len(vals)}, 中位数 {np.median(vals):.1f} (IQR {np.percentile(vals,25):.1f}-{np.percentile(vals,75):.1f})')
# 趋势检验 (Jonckheere)
from scipy.stats import kruskal
groups = [analy[analy['stage_group'] == g]['DUOX2_cpm'].values for g in ['early', 'intermediate', 'advanced']]
h, p_kw = kruskal(*groups)
print(f'Kruskal-Wallis p={p_kw:.2e}')
# 早期 vs 晚期
e = analy[analy['stage_group'] == 'early']['DUOX2_cpm']
a = analy[analy['stage_group'] == 'advanced']['DUOX2_cpm']
_, p_ea = stats.mannwhitneyu(e, a)
print(f'早期 vs 晚期: Mann-Whitney p={p_ea:.2e}')

# ===== 3. DUOX2 高低表达组 KM =====
print('\n=== 3. DUOX2 高低表达组 KM（中位数分组）===')
med = analy['DUOX2_cpm'].median()
analy['DUOX2_group'] = np.where(analy['DUOX2_cpm'] >= med, 'high', 'low')
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
h = analy[analy['DUOX2_group'] == 'high']; l = analy[analy['DUOX2_group'] == 'low']
kmf_h.fit(h['OS_time_days'], h['OS_event']); kmf_l.fit(l['OS_time_days'], l['OS_event'])
print(f'high: n={len(h)}, 死亡 {h["OS_event"].sum()}, 中位生存 {kmf_h.median_survival_time_:.0f}天')
print(f'low : n={len(l)}, 死亡 {l["OS_event"].sum()}, 中位生存 {kmf_l.median_survival_time_:.0f}天')
res = logrank_test(h['OS_time_days'], l['OS_time_days'], h['OS_event'], l['OS_event'])
print(f'log-rank p={res.p_value:.2e}')

# 单因素 Cox (DUOX2 连续/分组)
from lifelines import CoxPHFitter
df = analy[['OS_time_days', 'OS_event', 'DUOX2_cpm', 'DUOX2_group']].copy()
df['DUOX2_log2'] = np.log2(df['DUOX2_cpm'] + 1)
cph = CoxPHFitter()
cph.fit(df[['OS_time_days', 'OS_event', 'DUOX2_log2']], duration_col='OS_time_days', event_col='OS_event')
print('Cox(连续 log2DUOX2): HR=', cph.summary.iloc[0]['exp(coef)'], 'p=', cph.summary.iloc[0]['p'])

# ===== 4. DUOX2 共表达 =====
print('\n=== 4. DUOX2 共表达（肿瘤样本内，Pearson）===')
tumor_cpm = cpm.loc[tumor_ids].copy()
tumor_cpm_log2 = np.log2(tumor_cpm + 1)
duox2_vec = tumor_cpm_log2['DUOX2']
corrs = tumor_cpm_log2.corrwith(duox2_vec)
corrs = corrs.drop('DUOX2').sort_values(ascending=False)
print('Top 20 正相关:')
print(corrs.head(20).round(3).to_string())
print('Top 10 负相关:')
print(corrs.tail(10).round(3).to_string())
corrs.to_csv(f'{DATA}/DUOX2_coexpression_all.csv', header=['pearson_r'])
print(f'共表达基因数: {len(corrs)}')

# ===== 5. DUOX2 高低组 GSEA（用共表达排序做 prerank）=====
print('\n=== 5. GSEA（按 DUOX2 相关性排序）===')
rnk = pd.DataFrame({'gene': corrs.index, 'score': corrs.values})
rnk = rnk.dropna()
try:
    pre_res = gp.prerank(rnk=rnk,
                         gene_sets='KEGG_2021_Human',
                         outdir=None,
                         min_size=5, max_size=1000,
                         permutation_num=1000, seed=42)
    pre_res.res2d.to_csv(f'{DATA}/GSEA_DUOX2_KEGG.csv', index=False)
    # 只看 FDR<0.25
    sig = pre_res.res2d[pre_res.res2d['FDR q-val'] < 0.25].sort_values('NES', ascending=False)
    print(f'显著通路 (FDR<0.25): {len(sig)}')
    print(sig[['Term', 'NES', 'pval', 'FDR q-val']].head(15).to_string())
except Exception as e:
    print('GSEA KEGG err:', e)

try:
    pre_res2 = gp.prerank(rnk=rnk,
                          gene_sets='GO_Biological_Process_2023',
                          outdir=None,
                          min_size=5, max_size=1000,
                          permutation_num=1000, seed=42)
    pre_res2.res2d.to_csv(f'{DATA}/GSEA_DUOX2_GO.csv', index=False)
    sig2 = pre_res2.res2d[pre_res2.res2d['FDR q-val'] < 0.25].sort_values('NES', ascending=False)
    print(f'GO 显著 (FDR<0.25): {len(sig2)}')
    print(sig2[['Term', 'NES', 'pval', 'FDR q-val']].head(10).to_string())
except Exception as e:
    print('GSEA GO err:', e)

# ===== 6. ssGSEA 免疫浸润 =====
print('\n=== 6. ssGSEA 免疫浸润 ===')
# 下载免疫细胞基因集（LM22 风格 - 使用 immunologic signature）
try:
    ss = gp.ssgsea(data=tumor_cpm_log2.T,  # 基因 × 样本
                   gene_sets='KEGG_2021_Human',
                   outdir=None, sample_norm_method='rank', seed=42)
    print('ssGSEA 输出 shape:', ss.res2d.shape)
except Exception as e:
    print('ssGSEA err:', e)

# 简化免疫浸润：使用代表性免疫标记基因
immune_markers = {
    'CD8_T': ['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1', 'IFNG'],
    'CD4_T': ['CD4', 'IL7R', 'CCR7', 'LEF1'],
    'Treg': ['FOXP3', 'CTLA4', 'IL2RA', 'IKZF2'],
    'NK': ['NKG7', 'KLRD1', 'KLRK1', 'NCR1', 'GNLY'],
    'Macrophage_M1': ['NOS2', 'IL1B', 'TNF', 'CD80', 'CD86'],
    'Macrophage_M2': ['CD163', 'MRC1', 'MSR1', 'TGFB1', 'IL10', 'ARG1'],
    'DC': ['ITGAX', 'CD80', 'CD83', 'CCR7'],
    'B_cell': ['CD19', 'MS4A1', 'CD79A', 'CD79B'],
    'T_exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'PDCD1LG2'],
}
marker_avail = {k: [g for g in v if g in tumor_cpm_log2.columns] for k, v in immune_markers.items()}
print('可用标记基因:', {k: len(v) for k, v in marker_avail.items()})
ss_scores = pd.DataFrame(index=tumor_ids)
for k, genes in marker_avail.items():
    if len(genes) >= 3:
        ss_scores[k] = tumor_cpm_log2[genes].mean(axis=1)
# DUOX2 与免疫评分的相关
duox2_all = tumor_cpm_log2['DUOX2']
corr_out = []
for k in ss_scores.columns:
    r, p = stats.spearmanr(duox2_all.loc[ss_scores.index], ss_scores[k])
    corr_out.append((k, r, p))
    print(f'DUOX2 vs {k}: rho={r:.3f}, p={p:.2e}')
pd.DataFrame(corr_out, columns=['immune_cell', 'rho', 'p']).to_csv(f'{DATA}/DUOX2_immune_corr.csv', index=False)

# 免疫检查点
checkpoints = ['PDCD1', 'CD274', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'PDCD1LG2', 'SIGLEC15', 'IDO1', 'LGALS9']
print('\n免疫检查点与 DUOX2 相关:')
cp_out = []
for g in checkpoints:
    if g in tumor_cpm_log2.columns:
        r, p = stats.spearmanr(duox2_all, tumor_cpm_log2[g])
        cp_out.append((g, r, p))
        print(f'  {g}: rho={r:.3f}, p={p:.2e}')
pd.DataFrame(cp_out, columns=['checkpoint', 'rho', 'p']).to_csv(f'{DATA}/DUOX2_checkpoint_corr.csv', index=False)

print('\nDUOX2 深度分析完成')
