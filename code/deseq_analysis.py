# -*- coding: utf-8 -*-
"""差异表达分析：早期 vs 晚期 HCC（DESeq2/pydeseq2，raw counts）
修正原文问题：
- 用 raw counts 而非 FPKM（P0-7）
- 用修正后的 AJCC 分期分组（P0-1）
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

# 1. 读取数据
expr = pd.read_csv(f'{DATA}/expr_counts.tsv.gz', index_col=0, compression='gzip')
expr.index.name = 'sample_id'
sample_meta = pd.read_csv(f'{DATA}/sample_meta.tsv', sep='\t')
meta = pd.read_csv(f'{DATA}/meta_clinical_cleaned.tsv', sep='\t')

# 2. 构建分析样本集：原发肿瘤 + 有分期分组 + 有生存数据
tumor = sample_meta[sample_meta['sample_type'] == 'Primary Tumor']
analy = tumor.merge(meta[['case_id', 'stage_group', 'stage_derived', 'OS_time_days', 'OS_event']],
                    on='case_id', how='inner')
analy = analy[analy['stage_group'].notna()]

print(f'分析样本: {len(analy)}（早期 {sum(analy.stage_group=="early")}，'
      f'中期 {sum(analy.stage_group=="intermediate")}，晚期 {sum(analy.stage_group=="advanced")}）')

# 3. 提取表达子集（早期 vs 晚期）
subset = analy[analy['stage_group'].isin(['early', 'advanced'])]
X = expr.loc[subset['sample_id']].copy()

# 过滤：至少 10 个样本中 counts>0
keep_genes = (X > 0).sum(axis=0) >= 10
X = X.loc[:, keep_genes]
print(f'过滤后基因数: {X.shape[1]}')

# 4. 跑 DESeq2 (pydeseq2)
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

conditions = subset.set_index('sample_id')['stage_group'].astype(str)
counts = X  # 样本 × 基因

dds = DeseqDataSet(
    counts=counts,
    metadata=conditions.to_frame('condition'),
    design_factors='condition',
    ref_level=['condition', 'advanced'],   # 晚期为参考，早期 vs 晚期
    n_cpus=8,
)

print('拟合 DESeq2...')
dds.deseq2()

stats = DeseqStats(dds, contrast=['condition', 'early', 'advanced'], alpha=0.05, n_cpus=8)
stats.summary()
res = stats.results_df
res = res.dropna(subset=['padj'])
res = res[~res.index.str.startswith('N_')]

# 5. DEG 筛选 |log2FC|>1 & padj<0.05
res['DEG'] = (res['log2FoldChange'].abs() > 1) & (res['padj'] < 0.05)
deg = res[res['DEG']].copy()
up = deg[deg['log2FoldChange'] > 0]
down = deg[deg['log2FoldChange'] < 0]

print(f'\n=== DEG 结果（早期 vs 晚期，参考=晚期）===')
print(f'log2FC 方向: 正=早期上调（晚期下调）')
print(f'总 DEG: {len(deg)} | 上调(early>adv): {len(up)} | 下调(early<adv): {len(down)}')

# 保存
res.to_csv(f'{DATA}/deseq2_early_vs_advanced_all.csv')
deg.to_csv(f'{DATA}/deseq2_early_vs_advanced_DEG.csv')
print(f'\nDEG 保存: {DATA}/deseq2_early_vs_advanced_DEG.csv')
print('\nTop 20 上调:')
print(deg.sort_values('log2FoldChange', ascending=False).head(20)[['log2FoldChange', 'padj']].to_string())
