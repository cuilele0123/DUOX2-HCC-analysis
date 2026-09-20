# -*- coding: utf-8 -*-
"""GO/KEGG 富集分析（gseapy）+ DUOX2 高低表达组 GSEA"""
import pandas as pd
import numpy as np
import gseapy as gp
import warnings
warnings.filterwarnings('ignore')

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

# 1. 读取 DEG 结果
res = pd.read_csv(f'{DATA}/deseq2_early_vs_advanced_all.csv', index_col=0)
gene_map = pd.read_csv(f'{DATA}/gene_id_name_map.tsv', sep='\t')
gene_map = gene_map[gene_map['gene_id'].str.startswith('ENSG')].dropna()
id2name = dict(zip(gene_map['gene_id'], gene_map['gene_name']))
res['gene_name'] = res.index.map(id2name)

# DEG: 晚期上调 (log2FC<0, 参考=advanced) 与 早期上调 (log2FC>0)
deg = res[res['DEG']]
adv_up = deg[deg['log2FoldChange'] < 0]['gene_name'].dropna().tolist()   # 晚期上调
ear_up = deg[deg['log2FoldChange'] > 0]['gene_name'].dropna().tolist()   # 早期上调
print(f'晚期上调: {len(adv_up)} 个基因 | 早期上调: {len(ear_up)} 个基因')

# 2. GO 富集（晚期上调基因）
print('\n=== GO 富集（晚期上调基因）===')
go_adv = gp.enrichr(gene_list=adv_up,
                    gene_sets='GO_Biological_Process_2023',
                    organism='human', outdir=None)
go_adv.res2d.to_csv(f'{DATA}/GO_advanced_up.csv', index=False)
print(go_adv.res2d.head(10)[['Term', 'Adjusted P-value', 'Genes']].to_string())

print('\n=== GO 富集（早期上调基因）===')
go_ear = gp.enrichr(gene_list=ear_up,
                    gene_sets='GO_Biological_Process_2023',
                    organism='human', outdir=None)
go_ear.res2d.to_csv(f'{DATA}/GO_early_up.csv', index=False)
print(go_ear.res2d.head(10)[['Term', 'Adjusted P-value', 'Genes']].to_string())

# 3. KEGG 富集
print('\n=== KEGG 富集（晚期上调基因）===')
kegg_adv = gp.enrichr(gene_list=adv_up,
                      gene_sets='KEGG_2021_Human',
                      organism='human', outdir=None)
kegg_adv.res2d.to_csv(f'{DATA}/KEGG_advanced_up.csv', index=False)
print(kegg_adv.res2d.head(10)[['Term', 'Adjusted P-value', 'Genes']].to_string())

print('\n=== KEGG 富集（早期上调基因）===')
kegg_ear = gp.enrichr(gene_list=ear_up,
                      gene_sets='KEGG_2021_Human',
                      organism='human', outdir=None)
kegg_ear.res2d.to_csv(f'{DATA}/KEGG_early_up.csv', index=False)
print(kegg_ear.res2d.head(10)[['Term', 'Adjusted P-value', 'Genes']].to_string())

print('\n富集分析完成')
