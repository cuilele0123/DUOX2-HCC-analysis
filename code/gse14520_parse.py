# -*- coding: utf-8 -*-
"""GSE14520 外部验证 v2（正确处理 series_matrix 格式）"""
import gzip
import pandas as pd
import numpy as np
from scipy import stats
import sys, os, subprocess, urllib.request

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

# ===== 1. 解析表达矩阵 =====
print('=== 1. 解析 GSE14520 表达矩阵 ===')
sample_ids = None
expr_rows = []

with gzip.open(f'{DATA}/GSE14520-GPL3921_matrix.txt.gz', 'rt') as f:
    in_table = False
    for line in f:
        if line.startswith('!series_matrix_table_begin'):
            in_table = True
            continue
        if line.startswith('!series_matrix_table_end'):
            break
        if in_table:
            parts = line.rstrip('\n').split('\t')
            if sample_ids is None:
                sample_ids = [p.strip('"') for p in parts[1:]]
                print(f'样本数: {len(sample_ids)}')
            else:
                expr_rows.append(parts)

print(f'探针行: {len(expr_rows)}')
# 构建探针表达
probes = [r[0].strip('"') for r in expr_rows]
vals = np.array([[float(x) for x in r[1:]] for r in expr_rows])
expr = pd.DataFrame(vals, index=probes, columns=sample_ids)
print(f'表达矩阵: {expr.shape}')
expr.to_csv(f'{DATA}/GSE14520_expr_probes.tsv', sep='\t')
print('保存探针表达矩阵')

# ===== 2. 探针→基因映射 =====
print('\n=== 2. 探针→基因映射 (HG-U133A 2.0) ===')
# GPL3921 = Affymetrix HG-U133A 2.0。下载平台注释（带引号的 SOFT 平台表）
annot = None
annot_file = f'{DATA}/GPL3921_annot_probe.tsv'
if not os.path.exists(annot_file):
    print('下载 GPL3921 注释...')
    subprocess.run([sys.executable, f'{BASE}/download_parallel.py',
                    'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL3921/soft/GPL3921_family.soft.gz',
                    f'{DATA}/GPL3921_family.soft.gz'], capture_output=True, timeout=600)
    # 解析 SOFT 平台表（GPL 的 !platform_table_begin）
    if os.path.exists(f'{DATA}/GPL3921_family.soft.gz'):
        with gzip.open(f'{DATA}/GPL3921_family.soft.gz', 'rt') as f:
            in_tab = False
            rows = []
            hdr = None
            for line in f:
                if line.startswith('!platform_table_begin'):
                    in_tab = True; continue
                if line.startswith('!platform_table_end'):
                    break
                if in_tab:
                    parts = line.rstrip('\n').split('\t')
                    if hdr is None:
                        hdr = [p.strip('"') for p in parts]
                    else:
                        rows.append([p.strip('"') for p in parts])
            if hdr:
                gpl = pd.DataFrame(rows, columns=hdr)
                print(f'GPL 注释: {gpl.shape}, 列: {list(gpl.columns)[:10]}')
                # 找到 ID 和 Gene Symbol 列
                id_col = [c for c in gpl.columns if c.upper() in ('ID','PROBE ID','PROBEID')]
                sym_col = [c for c in gpl.columns if 'SYMBOL' in c.upper() or 'GENE' in c.upper()]
                print('ID 列:', id_col, '| 基因列:', sym_col)
                if id_col and sym_col:
                    annot = gpl[[id_col[0], sym_col[0]]].dropna()
                    annot.columns = ['probe', 'gene']
                    annot.to_csv(annot_file, sep='\t', index=False)
                    print(f'探针映射保存: {len(annot)}')

if annot is None and os.path.exists(annot_file):
    annot = pd.read_csv(annot_file, sep='\t')

# ===== 3. 提取目标基因 =====
print('\n=== 3. 提取目标基因 ===')
if annot is not None:
    target_genes = ['DUOX2', 'SLC16A3', 'SPP2', 'MMP7']
    gene_expr = {}
    for g in target_genes:
        probes_g = annot[annot['gene'].str.upper() == g]['probe'].tolist()
        if probes_g:
            vals_g = expr.loc[[p for p in probes_g if p in expr.index]]
            gene_expr[g] = vals_g.mean(axis=0)  # 多探针取均值
            print(f'{g}: 探针 {probes_g}, 表达样本 {len(vals_g.columns)}')
    if gene_expr:
        gene_df = pd.DataFrame(gene_expr)
        gene_df.to_csv(f'{DATA}/GSE14520_gene_expr.csv')
        print(f'\n目标基因表达矩阵: {gene_df.shape}')
        print(gene_df.head())
else:
    print('!!! 探针注释获取失败，使用内置 HG-U133A 2.0 注释')
    # 内置关键探针（HG-U133A 2.0 已知 ID）
    known = {'DUOX2': '203504_at', 'SLC16A3': '202234_s_at', 'SPP2': '209875_s_at', 'MMP7': '204259_at'}
    for g, p in known.items():
        if p in expr.index:
            print(f'{g}: {p} 提取成功')
            gene_df = expr.loc[[p for p in known.values() if p in expr.index]]
            gene_df.index = [g for g, p in known.items() if p in expr.index]
            gene_df.to_csv(f'{DATA}/GSE14520_gene_expr.csv')
            print(gene_df.head())
            break

print('\n完成')
