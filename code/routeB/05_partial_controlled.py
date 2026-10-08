#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线 B 补充分析：胆管区室是否充分解释 DUOX2 与免疫/IFN-γ 的关联。
问题：原稿把 DUOX2-免疫关联当作生物学属性；新框架主张其为区室构成共变的产物。
关键检验：DUOX2 ~ 免疫 | (肝细胞, 间质, 胆管)  —— 若趋零，则关联由胆管轴解释。
输出：GSE14520 与 TCGA 双队列结果，落盘 partial_correlation_controlled.tsv
"""
import os
import gzip
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.abspath(__file__))
TC = os.environ.get('DUOX2_DATA', os.path.join(_ROOT, '..', 'data'))

SIG = {
 'Cholangiocyte': ['KRT19', 'KRT7', 'KRT8', 'KRT18', 'EPCAM', 'SOX9', 'ANXA4', 'MUC1',
                   'MUC13', 'GCNT3', 'CFTR', 'CLDN4', 'SLC4A2', 'ONECUT1', 'HNF1B', 'JAG1'],
 'Hepatocyte':    ['ALB', 'APOA1', 'APOA2', 'APOB', 'TF', 'TTR', 'CYP2E1', 'CYP3A4',
                   'ASGR1', 'SERPINA1', 'HP', 'APOH', 'FGA', 'FGB', 'HNF4A', 'CYP2C9'],
 'Immune':        ['PTPRC', 'CD3E', 'CD3D', 'CD68', 'LYZ', 'CD14', 'NKG7', 'MS4A1',
                   'FCGR3B', 'S100A8', 'S100A9', 'CXCR2'],
 'Stromal':       ['COL1A1', 'COL1A2', 'ACTA2', 'PDGFRB', 'THY1', 'FAP', 'DCN'],
 'IFN-gamma':     ['STAT1', 'GBP1', 'CXCL9', 'CXCL10', 'IRF1', 'IDO1', 'HLA-DRA'],
}
z = lambda v: (v - v.mean()) / (v.std(ddof=0) + 1e-12)


def pcorr(x, y, covs):
    A = np.column_stack([np.ones(len(x))] + [c for c in covs])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def build_scores(M):
    S = {}
    for k, gs in SIG.items():
        gs = [g for g in gs if g in M.columns]
        S[k] = np.vstack([z(M[g].values.astype(float)) for g in gs]).mean(0)
    return S


rows = []


# 目标 → 协变量组合（公平对称：把其余三个区室全部作为协变量）
TARGET_COVS = {
    'Cholangiocyte': [('其余三区室', ['Hepatocyte', 'Stromal', 'Immune'])],
    'Immune':        [('其余三区室', ['Hepatocyte', 'Stromal', 'Cholangiocyte'])],
    'IFN-gamma':     [('其余三区室', ['Hepatocyte', 'Stromal', 'Immune']),
                      ('含胆管三区室', ['Hepatocyte', 'Stromal', 'Cholangiocyte'])],
    'Hepatocyte':    [('其余三区室', ['Cholangiocyte', 'Stromal', 'Immune'])],
    'Stromal':       [('其余三区室', ['Hepatocyte', 'Cholangiocyte', 'Immune'])],
}


def report(tag, D, S, n):
    print('=== %s n=%d ===' % (tag, n))
    print('%-16s %10s %12s' % ('区室', 'r(简单)', 'P'))
    for k in SIG:
        r, p = stats.pearsonr(D, S[k])
        print('%-16s %10.3f %12.2e' % (k, r, p))
        rows.append((tag, k, 'raw', r, p))
    print()
    print('%-16s %-16s %10s %12s   %s' % ('目标', '协变量', '偏 r', 'P', '判读'))
    for k, covsets in TARGET_COVS.items():
        for cname, keys in covsets:
            r, p = pcorr(D, S[k], [S[c] for c in keys])
            verdict = '独立保留' if p < 0.01 else ('边缘' if p < 0.05 else '消失')
            print('%-16s %-16s %10.3f %12.2e   %s' % (k, cname, r, p, verdict))
            rows.append((tag, k, '|' + '+'.join(keys), r, p))
    print()


# ---------------- TCGA-LIHC ----------------
M = pd.read_pickle(os.path.join(TC, 'logcpm_matrix.pkl'))
meta = pd.read_csv(os.path.join(TC, 'sample_meta.tsv'), sep='\t').set_index('sample_id').reindex(M.index)
tum = meta['sample_type'].eq('Primary Tumor').values
S = build_scores(M)
D = M['DUOX2'].values.astype(float)
report('TCGA-LIHC', D[tum], {k: v[tum] for k, v in S.items()}, int(tum.sum()))

# ---------------- GSE14520 ----------------
soft = os.path.join(TC, 'GSE14520_family.soft.gz')
wanted = {'DUOX2'} | {g for v in SIG.values() for g in v}
g2p = {}
in_tab = False
with gzip.open(soft, 'rt', errors='ignore') as f:
    for line in f:
        if line.startswith('!platform_table_begin'):
            in_tab = True; continue
        if line.startswith('!platform_table_end'):
            in_tab = False; continue
        if not in_tab:
            continue
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 11:
            continue
        sym = parts[10].strip()
        if sym in wanted:
            g2p.setdefault(sym, parts[0].strip())

mpath = os.path.join(TC, 'GSE14520-GPL3921_matrix.txt.gz')
hdr = None
with gzip.open(mpath, 'rt', errors='ignore') as f:
    for i, line in enumerate(f):
        if line.split('\t')[0].strip().strip('"').upper() in ('ID_REF', 'AFFYID'):
            hdr = i; break
assert hdr is not None, '未找到 GSE14520 矩阵表头行'
mat = pd.read_csv(mpath, sep='\t', skiprows=hdr, index_col=0, engine='python')
mat = mat.loc[:, [c for c in mat.columns if not str(c).startswith('Unnamed')]]
val = pd.read_csv(os.path.join(TC, 'GSE14520_validation_final.tsv'), sep='\t')
tum_ids = [g for g in val.loc[val['Tissue Type'].astype(str).str.contains('Tumor', na=False), 'gsm']
           if g in mat.columns]
T = mat[tum_ids].astype(float).T          # 样本 × 探针
G = pd.DataFrame({s: T[p] for s, p in g2p.items() if p in T.columns})
G = G.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
S2 = build_scores(G)
print('GSE14520：肿瘤 %d 例，探针映射到 %d 个目标基因' % (len(G), G.shape[1]))
report('GSE14520', G['DUOX2'].values.astype(float), S2, len(G))

pd.DataFrame(rows, columns=['cohort', 'target', 'adjustment', 'r', 'P']) \
    .to_csv(os.path.join(HERE, 'partial_correlation_controlled.tsv'), sep='\t', index=False)
print()
print('已落盘: partial_correlation_controlled.tsv')
