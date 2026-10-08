# -*- coding: utf-8 -*-
"""
路线 B 第一步 · 外部验证
问题：在 HBV 主导的中国 HCC 队列（GSE14520）中，DUOX2 是否同样追踪胆管上皮区室，
      而不是免疫区室或肝细胞？

关键设定（均经查证，不采用推测值）：
  · DUOX2 在 GPL3921 上的探针 = 219727_at（GSE14520_family.soft.gz 中 /GEN=DUOX2）
  · GSE14520 的 "HBV viral status" 编码含义引自
    BioMed Res Int. 2020;2020:4037639 Table 1 脚注：
      AVR-CC = active viral replication chronic carrier（活动性复制慢性携带者）
      CC     = chronic carrier（慢性携带者）
      N      = 非病毒性
    故 CC + AVR-CC 视为 HBV 相关 HCC（212/221 = 95.9%）
输出：results_GSE14520.txt（同时打印到 stdout）
"""
import os
import gzip, os, re, sys, math, csv
import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('DUOX2_DATA', os.path.join(_ROOT, '..', 'data'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_GSE14520.txt')
log = open(OUT, 'w')
def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s); log.write(s + '\n')

# ---------- 1. 平台表：探针 → 基因符号 ----------
TARGETS = {
 'Cholangiocyte': ['KRT19','KRT7','KRT8','KRT18','EPCAM','SOX9','ANXA4','MUC1','MUC13','GCNT3','CFTR','CLDN4','SLC4A2','ONECUT1','HNF1B','JAG1'],
 'Hepatocyte':    ['ALB','APOA1','APOA2','APOB','TF','TTR','CYP2E1','CYP3A4','ASGR1','SERPINA1','HP','APOH','FGA','FGB','HNF4A','CYP2C9'],
 'Immune':        ['PTPRC','CD3E','CD3D','CD68','LYZ','CD14','NKG7','MS4A1','FCGR3B','S100A8','S100A9','CXCR2'],
 'Stromal':       ['COL1A1','COL1A2','ACTA2','PDGFRB','THY1','FAP','DCN'],
 'Endothelial':   ['PECAM1','VWF','CDH5','ENG','KDR'],
 'Proliferation': ['MKI67','PCNA','TOP2A','CCNB1'],
 'IFNg':          ['STAT1','GBP1','CXCL9','CXCL10','IRF1','IDO1','HLA-DRA'],
}
wanted = {'DUOX2'} | {g for v in TARGETS.values() for g in v}
# 平台表是 TAB 分隔表格：第 0 列 = 探针 ID，第 10 列 = Gene Symbol（多映射用 " /// " 分隔）
probe2gene, gene2probes = {}, {}
in_table, nrows = False, 0
with gzip.open(os.path.join(BASE, 'GSE14520_family.soft.gz'), 'rt', errors='ignore') as f:
    for line in f:
        if line.startswith('!platform_table_begin'):
            in_table, nrows = True, 0; continue
        if line.startswith('!platform_table_end'):
            break
        if not in_table:
            continue
        nrows += 1
        if nrows == 1:      # 表头
            continue
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 11:
            continue
        probe, syms = parts[0].strip(), parts[10].strip()
        for sym in (s.strip() for s in syms.split('///')):
            if sym in wanted:
                probe2gene.setdefault(probe, sym)
                gene2probes.setdefault(sym, set()).add(probe)
P('平台表数据行数:', nrows - 1)
P('=== 1) 平台探针映射 ===')
P('DUOX2 探针:', sorted(gene2probes.get('DUOX2', [])))
for k, gs in TARGETS.items():
    miss = [g for g in gs if g not in gene2probes]
    P('  %-14s 覆盖 %2d/%2d  缺失: %s' % (k, len(gs)-len(miss), len(gs), miss if miss else '-'))

# ---------- 2. 表达矩阵（探针 × GSM）----------
P('\n=== 2) 载入表达矩阵 ===')
path = os.path.join(BASE, 'GSE14520_expr_probes.tsv')
with open(path) as f:
    rd = csv.reader(f, delimiter='\t')
    hdr = next(rd)
    gsms = [h.strip() for h in hdr[1:]]
    data = {}
    for row in rd:
        if len(row) < 2: continue
        probe = row[0].strip()
        if probe in probe2gene:
            try: data[probe] = np.array([float(x) if x not in ('','NA') else np.nan for x in row[1:]])
            except ValueError: pass
P('矩阵: %d 探针 × %d 样本' % (len(data), len(gsms)))
P('DUOX2 探针是否在矩阵中:', '219727_at' in data)

# ---------- 3. 样本注释 ----------
P('\n=== 3) 样本注释与 HBV 分层 ===')
rows = list(csv.DictReader(open(os.path.join(BASE, 'GSE14520_validation_final.tsv')), delimiter='\t'))
hbv = {}
for r in rows:
    g = (r.get('gsm') or '').strip()
    if g: hbv[g] = (r.get('HBV viral status') or '').strip()
import collections
P('HBV viral status 分布:', dict(collections.Counter(hbv.values())))

idx = {g: i for i, g in enumerate(gsms)}
def vec(gene):
    ps = [p for p in gene2probes.get(gene, []) if p in data]
    if not ps: return None
    return np.nanmean(np.vstack([data[p] for p in ps]), axis=0)

# ---------- 4. 相关性分析 ----------
def z(v):
    v = np.asarray(v, float); s = v.std(ddof=0)
    return (v - v.mean()) / (s if s > 0 else 1e-12)

def score(genes, sel):
    vs = [vec(g) for g in genes]
    vs = [v for v in vs if v is not None]
    return np.vstack([z(v[sel]) for v in vs]).mean(0)

def pcorr(x, y, cov):
    A = np.column_stack([np.ones(len(x)), cov])
    bx, *_ = np.linalg.lstsq(A, x, rcond=None); rx = x - A @ bx
    by, *_ = np.linalg.lstsq(A, y, rcond=None); ry = y - A @ by
    return stats.pearsonr(rx, ry)

duox2 = vec('DUOX2')
for label, keep in [('全部肿瘤 (n=221, 95.9% HBV 携带者)', lambda h: h in ('CC','AVR-CC','N','.')),
                    ('仅 HBV 相关 (CC + AVR-CC, n=212)',      lambda h: h in ('CC','AVR-CC')),
                    ('仅活动性复制 (AVR-CC, n=56)',           lambda h: h == 'AVR-CC'),
                    ('仅非活动携带 (CC, n=156)',              lambda h: h == 'CC')]:
    sel = np.array([keep(hbv.get(g, '')) for g in gsms])
    n = int(sel.sum())
    P('\n=== %s，n=%d ===' % (label, n))
    if n < 20:
        P('  样本过少，跳过统计'); continue
    S = {k: score(gs, sel) for k, gs in TARGETS.items()}
    d = duox2[sel]
    P('%-16s %9s %9s' % ('区室', 'r', 'P'))
    for k in TARGETS:
        r, p = stats.pearsonr(d, S[k]); P('%-16s %9.3f %9.2e' % (k, r, p))
    cov = np.column_stack([S['Hepatocyte'], S['Immune'], S['Stromal']])
    r, p = pcorr(d, S['Cholangiocyte'], cov)
    P('偏相关（控肝细胞+免疫+间质）DUOX2~Cholangiocyte: r=%6.3f P=%.2e' % (r, p))
    r, p = pcorr(d, S['IFNg'], cov)
    P('偏相关（控肝细胞+免疫+间质）DUOX2~IFNg       : r=%6.3f P=%.2e' % (r, p))

P('\n=== 4) 关键单基因相关（HBV 相关亚组）===')
sel = np.array([hbv.get(g, '') in ('CC','AVR-CC') for g in gsms])
d = duox2[sel]
for g in ['KRT19','KRT7','ANXA4','MUC1','GCNT3','SOX9','EPCAM','CFTR',
          'ALB','APOA1','CYP3A4','ASGR1','PTPRC','CD68','COL1A1']:
    v = vec(g)
    if v is None: P('  %-8s 无探针' % g); continue
    r, p = stats.pearsonr(d, v[sel]); P('  %-8s r=%6.3f  P=%.1e' % (g, r, p))

log.close()
print('\n结果已写入:', OUT)
