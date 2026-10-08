# -*- coding: utf-8 -*-
"""
路线 B 第一步 · 主图与统计输出
结论：在 HCC 中 DUOX2 追踪胆管/管状上皮区室，而非肝细胞或免疫细胞；
      与免疫/IFN-γ 的关联在区室校正后消失，与胆管的关联独立保留。

数据来源（均为公开数据，无湿实验）：
  · HPA 单细胞：https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip
                对应文件 HPA_DUOX2_single_cell_type.tsv
  · TCGA-LIHC：本地 GDC STAR-counts（424 样本），logCPM 矩阵
  · GSE14520 ：HBV 主导中国队列；HBV 编码含义引自
               BioMed Res Int. 2020;2020:4037639 Table 1（AVR-CC = 活动性复制慢性携带者）
输出：Figure_localization.png（300 dpi）、summary_stats.txt
"""
import os, csv, gzip, re, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.abspath(__file__))
TCGA = os.environ.get('DUOX2_DATA', os.path.join(_ROOT, '..', 'data'))
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'axes.linewidth': 0.8, 'font.size': 8,
                     'xtick.major.width': 0.8, 'ytick.major.width': 0.8})

SIG = {
 'Cholangiocyte': ['KRT19','KRT7','KRT8','KRT18','EPCAM','SOX9','ANXA4','MUC1','MUC13','GCNT3','CFTR','CLDN4','SLC4A2','ONECUT1','HNF1B','JAG1'],
 'Hepatocyte':    ['ALB','APOA1','APOA2','APOB','TF','TTR','CYP2E1','CYP3A4','ASGR1','SERPINA1','HP','APOH','FGA','FGB','HNF4A','CYP2C9'],
 'Immune':        ['PTPRC','CD3E','CD3D','CD68','LYZ','CD14','NKG7','MS4A1','FCGR3B','S100A8','S100A9','CXCR2'],
 'Stromal':       ['COL1A1','COL1A2','ACTA2','PDGFRB','THY1','FAP','DCN'],
 'Endothelial':   ['PECAM1','VWF','CDH5','ENG','KDR'],
 'Proliferation': ['MKI67','PCNA','TOP2A','CCNB1'],
 'IFN-gamma':     ['STAT1','GBP1','CXCL9','CXCL10','IRF1','IDO1','HLA-DRA'],
}
z = lambda v: (v - v.mean()) / (v.std(ddof=0) + 1e-12)
def pcorr(x, y, cov):
    A = np.column_stack([np.ones(len(x)), cov])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return stats.pearsonr(rx, ry)

# ---------------- 队列 1：TCGA-LIHC ----------------
M = pd.read_pickle(os.path.join(TCGA, 'logcpm_matrix.pkl'))
meta = pd.read_csv(os.path.join(TCGA, 'sample_meta.tsv'), sep='\t').set_index('sample_id').reindex(M.index)
tum = meta['sample_type'].eq('Primary Tumor').values
S1 = {k: np.vstack([z(M[g].values) for g in gs if g in M.columns]).mean(0) for k, gs in SIG.items()}
d1 = M['DUOX2'].values[tum]
r1 = {k: stats.pearsonr(d1, S1[k][tum]) for k in SIG}
cov1 = np.column_stack([S1['Hepatocyte'][tum], S1['Immune'][tum], S1['Stromal'][tum]])
p1 = {k: pcorr(d1, S1[k][tum], cov1) for k in SIG}

# ---------------- 队列 2：GSE14520（HBV 主导）----------------
wanted = {'DUOX2'} | {g for v in SIG.values() for g in v}
g2p = {}
in_t = False
with gzip.open(os.path.join(TCGA, 'GSE14520_family.soft.gz'), 'rt', errors='ignore') as f:
    for line in f:
        if line.startswith('!platform_table_begin'): in_t = True; first = True; continue
        if line.startswith('!platform_table_end'): break
        if not in_t: continue
        if first: first = False; continue
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 11: continue
        for s in (x.strip() for x in parts[10].split('///')):
            if s in wanted: g2p.setdefault(s, set()).add(parts[0].strip())
with open(os.path.join(TCGA, 'GSE14520_expr_probes.tsv')) as f:
    rd = csv.reader(f, delimiter='\t'); gsms = [x.strip() for x in next(rd)[1:]]
    mat = {}
    for row in rd:
        if row and row[0].strip() in {p for s in g2p.values() for p in s}:
            try: mat[row[0].strip()] = np.array([float(v) for v in row[1:]])
            except ValueError: pass
vec = lambda g: (np.nanmean(np.vstack([mat[p] for p in g2p[g] if p in mat]), 0)
                 if any(p in mat for p in g2p.get(g, [])) else None)
hbv = {}
for r in csv.DictReader(open(os.path.join(TCGA, 'GSE14520_validation_final.tsv')), delimiter='\t'):
    if (r.get('gsm') or '').strip(): hbv[r['gsm'].strip()] = (r.get('HBV viral status') or '').strip()
# 主分析：全部肿瘤（验证表中已登记的 221 例，不含癌旁）；敏感性：其中 HBV 相关亚组
tum_sel = np.array([g in hbv for g in gsms])
hbv_sel = np.array([hbv.get(g, '') in ('CC', 'AVR-CC') for g in gsms])
sel = tum_sel
assert tum_sel.sum() == 221 and hbv_sel.sum() == 212, \
    '样本集异常：肿瘤 %d、HBV 相关 %d' % (tum_sel.sum(), hbv_sel.sum())


def scores_for(mask):
    out = {}
    for k, gs in SIG.items():
        vs = [vec(g) for g in gs]; vs = [v for v in vs if v is not None]
        out[k] = np.vstack([z(v[mask]) for v in vs]).mean(0)
    return out


d2, S2 = vec('DUOX2')[sel], scores_for(sel)
r2 = {k: stats.pearsonr(d2, S2[k]) for k in SIG}
n2 = int(sel.sum())

# 敏感性：仅 HBV 相关（CC + AVR-CC）
d2h, S2h = vec('DUOX2')[hbv_sel], scores_for(hbv_sel)
r2h = {k: stats.pearsonr(d2h, S2h[k]) for k in SIG}
n2h = int(hbv_sel.sum())

# 对称偏相关：每个区室都以"其余三区室"为协变量，便于逐个解释
COVS = {'Cholangiocyte': ['Hepatocyte', 'Immune', 'Stromal'],
        'Hepatocyte':    ['Cholangiocyte', 'Immune', 'Stromal'],
        'Immune':        ['Hepatocyte', 'Stromal', 'Cholangiocyte'],
        'Stromal':       ['Hepatocyte', 'Immune', 'Cholangiocyte'],
        'IFN-gamma':     ['Hepatocyte', 'Immune', 'Stromal']}
p1s = {k: pcorr(d1, S1[k][tum], np.column_stack([S1[c][tum] for c in v])) for k, v in COVS.items()}
p2s = {k: pcorr(d2, S2[k], np.column_stack([S2[c] for c in v])) for k, v in COVS.items()}
p2hs = {k: pcorr(d2h, S2h[k], np.column_stack([S2h[c] for c in v])) for k, v in COVS.items()}
p1, p2 = p1s, p2s

# ---------------- HPA 单细胞 ----------------
hpa = pd.read_csv(os.path.join(HERE, 'HPA_DUOX2_single_cell_type.tsv'), sep='\t')
LIVER = ['cholangiocytes', 'mast cells', 'monocytes', 'b-cells', 'nk-cells', 't-cells',
         'plasma cells', 'neutrophils', 'macrophages', 'vascular endothelial cells',
         'hepatocytes', 'kupffer cells', 'hepatic stellate cells']
LBL = {'cholangiocytes': 'Cholangiocytes', 'mast cells': 'Mast cells', 'monocytes': 'Monocytes',
       'b-cells': 'B cells', 'nk-cells': 'NK cells', 't-cells': 'T cells',
       'plasma cells': 'Plasma cells', 'neutrophils': 'Neutrophils', 'macrophages': 'Macrophages',
       'vascular endothelial cells': 'Endothelial', 'hepatocytes': 'Hepatocytes',
       'kupffer cells': 'Kupffer cells', 'hepatic stellate cells': 'Hepatic stellate'}
vals = {LBL[c]: float(hpa.loc[hpa['cell_type'].eq(c), 'nCPM'].iloc[0]) for c in LIVER}

# ---------------- 出图 ----------------
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.55, wspace=0.35)

# (a) 单细胞
ax = fig.add_subplot(gs[0, :])
names = [LBL[c] for c in LIVER]; v = [vals[n] for n in names]
show = np.maximum(v, 0.02)
cols = ['#C0504D' if n == 'Cholangiocytes' else ('#4F81BD' if n in
        ('Hepatocytes', 'Kupffer cells', 'Hepatic stellate') else '#BFBFBF') for n in names]
ax.bar(range(len(names)), np.log10(show) + 2.0, bottom=-2.0, color=cols,
       edgecolor='black', linewidth=0.5, width=0.68)
for i, x in enumerate(v):
    ax.text(i, np.log10(max(x, 0.02)) + 0.06, ('%.1f' % x), ha='center', va='bottom', fontsize=6.4)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=42, ha='right', fontsize=7)
ax.set_ylabel('DUOX2 (nCPM, log$_{10}$)', fontsize=8)
ax.set_ylim(-2.0, 2.0)
ax.set_yticks([-2, -1, 0, 1, 2]); ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
ax.axhline(0, color='grey', lw=0.6, ls='--')
ax.set_title('a  Cell-type resolved DUOX2 expression (HPA single-cell type map)',
             fontsize=8, loc='left', fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

# (b) 区室相关（两队列）
ax = fig.add_subplot(gs[1, 0])
order = ['Cholangiocyte', 'Hepatocyte', 'Immune', 'Stromal', 'Endothelial', 'IFN-gamma']
x = np.arange(len(order)); w = 0.38
ax.bar(x - w/2, [r1[k][0] for k in order], w, label='TCGA-LIHC (n=%d)' % tum.sum(),
       color='#4F81BD', edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, [r2[k][0] for k in order], w, label='GSE14520 (n=%d)' % n2,
       color='#C0504D', edgecolor='black', linewidth=0.5)
for i, k in enumerate(order):
    for off, rr in ((x[i]-w/2, r1[k][1]), (x[i]+w/2, r2[k][1])):
        ax.text(off, (r1 if off == x[i]-w/2 else r2)[k][0] + (0.03 if (r1 if off == x[i]-w/2 else r2)[k][0] >= 0 else -0.06),
                '*' if rr < 0.05 else 'ns', ha='center', fontsize=7.5)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['Cholangio-\ncyte', 'Hepato-\ncyte', 'Immune', 'Stromal', 'Endo-\nthelial', r'IFN-$\gamma$'], fontsize=7)
ax.set_ylabel('Pearson $r$ with DUOX2', fontsize=8); ax.set_ylim(-0.45, 0.68)
ax.legend(fontsize=6.8, frameon=False, loc='upper right')
ax.set_title('b  Compartment scores', fontsize=8, loc='left', fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

# (c) 偏相关
ax = fig.add_subplot(gs[1, 1])
keys = ['Cholangiocyte', 'IFN-gamma']
x = np.arange(len(keys))
ax.bar(x - w/2, [p1[k][0] for k in keys], w, color='#4F81BD', edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, [p2[k][0] for k in keys], w, color='#C0504D', edgecolor='black', linewidth=0.5)
for i, k in enumerate(keys):
    ax.text(x[i]-w/2, p1[k][0] + 0.03, '*' if p1[k][1] < 0.05 else 'ns', ha='center', fontsize=7.5)
    ax.text(x[i]+w/2, p2[k][0] + (0.03 if p2[k][0] >= 0 else -0.07), '*' if p2[k][1] < 0.05 else 'ns', ha='center', fontsize=7.5)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(['Cholangiocyte', r'IFN-$\gamma$'], fontsize=7.5)
ax.set_ylabel('Adjusted partial $r$', fontsize=8); ax.set_ylim(-0.28, 0.58)
ax.set_title('c  Adjusted for hepatocyte, immune, stromal', fontsize=8, loc='left', fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

png = os.path.join(HERE, 'Figure_localization.png')
fig.savefig(png, dpi=312, bbox_inches='tight', facecolor='white')   # 排版宽 6.5 in → 1950 px = 300 dpi
print('图已保存:', png)

# ---------------- 统计输出 ----------------
L = []
def W(s):
    print(s); L.append(s)
W('=== HPA 单细胞：肝内各细胞类型 DUOX2 (nCPM) ===')
for n in names: W('  %-20s %s' % (n, vals[n]))
W('  胆管/肝细胞 倍数: %.0f 倍' % (27.0 / 0.1))
W('  胆管/T 细胞 倍数: %.0f 倍' % (27.0 / 0.7))
W('\n=== 区室相关（简单 r）===')
W('%-16s %18s %18s' % ('区室', 'TCGA-LIHC', 'GSE14520(n=%d)' % n2))
for k in SIG: W('%-16s %8.3f (P=%.1e) %8.3f (P=%.1e)' % (k, r1[k][0], r1[k][1], r2[k][0], r2[k][1]))
W('\n=== 对称偏相关（各自控制其余三区室）===')
W('%-16s %18s %18s' % ('区室', 'TCGA-LIHC', 'GSE14520(n=%d)' % n2))
for k in COVS:
    W('%-16s %8.3f (P=%.1e) %8.3f (P=%.1e)' % (k, p1s[k][0], p1s[k][1], p2s[k][0], p2s[k][1]))
W('\n=== 敏感性：GSE14520 仅 HBV 相关亚组 (CC+AVR-CC, n=%d) ===' % n2h)
W('%-16s %10s %10s %10s %10s' % ('区室', 'r', 'P', 'partial r', 'P'))
for k in SIG:
    W('%-16s %10.3f %10.1e %10.3f %10.1e'
      % (k, r2h[k][0], r2h[k][1], p2hs.get(k, (np.nan, np.nan))[0], p2hs.get(k, (np.nan, np.nan))[1]))
open(os.path.join(HERE, 'summary_stats.txt'), 'w').write('\n'.join(L) + '\n')
print('\n统计已写入 summary_stats.txt')
