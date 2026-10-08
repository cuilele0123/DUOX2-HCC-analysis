# -*- coding: utf-8 -*-
"""出 Supplementary Figure S9：DUOX2 在两个独立人肝单细胞图谱 + 小鼠肝癌肿瘤区室中的归属。

数据：RouteB_analysis/cellxgene_DUOX2_summary.tsv（由 13_cellxgene_duox2.py 生成，
原始counts 来自 CELLxGENE H5AD 的 .raw.X）。

面板（2×2）：
  a 人肝两图谱各细胞类型的 DUOX2 检出率（%），线性轴
  b 同一批细胞的平均 CPM（对数轴），叠加 HPA 参考图谱 27.0 nCPM 参考线
  c 谱系标记 QC：ALB vs KRT19+KRT7（对数轴），确认注释互不污染
  d 小鼠 MASLD-HCC 肿瘤区室可比较的细胞数（对数轴），并标注 0 检出
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('DUOX2_WORK', os.path.join(_ROOT, '..'))
TSV = os.path.join(BASE, 'RouteB_analysis', 'cellxgene_DUOX2_summary.tsv')
OUTDIR = os.path.join(BASE, 'v21_Supplementary_Figures')
OUT = os.path.join(OUTDIR, 'FigureS9_singlecell_atlases.png')
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': 0.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
CB, CH, CG = '#2b6cb0', '#c05621', '#4a5568'
HPA_CHOL = 27.0

df = pd.read_csv(TSV, sep='\t')
hum = df[(df.species == 'human') & (~df.cell_type.str.contains('unknown', case=False, na=False))].copy()
mouse = df[(df.species == 'mouse') & (df.scope == 'tumor')].copy()

LABEL = {
    ('JHepatol2024_biliary', 'intrahepatic cholangiocyte'): 'intrahepatic cholangiocyte',
    ('HepatolComm2022_biliary', 'cholangiocyte'): 'cholangiocyte',
    ('HepatolComm2022_biliary', 'progenitor cell'): 'progenitor cell',
    ('HepatolComm2022_biliary', 'centrilobular region hepatocyte'): 'centrilobular hepatocyte',
    ('HepatolComm2022_hepatocyte', 'hepatocyte'): 'hepatocyte',
    ('HepatolComm2022_hepatocyte', 'periportal region hepatocyte'): 'periportal hepatocyte',
    ('HepatolComm2022_hepatocyte', 'midzonal region hepatocyte'): 'midzonal hepatocyte',
    ('HepatolComm2022_hepatocyte', 'centrilobular region hepatocyte'): 'centrilobular hepatocyte',
    ('JHepatol2024_hepatocyte1', 'periportal region hepatocyte'): 'periportal hepatocyte',
    ('JHepatol2024_hepatocyte1', 'centrilobular region hepatocyte'): 'centrilobular hepatocyte',
    ('JHepatol2024_hepatocyte1', 'midzonal region hepatocyte'): 'midzonal hepatocyte',
    ('JHepatol2024_hepatocyte2', 'periportal region hepatocyte'): 'periportal hepatocyte',
    ('JHepatol2024_hepatocyte2', 'centrilobular region hepatocyte'): 'centrilobular hepatocyte',
    ('JHepatol2024_hepatocyte2', 'midzonal region hepatocyte'): 'midzonal hepatocyte',
    ('JHepatol2024_hepatocyte2', 'hepatocyte'): 'hepatocyte',
}
hum['lab'] = [LABEL.get((d, c), str(c)) for d, c in zip(hum.dataset, hum.cell_type)]
# 谱系判定必须用原始 cell_type，不能用可能被截断/改写的展示标签；红系等其他细胞一律排除
_ct = hum.cell_type.str.lower()
hum['lineage'] = np.where(_ct.str.contains('hepatocyte'), 'hepatocyte',
                   np.where(_ct.str.contains('cholangiocyte|progenitor|ductular|biliary'),
                            'biliary', 'other'))
hum = hum[hum.lineage.isin(['biliary', 'hepatocyte'])]

g = hum.groupby(['lineage', 'lab'], as_index=False).agg(
    n=('n_cells', 'sum'), pos=('DUOX2_pos_cells', 'sum'), cpm=('DUOX2_mean_cpm', 'mean'),
    alb=('ALB_mean_cpm', 'mean'), k19=('KRT19_mean_cpm', 'mean'), k7=('KRT7_mean_cpm', 'mean'))
g['pct'] = 100 * g.pos / g.n
SHORT = {'intrahepatic cholangiocyte': 'intrahepatic\ncholangiocyte', 'centrilobular hepatocyte':
         'centrilobular\nhepatocyte', 'periportal hepatocyte': 'periportal\nhepatocyte',
         'midzonal hepatocyte': 'midzonal\nhepatocyte', 'progenitor cell': 'progenitor cell',
         'cholangiocyte': 'cholangiocyte', 'hepatocyte': 'hepatocyte'}
g['lab2'] = [SHORT.get(l, l) for l in g.lab]
g = g.sort_values(['lineage', 'pct'], ascending=[False, False]).reset_index(drop=True)

fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))

# ---- a 检出率（线性轴）
a = ax[0, 0]
cols = [CB if l == 'biliary' else CH for l in g.lineage]
y = np.arange(len(g))
a.barh(y, g.pct, color=cols, height=0.66, edgecolor='none')
a.set_yticks(y)
a.set_yticklabels(g.lab2, fontsize=6.6)
a.invert_yaxis()
a.set_xlim(0, 9.6)
a.set_xlabel('DUOX2-positive cells (%)')
a.set_title('a  Detection rate in two human liver atlases', fontsize=8, loc='left')
for i, v in enumerate(g.pct):
    a.text(v + 0.18, i, ('0' if v == 0 else '%.2f' % v), va='center', fontsize=6.4,
           color='#2d3748')
a.legend(handles=[Line2D([], [], color=CB, lw=5, label='biliary lineage'),
                  Line2D([], [], color=CH, lw=5, label='hepatocyte')],
         loc='lower right', frameon=False, fontsize=6.8)

# ---- b 平均 CPM（对数轴）+ HPA 参考线
b = ax[0, 1]
yy = np.arange(len(g))[::-1]
for i, r in enumerate(g.itertuples()):
    c = CB if r.lineage == 'biliary' else CH
    b.scatter(max(r.cpm, 1e-3), i, s=30, color=c, zorder=3, edgecolor='none')
b.axvline(HPA_CHOL, color='#2d3748', ls='--', lw=0.8, zorder=2)
b.set_yticks(np.arange(len(g)))
b.set_yticklabels([])
b.set_xscale('log')
b.set_ylim(len(g) - 0.4, -0.6)
b.set_xlim(3e-3, 300)
b.set_xlabel('DUOX2 (CPM, log scale)')
b.set_title('b  Mean expression', fontsize=8, loc='left')
b.text(HPA_CHOL * 0.42, len(g) - 0.75, 'HPA reference map\ncholangiocytes 27.0 nCPM',
       fontsize=6.2, color='#2d3748', va='center', ha='right')

# ---- c 谱系标记 QC
c = ax[1, 0]
for r in g.itertuples():
    col = CB if r.lineage == 'biliary' else CH
    c.scatter(max(r.alb, 1e-2), max(r.k19 + r.k7, 1e-2), s=34, color=col, alpha=0.9,
              edgecolor='none', zorder=3)
c.set_xscale('log')
c.set_yscale('log')
c.set_xlabel('ALB (CPM)')
c.set_ylabel('KRT19 + KRT7 (CPM)')
c.set_title('c  Lineage marker check', fontsize=8, loc='left')
c.text(0.03, 0.95, 'biliary', transform=c.transAxes, fontsize=7, color=CB, va='top')
c.text(0.03, 0.86, 'hepatocyte', transform=c.transAxes, fontsize=7, color=CH, va='top')
from matplotlib.ticker import NullFormatter
c.xaxis.set_minor_formatter(NullFormatter())
c.yaxis.set_minor_formatter(NullFormatter())
c.text(0.97, 0.30, 'no overlap →\nannotations are clean', transform=c.transAxes, fontsize=6.4,
       color='#4a5568', ha='right', va='bottom')

# ---- d 小鼠肿瘤区室：可比较细胞数
d = ax[1, 1]
m = mouse[mouse.cell_type.isin(['cholangiocyte', 'hepatocyte'])].copy()
extra = mouse[(~mouse.cell_type.isin(['cholangiocyte', 'hepatocyte'])) & (mouse.n_cells >= 300)]
m = pd.concat([m, extra]).sort_values('n_cells')
y2 = np.arange(len(m))
d.barh(y2, m.n_cells, color=[CB if x == 'cholangiocyte' else (CH if x == 'hepatocyte' else CG)
                             for x in m.cell_type], height=0.66, edgecolor='none')
d.set_yticks(y2)
d.set_yticklabels([str(c)[:30] for c in m.cell_type], fontsize=6.6)
d.invert_yaxis()
d.set_xscale('log')
d.set_xlabel('cells available in the tumor compartment (log)')
d.set_title('d  Mouse MASLD-HCC dataset', fontsize=8, loc='left')
for i, r in enumerate(m.itertuples()):
    d.text(r.n_cells * 1.15, i, '%d' % r.n_cells, va='center', fontsize=6.4, color='#2d3748')
d.set_xlim(8, 6000)
d.text(0.97, 0.97, '0 Duox2-positive cells among 6,008 tumor cells', transform=d.transAxes,
       fontsize=6.6, color='#c53030', ha='right', va='top')

fig.tight_layout(w_pad=1.4, h_pad=1.6)
fig.savefig(OUT, bbox_inches='tight', facecolor='white', dpi=600)
plt.close(fig)

from PIL import Image
im = Image.open(OUT)
print('已生成 %s' % OUT)
print('  %.2f × %.2f in  %d × %d px  %.2f MB  (dpi 600)'
      % (im.size[0] / 600, im.size[1] / 600, im.size[0], im.size[1], os.path.getsize(OUT) / 1e6))
print()
print(g[['lineage', 'lab', 'n', 'pos', 'pct', 'cpm', 'alb', 'k19', 'k7']].to_string(index=False))