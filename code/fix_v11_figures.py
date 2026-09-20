# -*- coding: utf-8 -*-
"""
v11 修正三张图：
- Fig1: 火山图图例改为与正文一致的 DEG 定义 (|log2FC|>1 且 FDR<0.05): Up 345 / Down 954 / NS 29217
- Fig3: (A) 改为 GSE14520 肿瘤 vs 癌旁 DUOX2 (支撑正文 7.83 vs 8.12, P=0.002)
        (B) 保留固定 TCGA 系数 LASSO 风险 KM (p=0.0006)
- Fig5: 删除与 Fig7 重复的 LASSO 面板，只保留 DUOX2 中位数 KM (p=0.976)
样式与 make_all_figures_sci.py 完全一致 (clear sans-serif Helvetica/Arial, 300dpi)
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def FMT_P(p):
    """统一 P 值写法：大写 P、科学计数法用 x 10^n 上标形式。"""
    if p >= 0.001:
        return 'P = %.4g' % p
    e = int(np.floor(np.log10(p)))
    m = p / 10.0 ** e
    return 'P = %s $\\times$ 10$^{%d}$' % (('%.3g' % m), e)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')

# Scientific Reports 图片规范：clear sans-serif（Helvetica / Arial）
for p in ['/System/Library/Fonts/Helvetica.ttc',
          '/System/Library/Fonts/Supplemental/Arial.ttf',
          '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
          '/System/Library/Fonts/Supplemental/Arial Italic.ttf',
          '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf']:
    try: fm.fontManager.addfont(p)
    except Exception: pass
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 6.5          # 最终印刷字号（5–7 pt 区间）
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 7.0
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 7.0
plt.rcParams['xtick.labelsize'] = 6.3
plt.rcParams['ytick.labelsize'] = 6.3
plt.rcParams['legend.fontsize'] = 6.3
plt.rcParams['legend.frameon'] = False
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

C_HI = '#C0392B'; C_LO = '#2980B9'; C_NEUT = '#95A5A6'

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', name)

# ============ Fig 1: volcano（DEG 口径 = |l2fc|>1 且 FDR<0.05） ============
res = pd.read_csv(os.path.join(DATA, 'deseq2_early_vs_advanced_all.csv'), index_col=0)
gm = pd.read_csv(os.path.join(DATA, 'gene_id_name_map.tsv'), sep='\t')
id2name = dict(zip(gm.gene_id.astype(str).str.split('.').str[0], gm.gene_name))
res['gene_name'] = [id2name.get(str(i).split('.')[0], str(i)) for i in res.index]
res = res.dropna(subset=['padj', 'log2FoldChange'])
res['l2fc_adv'] = -res['log2FoldChange']   # 翻转为 advanced vs early
res['nlog10p'] = -np.log10(res['padj'])

sig = res[(res['l2fc_adv'].abs() > 1) & (res['padj'] < 0.05)]
up = sig[sig['l2fc_adv'] > 1]
down = sig[sig['l2fc_adv'] < -1]
ns = res.drop(sig.index)
print('Fig1 DEG counts: up %d, down %d, NS %d' % (len(up), len(down), len(ns)))
assert len(up) == 954 and len(down) == 345, 'DEG counts unexpected!'

fig, ax = plt.subplots(figsize=(3.22, 2.41))
ax.scatter(ns['l2fc_adv'], ns['nlog10p'], s=3.05, c=C_NEUT, alpha=0.45, linewidths=0.0,
           label=f'NS (n={len(ns):,})')
ax.scatter(down['l2fc_adv'], down['nlog10p'], s=4.36, c=C_LO, alpha=0.7, linewidths=0.0,
           label=f'Down in advanced (n={len(down):,})')
ax.scatter(up['l2fc_adv'], up['nlog10p'], s=4.36, c=C_HI, alpha=0.7, linewidths=0.0,
           label=f'Up in advanced (n={len(up):,})')
label_genes = ['DUOX2', 'KLK11', 'DUOXA2', 'TMC5', 'ZNF208', 'GIPR',
               'SLC16A3', 'MMP7', 'SPP2', 'NTS', 'CEACAM7']
offsets = {'DUOX2': (12, 10), 'KLK11': (12, 8), 'DUOXA2': (12, -8),
           'TMC5': (12, 10), 'ZNF208': (12, 8), 'GIPR': (12, -10),
           'SLC16A3': (-30, 10), 'MMP7': (-40, 8), 'SPP2': (12, -10),
           'NTS': (12, 8), 'CEACAM7': (12, -10)}
for g in label_genes:
    row = res[res['gene_name'] == g]
    if len(row) == 0: continue
    r = row.iloc[0]
    dx, dy = offsets.get(g, (8, 6))
    ax.annotate(g, (r['l2fc_adv'], r['nlog10p']), fontsize=6.3, xytext=(dx, dy),
                textcoords='offset points', color='black',
                arrowprops=dict(arrowstyle='-', color=C_NEUT, lw=0.35))
ax.axhline(-np.log10(0.05), color=C_NEUT, ls='--', lw=0.53)
ax.axvline(1, color=C_NEUT, ls='--', lw=0.53)
ax.axvline(-1, color=C_NEUT, ls='--', lw=0.53)
ax.set_xlabel('log$_{2}$ Fold Change (Advanced vs Early)')
ax.set_ylabel('$-$log$_{10}$ (adjusted P)')
ax.set_title('Volcano plot of DEGs between advanced and early HCC', pad=10)
ax.legend(loc='upper right', frameon=False, fontsize=5.9)
save(fig, 'Fig1_volcano_v11')

# ============ Fig 3: GSE14520 (A) tumor vs non-tumor + (B) fixed-coef KM ============
expr = pd.read_csv(os.path.join(DATA, 'GSE14520_gene_expr.csv'), index_col=0)
sup = pd.read_csv(os.path.join(DATA, 'GSE14520_Extra_Supplement.txt.gz'), sep='\t')
sup_affy = sup.dropna(subset=['Affy_GSM'])
m = expr.reset_index().merge(sup_affy[['Affy_GSM', 'Tissue Type']],
                             left_on='index', right_on='Affy_GSM', how='inner')
t = m[m['Tissue Type'] == 'Tumor']['DUOX2']
n = m[m['Tissue Type'] == 'Non-Tumor']['DUOX2']
from scipy import stats as sps
p_tn = sps.mannwhitneyu(t, n).pvalue
print('Fig3A: tumor n=%d median %.2f | non-tumor n=%d median %.2f | p=%.4g'
      % (len(t), t.median(), len(n), n.median(), p_tn))
assert abs(t.median() - 7.83) < 0.01 and abs(n.median() - 8.12) < 0.01

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
val = pd.read_csv(os.path.join(DATA, 'GSE14520_validation_final.tsv'), sep='\t')

fig, axes = plt.subplots(1, 2, figsize=(8.11, 3.31))
ax = axes[0]
bp = ax.boxplot([n.values, t.values], tick_labels=[f'Non-tumor\n(n={len(n)})', f'Tumor\n(n={len(t)})'],
                widths=0.5, patch_artist=True, showfliers=False,
                medianprops=dict(color='black', lw=0.92))
for patch, c in zip(bp['boxes'], [C_LO, C_HI]):
    patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor('black'); patch.set_linewidth(0.53)
for arr, x in [(n.values, 1), (t.values, 2)]:
    jitter = np.random.default_rng(7).normal(x, 0.06, len(arr))
    ax.scatter(jitter, arr, s=2.18, color='black', alpha=0.25, linewidths=0.0, zorder=3)
ax.set_ylabel('DUOX2 expression (RMA, log2)')
ax.set_title('Tumor vs adjacent non-tumor liver', pad=8)
ax.text(0.97, 0.95, f'Mann-Whitney {FMT_P(p_tn)}', transform=ax.transAxes,
        ha='right', va='top', fontsize=6.3, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.2'))

ax = axes[1]
hi = val[val['risk_g_f'] == 'high']; lo = val[val['risk_g_f'] == 'low']
def km(axx, d, c, lab):
    kmf = KaplanMeierFitter()
    kmf.fit(d['OS_months'], d['event'], label=lab)
    kmf.plot_survival_function(ax=axx, ci_show=False, color=c, linewidth=1.32)
km(ax, hi, C_HI, f'High risk (n={len(hi)})')
km(ax, lo, C_LO, f'Low risk (n={len(lo)})')
lr = logrank_test(hi['OS_months'], lo['OS_months'], hi['event'], lo['event'])
print('Fig3B log-rank p = %.2e' % lr.p_value)
ax.set_xlabel('Time (months)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
ax.set_title('LASSO risk score (fixed TCGA coefficients)', pad=8)
ax.text(0.97, 0.05, f'log-rank {FMT_P(lr.p_value)}', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=6.3, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.2'))
ax.legend(loc='upper right', fontsize=5.9)
save(fig, 'Fig3_GSE14520_v11')

# ============ Fig 5: 单面板 DUOX2 中位数 KM ============
surv = pd.read_pickle(os.path.join(DATA, 'survival_data.pkl')).set_index('sample_id')
cpm = pd.read_pickle(os.path.join(DATA, 'logcpm_matrix.pkl'))
d = surv.copy()
d['DUOX2_cpm'] = [cpm.loc[s, 'DUOX2'] for s in d.index]
med = d['DUOX2_cpm'].median()
d['g'] = np.where(d['DUOX2_cpm'] >= med, 'High', 'Low')
h = d[d['g'] == 'High']; l = d[d['g'] == 'Low']
lr5 = logrank_test(h['OS_time_days'], l['OS_time_days'], h['OS_event'], l['OS_event'])
print('Fig5: n=%d/%d, log-rank p=%.3f' % (len(h), len(l), lr5.p_value))
assert abs(lr5.p_value - 0.976) < 0.005

fig, ax = plt.subplots(figsize=(3.49, 2.50))
def km2(axx, dd, c, lab):
    kmf = KaplanMeierFitter()
    kmf.fit(dd['OS_time_days'], dd['OS_event'], label=lab)
    kmf.plot_survival_function(ax=axx, ci_show=False, color=c, linewidth=1.32)
km2(ax, h, C_HI, f'DUOX2 High (n={len(h)})')
km2(ax, l, C_LO, f'DUOX2 Low (n={len(l)})')
ax.set_xlabel('Time (days)'); ax.set_ylabel('Overall survival probability'); ax.set_ylim(0, 1.05)
ax.set_title('Overall survival by median DUOX2 expression', pad=8)
ax.text(0.97, 0.05, f'log-rank {FMT_P(lr5.p_value)} (NS)', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=6.3, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.2'))
ax.legend(loc='upper right', fontsize=5.9)
save(fig, 'Fig5_KM_duox2_v11')
print('ALL DONE')
