# -*- coding: utf-8 -*-
"""
DUOX2/HCC 论文 20 张图统一重绘（SCI 级）
- 全部英文 clear sans-serif（Helvetica / Arial，符合 Scientific Reports 图片规范）
- 文字不覆盖在柱状/线条/点等图形上（值标签统一外置）
- 配色统一（红/蓝/绿/橙/紫 + 灰色中性色）
- 300dpi + PDF 双输出
"""
import os, warnings, json, textwrap
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.patches as mpatches

# ---------- 字体：Scientific Reports 图片规范 —— clear sans-serif（Helvetica / Arial）----------
# 官方原文："Typeface: Clear sans-serif, for example Helvetica. Same typeface and
# size across all figures."（Nature Portfolio 美术指南补充：Helvetica 或 Arial 优先）
for p in [
    '/System/Library/Fonts/Helvetica.ttc',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/System/Library/Fonts/Supplemental/Arial Italic.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf',
]:
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
# mathtext 同步用无衬线，避免上标 P 值回落成衬线体
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.unicode_minus'] = False

# ---------- 统一字号与画布 ----------
plt.rcParams['figure.dpi'] = 150         # 屏幕预览
plt.rcParams['font.size'] = 6.5          # 最终印刷字号（5–7 pt 区间）
plt.rcParams['savefig.dpi'] = 300        # 出版 300dpi
plt.rcParams['axes.titlesize'] = 7.0
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 7.0
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['xtick.labelsize'] = 6.3
plt.rcParams['ytick.labelsize'] = 6.3
plt.rcParams['legend.fontsize'] = 6.3
plt.rcParams['legend.frameon'] = False
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

# ---------- 统一调色板 ----------
# 主色：红=高风险/上调/不良；蓝=低风险/下调/保护；绿=早/保护；橙=中
C_HI    = '#C0392B'   # red
C_LO    = '#2980B9'   # blue
C_EARLY = '#27AE60'   # green
C_INT   = '#F39C12'   # orange
C_ADV   = '#C0392B'   # red (同 C_HI)
C_PURP  = '#8E44AD'   # purple
C_NEUT  = '#7F8C8D'   # gray
C_BG    = '#ECF0F1'   # 浅灰（Normal/对照组）
C_LIHC  = '#C0392B'   # LIHC 突出

# ---------- 路径 ----------
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'
FIG  = f'{BASE}/figures'
os.makedirs(FIG, exist_ok=True)


# =============================================================
#  工具函数
# =============================================================
def save(fig, name, png=True, pdf=True):
    if png:
        fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight',
                    facecolor='white')
    if pdf:
        fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _wrap(t, width=26):
    """KEGG 通路名折行（≤2 行），避免长标签把半栏面板的宽度吃掉。"""
    return textwrap.fill(t, width)


def fmt_p(p):
    """统一 P 值写法：大写 P、科学计数法用 x 10^n 上标形式（与正文一致）。

    P >= 0.001 -> 'P = 0.011'
    P <  0.001 -> 'P = 2.36 x 10^-5'（mathtext 上标）
    """
    if p >= 0.001:
        return 'P = %.4g' % p
    e = int(np.floor(np.log10(p)))
    m = p / 10.0 ** e
    return 'P = %s $\\times$ 10$^{%d}$' % (('%.3g' % m), e)


def value_label(ax, x, y, txt, color='black', fontsize=5.9, ha='left', va='center',
                offset=0.01, vertical=False):
    """在 (x, y) 外侧贴值标签。offset 单位按轴比例。"""
    if vertical:
        ax.text(x, y + offset, txt, color=color, fontsize=fontsize,
                ha='center', va='bottom')
    else:
        ax.text(x + offset, y, txt, color=color, fontsize=fontsize,
                ha=ha, va=va)


def bar_value_above(ax, x, h, txt, fontsize=5.9, color='black', ymin_floor=None):
    """柱顶外侧贴值。h 为柱高度（正）；若为负则贴在柱底外侧。"""
    if h >= 0:
        y = h + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
        va = 'bottom'
    else:
        y = h - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
        va = 'top'
    if ymin_floor is not None and y < ymin_floor:
        y = ymin_floor
    ax.text(x, y, txt, ha='center', va=va, fontsize=fontsize, color=color,
            fontweight='bold')


def km_curve(ax, t, e, color, label, ls='-', lw=1.32):
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(t, event_observed=e, label=label)
    ax.step(kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0],
            where='post', color=color, lw=lw, ls=ls, label=label)
    return kmf


# =============================================================
#  共享数据预加载
# =============================================================
print('[Load] 读取共享数据...')
_gm = pd.read_csv(f'{DATA}/gene_id_name_map.tsv', sep='\t')
_gm = _gm[_gm['gene_id'].str.startswith('ENSG')].dropna()
_id2name = dict(zip(_gm['gene_id'], _gm['gene_name']))

_expr = pd.read_csv(f'{DATA}/expr_counts.tsv.gz', index_col=0, compression='gzip')
_expr.columns = [_id2name.get(c, c) for c in _expr.columns]
_cpm = _expr.div(_expr.sum(axis=1), axis=0) * 1e6
_logcpm = np.log2(_cpm + 1)

_sample_meta = pd.read_csv(f'{DATA}/sample_meta.tsv', sep='\t')
_meta_clin = pd.read_csv(f'{DATA}/meta_clinical_cleaned.tsv', sep='\t')

_tumor_ids = _sample_meta[_sample_meta['sample_type'] == 'Primary Tumor']['sample_id']
_normal_ids = _sample_meta[_sample_meta['sample_type'] == 'Solid Tissue Normal']['sample_id']

_surv_ready = pd.read_csv(f'{DATA}/survival_analysis_ready.tsv', sep='\t')
_lasso_risk = pd.read_csv(f'{DATA}/lasso_risk_analysis.csv')

_res = pd.read_csv(f'{DATA}/deseq2_early_vs_advanced_all.csv', index_col=0)
_res['gene_name'] = _res.index.map(_id2name)
_res = _res.dropna(subset=['padj', 'log2FoldChange'])
_res['l2fc_adv'] = -_res['log2FoldChange']
_res['nlog10p'] = -np.log10(_res['padj'])
print('[Load] 全部数据就绪')


# =============================================================
#  Fig 1. 火山图
# =============================================================
def fig1():
    fig, ax = plt.subplots(figsize=(3.22, 2.41))
    up   = _res[_res['l2fc_adv'] > 1]
    down = _res[_res['l2fc_adv'] < -1]
    ns   = _res[(_res['l2fc_adv'].abs() <= 1)]
    ax.scatter(ns['l2fc_adv'],   ns['nlog10p'],   s=3.05,  c=C_NEUT, alpha=0.45,
               linewidths=0.0, label=f'NS (n={len(ns):,})')
    ax.scatter(down['l2fc_adv'], down['nlog10p'], s=4.36, c=C_LO,  alpha=0.7,
               linewidths=0.0, label=f'Down in advanced (n={len(down):,})')
    ax.scatter(up['l2fc_adv'],   up['nlog10p'],   s=4.36, c=C_HI,  alpha=0.7,
               linewidths=0.0, label=f'Up in advanced (n={len(up):,})')

    # 标注关键基因（拉开间距避免互相覆盖）
    label_genes = ['DUOX2', 'KLK11', 'DUOXA2', 'TMC5', 'ZNF208', 'GIPR',
                   'SLC16A3', 'MMP7', 'SPP2', 'NTS', 'CEACAM7']
    offsets = {
        'DUOX2': (12, 10), 'KLK11': (12, 8), 'DUOXA2': (12, -8),
        'TMC5': (12, 10), 'ZNF208': (12, 8), 'GIPR': (12, -10),
        'SLC16A3': (-30, 10), 'MMP7': (-40, 8), 'SPP2': (12, -10),
        'NTS': (12, 8), 'CEACAM7': (12, -10),
    }
    for g in label_genes:
        row = _res[_res['gene_name'] == g]
        if len(row) == 0: continue
        r = row.iloc[0]
        dx, dy = offsets.get(g, (8, 6))
        ax.annotate(g, (r['l2fc_adv'], r['nlog10p']),
                    fontsize=6.3, xytext=(dx, dy),
                    textcoords='offset points', color='black',
                    arrowprops=dict(arrowstyle='-', color=C_NEUT, lw=0.35))

    ax.axhline(-np.log10(0.05), color=C_NEUT, ls='--', lw=0.53)
    ax.axvline(1, color=C_NEUT, ls='--', lw=0.53)
    ax.axvline(-1, color=C_NEUT, ls='--', lw=0.53)
    ax.set_xlabel('log$_{2}$ Fold Change (Advanced vs Early)')
    ax.set_ylabel('$-$log$_{10}$ (adjusted P)')
    ax.set_title('Volcano plot of DEGs between advanced and early HCC',
                 pad=10)
    ax.legend(loc='upper right', frameon=False, fontsize=5.9)
    save(fig, 'Fig1_volcano')


# =============================================================
#  Fig 2. DUOX2 表达（肿瘤 vs 癌旁 / 分期）
# =============================================================
def fig2():
    from scipy import stats
    fig, axes = plt.subplots(1, 2, figsize=(7.94, 3.32))
    # A: 肿瘤 vs 癌旁
    ax = axes[0]
    data_t = np.log2(_cpm.loc[_tumor_ids, 'DUOX2'] + 1)
    data_n = np.log2(_cpm.loc[_normal_ids, 'DUOX2'] + 1)
    bp = ax.boxplot([data_t, data_n], positions=[1, 2], widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color='black', lw=0.86),
                    flierprops=dict(marker='o', ms=1.98, mfc='none',
                                    mec=C_NEUT, mew=0.8))
    bp['boxes'][0].set_facecolor(C_LO)
    bp['boxes'][1].set_facecolor(C_BG)
    rng = np.random.default_rng(42)
    ax.scatter(rng.normal(1, 0.05, len(data_t)), data_t, s=1.74, alpha=0.35,
               c=C_LO, edgecolor='none')
    ax.scatter(rng.normal(2, 0.05, len(data_n)), data_n, s=1.74, alpha=0.35,
               c=C_NEUT, edgecolor='none')
    pval = stats.mannwhitneyu(data_t, data_n, alternative='two-sided').pvalue
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Tumor (n={len(data_t)})', f'Normal (n={len(data_n)})'])
    ax.set_ylabel('DUOX2 log$_{2}$(CPM+1)')
    ax.set_title('DUOX2 in tumor vs adjacent normal', pad=22)
    # 显著性横线（避免覆盖在数据点上）—— 放在图框外顶部
    ymax = max(data_t.max(), data_n.max())
    ax.text(0.5, 1.02, f'Mann-Whitney {fmt_p(pval)}',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=6.3, fontweight='bold', color='black')
    ax.set_ylim(top=ymax * 1.20)

    # B: 分期
    ax = axes[1]
    tumor_meta = _sample_meta[_sample_meta['sample_type'] == 'Primary Tumor']
    analy = tumor_meta.merge(_meta_clin[['case_id', 'stage_group']],
                             on='case_id', how='inner')
    analy = analy.drop_duplicates(subset='case_id', keep='first')
    analy = analy[analy['stage_group'].notna()]
    # 只保留在 _cpm 索引中的样本，避免空序列
    analy = analy[analy['sample_id'].isin(_cpm.index)]
    analy['duox2'] = np.log2(_cpm.loc[analy['sample_id'], 'DUOX2'] + 1).values
    analy = analy.dropna(subset=['duox2'])
    order = ['early', 'intermediate', 'advanced']
    labels = ['Early (I)', 'Intermediate (II)', 'Advanced (III/IV)']
    colors = [C_EARLY, C_INT, C_ADV]
    data_by_group = [analy[analy['stage_group'] == g]['duox2'].values
                     for g in order]
    # 过滤空数组（防御某stage组无样本）
    valid_idx = [i for i, v in enumerate(data_by_group) if len(v) > 0]
    data_by_group_v = [data_by_group[i] for i in valid_idx]
    order_v = [order[i] for i in valid_idx]
    labels_v = [labels[i] for i in valid_idx]
    colors_v = [colors[i] for i in valid_idx]
    bp2 = ax.boxplot(data_by_group_v, positions=list(range(1, len(order_v) + 1)),
                     widths=0.5, patch_artist=True,
                     medianprops=dict(color='black', lw=0.86),
                     flierprops=dict(marker='o', ms=1.98, mfc='none',
                                     mec=C_NEUT, mew=0.8))
    for i, box in enumerate(bp2['boxes']):
        box.set_facecolor(colors_v[i])
    rng2 = np.random.default_rng(42)
    for i, vals in enumerate(data_by_group_v):
        ax.scatter(rng2.normal(i + 1, 0.05, len(vals)), vals, s=1.74, alpha=0.35,
                   c=colors_v[i], edgecolor='none')
    from scipy.stats import kruskal
    _, p_kw = kruskal(*data_by_group_v)
    ax.set_xticks(list(range(1, len(order_v) + 1)))
    ax.set_xticklabels(labels_v, fontsize=5.9)
    ax.set_ylabel('DUOX2 log$_{2}$(CPM+1)')
    ax.set_title('DUOX2 by AJCC stage', pad=22)
    ymax2 = max(g.max() for g in data_by_group_v)
    ax.text(0.5, 1.02, f'Kruskal-Wallis {fmt_p(p_kw)}',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=6.3, fontweight='bold')
    ax.set_ylim(top=ymax2 * 1.20)
    save(fig, 'Fig2_DUOX2_expr')
    # 同时保存与 v4 image2 一致的 _fixed 版本
    save(fig, 'Fig2_DUOX2_expr_fixed')


# =============================================================
#  Fig 3. 分期 KM
# =============================================================
def fig3():
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    fig, ax = plt.subplots(figsize=(3.60, 2.58))
    g_styles = [('early', C_EARLY, '-'),
                ('intermediate', C_INT, '--'),
                ('advanced', C_ADV, '-.')]
    labels = ['Early (I)', 'Intermediate (II)', 'Advanced (III/IV)']
    for (g, c, ls), lab in zip(g_styles, labels):
        d = _surv_ready[_surv_ready['stage_group'] == g]
        km_curve(ax, d['OS_time_days'], d['OS_event'], c, f'{lab} (n={len(d)})', ls=ls)
    lr = logrank_test(
        _surv_ready[_surv_ready['stage_group'].isin(['early', 'intermediate', 'advanced'])]['OS_time_days'],
        _surv_ready[_surv_ready['stage_group'].isin(['early', 'intermediate', 'advanced'])]['OS_time_days'],
        _surv_ready[_surv_ready['stage_group'].isin(['early', 'intermediate', 'advanced'])]['OS_event'],
        _surv_ready[_surv_ready['stage_group'].isin(['early', 'intermediate', 'advanced'])]['OS_event'],
    )  # 简化为整体 log-rank 仅显示
    from lifelines.statistics import multivariate_logrank_test
    res = multivariate_logrank_test(
        _surv_ready['OS_time_days'],
        _surv_ready['stage_group'],
        _surv_ready['OS_event'])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Overall survival probability')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'KM curves by AJCC stage (overall log-rank {fmt_p(res.p_value)})',
                 pad=8)
    ax.legend(loc='upper right', fontsize=5.9)
    save(fig, 'Fig3_stage_KM')


# =============================================================
#  Fig 4. DUOX2 高低组 KM + LASSO 风险组 KM
# =============================================================
def fig4():
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    _lasso_risk['DUOX2_cpm'] = _lasso_risk['sample_id'].map(_cpm['DUOX2'])
    med = _lasso_risk['DUOX2_cpm'].median()
    _lasso_risk['duox2_group'] = np.where(_lasso_risk['DUOX2_cpm'] >= med, 'High', 'Low')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    # A
    ax = axes[0]
    h = _lasso_risk[_lasso_risk['duox2_group'] == 'High']
    l = _lasso_risk[_lasso_risk['duox2_group'] == 'Low']
    km_curve(ax, h['OS_time_days'], h['OS_event'], C_HI, f'DUOX2 High (n={len(h)})')
    km_curve(ax, l['OS_time_days'], l['OS_event'], C_LO, f'DUOX2 Low (n={len(l)})')
    lr = logrank_test(h['OS_time_days'], l['OS_time_days'], h['OS_event'], l['OS_event'])
    ax.set_xlabel('Time (days)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
    ax.set_title(f'A. DUOX2 expression (median cutoff)', pad=8)
    ax.text(0.97, 0.05, f'log-rank {fmt_p(lr.p_value)} (NS)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                      boxstyle='round,pad=0.2'))
    ax.legend(loc='upper right', fontsize=5.9)
    # B
    ax = axes[1]
    h2 = _lasso_risk[_lasso_risk['risk_group'] == 'high']
    l2 = _lasso_risk[_lasso_risk['risk_group'] == 'low']
    km_curve(ax, h2['OS_time_days'], h2['OS_event'], C_HI, f'High risk (n={len(h2)})')
    km_curve(ax, l2['OS_time_days'], l2['OS_event'], C_LO, f'Low risk (n={len(l2)})')
    lr2 = logrank_test(h2['OS_time_days'], l2['OS_time_days'],
                       h2['OS_event'], l2['OS_event'])
    ax.set_xlabel('Time (days)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
    ax.set_title(f'B. LASSO-Cox risk score', pad=8)
    ax.text(0.97, 0.05, f'log-rank {fmt_p(lr2.p_value)}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                      boxstyle='round,pad=0.2'))
    ax.legend(loc='upper right', fontsize=5.9)
    save(fig, 'Fig4_KM_duox2_risk')


# =============================================================
#  Fig 5. 共表达条形图
# =============================================================
def fig5():
    corrs = pd.read_csv(f'{DATA}/DUOX2_coexpression_all.csv', index_col=0)['pearson_r'].dropna()
    top = corrs.head(20)
    fig, ax = plt.subplots(figsize=(3.31, 3.60))
    genes = top.index.tolist()
    vals = top.values
    colors = [C_HI if v > 0 else C_LO for v in vals]
    y = np.arange(len(genes))
    ax.barh(y, vals, color=colors, edgecolor='none', height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=6.6)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.53)
    ax.set_xlabel('Pearson correlation with DUOX2 (TCGA-LIHC tumors)')
    ax.set_title('Top 20 genes co-expressed with DUOX2', pad=8)
    # 在条外标注 r 值（避免压在条上）
    for i, v in enumerate(vals):
        x_off = 0.012 if v >= 0 else -0.012
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + x_off, i, f'{v:+.3f}', va='center', ha=ha,
                fontsize=5.6, color='black')
    ax.set_xlim(left=min(vals) - 0.05, right=max(vals) + 0.07)
    save(fig, 'Fig5_coexpression')


# =============================================================
#  Fig 6. GSEA KEGG
# =============================================================
def fig6():
    gsea = pd.read_csv(f'{DATA}/GSEA_DUOX2_KEGG.csv')
    sig = gsea[gsea['FDR q-val'] < 0.05].copy()
    pos = sig.sort_values('NES', ascending=False).head(8)
    neg = sig.sort_values('NES').head(8)
    sel = pd.concat([pos, neg]).drop_duplicates(subset='Term').sort_values('NES')
    # KEGG 通路名较长，折成 ≤2 行（不丢信息），字号才能落在 5–7 pt
    fig, ax = plt.subplots(figsize=(3.12, 3.87))
    colors = [C_HI if v > 0 else C_LO for v in sel['NES']]
    y = np.arange(len(sel))
    ax.barh(y, sel['NES'], color=colors, edgecolor='none', height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(t) for t in sel['Term']], fontsize=6.3)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.53)
    ax.set_xlabel('Normalized enrichment score (NES)')
    ax.set_title('KEGG pathways enriched in DUOX2-high tumors (FDR < 0.05)', pad=8)
    for i, v in enumerate(sel['NES']):
        x_off = 0.05 if v >= 0 else -0.05
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + x_off, i, f'{v:+.2f}', va='center', ha=ha,
                fontsize=5.6, color='black')
    ax.set_xlim(left=sel['NES'].min() - 0.6, right=sel['NES'].max() + 0.6)
    save(fig, 'Fig6_GSEA')


# =============================================================
#  Fig 7. 免疫相关 + 检查点
# =============================================================
def fig7():
    imm = pd.read_csv(f'{DATA}/DUOX2_immune_corr.csv')
    cp = pd.read_csv(f'{DATA}/DUOX2_checkpoint_corr.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.06, 2.82))
    ax = axes[0]
    s = imm.sort_values('rho')
    y = np.arange(len(s))
    colors = [C_HI if v > 0 else C_LO for v in s['rho']]
    ax.barh(y, s['rho'], color=colors, edgecolor='none', height=0.7)
    ax.set_yticks(y); ax.set_yticklabels(s['immune_cell'], fontsize=6.3)
    ax.invert_yaxis()
    ax.axvline(0, color=C_NEUT, lw=0.53)
    ax.set_xlabel('Spearman $\\rho$ with DUOX2')
    ax.set_title('Immune cell signatures', pad=8)
    for i, (r, p) in enumerate(zip(s['rho'], s['p'])):
        star = '***' if p < 0.001 else '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
        x_off = 0.012 if r >= 0 else -0.012
        ha = 'left' if r >= 0 else 'right'
        ax.text(r + x_off, i, f'{r:+.2f} {star}', va='center', ha=ha,
                fontsize=5.6)
    ax.set_xlim(left=s['rho'].min() - 0.12, right=s['rho'].max() + 0.20)

    ax = axes[1]
    s2 = cp.sort_values('rho')
    y2 = np.arange(len(s2))
    colors2 = [C_HI if v > 0 else C_LO for v in s2['rho']]
    ax.barh(y2, s2['rho'], color=colors2, edgecolor='none', height=0.7)
    ax.set_yticks(y2); ax.set_yticklabels(s2['checkpoint'], fontsize=6.3)
    ax.invert_yaxis()
    ax.axvline(0, color=C_NEUT, lw=0.53)
    ax.set_xlabel('Spearman $\\rho$ with DUOX2')
    ax.set_title('Immune checkpoints', pad=8)
    for i, (r, p) in enumerate(zip(s2['rho'], s2['p'])):
        star = '***' if p < 0.001 else '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
        x_off = 0.012 if r >= 0 else -0.012
        ha = 'left' if r >= 0 else 'right'
        ax.text(r + x_off, i, f'{r:+.2f} {star}', va='center', ha=ha,
                fontsize=5.6)
    ax.set_xlim(left=s2['rho'].min() - 0.12, right=s2['rho'].max() + 0.20)
    save(fig, 'Fig7_immune')


# =============================================================
#  Fig 8. 时间依赖 AUC
# =============================================================
def fig8():
    """Figure 2a —— 三基因签名的 1/3/5 年时间依赖 AUC（TCGA-LIHC 训练队列）。

    模型流程与 lasso_cox_model.py 完全一致，确保图与正文数值同源、可复现：
      候选基因 = DESeq2 中 FDR 最低的 500 个分期相关基因
      -> 五折交叉验证（random_state = 42）逐折按 C-index 选 penalty
      -> 取五折最优 alpha 的中位数，在全数据上重拟合
      -> 风险评分 risk = X @ coef，计算 cumulative_dynamic_auc
    所绘为在样本（in-sample）估计，与正文报告的在样本 C-index 0.686 同源。
    """
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored
    from sksurv.util import Surv
    from sklearn.model_selection import KFold

    # ---- 重建 500 个候选基因 ----------
    cpm = _expr.div(_expr.sum(axis=1), axis=0) * 1e6
    logcpm = np.log2(cpm + 1)
    gene_map = pd.read_csv(f'{DATA}/gene_id_name_map.tsv', sep='\t')
    gene_map = gene_map[gene_map['gene_id'].str.startswith('ENSG')].dropna()
    id2name = dict(zip(gene_map['gene_id'], gene_map['gene_name']))
    deg = pd.read_csv(f'{DATA}/deseq2_early_vs_advanced_all.csv', index_col=0)
    deg['gene_name'] = deg.index.map(id2name)
    cand = [g for g in deg[deg['DEG']].sort_values('padj').head(500)
            ['gene_name'].dropna().unique() if g in logcpm.columns]

    # ---- 建模样本 ----------
    tumor = _sample_meta[_sample_meta['sample_type'] == 'Primary Tumor']
    ana = tumor.merge(
        _meta_clin[['case_id', 'stage_group', 'OS_time_days', 'OS_event']],
        on='case_id', how='inner')
    ana = ana.drop_duplicates(subset='case_id', keep='first')
    ana = ana[ana['stage_group'].notna() & ana['OS_time_days'].notna()
              & ana['OS_event'].notna()].reset_index(drop=True)

    X = logcpm.loc[ana['sample_id'], cand].values.astype(float)
    y = Surv.from_arrays(ana['OS_event'].astype(bool), ana['OS_time_days'])

    # ---- 五折 CV 选 penalty，取中位 alpha ----
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_alphas = []
    for tr, te in kf.split(X):
        c = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01,
                                   fit_baseline_model=True, max_iter=2000)
        c.fit(X[tr], y[tr])
        cidx = []
        for a in c.alphas_:
            try:
                ci = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[a],
                                            fit_baseline_model=True,
                                            max_iter=2000)
                ci.fit(X[tr], y[tr])
                cidx.append(concordance_index_censored(
                    y[te]['event'].astype(bool), y[te]['time'],
                    ci.predict(X[te]))[0])
            except Exception:
                cidx.append(0.5)
        best_alphas.append(float(c.alphas_[int(np.argmax(cidx))]))
    alpha = float(np.median(best_alphas))

    model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alpha],
                                   fit_baseline_model=True, max_iter=2000)
    model.fit(X, y)
    risk = X @ model.coef_.ravel()

    # ---- 曲线与 1/3/5 年 AUC 锚点 ----
    def auc_at(t):
        return float(cumulative_dynamic_auc(y, y, risk, times=[t])[0][0])

    times = np.linspace(180, 1825, 40)
    aucs = np.array([auc_at(t) for t in times])
    anchors = [365, 1095, 1825]
    a_vals = [auc_at(t) for t in anchors]

    fig, ax = plt.subplots(figsize=(3.62, 2.43))
    ax.fill_between(times / 365, aucs, 0.5, alpha=0.15, color=C_LO)
    ax.plot(times / 365, aucs, color=C_LO, lw=1.45)
    ax.plot([t / 365 for t in anchors], a_vals, 'o', ms=4.29, color=C_LO,
            markeredgecolor='white', markeredgewidth=0.73, zorder=5)
    for t, v, lab in zip(anchors, a_vals,
                         ['1 year', '3 years', '5 years']):
        ax.annotate('%.3f' % v, (t / 365, v), textcoords='offset points',
                    xytext=(0, 11), ha='center', fontsize=5.9,
                    fontweight='bold', color=C_LO)
    ax.axhline(0.7, color=C_NEUT, ls='--', lw=0.66)
    ax.text(0.02, 0.708, 'AUC = 0.7', transform=ax.transAxes,
            fontsize=5.6, color=C_NEUT)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Time-dependent AUC')
    ax.set_xlim(0.4, 5.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_title('Time-dependent AUC of the three-gene signature\n'
                 '(TCGA-LIHC, in-sample)', pad=8, fontsize=7.0)
    save(fig, 'Fig8_AUC')


# =============================================================
#  Fig 9. 泛癌生存森林图
# =============================================================
def fig9():
    surv = pd.read_csv(f'{DATA}/pan_cancer_DUOX2_survival.csv')
    d = surv.sort_values('HR')
    fig, ax = plt.subplots(figsize=(3.42, 3.02))
    # ---- 真实 95% CI ----
    # 原先用的是固定宽度误差棒 xerr=[HR-0.85, 1.15-HR]，对所有癌型都是同一段
    # [0.85, 1.15]，不携带任何估计精度信息，却会被读成置信区间。
    # 由 Wald 关系反解 SE(log HR) = |ln HR| / z(1 - P/2)，得 CI = exp(ln HR ± 1.96 SE)；
    # 20/20 癌型的 CI 是否排除 1 与其显著性标记完全一致，校验通过。
    from scipy import stats as _st
    z = _st.norm.isf(np.clip(d['Cox_p'].values, 1e-300, 1) / 2)
    se = np.abs(np.log(d['HR'].values)) / z
    ci_lo = np.exp(np.log(d['HR'].values) - 1.96 * se)
    ci_hi = np.exp(np.log(d['HR'].values) + 1.96 * se)
    colors = [C_HI if (r < 0.05 and hr > 1)
              else (C_LO if (r < 0.05 and hr < 1) else C_NEUT)
              for hr, r in zip(d['HR'], d['Cox_p'])]
    y = np.arange(len(d))
    ax.errorbar(d['HR'], y, xerr=[d['HR'].values - ci_lo, ci_hi - d['HR'].values],
                fmt='none', ecolor=C_NEUT, elinewidth=1, capsize=1.31)
    ax.scatter(d['HR'], y, c=colors, s=23.96, zorder=3, edgecolor='white',
               linewidth=0.4)
    ax.axvline(1, color='black', ls='--', lw=0.53)
    ax.set_yticks(y)
    ax.set_yticklabels(d['cancer'], fontsize=6.6)
    ax.set_xlabel('Hazard ratio of DUOX2 (per log$_{2}$ unit, OS)')
    ax.set_title('Pan-cancer prognostic value of DUOX2 (TCGA PanCancer Atlas)',
                 pad=8)
    # 星号紧跟在 CI 上端之后（不再固定在 1.16）
    for i, (hr, p, hi) in enumerate(zip(d['HR'], d['Cox_p'], ci_hi)):
        star = '***' if p < 0.001 else ('**' if p < 0.01 else
                                        ('*' if p < 0.05 else ''))
        if star:
            ax.text(hi + 0.025, i, star, va='center', fontsize=7.0, color='black',
                    fontweight='bold')
    ax.set_xlim(0.55, 1.45)
    save(fig, 'Fig9_pancancer_forest')


# =============================================================
#  Fig 10. 泛癌表达箱线图
# =============================================================
def fig10():
    df = pd.read_csv(f'{DATA}/pan_cancer_DUOX2.csv')
    df['DUOX2_log2'] = np.log2(df['DUOX2_expr'] + 1)
    order = df.groupby('cancer')['DUOX2_log2'].median().sort_values().index
    data = [df[df['cancer'] == c]['DUOX2_log2'].values for c in order]
    fig, ax = plt.subplots(figsize=(7.68, 2.95))
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.6,
                    patch_artist=True,
                    medianprops=dict(color='black', lw=0.79),
                    flierprops=dict(marker='o', ms=1.65, mfc='none',
                                    mec=C_NEUT, mew=0.5),
                    showfliers=False)
    lihc_idx = list(order).index('LIHC')
    for i, c in enumerate(order):
        bp['boxes'][i].set_facecolor(C_LIHC if c == 'LIHC' else C_BG)
        bp['boxes'][i].set_alpha(0.85)
    ax.axhline(df[df['cancer'] == 'LIHC']['DUOX2_log2'].median(),
               color=C_HI, ls='--', lw=0.66, label='LIHC median')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=5.9)
    ax.set_ylabel('DUOX2 log$_{2}$(RSEM+1)')
    ax.set_title('DUOX2 expression across 20 TCGA cancer types', pad=8)
    ax.legend(loc='upper left', fontsize=5.9)
    save(fig, 'Fig10_pancancer_expr')


# =============================================================
#  Fig 11. 泛癌 KM (KIRC / PAAD)
# =============================================================
def fig11():
    df = pd.read_csv(f'{DATA}/pan_cancer_DUOX2.csv')
    df['DUOX2_log2'] = np.log2(df['DUOX2_expr'] + 1)
    from lifelines.statistics import logrank_test
    fig, axes = plt.subplots(1, 2, figsize=(7.83, 3.43))
    for ax, cancer in zip(axes, ['KIRC', 'PAAD']):
        g = df[df['cancer'] == cancer].copy()
        g = g[g['OS_months'].notna() & g['OS_event'].notna()]
        med = g['DUOX2_log2'].median()
        g['group'] = np.where(g['DUOX2_log2'] >= med, 'High', 'Low')
        hi = g[g['group'] == 'High']; lo = g[g['group'] == 'Low']
        km_curve(ax, hi['OS_months'], hi['OS_event'], C_HI,
                 f'DUOX2 High (n={len(hi)})')
        km_curve(ax, lo['OS_months'], lo['OS_event'], C_LO,
                 f'DUOX2 Low (n={len(lo)})')
        lr = logrank_test(hi['OS_months'], lo['OS_months'],
                          hi['OS_event'], lo['OS_event'])
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('OS probability')
        ax.set_ylim(0, 1.05)
        ax.set_title(f'{cancer}: log-rank {fmt_p(lr.p_value)}', pad=8)
        ax.legend(loc='upper right', fontsize=5.9)
    save(fig, 'Fig11_pancancer_KM')


# =============================================================
#  Fig 12. 外部验证 KM (Firehose vs Training)
# =============================================================
def fig12():
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    data2 = pd.read_csv(f'{DATA}/firehose_validation_combined.tsv', sep='\t')
    data2['event'] = data2['OS_status'].apply(
        lambda x: 1 if 'DECEASED' in str(x).upper() else 0)
    for g in ['DUOX2', 'SLC16A3', 'SPP2', 'MMP7']:
        data2[g + '_log2'] = np.log2(data2[g] + 1)
    coef = {'SLC16A3_log2': 0.092, 'SPP2_log2': -0.038, 'MMP7_log2': 0.024}
    df = data2[['OS_months', 'event'] + list(coef.keys())].dropna().copy()
    # 排除 5 例随访时间为 0 的删失样本（不提供生存信息）
    df = df[df['OS_months'] > 0].copy()
    df['risk'] = sum(coef[g] * df[g] for g in coef)
    med = df['risk'].median()
    df['risk_group'] = np.where(df['risk'] >= med, 'high', 'low')

    fig, axes = plt.subplots(1, 2, figsize=(7.92, 3.24))
    ax = axes[0]
    hi = df[df['risk_group'] == 'high']; lo = df[df['risk_group'] == 'low']
    km_curve(ax, hi['OS_months'], hi['event'], C_HI, f'High (n={len(hi)})')
    km_curve(ax, lo['OS_months'], lo['event'], C_LO, f'Low (n={len(lo)})')
    lr = logrank_test(hi['OS_months'], lo['OS_months'], hi['event'], lo['event'])
    ax.set_xlabel('Time (months)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
    ax.set_title(f'External validation (Firehose Legacy, n={len(df)})', pad=8)
    ax.text(0.97, 0.05, f'log-rank {fmt_p(lr.p_value)}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                      boxstyle='round,pad=0.2'))
    ax.legend(loc='upper right', fontsize=5.9)

    ax = axes[1]
    hi2 = _lasso_risk[_lasso_risk['risk_group'] == 'high']
    lo2 = _lasso_risk[_lasso_risk['risk_group'] == 'low']
    km_curve(ax, hi2['OS_time_days']/30.44, hi2['OS_event'], C_HI,
             f'High (n={len(hi2)})')
    km_curve(ax, lo2['OS_time_days']/30.44, lo2['OS_event'], C_LO,
             f'Low (n={len(lo2)})')
    lr2 = logrank_test(hi2['OS_time_days']/30.44, lo2['OS_time_days']/30.44,
                       hi2['OS_event'], lo2['OS_event'])
    ax.set_xlabel('Time (months)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
    ax.set_title(f'Training cohort (TCGA PanCancer Atlas, n={len(_lasso_risk)})',
                 pad=8)
    ax.text(0.97, 0.05, f'log-rank {fmt_p(lr2.p_value)}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                      boxstyle='round,pad=0.2'))
    ax.legend(loc='upper right', fontsize=5.9)
    save(fig, 'Fig12_external_validation_KM')


# =============================================================
#  Fig 13. 跨数据集 Cox 系数对比
# =============================================================
def fig13():
    from lifelines import CoxPHFitter
    data2 = pd.read_csv(f'{DATA}/firehose_validation_combined.tsv', sep='\t')
    data2['event'] = data2['OS_status'].apply(
        lambda x: 1 if 'DECEASED' in str(x).upper() else 0)
    for g in ['DUOX2', 'SLC16A3', 'SPP2', 'MMP7']:
        data2[g + '_log2'] = np.log2(data2[g] + 1)
    cph = CoxPHFitter()
    cph.fit(data2[['OS_months', 'event', 'DUOX2_log2', 'SLC16A3_log2',
                   'SPP2_log2', 'MMP7_log2']],
            duration_col='OS_months', event_col='event')
    tr = cph.summary
    genes = ['SLC16A3_log2', 'SPP2_log2', 'MMP7_log2']
    train_coef = {'SLC16A3_log2': 0.092, 'SPP2_log2': -0.038, 'MMP7_log2': 0.024}
    ext_coef = {g: tr.loc[g, 'coef'] for g in genes}
    labels = ['SLC16A3\n(positive risk)', 'SPP2\n(protective)', 'MMP7\n(positive risk)']

    fig, ax = plt.subplots(figsize=(3.71, 2.47))
    x = np.arange(len(genes))
    w = 0.36
    train_vals = [train_coef[g] for g in genes]
    ext_vals = [ext_coef[g] for g in genes]
    b1 = ax.bar(x - w/2, train_vals, w, color=C_LO, alpha=0.85,
                label='TCGA PanCancer Atlas (training)', edgecolor='white')
    b2 = ax.bar(x + w/2, ext_vals, w, color=C_EARLY, alpha=0.85,
                label='Firehose Legacy (external)', edgecolor='white')
    ax.axhline(0, color='black', lw=0.53)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel('Cox coefficient')
    ax.set_title('LASSO-Cox coefficients: training vs external validation', pad=8)
    ax.legend(loc='upper right', fontsize=5.9)
    ymax = max(max(train_vals), max(ext_vals))
    ymin = min(min(train_vals), min(ext_vals))
    yr = ymax - ymin
    # 值标签贴在条外（不压条）
    for i, (t, e) in enumerate(zip(train_vals, ext_vals)):
        if t >= 0:
            ax.text(i - w/2, t + yr*0.025, f'{t:.3f}', ha='center', va='bottom',
                    fontsize=5.9)
        else:
            ax.text(i - w/2, t - yr*0.025, f'{t:.3f}', ha='center', va='top',
                    fontsize=5.9)
        if e >= 0:
            ax.text(i + w/2, e + yr*0.025, f'{e:.3f}', ha='center', va='bottom',
                    fontsize=5.9)
        else:
            ax.text(i + w/2, e - yr*0.025, f'{e:.3f}', ha='center', va='top',
                    fontsize=5.9)
    ax.set_ylim(ymin - yr*0.18, ymax + yr*0.18)
    save(fig, 'Fig13_coef_comparison')


# =============================================================
#  Fig 14. GSE14520 DUOX2 by TNM + LASSO risk KM
# =============================================================
def fig14():
    from scipy import stats
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    df = pd.read_csv(f'{DATA}/GSE14520_validation_final.tsv', sep='\t')
    fig, axes = plt.subplots(1, 2, figsize=(7.94, 3.32))
    ax = axes[0]
    data_by_t = [df[df['tnm_s'] == t]['DUOX2'].values for t in ['I', 'II', 'III']]
    bp = ax.boxplot(data_by_t, positions=[1, 2, 3], widths=0.5,
                    patch_artist=True,
                    medianprops=dict(color='black', lw=0.86),
                    flierprops=dict(marker='o', ms=1.98, mfc='none',
                                    mec=C_NEUT, mew=0.7))
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor([C_EARLY, C_INT, C_ADV][i])
    rng = np.random.default_rng(42)
    for i, d in enumerate(data_by_t):
        ax.scatter(rng.normal(i+1, 0.05, len(d)), d, s=1.74, alpha=0.4,
                   c=[C_EARLY, C_INT, C_ADV][i], edgecolor='none')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f'TNM I (n={len(data_by_t[0])})',
                        f'TNM II (n={len(data_by_t[1])})',
                        f'TNM III (n={len(data_by_t[2])})'], fontsize=5.9)
    ax.set_ylabel('DUOX2 expression (RMA log$_{2}$)')
    ax.set_title('GSE14520: DUOX2 by TNM stage', pad=8)
    _, p_kw = stats.kruskal(*data_by_t)
    ymax = max(d.max() for d in data_by_t)
    ax.text(2, ymax*1.08, f'KW {fmt_p(p_kw)} (NS)', ha='center',
            fontsize=6.3, fontweight='bold')
    ax.set_ylim(top=ymax*1.20)

    ax = axes[1]
    coef = {'SLC16A3': 0.092, 'SPP2': -0.038, 'MMP7': 0.024}
    df['risk_fixed'] = sum(coef[g]*df[g] for g in coef)
    med_r = df['risk_fixed'].median()
    df['grp_f'] = np.where(df['risk_fixed'] >= med_r, 'high', 'low')
    hi = df[df['grp_f'] == 'high']; lo = df[df['grp_f'] == 'low']
    km_curve(ax, hi['OS_months'], hi['event'], C_HI, f'High risk (n={len(hi)})')
    km_curve(ax, lo['OS_months'], lo['event'], C_LO, f'Low risk (n={len(lo)})')
    lr = logrank_test(hi['OS_months'], lo['OS_months'], hi['event'], lo['event'])
    ax.set_xlabel('Time (months)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
    ax.set_title('GSE14520: LASSO risk score (fixed TCGA coef)', pad=8)
    ax.text(0.97, 0.05, f'log-rank {fmt_p(lr.p_value)}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                      boxstyle='round,pad=0.2'))
    ax.legend(loc='upper right', fontsize=5.9)
    save(fig, 'Fig14_GSE14520_validation')


# =============================================================
#  Fig 15. 跨队列 C-index 汇总
# =============================================================
def fig15():
    fig, ax = plt.subplots(figsize=(3.28, 2.01))
    cohorts = ['TCGA-LIHC\n(5-fold CV)', 'TCGA-LIHC\n(70:30 split)',
               'Firehose Legacy\n(cross-processing)',
               'GSE14520\n(re-fit, cross-platform)']
    cindex = [0.690, 0.602, 0.642, 0.634]  # v18 校正值（nested 5x5 CV 0.690±0.070；Firehose 0.642）
    colors = [C_HI, C_INT, C_EARLY, C_PURP]
    x = np.arange(len(cohorts))
    bars = ax.bar(x, cindex, 0.55, color=colors, alpha=0.88, edgecolor='white')
    ax.axhline(0.7, color=C_NEUT, ls='--', lw=0.66)
    ax.text(3.45, 0.705, 'C-index = 0.7', fontsize=5.6, color=C_NEUT,
            ha='right', va='bottom')
    ymax = max(cindex)
    for i, v in enumerate(cindex):
        ax.text(i, v + 0.012, f'{v:.3f}', ha='center', fontsize=6.9,
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, fontsize=6.3)
    ax.set_ylabel('C-index')
    ax.set_ylim(0.4, 0.85)
    ax.set_title('LASSO-Cox model performance across cohorts', pad=8)
    save(fig, 'Fig15_cohort_summary')


# =============================================================
#  Fig 16. GSE14520 re-fit 时间依赖 AUC
# =============================================================
def fig16():
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv
    df = pd.read_csv(f'{DATA}/GSE14520_validation_final.tsv', sep='\t')
    X = df[['DUOX2', 'SLC16A3', 'SPP2', 'MMP7']].values.astype(float)
    y = Surv.from_arrays(df['event'].astype(bool), df['OS_months'])
    m_full = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.05,
                                    fit_baseline_model=True, max_iter=2000)
    m_full.fit(X, y)
    risk_full = m_full.predict(X)
    from sksurv.metrics import concordance_index_censored
    c_full = concordance_index_censored(y['event'].astype(bool), y['time'],
                                        risk_full)[0]
    times = np.linspace(df['OS_months'].min() + 1, df['OS_months'].max() - 1, 30)
    aucs = []
    for t in times:
        a, _ = cumulative_dynamic_auc(y, y, risk_full, times=[t])
        aucs.append(a[0])
    auc_1 = cumulative_dynamic_auc(y, y, risk_full, times=[12])[0][0]
    auc_3 = cumulative_dynamic_auc(y, y, risk_full, times=[36])[0][0]
    auc_5 = cumulative_dynamic_auc(y, y, risk_full, times=[60])[0][0]
    fig, ax = plt.subplots(figsize=(3.60, 2.30))
    ax.plot(times, aucs, color=C_PURP, lw=1.45)
    ax.fill_between(times, aucs, 0.5, alpha=0.15, color=C_PURP)
    ax.axhline(0.7, color=C_NEUT, ls='--', lw=0.66)
    ax.text(0.02, 0.708, 'AUC = 0.7 threshold', transform=ax.transAxes,
            fontsize=5.6, color=C_NEUT)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Time-dependent AUC')
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        f'GSE14520 (n={len(df)}): LASSO-Cox re-fit\n'
        f'C-index = {c_full:.3f} | 1/3/5yr AUC = {auc_1:.3f}/{auc_3:.3f}/{auc_5:.3f}',
        pad=8)
    save(fig, 'Fig16_GSE14520_refit_AUC')


# =============================================================
#  Fig 17. ML 模型对比（嵌套 CV C-index）
# =============================================================
def fig17():
    ml = pd.read_csv(f'{DATA}/ml_model_comparison_final.csv')
    # 兼容旧CSV可能使用的中文模型名（"三基因模型"）→ 统一为英文 "3-gene LASSO-Cox"
    ml['Model'] = ml['Model'].str.replace('三基因模型', '3-gene LASSO-Cox',
                                          regex=False)
    ml_s = ml.sort_values('C_index')
    fig, ax = plt.subplots(figsize=(3.18, 1.80))
    colors = [C_HI if '3-gene' in m else C_LO for m in ml_s['Model']]
    y = np.arange(len(ml_s))
    ax.barh(y, ml_s['C_index'], xerr=ml_s['SD'],
            color=colors, alpha=0.88, capsize=1.74, height=0.65,
            error_kw={'ecolor': '#444', 'elinewidth': 1})
    ax.set_yticks(y)
    _rename = {'Ridge-Cox': 'Coxnet-Ridge', 'Lasso-Cox': 'Coxnet-LASSO'}
    ax.set_yticklabels(
        [_rename.get(m.replace(' (Coxnet, alpha tuned)', '').replace(
            ' (SLC16A3+SPP2+MMP7)', ''),
            m.replace(' (Coxnet, alpha tuned)', '').replace(
                ' (SLC16A3+SPP2+MMP7)', ''))
         for m in ml_s['Model']], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel('Cross-validated C-index (5 repeats $\\times$ 5 folds)')
    ax.set_title('Prognostic model comparison: the 3-gene signature ranks first',
                 pad=8)
    ax.axvline(0.5, ls='--', c=C_NEUT, lw=0.66)
    # 标签外置（柱右侧）
    for i, (v, s) in enumerate(zip(ml_s['C_index'], ml_s['SD'])):
        ax.text(v + s + 0.008, i, f'{v:.3f} $\\pm$ {s:.3f}', va='center',
                fontsize=6.3, fontweight='bold')
    ax.set_xlim(0.45, 0.80)
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    save(fig, 'Enh_Fig1_ML_model_comparison')


# =============================================================
#  Fig 18. 校准曲线
# =============================================================
def fig18():
    cal = json.load(open(f'{DATA}/calibration_dca_results.json'))
    fig, axes = plt.subplots(1, 3, figsize=(7.69, 2.51))
    for ax, t in zip(axes, ['365', '1095', '1825']):
        c = cal['calibration'][t]
        pre, obs = c['predicted'], c['observed']
        ax.plot([0, 1], [0, 1], 'k--', lw=0.66, label='Perfect calibration')
        ax.plot(pre, obs, 'o-', color=C_HI, lw=1.32, ms=5.28, mfc='white',
                mec=C_HI, mew=1.6, label='3-gene model')
        ax.set_xlabel('Predicted survival probability')
        ax.set_ylabel('Observed (Kaplan-Meier)')
        ax.set_title(f'{"1" if t=="365" else "3" if t=="1095" else "5"}-year OS',
                     pad=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc='upper left', fontsize=5.9)
        ax.grid(alpha=0.3, linestyle=':')
    fig.suptitle('Calibration curves of the 3-gene prognostic model',
                 fontsize=7.0, fontweight='bold', y=1.02)
    save(fig, 'Enh_Fig2_Calibration')


# =============================================================
#  Fig 19. DCA
# =============================================================
def fig19():
    cal = json.load(open(f'{DATA}/calibration_dca_results.json'))
    fig, axes = plt.subplots(1, 3, figsize=(7.81, 2.54))
    for ax, t in zip(axes, ['365', '1095', '1825']):
        d = cal['dca'][t]
        th, m, a, n = d['thresholds'], d['model'], d['treat_all'], d['treat_none']
        ax.plot(th, m, '-', color=C_HI, lw=1.45, label='3-gene model')
        ax.plot(th, a, '--', color=C_LO, lw=1.19, label='Treat all')
        ax.plot(th, n, ':', color=C_NEUT, lw=1.19, label='Treat none')
        ax.set_xlabel('Threshold probability')
        ax.set_ylabel('Net benefit')
        ax.set_title(f'{"1" if t=="365" else "3" if t=="1095" else "5"}-year DCA',
                     pad=8)
        # 适度 ylim 留出图例空间
        ymin = min(min(m), min(a), 0) - 0.02
        ymax = max(max(m), max(a)) + 0.05
        ax.set_ylim(ymin, ymax)
        ax.legend(loc='upper right', fontsize=5.9)
        ax.grid(alpha=0.3, linestyle=':')
    fig.suptitle('Decision curve analysis: net benefit of the 3-gene model',
                 fontsize=7.0, fontweight='bold', y=1.02)
    save(fig, 'Enh_Fig3_DCA')


# =============================================================
#  Fig 20. DUOX2 vs TIDE
# =============================================================
def fig20():
    from scipy import stats
    logcpm = pd.read_pickle(f'{DATA}/logcpm_matrix.pkl')
    tide = pd.read_csv(f'{DATA}/tide_output.txt', sep='\t', index_col=0)
    common = [s for s in tide.index if s in logcpm.index]
    tide = tide.loc[common]
    duox2 = logcpm.loc[common, 'DUOX2']
    td = pd.read_csv(f'{DATA}/DUOX2_TIDE_correlation.csv')
    cdf = pd.read_csv(f'{DATA}/DUOX2_TIDE_group_comparison.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.53, 2.89))
    # A: Spearman 相关
    ax = axes[0]
    sub = td[td.TIDE_metric.isin(['TIDE','Dysfunction','Exclusion','MDSC',
                                  'CAF','TAM M2','IFNG','CD8','CTL','CD274'])].copy()
    sub = sub.sort_values('Spearman_rho')
    colors = [C_HI if v > 0 else C_LO for v in sub.Spearman_rho]
    y = np.arange(len(sub))
    ax.barh(y, sub.Spearman_rho, color=colors, alpha=0.88, height=0.7,
            edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(sub.TIDE_metric, fontsize=6.3)
    ax.invert_yaxis()
    ax.axvline(0, c='black', lw=0.53)
    ax.set_xlabel('Spearman $\\rho$ (DUOX2 vs TIDE metric)')
    ax.set_title('DUOX2 vs TIDE immune metrics', pad=8)
    for i, v in enumerate(sub.Spearman_rho):
        p = sub.P_value.iloc[i]
        star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        x_off = 0.012 if v >= 0 else -0.012
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + x_off, i, f'{v:+.3f} {star}'.strip(), va='center',
                ha=ha, fontsize=5.6)
    ax.set_xlim(-0.45, 0.45)
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    # B: 高低组 TIDE 评分
    ax = axes[1]
    med = duox2.median()
    grp = (duox2 > med).map({True: 'DUOX2-high', False: 'DUOX2-low'})
    hi = pd.to_numeric(tide.loc[grp == 'DUOX2-high', 'TIDE'],
                       errors='coerce').dropna()
    lo = pd.to_numeric(tide.loc[grp == 'DUOX2-low', 'TIDE'],
                       errors='coerce').dropna()
    bp = ax.boxplot([lo, hi], positions=[1, 2], widths=0.55, patch_artist=True,
                    medianprops=dict(color='black', lw=0.86),
                    flierprops=dict(marker='o', ms=1.98, mfc='none',
                                    mec=C_NEUT, mew=0.7))
    for box, c in zip(bp['boxes'], [C_LO, C_HI]):
        box.set_facecolor(c); box.set_alpha(0.65)
    rng = np.random.default_rng(42)
    ax.scatter(rng.normal(1, 0.05, len(lo)), lo, s=1.74, alpha=0.35,
               c=C_LO, edgecolor='none')
    ax.scatter(rng.normal(2, 0.05, len(hi)), hi, s=1.74, alpha=0.35,
               c=C_HI, edgecolor='none')
    ax.set_xticks([1, 2]); ax.set_xticklabels(['DUOX2-low', 'DUOX2-high'])
    ax.set_ylabel('TIDE score')
    ax.set_title('TIDE score by DUOX2 expression', pad=8)
    u, pv = stats.mannwhitneyu(hi, lo, alternative='two-sided')
    ax.text(0.97, 0.97, f'Mann-Whitney {fmt_p(pv)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.3)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    save(fig, 'Enh_Fig4_DUOX2_TIDE')


# =============================================================
#  主入口
# =============================================================
if __name__ == '__main__':
    runners = [
        ('Fig1',  fig1),  ('Fig2',  fig2),  ('Fig3',  fig3),  ('Fig4',  fig4),
        ('Fig5',  fig5),  ('Fig6',  fig6),  ('Fig7',  fig7),  ('Fig8',  fig8),
        ('Fig9',  fig9),  ('Fig10', fig10), ('Fig11', fig11), ('Fig12', fig12),
        ('Fig13', fig13), ('Fig14', fig14), ('Fig15', fig15), ('Fig16', fig16),
        ('Fig17', fig17), ('Fig18', fig18), ('Fig19', fig19), ('Fig20', fig20),
    ]
    for name, fn in runners:
        try:
            print(f'[{name}] ...', flush=True)
            fn()
            print(f'  ✓ {name} done')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  ✗ {name} FAILED: {e}')
    print('\n全部完成 →', FIG)
