# -*- coding: utf-8 -*-
"""
Figure 21 — 肿瘤含量（上皮/基质比例）校正后的 DUOX2-TIDE 关联
4 面板，与全文图风格统一（TNR / 300dpi / 统一配色）

A  校正前后 rho 对比（哑铃图，E/S ratio 为协变量）
B  过度校正诊断（TIDE 与各细胞成分评分的共享方差 r2）
C  分层验证（按 E/S ratio 三分位）
D  细胞来源定量验证（DUOX2 与上皮/基质/免疫评分的相关强度 + bootstrap 差异）
"""
import os, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# Scientific Reports 图片规范：clear sans-serif（Helvetica / Arial）
for p in ['/System/Library/Fonts/Helvetica.ttc',
          '/System/Library/Fonts/Supplemental/Arial.ttf',
          '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
          '/System/Library/Fonts/Supplemental/Arial Italic.ttf',
          '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf']:
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.unicode_minus'] = False
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

C_HI = '#C0392B'
C_LO = '#2980B9'
C_EARLY = '#27AE60'
C_INT = '#F39C12'
C_PURP = '#8E44AD'
C_NEUT = '#7F8C8D'
C_BG = '#ECF0F1'

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'
FIG = f'{BASE}/figures'
os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def ci_spearman(r, n):
    """Fisher z 近似 95%CI"""
    if abs(r) >= 1:
        return r, r
    zf = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return np.tanh(zf - 1.96 * se), np.tanh(zf + 1.96 * se)


df = pd.read_csv(f'{DATA}/purity_scores_tide.csv', index_col=0)
summ = json.load(open(f'{DATA}/purity_adjusted_summary.json'))
adj = pd.read_csv(f'{DATA}/DUOX2_TIDE_purity_adjusted.csv')
strat = pd.read_csv(f'{DATA}/DUOX2_TIDE_stratified_purity.csv')
n = len(df)


def partial_spearman(x, y, covars):
    C = np.asarray(covars, float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    nn, k = len(x), C.shape[1]
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Cr = sm.add_constant(np.column_stack([stats.rankdata(C[:, j]) for j in range(k)]))
    rxr = sm.OLS(rx, Cr).fit().resid
    ryr = sm.OLS(ry, Cr).fit().resid
    r = np.corrcoef(rxr, ryr)[0, 1]
    d = nn - 2 - k
    t = r * np.sqrt(d / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), d)


fig, axes = plt.subplots(2, 2, figsize=(7.13, 5.51))

# ---------------- A. 校正前后哑铃图 ----------------
ax = axes[0, 0]
show = ['TIDE', 'Dysfunction', 'Exclusion', 'CAF', 'CD8', 'IFNG',
        'TAM M2', 'MSI Score']
ypos = np.arange(len(show))[::-1]
ax.axvline(0, color=C_NEUT, lw=0.66, ls='--', zorder=1)
for i, m in enumerate(show):
    row = adj[adj['metric'] == m].iloc[0]
    r0, r1 = row['rho_raw'], row['rho_ES_ratio']
    ax.plot([r0, r1], [ypos[i], ypos[i]], color='#BDC3C7', lw=1.45, zorder=2)
    ax.scatter([r0], [ypos[i]], s=22.65, color=C_NEUT, zorder=3,
               edgecolor='white', linewidth=0.53)
    p1 = row['p_ES_ratio']
    ax.scatter([r1], [ypos[i]], s=28.75, color=C_HI, zorder=4,
               marker='D', edgecolor='white', linewidth=0.53)
    star = '**' if p1 < 0.01 else ('*' if p1 < 0.05 else 'NS')
    xr = ax.get_xlim()
    ax.text(0.50, ypos[i], f'{r1:+.3f}{star}', va='center', ha='left',
            fontsize=5.7, color=C_HI)
ax.set_yticks(ypos)
ax.set_yticklabels(show, fontsize=6.3)
ax.set_xlim(-0.50, 0.60)
ax.set_xlabel('Spearman rho with DUOX2 expression')
ax.set_title('Association before vs. after adjustment', pad=10)
ax.scatter([], [], s=22.65, color=C_NEUT, label='Unadjusted', edgecolor='white', linewidth=0.53)
ax.scatter([], [], s=28.75, color=C_HI, marker='D', label='Adjusted for E/S ratio',
           edgecolor='white', linewidth=0.53)
ax.legend(loc='upper left', fontsize=5.9, frameon=False, ncol=1,
          bbox_to_anchor=(0.005, 0.34))

# ---------------- B. 过度校正诊断 ----------------
ax = axes[0, 1]
vars_ = ['epi', 'stromal', 'immune', 'ES_ratio', 'ESTIMATE_like']
labels = ['Epithelial', 'Stromal', 'Immune', 'E/S ratio', 'ESTIMATE-like']
r2 = []
for v in vars_:
    r, _ = stats.spearmanr(df['TIDE'], df[v])
    r2.append(r ** 2)
r2 = np.array(r2)
cols = [C_EARLY, C_INT, C_INT, C_EARLY, C_HI]
bars = ax.bar(np.arange(len(vars_)), r2 * 100, color=cols, width=0.62,
              edgecolor='white', linewidth=0.53)
for i, (b, v) in enumerate(zip(bars, r2)):
    ax.text(i, (v * 100) + 1.6, f'{v*100:.1f}%', ha='center', va='bottom',
            fontsize=6.1, fontweight='bold',
            color=C_HI if v > 0.25 else '#2C3E50')
ax.set_xticks(np.arange(len(vars_)))
ax.set_xticklabels(labels, fontsize=6.3)
ax.set_ylabel('Shared variance with TIDE score (r$^{2}$, %)')
ax.set_title('Why E/S ratio is the appropriate covariate', pad=10)
# 注释框用白底覆盖绘制，会盖住最高柱顶部的百分比标签；
# 抬高上限并把框移到顶部，使两者不再重叠。
ax.set_ylim(0, max(r2 * 100) * 1.62)
ax.text(0.5, 0.97,
        'Stromal / immune scores share 28-49% variance with TIDE;\n'
        'adjusting for them removes TIDE itself (over-adjustment).\n'
        'E/S ratio shares only 9.2% and is therefore appropriate.',
        transform=ax.transAxes, ha='center', va='top', fontsize=5.8,
        color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.5', fc='#FDFEFE', ec='#BDC3C7', lw=0.53))

# ---------------- C. 分层验证 ----------------
ax = axes[1, 0]
yp = np.arange(len(strat))[::-1]
ax.axvline(0, color=C_NEUT, lw=0.66, ls='--', zorder=1)
for i, row in strat.iterrows():
    r, p, nn = row['rho'], row['p'], int(row['n'])
    lo, hi = ci_spearman(r, nn)
    col = C_HI if p < 0.05 else C_NEUT
    ax.plot([lo, hi], [yp[i], yp[i]], color=col, lw=1.32, zorder=2)
    ax.scatter([r], [yp[i]], s=38.33, color=col, marker='s', zorder=3,
               edgecolor='white', linewidth=0.53)
    star = '*' if p < 0.05 else 'NS'
    ax.text(0.46, yp[i], f'rho={r:+.3f} (p={p:.3f}) {star}  n={nn}',
            va='center', ha='left', fontsize=5.7, color=col)
ax.set_yticks(yp)
ax.set_yticklabels([f'{g} E/S' for g in strat['group']], fontsize=6.3)
ax.set_xlim(-0.12, 0.78)
ax.set_xlabel('Spearman rho (DUOX2 vs. TIDE) within stratum')
ax.set_title('Stratified by epithelial/stromal ratio', pad=10)
ax.text(0.5, -0.20, f'DUOX2 x stratum interaction p = {summ["interaction_p"]:.3f} (no heterogeneity)',
        transform=ax.transAxes, ha='center', va='top', fontsize=5.8, color=C_NEUT)

# ---------------- D. 细胞来源定量验证 ----------------
ax = axes[1, 1]
pairs = [('epi', 'Epithelial', C_HI), ('stromal', 'Stromal', C_INT),
         ('immune', 'Immune', C_LO)]
rhos, los, his, names, cols = [], [], [], [], []
for v, lab, col in pairs:
    r, _ = stats.spearmanr(df['DUOX2'], df[v])
    lo, hi = ci_spearman(r, n)
    rhos.append(r); los.append(lo); his.append(hi); names.append(lab); cols.append(col)
xp = np.arange(len(pairs))
ax.bar(xp, rhos, color=cols, width=0.56, edgecolor='white', linewidth=0.53)
for i in range(len(pairs)):
    ax.plot([xp[i], xp[i]], [los[i], his[i]], color='#2C3E50', lw=0.99)
    ax.plot([xp[i] - 0.07, xp[i] + 0.07], [his[i], his[i]], color='#2C3E50', lw=0.99)
    ax.plot([xp[i] - 0.07, xp[i] + 0.07], [los[i], los[i]], color='#2C3E50', lw=0.99)
    ax.text(xp[i], his[i] + 0.022, f'{rhos[i]:+.3f}', ha='center', va='bottom',
            fontsize=6.2, fontweight='bold', color=cols[i])
ax.set_xticks(xp)
ax.set_xticklabels(names, fontsize=6.5)
ax.set_ylabel('Spearman rho with DUOX2 expression')
ax.set_title('Cell-of-origin: DUOX2 tracks the epithelial compartment',
             pad=10)
ax.set_ylim(0, max(his) * 1.72)   # 同上：给注释框留出空间
bs = summ['L1b']['diff_epi_immune']
ax.text(0.5, 0.97,
        f'rho(epithelial) - rho(immune) = {bs["d"]:+.3f}\n'
        f'95% CI [{bs["lo"]:+.3f}, {bs["hi"]:+.3f}], bootstrap p = {bs["p"]:.3f}\n'
        'Consistent with HPA: DUOX2 not detected in immune cells',
        transform=ax.transAxes, ha='center', va='top', fontsize=5.8,
        color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.5', fc='#FDFEFE', ec='#BDC3C7', lw=0.53))

plt.tight_layout(h_pad=2.6, w_pad=2.4)
save(fig, 'Fig21_purity_adjusted_TIDE')
print('Figure 21 saved.')
