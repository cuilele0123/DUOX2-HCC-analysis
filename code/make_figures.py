# -*- coding: utf-8 -*-
"""生成论文出版级图表（300dpi PNG + PDF）"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 11
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 8.5
rcParams['ytick.labelsize'] = 8.5
rcParams['legend.fontsize'] = 8.5
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'
FIG = f'{BASE}/figures'
os.makedirs(FIG, exist_ok=True)

C1, C2, C3, C4 = '#C0392B', '#2980B9', '#27AE60', '#8E44AD'  # 中国股市红涨绿跌风格保留

# ============ 图1：火山图 ============
print('图1 火山图...')
res = pd.read_csv(f'{DATA}/deseq2_early_vs_advanced_all.csv', index_col=0)
gene_map = pd.read_csv(f'{DATA}/gene_id_name_map.tsv', sep='\t')
gene_map = gene_map[gene_map['gene_id'].str.startswith('ENSG')].dropna()
id2name = dict(zip(gene_map['gene_id'], gene_map['gene_name']))
res['gene_name'] = res.index.map(id2name)

res = res.dropna(subset=['padj', 'log2FoldChange'])
# 晚期上调 = log2FC<0 (参考=advanced)。为符合阅读习惯，将方向转为"晚期 vs 早期"（晚期上调为正）
res['l2fc_adv'] = -res['log2FoldChange']  # 正值=晚期上调
res['nlog10p'] = -np.log10(res['padj'])

fig, ax = plt.subplots(figsize=(7, 5.5))
up = res[res['l2fc_adv'] > 1]
down = res[res['l2fc_adv'] < -1]
ns = res[(res['l2fc_adv'].abs() <= 1)]
ax.scatter(ns['l2fc_adv'], ns['nlog10p'], s=6, c='#BDC3C7', alpha=0.5, linewidths=0, label=f'NS (n={len(ns)})')
ax.scatter(down['l2fc_adv'], down['nlog10p'], s=8, c=C3, alpha=0.7, linewidths=0, label=f'Down in advanced (n={len(down)})')
ax.scatter(up['l2fc_adv'], up['nlog10p'], s=8, c=C1, alpha=0.7, linewidths=0, label=f'Up in advanced (n={len(up)})')
# 标注关键基因
for g in ['DUOX2', 'KLK11', 'DUOXA2', 'TMC5', 'ZNF208', 'GIPR', 'SLC16A3', 'MMP7', 'SPP2', 'NTS', 'CEACAM7']:
    row = res[res['gene_name'] == g]
    if len(row):
        r = row.iloc[0]
        ax.annotate(g, (r['l2fc_adv'], r['nlog10p']), fontsize=8, xytext=(5, 5),
                    textcoords='offset points', color='black')
ax.axhline(-np.log10(0.05), color='grey', ls='--', lw=0.8)
ax.axvline(1, color='grey', ls='--', lw=0.8); ax.axvline(-1, color='grey', ls='--', lw=0.8)
ax.set_xlabel('log2 Fold Change (Advanced vs Early)')
ax.set_ylabel('-log10 (adjusted P)')
ax.set_title('Volcano plot of DEGs between advanced and early HCC')
ax.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.savefig(f'{FIG}/Fig1_volcano.png', dpi=300)
plt.savefig(f'{FIG}/Fig1_volcano.pdf')
plt.close()

# ============ 图2：DUOX2 表达 ============
print('图2 DUOX2 表达...')
expr = pd.read_csv(f'{DATA}/expr_counts.tsv.gz', index_col=0, compression='gzip')
sample_meta = pd.read_csv(f'{DATA}/sample_meta.tsv', sep='\t')
expr.columns = [id2name.get(c, c) for c in expr.columns]
cpm = expr.div(expr.sum(axis=1), axis=0) * 1e6
meta = pd.read_csv(f'{DATA}/meta_clinical_cleaned.tsv', sep='\t')
tumor_ids = sample_meta[sample_meta['sample_type'] == 'Primary Tumor']['sample_id']
normal_ids = sample_meta[sample_meta['sample_type'] == 'Solid Tissue Normal']['sample_id']

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
# A: 肿瘤 vs 癌旁
ax = axes[0]
data_t = np.log2(cpm.loc[tumor_ids, 'DUOX2'] + 1)
data_n = np.log2(cpm.loc[normal_ids, 'DUOX2'] + 1)
bp = ax.boxplot([data_t, data_n], positions=[1, 2], widths=0.5, patch_artist=True,
                medianprops=dict(color='black', lw=1.2))
bp['boxes'][0].set_facecolor(C2); bp['boxes'][1].set_facecolor('#ECF0F1')
ax.scatter(np.random.normal(1, 0.06, len(data_t)), data_t, s=4, alpha=0.4, c=C2)
ax.scatter(np.random.normal(2, 0.06, len(data_n)), data_n, s=4, alpha=0.4, c='grey')
from scipy import stats
pval = stats.mannwhitneyu(data_t, data_n).pvalue
ax.set_xticks([1, 2]); ax.set_xticklabels([f'Tumor (n={len(data_t)})', f'Normal (n={len(data_n)})'])
ax.set_ylabel('DUOX2 log2(CPM+1)')
ax.set_title(f'A. DUOX2 in tumor vs normal\nMann-Whitney p={pval:.3f}')
# B: 分期
ax = axes[1]
tumor_meta = sample_meta[sample_meta['sample_type'] == 'Primary Tumor']
analy = tumor_meta.merge(meta[['case_id', 'stage_group']], on='case_id', how='inner')
analy = analy.drop_duplicates(subset='case_id', keep='first')
analy = analy[analy['stage_group'].notna()]
analy['duox2'] = np.log2(cpm.loc[analy['sample_id'], 'DUOX2'] + 1)
order = ['early', 'intermediate', 'advanced']
labels = ['Early (I)', 'Intermediate (II)', 'Advanced (III/IV)']
colors = [C3, '#F39C12', C1]
data_by_group = [analy[analy['stage_group'] == g]['duox2'].values for g in order]
bp2 = ax.boxplot(data_by_group, positions=[1, 2, 3], widths=0.5, patch_artist=True,
                 medianprops=dict(color='black', lw=1.2))
for i, box in enumerate(bp2['boxes']):
    box.set_facecolor(colors[i])
for i, g in enumerate(order):
    vals = data_by_group[i]
    ax.scatter(np.random.normal(i + 1, 0.06, len(vals)), vals, s=4, alpha=0.4, c=colors[i])
from scipy.stats import kruskal
h, p_kw = kruskal(*data_by_group)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('DUOX2 log2(CPM+1)')
ax.set_title(f'B. DUOX2 by AJCC stage\nKruskal-Wallis p={p_kw:.4f}')
plt.tight_layout()
plt.savefig(f'{FIG}/Fig2_DUOX2_expr.png', dpi=300)
plt.savefig(f'{FIG}/Fig2_DUOX2_expr.pdf')
plt.close()

# ============ 图3：分期 KM ============
print('图3 分期KM...')
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
surv_ready = pd.read_csv(f'{DATA}/survival_analysis_ready.tsv', sep='\t')

fig, ax = plt.subplots(figsize=(6.5, 5))
kmf = KaplanMeierFitter()
g_styles = [('early', C3, '-'), ('intermediate', '#F39C12', '--'), ('advanced', C1, '-.')]
for g, c, ls in g_styles:
    d = surv_ready[surv_ready['stage_group'] == g]
    kmf.fit(d['OS_time_days'], event_observed=d['OS_event'], label=labels[order.index(g)])
    ax.step(kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0],
            where='post', color=c, lw=2, ls=ls, label=f'{labels[order.index(g)]} (n={len(d)})')
ax.set_xlabel('Time (days)'); ax.set_ylabel('Overall survival probability')
ax.set_ylim(0, 1.05)
ax.set_title('KM curves by AJCC stage\noverall log-rank p=1.77e-03')
ax.legend(frameon=False, loc='upper right')
plt.tight_layout()
plt.savefig(f'{FIG}/Fig3_stage_KM.png', dpi=300)
plt.savefig(f'{FIG}/Fig3_stage_KM.pdf')
plt.close()

# ============ 图4：DUOX2 高低组 KM ============
print('图4 DUOX2 KM...')
lasso_risk = pd.read_csv(f'{DATA}/lasso_risk_analysis.csv')
duox2_analy = pd.read_csv(f'{DATA}/lasso_risk_analysis.csv')
# 从 lasso_risk_analysis 重建 DUOX2 分组（含 DUOX2_cpm 需重新计算）
expr2 = pd.read_csv(f'{DATA}/expr_counts.tsv.gz', index_col=0, compression='gzip')
expr2.columns = [id2name.get(c, c) for c in expr2.columns]
cpm2 = expr2.div(expr2.sum(axis=1), axis=0) * 1e6
lasso_risk['DUOX2_cpm'] = lasso_risk['sample_id'].map(cpm2['DUOX2'])
med = lasso_risk['DUOX2_cpm'].median()
lasso_risk['duox2_group'] = np.where(lasso_risk['DUOX2_cpm'] >= med, 'High', 'Low')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
# A: DUOX2 高低组
ax = axes[0]
kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
h = lasso_risk[lasso_risk['duox2_group'] == 'High']; l = lasso_risk[lasso_risk['duox2_group'] == 'Low']
kmf_h.fit(h['OS_time_days'], h['OS_event']); kmf_l.fit(l['OS_time_days'], l['OS_event'])
ax.step(kmf_h.survival_function_.index, kmf_h.survival_function_.iloc[:, 0], where='post', color=C1, lw=2, label=f'DUOX2 High (n={len(h)})')
ax.step(kmf_l.survival_function_.index, kmf_l.survival_function_.iloc[:, 0], where='post', color=C2, lw=2, label=f'DUOX2 Low (n={len(l)})')
lr = logrank_test(h['OS_time_days'], l['OS_time_days'], h['OS_event'], l['OS_event'])
ax.set_xlabel('Time (days)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
ax.set_title(f'A. DUOX2 expression (median cutoff)\nlog-rank p={lr.p_value:.4f} (NS)')
ax.legend(frameon=False)
# B: LASSO 风险组
ax = axes[1]
kmf_h2 = KaplanMeierFitter(); kmf_l2 = KaplanMeierFitter()
h2 = lasso_risk[lasso_risk['risk_group'] == 'high']; l2 = lasso_risk[lasso_risk['risk_group'] == 'low']
kmf_h2.fit(h2['OS_time_days'], h2['OS_event']); kmf_l2.fit(l2['OS_time_days'], l2['OS_event'])
ax.step(kmf_h2.survival_function_.index, kmf_h2.survival_function_.iloc[:, 0], where='post', color=C1, lw=2, label=f'High risk (n={len(h2)})')
ax.step(kmf_l2.survival_function_.index, kmf_l2.survival_function_.iloc[:, 0], where='post', color=C2, lw=2, label=f'Low risk (n={len(l2)})')
lr2 = logrank_test(h2['OS_time_days'], l2['OS_time_days'], h2['OS_event'], l2['OS_event'])
ax.set_xlabel('Time (days)'); ax.set_ylabel('OS probability'); ax.set_ylim(0, 1.05)
ax.set_title(f'B. LASSO-Cox risk score\nlog-rank p={lr2.p_value:.2e}')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'{FIG}/Fig4_KM_duox2_risk.png', dpi=300)
plt.savefig(f'{FIG}/Fig4_KM_duox2_risk.pdf')
plt.close()

# ============ 图5：共表达热图 ============
print('图5 共表达热图...')
corrs = pd.read_csv(f'{DATA}/DUOX2_coexpression_all.csv', index_col=0)['pearson_r'].dropna()
top = corrs.head(20)
fig, ax = plt.subplots(figsize=(6, 5))
genes = top.index.tolist()
vals = top.values
colors_bar = [C1 if v > 0 else C3 for v in vals]
ax.barh(range(len(genes)), vals, color=colors_bar, edgecolor='none')
ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=8.5)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Pearson correlation with DUOX2')
ax.set_title('Top 20 genes co-expressed with DUOX2 in TCGA-LIHC')
plt.tight_layout()
plt.savefig(f'{FIG}/Fig5_coexpression.png', dpi=300)
plt.savefig(f'{FIG}/Fig5_coexpression.pdf')
plt.close()

# ============ 图6：GSEA 关键通路 ============
print('图6 GSEA...')
gsea = pd.read_csv(f'{DATA}/GSEA_DUOX2_KEGG.csv')
sig = gsea[gsea['FDR q-val'] < 0.05].copy()
pos = sig.sort_values('NES', ascending=False).head(8)
neg = sig.sort_values('NES').head(8)
sel = pd.concat([pos, neg]).drop_duplicates(subset='Term')
sel = sel.sort_values('NES')
fig, ax = plt.subplots(figsize=(8, 6.5))
colors_bar = [C1 if v > 0 else C3 for v in sel['NES']]
ax.barh(range(len(sel)), sel['NES'], color=colors_bar)
ax.set_yticks(range(len(sel)))
ax.set_yticklabels([t[:48] for t in sel['Term']], fontsize=7.5)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Normalized enrichment score (NES)')
ax.set_title('KEGG pathways enriched in DUOX2-high tumors (FDR<0.05)')
plt.tight_layout()
plt.savefig(f'{FIG}/Fig6_GSEA.png', dpi=300)
plt.savefig(f'{FIG}/Fig6_GSEA.pdf')
plt.close()

# ============ 图7：免疫相关 ============
print('图7 免疫相关...')
imm = pd.read_csv(f'{DATA}/DUOX2_immune_corr.csv')
cp = pd.read_csv(f'{DATA}/DUOX2_checkpoint_corr.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
imm_sorted = imm.sort_values('rho')
ax.barh(imm_sorted['immune_cell'], imm_sorted['rho'],
        color=[C1 if v > 0 else C3 for v in imm_sorted['rho']])
ax.set_xlabel('Spearman rho with DUOX2')
ax.set_title('A. Immune cell signatures')
for i, (r, p) in enumerate(zip(imm_sorted['rho'], imm_sorted['p'])):
    star = '*' if p < 0.05 else 'ns'
    ax.text(r + 0.005, i, star, va='center', fontsize=8)
ax.axvline(0, color='grey', lw=0.8)
ax = axes[1]
cp_sorted = cp.sort_values('rho')
ax.barh(cp_sorted['checkpoint'], cp_sorted['rho'],
        color=[C1 if v > 0 else C3 for v in cp_sorted['rho']])
ax.set_xlabel('Spearman rho with DUOX2')
ax.set_title('B. Immune checkpoints')
for i, (r, p) in enumerate(zip(cp_sorted['rho'], cp_sorted['p'])):
    star = '*' if p < 0.05 else 'ns'
    ax.text(r + 0.005, i, star, va='center', fontsize=8)
ax.axvline(0, color='grey', lw=0.8)
plt.tight_layout()
plt.savefig(f'{FIG}/Fig7_immune.png', dpi=300)
plt.savefig(f'{FIG}/Fig7_immune.pdf')
plt.close()

# ============ 图8：LASSO 时间依赖 AUC ============
print('图8 AUC...')
# 重算时间依赖AUC（图8）
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
lasso_risk2 = pd.read_csv(f'{DATA}/lasso_risk_analysis.csv')
genes_model = pd.read_csv(f'{DATA}/lasso_cox_model_genes.csv')
mod_genes = genes_model['gene'].tolist()
logcpm2 = np.log2(cpm2 + 1)
X2 = logcpm2.loc[lasso_risk2['sample_id'], mod_genes].values.astype(float)
y2 = Surv.from_arrays(lasso_risk2['OS_event'].astype(bool), lasso_risk2['OS_time_days'])
model = CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True, max_iter=2000)
model.fit(X2, y2)
risk = model.predict(X2)
from sksurv.metrics import cumulative_dynamic_auc
times = np.linspace(100, 1825, 30)
try:
    aucs = []
    for t in times:
        auc, _ = cumulative_dynamic_auc(y2, y2, risk, times=[t])
        aucs.append(auc[0])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(times / 365, aucs, color=C2, lw=2.2)
    ax.fill_between(times / 365, aucs, 0.5, alpha=0.15, color=C2)
    ax.axhline(0.7, color='grey', ls='--', lw=1)
    ax.text(0.1, 0.708, 'AUC=0.7 threshold', fontsize=8, color='grey')
    ax.set_xlabel('Time (years)'); ax.set_ylabel('Time-dependent AUC')
    ax.set_ylim(0.4, 1.0)
    ax.set_title('LASSO-Cox prognostic model: time-dependent AUC')
    plt.tight_layout()
    plt.savefig(f'{FIG}/Fig8_AUC.png', dpi=300)
    plt.savefig(f'{FIG}/Fig8_AUC.pdf')
    plt.close()
    print('  AUC 1yr/3yr/5yr:', [f'{a:.3f}' for a in [aucs[np.argmin(abs(times-365))], aucs[np.argmin(abs(times-1095))], aucs[np.argmin(abs(times-1825))]]])
except Exception as e:
    print('AUC err:', e)

print('\n全部图表生成完成:', FIG)
