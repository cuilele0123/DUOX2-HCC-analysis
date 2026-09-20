# -*- coding: utf-8 -*-
"""
v16 补充：GSE14520 固定系数方向检验 + 交叉验证 C-index
问题：TCGA 固定系数在 GSE14520 上产生显著但方向反转的分离，
      文中 "validated in one of two independent cohorts" 的表述需要修正。
"""
import os, json
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')
RNG = np.random.default_rng(20260914)
COEF = {'SLC16A3': 0.092, 'SPP2': -0.038, 'MMP7': 0.024}
GENES = list(COEF)
OUT = {}


def c_index(t, e, s):
    t = np.asarray(t, float); e = np.asarray(e, float); s = np.asarray(s, float)
    return float(concordance_index_censored(e.astype(bool), t, s)[0])


def boot_ci(t, e, s, n=2000):
    t = np.asarray(t, float); e = np.asarray(e, float); s = np.asarray(s, float)
    vals = []; k = len(t)
    for _ in range(n):
        idx = RNG.integers(0, k, k)
        if e[idx].sum() < 5:
            continue
        vals.append(concordance_index_censored(e[idx].astype(bool), t[idx], s[idx])[0])
    return (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3))


ge = pd.read_csv(os.path.join(DATA, 'GSE14520_gene_expr.csv'), index_col=0)
clin = pd.read_csv(os.path.join(DATA, 'GSE14520_Extra_Supplement.txt.gz'), sep='\t', compression='gzip')
clin['gsm'] = clin['Affy_GSM'].astype(str)
mt = clin[clin['gsm'].isin(ge.index)].copy()
mt = mt[mt['Tissue Type'] == 'Tumor']
for g in GENES:
    mt[g] = mt['gsm'].map(ge[g])
mt['OS_months'] = pd.to_numeric(mt['Survival months'], errors='coerce')
mt['event'] = pd.to_numeric(mt['Survival status'], errors='coerce')
mt = mt.dropna(subset=['OS_months', 'event'] + GENES)
mt = mt[mt['OS_months'] > 0].copy()
mt['risk'] = sum(COEF[g] * mt[g] for g in GENES)
med = mt['risk'].median()
mt['grp'] = np.where(mt['risk'] >= med, 'high', 'low')

# ---- 1. 固定系数：方向与效应量
lr = logrank_test(mt[mt.grp == 'high'].OS_months, mt[mt.grp == 'low'].OS_months,
                  mt[mt.grp == 'high'].event, mt[mt.grp == 'low'].event)
cph = CoxPHFitter().fit(mt[['OS_months', 'event', 'risk']].astype(float), 'OS_months', 'event')
beta = float(cph.params_['risk']); sd = float(mt['risk'].std())
kmf = KaplanMeierFitter()
meds = {}
for g in ['high', 'low']:
    s = mt[mt.grp == g]
    kmf.fit(s.OS_months, s.event)
    meds[g] = (None if not np.isfinite(kmf.median_survival_time_) else round(float(kmf.median_survival_time_), 1))
OUT['gse14520_fixed'] = dict(
    n=int(len(mt)), events=int(mt.event.sum()),
    logrank_p=round(float(lr.p_value), 5),
    median_os_high=meds['high'], median_os_low=meds['low'],
    c_index=round(c_index(mt.OS_months, mt.event, mt.risk), 3),
    hr_per_SD=round(float(np.exp(beta * sd)), 3),
    hr_p=float(cph.summary.loc['risk', 'p']),
    direction='reversed (higher score = better survival)')
OUT['gse14520_fixed']['ci95'] = list(boot_ci(mt.OS_months, mt.event, mt.risk))

# ---- 2. 队列内重拟合（refit）：in-sample 与 5 折交叉验证
X = mt[GENES].astype(float)
cph2 = CoxPHFitter().fit(mt[['OS_months', 'event'] + GENES].astype(float), 'OS_months', 'event')
insample = cph2.predict_partial_hazard(mt[GENES].astype(float)).values
OUT['gse14520_refit'] = dict(
    n=int(len(mt)), events=int(mt.event.sum()),
    c_index_insample=round(c_index(mt.OS_months, mt.event, insample), 3),
    coefficients={g: round(float(cph2.params_[g]), 3) for g in GENES})

oof = np.zeros(len(mt))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in skf.split(X, mt['event']):
    tr_df = mt.iloc[tr][['OS_months', 'event'] + GENES].astype(float)
    try:
        m = CoxPHFitter().fit(tr_df, 'OS_months', 'event')
        oof[te] = m.predict_partial_hazard(mt.iloc[te][GENES].astype(float)).values
    except Exception:
        oof[te] = np.mean(oof[tr])
OUT['gse14520_refit']['c_index_cv5'] = round(c_index(mt.OS_months, mt.event, oof), 3)
OUT['gse14520_refit']['cv5_ci95'] = list(boot_ci(mt.OS_months, mt.event, oof))
# 交叉验证下的分组 log-rank
g2 = np.where(oof >= np.median(oof), 'high', 'low')
lr2 = logrank_test(mt.OS_months[g2 == 'high'], mt.OS_months[g2 == 'low'],
                   mt.event[g2 == 'high'], mt.event[g2 == 'low'])
OUT['gse14520_refit']['cv5_logrank_p'] = round(float(lr2.p_value), 4)

json.dump(OUT, open(os.path.join(DATA, 'v16_gse14520_check.json'), 'w'), indent=2)

# ---- 3. 图：Figure 24
# Scientific Reports 图片规范：clear sans-serif（Helvetica / Arial）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.size'] = 5.9
plt.rcParams['axes.titlesize'] = 7.0
plt.rcParams['axes.labelsize'] = 7.0
fig, axes = plt.subplots(1, 2, figsize=(7.73, 2.76))

ax = axes[0]
for g, col in [('high', '#c00000'), ('low', '#1f4e79')]:
    s = mt[mt.grp == g]
    kmf.fit(s.OS_months, s.event, label='High risk score (n = %d)' % len(s) if g == 'high'
            else 'Low risk score (n = %d)' % len(s))
    kmf.plot_survival_function(ax=ax, ci_show=False, color=col, lw=1.06)
ax.set_xlabel('Overall survival (months)')
ax.set_ylabel('Survival probability')
ax.set_title('GSE14520: TCGA coefficients give reversed separation', loc='left', fontsize=6.6)
ax.text(0.98, 0.30, 'log-rank P = 5.5 x 10$^{-4}$\nbut high score = better survival',
        transform=ax.transAxes, ha='right', va='top', fontsize=5.3, color='#c00000')
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25, ls=':')

ax = axes[1]
labels = ['TCGA-LIHC\n(5-fold CV, training)', 'GSE14520\n(refit, in-sample)',
          'GSE14520\n(refit, 5-fold CV)', 'GSE14520\n(fixed coefficients)',
          'GSE76427\n(fixed coefficients)']
vals = [0.690, 0.634,
        OUT['gse14520_refit']['c_index_cv5'], OUT['gse14520_fixed']['c_index'], 0.482]
cis = [(0.620, 0.760), (0.573, 0.700), OUT['gse14520_refit']['cv5_ci95'],
       OUT['gse14520_fixed']['ci95'], (0.335, 0.642)]
ys = np.arange(len(labels))[::-1]
cols = ['#1f4e79', '#7f7f7f', '#7f7f7f', '#c00000', '#c00000']
for y, v, ci, c in zip(ys, vals, cis, cols):
    ax.plot([ci[0], ci[1]], [y, y], color=c, lw=1.06, alpha=0.8)
    ax.plot(v, y, 'o', color=c, ms=3.96)
    ax.text(0.80, y + 0.14, '%.3f (95%% CI %.3f\u2013%.3f)' % (v, ci[0], ci[1]), fontsize=5.0)
ax.axvline(0.5, color='#000000', ls='--', lw=0.66)
ax.text(0.5, -0.55, '0.5', fontsize=5.0, ha='center')
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=5.0)
ax.set_xlabel('C-index')
ax.set_xlim(0.22, 0.86)
ax.set_ylim(-0.8, len(labels) - 0.2)
ax.set_title('Discrimination of the three-gene model across evaluations', loc='left', fontsize=6.6)
ax.grid(axis='x', alpha=0.25, ls=':')

for a in axes:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.tick_params(width=0.4)

plt.tight_layout()
out = os.path.join(FIG, 'fig24_direction_ci.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('saved', out)
print(json.dumps(OUT, indent=2))
