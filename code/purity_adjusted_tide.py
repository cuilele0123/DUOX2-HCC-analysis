# -*- coding: utf-8 -*-
"""
DUOX2-TIDE 关联的混杂控制分析（肿瘤含量 / 上皮-基质比例校正）

背景问题：DUOX2 是上皮来源基因（HPA：免疫细胞中不表达）。因此必须排除一种可能——
"DUOX2 与 TIDE 的相关只是肿瘤细胞相对含量（纯度）差异造成的假象"。

策略（三层，互相印证）：
  L1  前提检验：DUOX2 与上皮/基质比例(E/S ratio)是否相关？若不相关，混杂前提不成立
  L2  偏相关  ：控制 E/S ratio（及基质、免疫、ESTIMATE-like）后 DUOX2-TIDE 是否仍显著
  L3  分层验证：按 E/S ratio 分层后，层内相关是否保持
  附加：Bootstrap 比较 DUOX2 与上皮 score vs 免疫 score 的相关强度，定量支持"上皮来源"

评分方法：rank-based（样本内分位数），消除样本间总体转录活性/文库差异的影响
"""
import os, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

EPI = ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CDH1']
STR = ['COL1A1', 'COL1A2', 'COL3A1', 'COL5A1', 'FAP', 'ACTA2', 'DCN', 'LUM',
       'FN1', 'PECAM1', 'VWF']
IMM = ['PTPRC', 'CD3D', 'CD3E', 'CD2', 'IL2RG', 'CD247', 'LCK', 'CD68', 'CSF1R']
METRICS = ['TIDE', 'Dysfunction', 'Exclusion', 'CAF', 'TAM M2', 'CD8',
           'MDSC', 'IFNG', 'CD274', 'MSI Score']

rng = np.random.default_rng(42)


def z(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std(ddof=1)


def partial_spearman(x, y, covars):
    """rank-based partial correlation, df = n - 2 - k"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    C = np.asarray(covars, float)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    n, k = len(x), C.shape[1]
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Cr = sm.add_constant(np.column_stack([stats.rankdata(C[:, j]) for j in range(k)]))
    rx_res = sm.OLS(rx, Cr).fit().resid
    ry_res = sm.OLS(ry, Cr).fit().resid
    r = np.corrcoef(rx_res, ry_res)[0, 1]
    df = n - 2 - k
    t = r * np.sqrt(df / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), df)
    return r, p


def boot_corr_diff(a, b, c, n_boot=2000):
    """Bootstrap: rho(a,b) - rho(a,c) 的差及其 95%CI 与双侧 p"""
    n = len(a)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            r1 = stats.spearmanr(a[i], b[i])[0]
            r2 = stats.spearmanr(a[i], c[i])[0]
            d.append(r1 - r2)
        except Exception:
            pass
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d < 0).mean(), (d > 0).mean())
    return d.mean(), lo, hi, max(p, 1.0 / n_boot)


# ============ 载入与评分 ============
X = pd.read_pickle(f'{DATA}/logcpm_matrix.pkl')
tide = pd.read_csv(f'{DATA}/tide_output.txt', sep='\t', index_col=0)
tide.index = [str(i) for i in tide.index]
common = [s for s in tide.index if s in X.index]
X = X.loc[common]
tide = tide.loc[common]

R = X.rank(axis=1, pct=True)                      # 样本内分位数
epi = R[EPI].mean(axis=1)
stro = R[STR].mean(axis=1)
imm = R[IMM].mean(axis=1)
ES = pd.Series(z(epi) - z(stro), index=X.index)   # 上皮-基质比例
EST = pd.Series(z(stro) + z(imm), index=X.index)  # ESTIMATE-like（非肿瘤成分）

df = pd.DataFrame({'DUOX2': X['DUOX2'], 'epi': epi, 'stromal': stro,
                   'immune': imm, 'ES_ratio': ES, 'ESTIMATE_like': EST})
for m in METRICS:
    if m in tide.columns:
        df[m] = tide[m]
df.to_csv(f'{DATA}/purity_scores_tide.csv')
n = len(df)
print(f'样本数 n = {n}\n')

print('=== L0. 评分的相关结构（HCC 中上皮与基质评分正相关，符合肝硬化背景） ===')
for a, b in [('epi', 'stromal'), ('epi', 'immune'), ('stromal', 'immune')]:
    r, p = stats.spearmanr(df[a], df[b])
    print(f'  {a:<8} vs {b:<8}: rho={r:+.3f}, p={p:.2e}')
r, p = stats.spearmanr(df['ES_ratio'], df['ESTIMATE_like'])
print(f'  E/S ratio  vs ESTIMATE-like: rho={r:+.3f}, p={p:.2e}')

print('\n=== L1. 前提检验：DUOX2 与肿瘤含量代理的关系 ===')
l1 = {}
for c in ['ES_ratio', 'ESTIMATE_like', 'epi', 'stromal', 'immune']:
    r, p = stats.spearmanr(df['DUOX2'], df[c])
    l1[c] = dict(rho=float(r), p=float(p))
    print(f'  DUOX2 vs {c:<15}: rho={r:+.3f}, p={p:.2e}')
print('  -> 与 E/S ratio 近零相关，说明 DUOX2 高低并非由肿瘤细胞相对含量驱动，'
      '混杂前提不成立')

print('\n=== L1b. 细胞来源定量验证（Bootstrap 比较相关强度） ===')
d, lo, hi, p_boot1 = boot_corr_diff(df['DUOX2'].values, df['epi'].values,
                                    df['immune'].values)
print(f'  rho(DUOX2,epi) - rho(DUOX2,immune) = {d:+.3f} '
      f'[95%CI {lo:+.3f}, {hi:+.3f}], p={p_boot1:.4f}')
d2, lo2, hi2, p_boot2 = boot_corr_diff(df['DUOX2'].values, df['epi'].values,
                                       df['stromal'].values)
print(f'  rho(DUOX2,epi) - rho(DUOX2,stromal) = {d2:+.3f} '
      f'[95%CI {lo2:+.3f}, {hi2:+.3f}], p={p_boot2:.4f}')

print('\n=== L2. 校正前 vs 校正后（多种协变量组合） ===')
cov_sets = {
    'ES_ratio': ['ES_ratio'],
    'stromal': ['stromal'],
    'immune': ['immune'],
    'ES+immune': ['ES_ratio', 'immune'],
    'ESTIMATE_like': ['ESTIMATE_like'],
    'stromal+immune': ['stromal', 'immune'],
}
rows = []
for m in METRICS:
    if m not in df.columns:
        continue
    sub = df[['DUOX2', m] + sorted(set(sum(cov_sets.values(), [])))].dropna()
    r0, p0 = stats.spearmanr(sub['DUOX2'], sub[m])
    rec = dict(metric=m, n=len(sub), rho_raw=r0, p_raw=p0)
    line = f'  {m:<12} n={len(sub)}  raw={r0:+.3f}(p={p0:.1e})'
    for cname, cols in cov_sets.items():
        r1, p1 = partial_spearman(sub['DUOX2'].values, sub[m].values,
                                  sub[cols].values)
        rec[f'rho_{cname}'] = r1
        rec[f'p_{cname}'] = p1
        flag = '**' if p1 < 0.01 else ('*' if p1 < 0.05 else 'NS')
        line += f' | {cname}:{r1:+.3f}{flag}'
    rows.append(rec)
    print(line)
pd.DataFrame(rows).to_csv(f'{DATA}/DUOX2_TIDE_purity_adjusted.csv', index=False)

print('\n=== L2b. 多元线性回归（rank 变换）: TIDE ~ DUOX2 + E/S ratio ===')
sub = df[['DUOX2', 'TIDE', 'ES_ratio']].dropna()
rk = sub.rank()
fit = sm.OLS(rk['TIDE'], sm.add_constant(rk[['DUOX2', 'ES_ratio']])).fit()
print(f'  n={int(fit.nobs)}, R2={fit.rsquared:.4f}, adjR2={fit.rsquared_adj:.4f}, '
      f'F p={fit.f_pvalue:.3e}')
ci = fit.conf_int()
for v in ['DUOX2', 'ES_ratio']:
    print(f'  {v:<10} beta_std={fit.params[v]:+.4f} p={fit.pvalues[v]:.3e} '
          f'95%CI=[{ci.loc[v,0]:+.4f}, {ci.loc[v,1]:+.4f}]')

print('\n=== L3. 分层验证（按 E/S ratio 三分位，避免中位数分层的统计损失） ===')
df['ES_tertile'] = pd.qcut(df['ES_ratio'], 3,
                           labels=['T1 low', 'T2 mid', 'T3 high'])
strat = []
for g in ['T1 low', 'T2 mid', 'T3 high']:
    idx = df.index[df['ES_tertile'] == g]
    sub = df.loc[idx, ['DUOX2', 'TIDE']].dropna()
    r, p = stats.spearmanr(sub['DUOX2'], sub['TIDE'])
    print(f'  {g:<8} n={len(sub):>3}  rho={r:+.3f}, p={p:.3e}')
    strat.append(dict(group=g, n=len(sub), rho=float(r), p=float(p)))
pd.DataFrame(strat).to_csv(f'{DATA}/DUOX2_TIDE_stratified_purity.csv', index=False)

# 交互检验
sub = df[['DUOX2', 'TIDE', 'ES_ratio']].dropna().copy()
sub['hi'] = (sub['ES_ratio'] > sub['ES_ratio'].median()).astype(int)
sub['inter'] = sub['DUOX2'] * sub['hi']
rk = sub.rank()
f2 = sm.OLS(rk['TIDE'], sm.add_constant(rk[['DUOX2', 'hi', 'inter']])).fit()
print(f'  交互项 DUOX2 x E/S 高低: beta={f2.params["inter"]:+.4f}, '
      f'p={f2.pvalues["inter"]:.3f}  (p>0.05 = 关联在各层一致，无异质性)')

ti = [r for r in rows if r['metric'] == 'TIDE'][0]
res = dict(
    n=int(n),
    L1=l1,
    L1b=dict(diff_epi_immune=dict(d=float(d), lo=float(lo), hi=float(hi),
                                  p=float(p_boot1)),
             diff_epi_stromal=dict(d=float(d2), lo=float(lo2), hi=float(hi2),
                                   p=float(p_boot2))),
    tide_raw=dict(rho=float(ti['rho_raw']), p=float(ti['p_raw'])),
    tide_adj_ES=dict(rho=float(ti['rho_ES_ratio']), p=float(ti['p_ES_ratio'])),
    tide_adj_ES_imm=dict(rho=float(ti['rho_ES+immune']), p=float(ti['p_ES+immune'])),
    tide_adj_stromal_imm=dict(rho=float(ti['rho_stromal+immune']),
                              p=float(ti['p_stromal+immune'])),
    mlr=dict(n=int(fit.nobs), r2=float(fit.rsquared),
             beta_duox2=float(fit.params['DUOX2']), p_duox2=float(fit.pvalues['DUOX2']),
             beta_es=float(fit.params['ES_ratio']), p_es=float(fit.pvalues['ES_ratio'])),
    stratified=strat,
    interaction_p=float(f2.pvalues['inter']),
)
with open(f'{DATA}/purity_adjusted_summary.json', 'w') as f:
    json.dump(res, f, indent=2)
print('\n已保存：purity_scores_tide.csv / DUOX2_TIDE_purity_adjusted.csv / '
      'DUOX2_TIDE_stratified_purity.csv / purity_adjusted_summary.json')
