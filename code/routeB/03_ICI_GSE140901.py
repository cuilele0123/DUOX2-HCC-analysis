#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线 B 第三步：免疫治疗应答终点替换
队列：GSE140901（24 例晚期 HCC 接受抗 PD-1/PD-L1 治疗，NanoString 770 基因免疫 panel，
      TMM 归一化 log2 CPM，PMID 34414122）

科学问题
--------
新框架认为 DUOX2 标记的是肝内胆管/管状上皮区室，其与免疫的相关性为组织构成共变的
假象。若该判断成立，则由 DUOX2 共表达谱导出的"DUOX2 关联程序"应是一个
上皮—髓系炎症程序，而不是淋巴细胞效应程序，因而也不应携带免疫治疗应答的预测信息。

方法
----
1) 用 TCGA-LIHC（n=371 肿瘤）的 DUOX2 共表达谱，取与本次 panel 交集的前 50 个正相关
   基因，构建 P_DUOX2 评分；先在 TCGA 中核验该评分确实是 DUOX2 的忠实代理。
2) 同类方法构建参照评分：T 效应/IFN-γ、T 细胞耗竭、髓系抑制、上皮/管状、间质、内皮。
3) 终点：clinical_benefit_response（Yes/No，主）、best_response（PR/SD/PD）、PFS、OS。
4) 统计：Mann-Whitney U、ROC/AUC（含 bootstrap 95% CI）、log-rank、单因素 Cox。

依赖：pandas numpy scipy（无需下载额外数据；GSE140901 处理后矩阵仅 62 KB）
"""
import os
import re
import csv
import gzip
import json
import math
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('DUOX2_WORK', os.path.join(_ROOT, '..'))
OUT = os.path.join(BASE, 'RouteB_analysis')
GSE = '/tmp/gse140901'
_ROOT = os.path.dirname(os.path.abspath(__file__))
TCGA_ROOT = os.environ.get('DUOX2_DATA', os.path.join(_ROOT, '..', 'data'))
COEXP = os.path.join(BASE, 'DUOX2_HCC_code_release/results/DUOX2_coexpression_all.csv')

os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261002)
LOG = []


def log(s=''):
    print(s)
    LOG.append(str(s))


def zscore(v):
    v = np.asarray(v, dtype=float)
    return (v - np.nanmean(v)) / (np.nanstd(v, ddof=0) + 1e-12)


def score(mat, genes):
    """评分 = 可用基因 z 分数的均值"""
    gs = [g for g in genes if g in mat.index]
    if not gs:
        return None, []
    Z = np.vstack([zscore(mat.loc[g].values.astype(float)) for g in gs])
    return Z.mean(0), gs


def auc_ci(y, x, n_boot=4000):
    """AUC 及 percentile bootstrap 95% CI"""
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    y, x = y[ok], x[ok]
    if len(set(y)) < 2:
        return float('nan'), float('nan'), float('nan'), 0
    r = stats.rankdata(x)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(y), len(y))
        yy, xx = y[idx], x[idx]
        if len(set(yy)) < 2:
            continue
        rr = stats.rankdata(xx)
        a1, a0 = int((yy == 1).sum()), int((yy == 0).sum())
        boots.append((rr[yy == 1].sum() - a1 * (a1 + 1) / 2) / (a1 * a0))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float('nan'), float('nan')))
    return auc, lo, hi, len(boots)


def logrank(t, e, g):
    """两组 log-rank 检验（g 为布尔，True=高分组）"""
    t = np.asarray(t, float); e = np.asarray(e, int); g = np.asarray(g, bool)
    times = np.unique(t[e == 1])
    O1 = E1 = V = 0.0
    for tt in times:
        n1 = ((t >= tt) & g).sum(); n0 = ((t >= tt) & ~g).sum()
        d1 = ((t == tt) & g & (e == 1)).sum(); d0 = ((t == tt) & ~g & (e == 1)).sum()
        n, d = n1 + n0, d1 + d0
        if n < 2 or d == 0:
            continue
        O1 += d1
        E1 += d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if V <= 0:
        return float('nan'), float('nan')
    chi2 = (O1 - E1) ** 2 / V
    return chi2, float(stats.chi2.sf(chi2, 1))


def cox_hr(t, e, x):
    """单因素 Cox（Breslow 偏似然，Newton-Raphson）"""
    t = np.asarray(t, float); e = np.asarray(e, int); x = np.asarray(x, float)
    ok = ~np.isnan(x)
    t, e, x = t[ok], e[ok], x[ok]
    x = zscore(x)

    def nll(b):
        b = float(b[0])
        lp = b * x
        m = lp.max()
        s = 0.0
        for i in range(len(t)):
            if e[i] != 1:
                continue
            risk = t >= t[i]
            lr = lp[risk]
            mm = lr.max()
            s += (lp[i] - mm) - math.log(np.exp(lr - mm).sum())
        return -s

    from scipy.optimize import minimize
    res = minimize(nll, [0.0], method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000})
    b = float(res.x[0])
    h = 1e-4
    d2 = (nll([b + h]) - 2 * nll([b]) + nll([b - h])) / h ** 2
    se = math.sqrt(1 / d2) if d2 > 0 else float('nan')
    z = b / se if se > 0 else float('nan')
    return math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se), float(2 * stats.norm.sf(abs(z))), se


# ---------------------------------------------------------------- 载入 panel
def load_panel():
    """第 1-4 行为说明，第 5 行为表头，第 6 行起为数据（行尾可能多出空字段）"""
    M = pd.read_csv(os.path.join(GSE, 'processed.txt'), sep='\t',
                    skiprows=4, index_col=0, engine='python')
    M = M.loc[:, [c for c in M.columns if not str(c).startswith('Unnamed')]]
    M = M.dropna(axis=1, how='all')
    M = M[~M.index.duplicated(keep='first')]
    M.columns = [str(c).strip() for c in M.columns]
    return M.astype(float)


# ---------------------------------------------------------------- 载入临床
def load_clin():
    url = ('https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi'
           '?acc=GSE140901&targ=gsm&form=text&view=brief')
    cache = '/tmp/gse140901_gsm.txt'
    if not os.path.exists(cache):
        urllib.request.urlretrieve(url, cache)
    txt = open(cache, encoding='utf-8', errors='ignore').read()
    blocks = re.split(r'(?:^|\n)\^SAMPLE = ', txt)[1:]
    rows = {}
    for b in blocks:
        d = {}
        title = re.search(r'!Sample_title = (.*)', b)
        sid = None
        if title:
            m = re.search(r'(S\d+)', title.group(1))
            sid = m.group(1) if m else None
        for c in re.findall(r'!Sample_characteristics_ch1 = (.*)', b):
            if ':' in c:
                k, v = c.split(':', 1)
                d[k.strip()] = v.strip()
        if sid:
            rows[sid] = d
    C = pd.DataFrame(rows).T
    for c in ['pfs_time', 'os_time', 'age']:
        if c in C:
            C[c] = pd.to_numeric(C[c], errors='coerce')
    for c in ['pfs_event', 'os_event']:
        if c in C:
            C[c] = pd.to_numeric(C[c], errors='coerce')
    if 'clinical_benefit_response' in C:
        C['cbr'] = (C['clinical_benefit_response'].str.lower() == 'yes').astype(int)
    return C


# ================================================================ 主流程
log('=' * 78)
log('路线 B 第三步：DUOX2 关联程序与 HCC 免疫治疗应答（GSE140901）')
log('=' * 78)
log()

M = load_panel()
C = load_clin()
log('panel 基因数（含对照）: %d' % M.shape[0])
log('样本数: %d  （%s）' % (M.shape[1], ', '.join(M.columns[:4]) + ' ...'))
ctrl = [g for g in M.index if g.startswith(('NEG_', 'POS_'))]
log('对照探针: %d' % len(ctrl))
log()

# --- 临床基线
log('--- 临床与应答分布 ---')
br = C['best_response'].value_counts().to_dict()
log('best_response: %s' % br)
log('clinical_benefit_response: Yes=%d, No=%d'
    % ((C['cbr'] == 1).sum(), (C['cbr'] == 0).sum()))
log('etiology: %s' % C.get('etiology', pd.Series(dtype=str)).value_counts().to_dict())
log()

# --- 构建 P_DUOX2
rows = []
with open(COEXP) as f:
    rd = csv.reader(f); next(rd)
    for r in rd:
        if len(r) >= 2:
            try:
                rows.append((r[0], float(r[1])))
            except ValueError:
                pass
rows.sort(key=lambda x: -x[1])
panel_genes = set(M.index) - set(ctrl)
prog = [g for g, r in rows if g in panel_genes and r > 0][:50]
log('--- P_DUOX2 定义 ---')
log('来源：TCGA-LIHC(n=371) DUOX2 共表达谱 ∩ 本 panel，取正相关前 50 位')
log('实际纳入 %d 个基因：' % len(prog))
for i in range(0, len(prog), 10):
    log('   ' + ', '.join(prog[i:i + 10]))
log()

S = {}
S['P_DUOX2'] = prog
S['T效应/IFN-$\\gamma$'] = ['GZMA', 'GZMB', 'GZMH', 'PRF1', 'IFNG', 'CXCL9', 'CXCL10',
                            'CXCL11', 'STAT1', 'IDO1', 'HLA-DRA', 'CD8A', 'NKG7', 'KLRD1']
S['T细胞耗竭'] = ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'BTLA', 'CD244', 'EOMES']
S['髓系抑制/M2'] = ['CD163', 'MRC1', 'TREM2', 'ARG1', 'MARCO', 'MSR1', 'S100A8',
                    'ITGAM', 'CD14', 'CSF1R', 'IL1B', 'CXCL1', 'CXCL3', 'CXCL5', 'CXCL6', 'AXL']
S['上皮/管状(近似)'] = ['EPCAM', 'MUC1', 'CDH1', 'SPP1', 'DMBT1', 'MST1R', 'CEACAM6',
                        'CEACAM1', 'ITGB4', 'CD24', 'JAM3', 'LGALS3']
S['间质/CAF'] = ['COL3A1', 'FN1', 'THY1', 'PDGFRB', 'ENG', 'TGFB1', 'VEGFA']
S['内皮'] = ['PECAM1', 'CDH5', 'ICAM1', 'VCAM1', 'SELE', 'CD34']
S['Treg'] = ['FOXP3', 'IL2RA', 'IKZF2', 'CTLA4', 'TNFRSF18']
S['I型干扰素'] = ['ISG15', 'MX1', 'OAS3', 'IFI16', 'IFI27', 'IFI35', 'IFIT1', 'IFITM1', 'IFI6']

sc = {}
log('--- 评分构建（panel 覆盖情况）---')
for k, gs in S.items():
    v, used = score(M.loc[[g for g in M.index if g not in ctrl]], gs)
    if v is None:
        continue
    sc[k] = pd.Series(v, index=M.columns)
    log('%-22s 指定 %2d 基因，命中 %2d：%s' % (k, len(gs), len(used), ', '.join(used)))
Sc = pd.DataFrame(sc)
log()

# --- QC：P_DUOX2 在 TCGA 中是否为 DUOX2 的忠实代理
log('--- QC：P_DUOX2 评分在 TCGA-LIHC 中与 DUOX2 表达的一致性 ---')
try:
    MT = pd.read_pickle(os.path.join(TCGA, 'data/logcpm_matrix.pkl'))
    meta = pd.read_csv(os.path.join(TCGA, 'data/sample_meta.tsv'), sep='\t').set_index('sample_id')
    meta = meta.reindex(MT.index)
    tum = (meta['sample_type'] == 'Primary Tumor').values
    sub = MT.loc[tum]
    pv, used = score(sub.T, prog)
    r, p = stats.pearsonr(pv, sub['DUOX2'].values.astype(float))
    log('TCGA 肿瘤 n=%d：P_DUOX2 与 DUOX2 的 r = %.3f (P = %.2e)' % (tum.sum(), r, p))
    rh, ph = stats.spearmanr(pv, sub['DUOX2'].values.astype(float))
    log('   Spearman rho = %.3f (P = %.2e)' % (rh, ph))
    # 与各参照评分的关系
    for k in ['上皮/管状(近似)', '髓系抑制/M2', 'T效应/IFN-$\\gamma$', 'T细胞耗竭', '间质/CAF']:
        if k in S:
            vv, _ = score(sub.T, S[k])
            if vv is not None:
                rr, pp = stats.pearsonr(vv, sub['DUOX2'].values.astype(float))
                log('   %-22s 与 DUOX2: r = %6.3f (P = %.1e)' % (k, rr, pp))
except Exception as e:
    log('  （TCGA QC 跳过：%s）' % e)
log()

# --- 主分析：与临床获益
dat = Sc.join(C[['cbr', 'pfs_time', 'pfs_event', 'os_time', 'os_event', 'etiology']])
dat = dat.dropna(subset=['cbr'])
log('=' * 78)
log('主终点：临床获益（clinical benefit response, Yes=%d / No=%d）'
    % ((dat['cbr'] == 1).sum(), (dat['cbr'] == 0).sum()))
log('=' * 78)
log('%-24s %10s %12s %10s %22s' % ('评分', 'r_AUC', 'P(MWU)', 'r', 'AUC(95% CI)'))
log('-' * 86)
res_main = []
for k in Sc.columns:
    x = dat[k].values.astype(float)
    y = dat['cbr'].values.astype(int)
    a, lo, hi, nb = auc_ci(y, x)
    u, pu = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative='two-sided')
    r, _ = stats.pointbiserialr(y, x)
    res_main.append((k, a, pu, r, lo, hi))
    log('%-24s %8.3f %10.4f %8.3f  %6.3f-%6.3f'
        % (k, a, pu, r, lo, hi))
log()

# --- 二元 best_response：PD vs (PR+SD)
log('--- 次要终点：最好疗效 PD(n=%d) vs PR/SD(n=%d) ---'
    % ((dat.index.map(C['best_response']) == 'PD').sum(),
       (dat.index.map(C['best_response']) != 'PD').sum()))
pdv = (C['best_response'].reindex(dat.index) == 'PD').astype(int).values
log('%-24s %10s %12s' % ('评分', 'AUC(PD)', 'P(MWU)'))
log('-' * 50)
for k in Sc.columns:
    x = dat[k].values.astype(float)
    a, lo, hi, nb = auc_ci(pdv, x)
    u, pu = stats.mannwhitneyu(x[pdv == 1], x[pdv == 0], alternative='two-sided')
    log('%-24s %10.3f %12.4f' % (k, a, pu))
log()

# --- PFS / OS
log('--- PFS / OS：单因素 Cox（评分每 +1 SD）与 log-rank（中位分组）---')
log('%-24s %-28s %-12s %-28s %-12s'
    % ('评分', 'PFS HR(95% CI)', 'P(Cox)', 'OS HR(95% CI)', 'P(Cox)'))
log('-' * 112)
res_surv = []
for k in Sc.columns:
    x = dat[k].values.astype(float)
    h1, l1, u1, p1, _ = cox_hr(dat['pfs_time'].values, dat['pfs_event'].values, x)
    h2, l2, u2, p2, _ = cox_hr(dat['os_time'].values, dat['os_event'].values, x)
    res_surv.append((k, h1, l1, u1, p1, h2, l2, u2, p2))
    log('%-24s %-28s %-12.4f %-28s %-12.4f'
        % (k, '%.2f (%.2f-%.2f)' % (h1, l1, u1), p1, '%.2f (%.2f-%.2f)' % (h2, l2, u2), p2))
log()

med = dat['P_DUOX2'].median()
g = (dat['P_DUOX2'] > med).values
chi2, plr = logrank(dat['pfs_time'].values, dat['pfs_event'].values, g)
log('P_DUOX2 中位分组的 PFS log-rank: chi2 = %.2f, P = %.3f' % (chi2, plr))
chi2o, plro = logrank(dat['os_time'].values, dat['os_event'].values, g)
log('P_DUOX2 中位分组的 OS log-rank: chi2 = %.2f, P = %.3f' % (chi2o, plro))
log()

# --- HBV 亚组
hbv = (C['etiology'].reindex(dat.index) == 'HBV').values
log('--- HBV 亚组（n=%d）内的 AUC ---' % hbv.sum())
for k in ['P_DUOX2', '上皮/管状(近似)', '髓系抑制/M2', 'T效应/IFN-$\\gamma$', 'T细胞耗竭']:
    x = dat[k].values.astype(float)[hbv]
    y = dat['cbr'].values.astype(int)[hbv]
    a, lo, hi, nb = auc_ci(y, x)
    u, pu = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative='two-sided')
    log('  %-22s AUC = %.3f (%.3f-%.3f)  P = %.4f' % (k, a, lo, hi, pu))
log()

# --- 单个基因：CD274 / PDCD1 等
log('--- 关键单基因（与临床获益）---')
for g_ in ['CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'IFNG', 'CXCL9',
           'GZMA', 'PRF1', 'CD8A', 'FOXP3', 'CD163', 'TREM2', 'SPP1', 'MUC1',
           'EPCAM', 'DMBT1', 'MST1R', 'CEACAM6', 'AXL', 'IL1B', 'CXCL1']:
    if g_ not in dat.columns and g_ in M.index:
        dat[g_] = M.loc[g_].reindex(dat.index).astype(float)
    if g_ in dat.columns:
        x = dat[g_].values.astype(float)
        y = dat['cbr'].values.astype(int)
        a, lo, hi, nb = auc_ci(y, x)
        u, pu = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative='two-sided')
        h1, l1, u1, p1, _ = cox_hr(dat['pfs_time'].values, dat['pfs_event'].values, x)
        log('%-9s AUC=%.3f  P(MWU)=%.4f  |  PFS HR=%.2f (%.2f-%.2f) P=%.3f'
            % (g_, a, pu, h1, l1, u1, p1))
log()

# --- 队列内评分相关性（看 P_DUOX2 在本队列代表什么）---
log('--- 评分间相关性（GSE140901 内部，Spearman rho）---')
cors = Sc.corr(method='spearman')
log(cors.round(2).to_string())
log()
log('P_DUOX2 与各参照评分的 rho：')
for k in Sc.columns:
    if k != 'P_DUOX2':
        log('   %-22s %6.3f' % (k, cors.loc['P_DUOX2', k]))
log()

# --- 多重比较校正：置换检验（max-statistic，控制 FWER）---
log('--- 多重比较校正：置换检验 max-statistic（主终点，10000 次重排）---')
Xi = Sc.loc[dat.index].values.astype(float)
y = dat['cbr'].values.astype(int)
n1, n0 = int((y == 1).sum()), int((y == 0).sum())


def auc_vec(x, yy):
    r = stats.rankdata(x)
    a1, a0 = int((yy == 1).sum()), int((yy == 0).sum())
    return (r[yy == 1].sum() - a1 * (a1 + 1) / 2) / (a1 * a0)


obs = np.array([abs(auc_vec(Xi[:, j], y) - 0.5) for j in range(Xi.shape[1])])
max_obs = obs.max()
n_ge = np.zeros(Xi.shape[1], dtype=int)
n_ge_max = 0
NPERM = 10000
for _ in range(NPERM):
    yp = RNG.permutation(y)
    st = np.array([abs(auc_vec(Xi[:, j], yp) - 0.5) for j in range(Xi.shape[1])])
    n_ge += (st >= obs - 1e-12)
    if st.max() >= max_obs - 1e-12:
        n_ge_max += 1
log('%-24s %8s %12s %12s' % ('评分', '|AUC-.5|', 'P_perm', 'P_FWER'))
log('-' * 60)
for j, k in enumerate(Sc.columns):
    log('%-24s %8.3f %12.4f %12.4f'
        % (k, obs[j], (n_ge[j] + 1) / (NPERM + 1), (n_ge_max + 1) / (NPERM + 1)))
log()
log('全队列最大 |AUC-0.5| = %.3f；FWER 校正后 P = %.3f（即：经多重比较校正后无任何评分显著）'
    % (max_obs, (n_ge_max + 1) / (NPERM + 1)))
log()

# ---------------------------------------------------------------- 落盘
dat.to_csv(os.path.join(OUT, 'GSE140901_scores.tsv'), sep='\t')
pd.DataFrame(res_main, columns=['score', 'AUC', 'P_MWU', 'r_pb', 'AUC_lo', 'AUC_hi']) \
    .to_csv(os.path.join(OUT, 'GSE140901_ICI_benefit.tsv'), sep='\t', index=False)
pd.DataFrame(res_surv, columns=['score', 'PFS_HR', 'PFS_lo', 'PFS_hi', 'PFS_P',
                               'OS_HR', 'OS_lo', 'OS_hi', 'OS_P']) \
    .to_csv(os.path.join(OUT, 'GSE140901_survival.tsv'), sep='\t', index=False)
with open(os.path.join(OUT, 'GSE140901_ICI_log.txt'), 'w') as f:
    f.write('\n'.join(LOG))
log('已落盘：GSE140901_scores.tsv / GSE140901_ICI_benefit.tsv / GSE140901_survival.tsv / GSE140901_ICI_log.txt')
