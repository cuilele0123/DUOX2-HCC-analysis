# -*- coding: utf-8 -*-
"""
机器学习模型对比：验证三基因 LASSO-Cox 模型的稳健性
严格嵌套交叉验证，防止数据泄露
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(_ROOT, 'data')

# ---------- 1. 加载 ----------
logcpm = pd.read_pickle(BASE + '/logcpm_matrix.pkl')
surv   = pd.read_pickle(BASE + '/survival_data.pkl')
surv = surv.set_index('sample_id')

# DEG 结果，取 top 500 by FDR 作为候选特征（与原文方法一致）
deg = pd.read_csv(BASE + '/deseq2_early_vs_advanced_all.csv')
deg.columns = [c.strip() for c in deg.columns]
print('DEG 文件列:', list(deg.columns)[:8])

# 基因名列处理
if 'gene_name' in deg.columns:
    deg['gene'] = deg['gene_name']
elif 'Unnamed: 0' in deg.columns:
    deg['gene'] = deg['Unnamed: 0'].astype(str).str.split('.').str[0]
    gm = pd.read_csv(BASE + '/gene_id_name_map.tsv', sep='\t')
    id2name = dict(zip(gm['gene_id'].astype(str).str.split('.').str[0], gm['gene_name']))
    deg['gene'] = deg['gene'].map(id2name).fillna(deg['gene'])
else:
    deg['gene'] = deg.index.astype(str)

padj_col = [c for c in deg.columns if c.lower() in ('padj','fdr','qval','q_value')][0]
deg = deg.dropna(subset=[padj_col])
top500 = deg.nsmallest(500, padj_col)['gene'].tolist()
top500 = [g for g in top500 if g in logcpm.columns]
print('Top500 候选基因(可匹配): %d' % len(top500))

# ---------- 2. 构建 X, y ----------
samples = [s for s in surv.index if s in logcpm.index]
X = logcpm.loc[samples, top500].astype(float)
y_df = surv.loc[samples, ['OS_event', 'OS_time_days']].copy()
y_df['OS_event'] = y_df['OS_event'].astype(bool)
y = Surv.from_arrays(event=y_df['OS_event'].values, time=y_df['OS_time_days'].values)
print('建模矩阵: X %s | 事件 %d / %d' % (str(X.shape), y_df['OS_event'].sum(), len(y_df)))

# ---------- 3. 单变量 Cox 筛选（内层） ----------
def univariate_screen(X_tr, y_tr, top_n=50):
    """在训练折内做单变量 Cox 筛选，返回 top_n 基因"""
    df = X_tr.copy()
    df['__t'] = [t for (e, t) in y_tr]
    df['__e'] = [1.0 if e else 0.0 for (e, t) in y_tr]
    ps = {}
    for g in X_tr.columns:
        try:
            sub = df[[g, '__t', '__e']].dropna()
            if sub[g].std() == 0 or len(sub) < 20:
                continue
            cph = CoxPHFitter()
            cph.fit(sub, duration_col='__t', event_col='__e', formula=g)
            ps[g] = cph.summary.loc[g, 'p']
        except Exception:
            continue
    s = pd.Series(ps).sort_values()
    return list(s.index[:top_n])

# ---------- 4. 模型 ----------
def build_models(seed):
    return {
        'Lasso-Cox (Coxnet)': Pipeline([
            ('sc', StandardScaler()),
            ('m', CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=50, max_iter=100000))
        ]),
        'Ridge-Cox (Coxnet)': Pipeline([
            ('sc', StandardScaler()),
            ('m', CoxnetSurvivalAnalysis(l1_ratio=0.01, alpha_min_ratio=0.01, n_alphas=50, max_iter=100000))
        ]),
        'Random Survival Forest': Pipeline([
            ('sc', StandardScaler()),
            ('m', RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                                       max_features='sqrt', random_state=seed, n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline([
            ('sc', StandardScaler()),
            ('m', GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05,
                                                   max_depth=2, subsample=0.8, random_state=seed))
        ]),
        'Survival SVM': Pipeline([
            ('sc', StandardScaler()),
            ('m', FastSurvivalSVM(max_iter=200, tol=1e-5, random_state=seed))
        ]),
    }

# ---------- 5. 重复 5 次 5 折 CV ----------
N_REPEAT, N_FOLD = 5, 5
results = {k: [] for k in build_models(0).keys()}
results['三基因模型 (SLC16A3+SPP2+MMP7)'] = []

THREE_GENES = ['SLC16A3', 'SPP2', 'MMP7']
COEF = {'SLC16A3': 0.0920072109058105, 'SPP2': -0.03765527210177806, 'MMP7': 0.023524515638499957}

for rep in range(N_REPEAT):
    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=42 + rep)
    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # --- 三基因模型（固定系数，来自原文训练，独立评估其泛化） ---
        rs_te = sum(COEF[g] * X_te[g] for g in THREE_GENES if g in X_te.columns)
        ev_te = np.array([e for (e, t) in y_te])
        tm_te = np.array([t for (e, t) in y_te])
        if len(np.unique(ev_te)) > 1:
            c3 = concordance_index_censored(ev_te, tm_te, rs_te.values)[0]
            results['三基因模型 (SLC16A3+SPP2+MMP7)'].append(c3)

        # --- 候选基因筛选（内层，仅用训练折） ---
        sel = univariate_screen(X_tr, y_tr, top_n=50)

        # --- 各模型 ---
        for name, mdl in build_models(42 + rep).items():
            try:
                mdl.fit(X_tr[sel], y_tr)
                pred = mdl.predict(X_te[sel])
                c = concordance_index_censored(ev_te, tm_te, pred)[0]
                results[name].append(c)
            except Exception as e:
                print('  [跳过 %s] %s' % (name, str(e)[:60]))

# ---------- 6. 汇总 ----------
print()
print('=' * 66)
print('5 次重复 × 5 折交叉验证 C-index（均值 ± 标准差）')
print('=' * 66)
rows = []
for name, vals in results.items():
    if len(vals) == 0:
        continue
    rows.append({'Model': name, 'C_index_mean': np.mean(vals),
                 'C_index_sd': np.std(vals), 'n_folds': len(vals)})
res = pd.DataFrame(rows).sort_values('C_index_mean', ascending=False)
for _, r in res.iterrows():
    print('%-38s %.4f ± %.4f  (n=%d)' % (r['Model'], r['C_index_mean'], r['C_index_sd'], r['n_folds']))

res.to_csv(BASE + '/ml_model_comparison_results.csv', index=False)
print()
print('已保存: ml_model_comparison_results.csv')
