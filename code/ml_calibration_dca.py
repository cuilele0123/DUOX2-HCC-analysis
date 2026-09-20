# -*- coding: utf-8 -*-
"""
增强分析1+2: (1) 修正后的ML模型对比(嵌套CV, Coxnet调alpha, 含临床基线)
              (2) 三基因模型校准曲线与DCA决策曲线
"""
import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter, KaplanMeierFitter

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(_ROOT, 'data')
FIG  = os.path.join(_ROOT, 'figures')

THREE = ['SLC16A3', 'SPP2', 'MMP7']
COEF  = {'SLC16A3': 0.0920072109058105, 'SPP2': -0.03765527210177806,
         'MMP7': 0.023524515638499957}

# ---------- 1. 加载并过滤 ----------
logcpm = pd.read_pickle(BASE + '/logcpm_matrix.pkl')
surv   = pd.read_pickle(BASE + '/survival_data.pkl').set_index('sample_id')
surv   = surv[surv['OS_time_days'] > 0]          # 过滤 time<=0
print('过滤 OS_time<=0 后: %d 样本, 事件 %d' % (len(surv), surv['OS_event'].sum()))

# 忠实复现原文候选集: DEG显著基因(|log2FC|>1 & padj<0.05)中 padj 最小的500个
_gm = pd.read_csv(BASE + '/gene_id_name_map.tsv', sep='\t')
_gm = _gm[_gm['gene_id'].str.startswith('ENSG')].dropna()
_id2name_v = dict(zip(_gm['gene_id'], _gm['gene_name']))   # key 带版本号，与原文一致
deg = pd.read_csv(BASE + '/deseq2_early_vs_advanced_all.csv', index_col=0)
deg['gene'] = deg.index.map(_id2name_v)
deg_sig = deg[deg['DEG']].sort_values('padj').head(500)
top500 = [g for g in deg_sig['gene'].dropna().unique() if g in logcpm.columns]
print('候选基因集(原文忠实复现): %d 个 | 含SPP2: %s' % (len(top500), 'SPP2' in top500))

samples = [s for s in surv.index if s in logcpm.index]
X   = logcpm.loc[samples, top500].astype(float)
ydf = surv.loc[samples, ['OS_event','OS_time_days']].copy()
ydf['OS_event'] = ydf['OS_event'].astype(bool)
y = Surv.from_arrays(event=ydf['OS_event'].values, time=ydf['OS_time_days'].values)
print('建模矩阵: %s | 事件 %d' % (str(X.shape), ydf['OS_event'].sum()))

# ---------- 2. Coxnet 内层选 alpha ----------
def select_alpha(Xa, ya, l1_ratio, seed=0, n_in=3):
    kf = KFold(n_in, shuffle=True, random_state=seed)
    acc = {}
    for tr, te in kf.split(Xa):
        try:
            cn = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01,
                                        n_alphas=40, max_iter=100000)
            cn.fit(Xa.iloc[tr], ya[tr])
        except Exception:
            continue
        ev = np.array([e for e,t in ya[te]]); tm = np.array([t for e,t in ya[te]])
        if len(np.unique(ev)) < 2: continue
        for a in cn.alphas_:
            try:
                c = concordance_index_censored(ev, tm, cn.predict(Xa.iloc[te], alpha=a))[0]
                acc.setdefault(a, []).append(c)
            except Exception:
                pass
    return max({a: np.mean(v) for a,v in acc.items()}.items(), key=lambda kv: kv[1])[0] if acc else None

def make_models(seed):
    return {
      'Lasso-Cox (Coxnet, alpha tuned)': ('coxnet', 1.0),
      'Ridge-Cox (Coxnet, alpha tuned)': ('coxnet', 0.01),
      'Random Survival Forest': ('rsf', RandomSurvivalForest(
          n_estimators=300, min_samples_leaf=15, max_features='sqrt',
          random_state=seed, n_jobs=-1)),
      'Gradient Boosting': ('gb', GradientBoostingSurvivalAnalysis(
          n_estimators=300, learning_rate=0.05, max_depth=2,
          subsample=0.8, random_state=seed)),
      'Survival SVM': ('svm', FastSurvivalSVM(max_iter=200, tol=1e-5, random_state=seed)),
    }

# ---------- 3. 重复5次5折CV ----------
results = {}
N_REP, N_FOLD = 5, 5
for rep in range(N_REP):
    kf = KFold(N_FOLD, shuffle=True, random_state=42+rep)
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        ev = np.array([e for e,t in yte]); tm = np.array([t for e,t in yte])
        if len(np.unique(ev)) < 2: continue

        # 三基因（固定系数）
        Xte3 = logcpm.loc[Xte.index, THREE]
        rs = sum(COEF[g]*Xte3[g] for g in THREE)
        results.setdefault('三基因模型 (SLC16A3+SPP2+MMP7)', []).append(
            concordance_index_censored(ev, tm, rs.values)[0])

        # 临床基线（stage + age + gender）
        try:
            clin = surv.loc[samples]
            cdf = pd.DataFrame(index=X.index)
            cdf['stage_num'] = clin['stage_group'].map({'early':1,'intermediate':2,'advanced':3}).fillna(2)
            cdf['age'] = pd.to_numeric(survival_age := clin['AGE'] if 'AGE' in clin.columns else 60, errors='coerce').fillna(60)
            cdf['male'] = (clin['gender'].astype(str).str.upper().str.startswith('M')).astype(float) if 'gender' in clin.columns else 0.0
            XCtr, XCte = cdf.iloc[tr], cdf.iloc[te]
            pl = Pipeline([('sc',StandardScaler()),('m',CoxnetSurvivalAnalysis(
                l1_ratio=0.5, alpha_min_ratio=0.05, n_alphas=30, max_iter=100000))])
            pl.fit(XCtr, ytr)
            results.setdefault('临床基线 (stage+age+sex)', []).append(
                concordance_index_censored(ev, tm, pl.predict(XCte))[0])
        except Exception as e:
            pass

        # 各ML模型（全部500特征，公平对比）
        for name,(kind,obj) in make_models(42+rep).items():
            try:
                if kind == 'coxnet':
                    cn = CoxnetSurvivalAnalysis(l1_ratio=obj, alpha_min_ratio=0.01,
                                                n_alphas=40, max_iter=100000)
                    cn.fit(Xtr, ytr)
                    a = select_alpha(Xtr, ytr, obj, seed=42+rep)
                    pred = cn.predict(Xte, alpha=a) if a else cn.predict(Xte)
                else:
                    pl = Pipeline([('sc',StandardScaler()),('m',obj)])
                    pl.fit(Xtr, ytr); pred = pl.predict(Xte)
                results.setdefault(name, []).append(
                    concordance_index_censored(ev, tm, pred)[0])
            except Exception as e:
                print('  [skip %s] %s' % (name, str(e)[:50]))

print()
print('='*68)
print('5次重复 × 5折交叉验证 C-index（嵌套CV，均值 ± SD）')
print('='*68)
rows = [{'Model':k,'C_index':np.mean(v),'SD':np.std(v),'n':len(v)}
        for k,v in results.items() if v]
res = pd.DataFrame(rows).sort_values('C_index', ascending=False)
for _,r in res.iterrows():
    print('%-36s %.4f ± %.4f (n=%d)' % (r['Model'], r['C_index'], r['SD'], r['n']))
res.to_csv(BASE+'/ml_model_comparison_final.csv', index=False)

# ---------- 4. 校准曲线 + DCA（三基因模型） ----------
X3all = logcpm.loc[X.index, THREE]
rs_all = sum(COEF[g]*X3all[g] for g in THREE)
cdf = pd.DataFrame({'rs': rs_all.values,
                    'T': ydf['OS_time_days'].values,
                    'E': ydf['OS_event'].astype(int).values}, index=X.index)
cph = CoxPHFitter()
cph.fit(cdf, duration_col='T', event_col='E')
print()
print('三基因 Cox 重新拟合: HR=%.4f (95%%CI %.4f-%.4f), p=%.3g' % (
    cph.hazard_ratios_['rs'],
    np.exp(cph.confidence_intervals_.loc['rs'].iloc[0]),
    np.exp(cph.confidence_intervals_.loc['rs'].iloc[1]),
    cph.summary.loc['rs','p']))

# 校准：1/3/5年
calib = {}
for t in [365, 1095, 1825]:
    sf = cph.predict_survival_function(cdf, times=[t])
    pred_surv = sf.iloc[0].values
    grp = pd.qcut(pred_surv, 5, labels=False, duplicates='drop')
    obs, pre = [], []
    for g in sorted(pd.unique(grp)):
        m = grp == g
        kmf = KaplanMeierFitter()
        kmf.fit(cdf['T'][m], cdf['E'][m])
        s = kmf.predict(t)
        obs.append(float(s) if np.isfinite(s) else np.nan)
        pre.append(float(pred_surv[m].mean()))
    calib[t] = {'predicted': pre, 'observed': obs}
    print('  %d年校准  预测: %s' % (t//365, ' '.join('%.2f'%v for v in pre)))
    print('          实际: %s' % (' '.join('%.2f'%v for v in obs)))

# DCA（IPCW 加权）
def dca(cdf, surv_prob, t, thresholds):
    T, E = cdf['T'].values, cdf['E'].values
    # 删失分布 KM
    kmf_c = KaplanMeierFitter(); kmf_c.fit(T, 1-E)
    G_T = np.array([kmf_c.predict(x) if x < kmf_c._estimation_method else 1e-6 for x in T])
    G_T = np.clip(G_T, 1e-6, 1)
    G_t = float(kmf_c.predict(t)) if t < T.max() else 1e-6
    G_t = max(G_t, 1e-6)
    p = 1 - surv_prob
    n = len(T)
    nb_model, nb_all = [], []
    for pt in thresholds:
        hi = p >= pt
        tp = np.sum((T <= t) & (E==1) & hi / G_T)
        fp = np.sum((T >  t) & hi / G_t)
        nb_model.append(tp/n - fp/n * (pt/(1-pt)))
        prev = np.sum((T<=t)&(E==1)/G_T)/n
        nb_all.append(prev - (1-prev)*(pt/(1-pt)))
    return nb_model, nb_all

thresholds = np.arange(0.05, 0.95, 0.05)
dca_res = {}
for t in [365, 1095, 1825]:
    sf = cph.predict_survival_function(cdf, times=[t])
    nb_m, nb_a = dca(cdf, sf.iloc[0].values, t, thresholds)
    dca_res[t] = {'thresholds': thresholds.tolist(), 'model': nb_m, 'treat_all': nb_a}

json.dump({'calibration': {str(k): v for k,v in calib.items()},
           'dca': {str(k): v for k,v in dca_res.items()}},
          open(BASE+'/calibration_dca_results.json','w'), indent=2)
print()
print('已保存: ml_model_comparison_final.csv, calibration_dca_results.json')
