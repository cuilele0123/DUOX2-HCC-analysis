# -*- coding: utf-8 -*-
"""修正版：忠实复现原文候选集；三基因取完整表达；Coxnet调alpha；校准曲线+DCA"""
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
DATA=os.path.join(_ROOT, 'data')
THREE=['SLC16A3','SPP2','MMP7']
COEF={'SLC16A3':0.0920072109058105,'SPP2':-0.03765527210177806,'MMP7':0.023524515638499957}

logcpm=pd.read_pickle(DATA+'/logcpm_matrix.pkl')
surv=pd.read_pickle(DATA+'/survival_data.pkl').set_index('sample_id')
surv=surv[surv['OS_time_days']>0]
genes=list(np.load(DATA+'/candidate_genes_500.npy', allow_pickle=True))
genes=[g for g in genes if g in logcpm.columns]
samples=[s for s in surv.index if s in logcpm.index]

X=logcpm.loc[samples, genes].astype(float)
X3=logcpm.loc[samples, THREE].astype(float)          # 三基因：完整表达
ydf=surv.loc[samples,['OS_event','OS_time_days']].copy()
ydf['OS_event']=ydf['OS_event'].astype(bool)
y=Surv.from_arrays(event=ydf['OS_event'].values, time=ydf['OS_time_days'].values)
print('候选基因: %d | 样本: %d | 事件: %d' % (X.shape[1], X.shape[0], ydf['OS_event'].sum()))
print('三基因齐全:', all(g in X3.columns for g in THREE))

def select_alpha(Xa, ya, l1_ratio, seed):
    try:
        cn=CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01, n_alphas=40, max_iter=100000)
        cn.fit(Xa, ya); grid=cn.alphas_[::4]
    except Exception: return None
    acc={a:[] for a in grid}
    for tr,te in KFold(3, shuffle=True, random_state=seed).split(Xa):
        try:
            c2=CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01, n_alphas=40, max_iter=100000)
            c2.fit(Xa.iloc[tr], ya[tr])
        except Exception: continue
        ev=np.array([e for e,t in ya[te]]); tm=np.array([t for e,t in ya[te]])
        if len(np.unique(ev))<2: continue
        for a in grid:
            try: acc[a].append(concordance_index_censored(ev,tm,c2.predict(Xa.iloc[te],alpha=a))[0])
            except Exception: pass
    m={a:np.mean(v) for a,v in acc.items() if v}
    return max(m,key=m.get) if m else None

def models(seed):
    return {'Lasso-Cox (Coxnet)':('coxnet',1.0),'Ridge-Cox (Coxnet)':('coxnet',0.01),
            'Random Survival Forest':('m',RandomSurvivalForest(n_estimators=300,min_samples_leaf=15,
                max_features='sqrt',random_state=seed,n_jobs=-1)),
            'Gradient Boosting':('m',GradientBoostingSurvivalAnalysis(n_estimators=300,learning_rate=0.05,
                max_depth=2,subsample=0.8,random_state=seed)),
            'Survival SVM':('m',FastSurvivalSVM(max_iter=200,tol=1e-5,random_state=seed))}

res={}
for rep in range(5):
    for tr,te in KFold(5,shuffle=True,random_state=42+rep).split(X):
        Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y[tr],y[te]
        X3tr,X3te=X3.iloc[tr],X3.iloc[te]
        ev=np.array([e for e,t in yte]); tm=np.array([t for e,t in yte])
        if len(np.unique(ev))<2: continue
        # 三基因固定系数
        rs=sum(COEF[g]*X3te[g] for g in THREE)
        res.setdefault('三基因模型 (SLC16A3+SPP2+MMP7)',[]).append(
            concordance_index_censored(ev,tm,rs.values)[0])
        for name,(kind,obj) in models(42+rep).items():
            try:
                if kind=='coxnet':
                    cn=CoxnetSurvivalAnalysis(l1_ratio=obj,alpha_min_ratio=0.01,n_alphas=40,max_iter=100000)
                    cn.fit(Xtr,ytr); a=select_alpha(Xtr,ytr,obj,42+rep)
                    pred=cn.predict(Xte,alpha=a) if a else cn.predict(Xte)
                else:
                    pl=Pipeline([('sc',StandardScaler()),('m',obj)]); pl.fit(Xtr,ytr); pred=pl.predict(Xte)
                res.setdefault(name,[]).append(concordance_index_censored(ev,tm,pred)[0])
            except Exception as e: print('  [skip %s] %s'%(name,str(e)[:40]))

print(); print('='*66); print('5次重复×5折CV C-index（忠实复现原文候选集）'); print('='*66)
rows=[{'Model':k,'C_index':np.mean(v),'SD':np.std(v),'n':len(v)} for k,v in res.items() if v]
out=pd.DataFrame(rows).sort_values('C_index',ascending=False)
for _,r in out.iterrows(): print('%-34s %.4f ± %.4f (n=%d)'%(r['Model'],r['C_index'],r['SD'],r['n']))
out.to_csv(DATA+'/ml_model_comparison_final.csv',index=False)

# ---- 校准 + DCA ----
rs_all=sum(COEF[g]*X3[g] for g in THREE)
cdf=pd.DataFrame({'rs':rs_all.values,'T':ydf['OS_time_days'].values,'E':ydf['OS_event'].astype(int).values},index=X3.index)
cph=CoxPHFitter(); cph.fit(cdf,duration_col='T',event_col='E')
print(); print('三基因Cox: HR=%.4f (95%%CI %.4f-%.4f) p=%.3g'%(cph.hazard_ratios_['rs'],
    np.exp(cph.confidence_intervals_.loc['rs'].iloc[0]),np.exp(cph.confidence_intervals_.loc['rs'].iloc[1]),
    cph.summary.loc['rs','p']))
calib={}
for t in [365,1095,1825]:
    ps=cph.predict_survival_function(cdf,times=[t]).iloc[0].values
    grp=pd.qcut(ps,5,labels=False,duplicates='drop'); obs,pre=[],[]
    for g in sorted(pd.unique(grp)):
        m=grp==g; kmf=KaplanMeierFitter(); kmf.fit(cdf['T'][m],cdf['E'][m])
        s=kmf.predict(t); obs.append(float(s) if np.isfinite(s) else None); pre.append(float(ps[m].mean()))
    calib[t]={'predicted':pre,'observed':obs}
    print('  %d年 预测 %s'%(t//365,' '.join('%.2f'%v for v in pre)))
    print('      实际 %s'%(' '.join('%.2f'%v for v in obs if v is not None)))
def dca(cdf,sp,t,ths):
    T,E=cdf['T'].values,cdf['E'].values; n=len(T)
    kc=KaplanMeierFitter(); kc.fit(T,1-E)
    G_T=np.clip(np.array([kc.predict(min(x,T.max()-1)) for x in T]),1e-6,1)
    G_t=max(float(kc.predict(t)) if t<T.max() else 1e-6,1e-6)
    p=1-sp; nb,nb_all=[],[]
    for pt in ths:
        hi=p>=pt
        tp=np.sum(((T<=t)&(E==1)&hi)/G_T); fp=np.sum(((T>t)&hi)/G_t)
        nb.append(tp/n-fp/n*(pt/(1-pt)))
        prev=np.sum(((T<=t)&(E==1))/G_T)/n; nb_all.append(prev-(1-prev)*(pt/(1-pt)))
    return nb,nb_all
ths=np.arange(0.05,0.95,0.05); d={}
for t in [365,1095,1825]:
    sp=cph.predict_survival_function(cdf,times=[t]).iloc[0].values
    m,a=dca(cdf,sp,t,ths); d[t]={'thresholds':ths.tolist(),'model':m,'treat_all':a}
json.dump({'calibration':{str(k):v for k,v in calib.items()},'dca':{str(k):v for k,v in d.items()}},
          open(DATA+'/calibration_dca_results.json','w'),indent=2)
print(); print('已保存 ml_model_comparison_final.csv, calibration_dca_results.json')
