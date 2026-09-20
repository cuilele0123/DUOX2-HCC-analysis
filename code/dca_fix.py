# -*- coding: utf-8 -*-
"""修复DCA的bug，重跑校准+DCA（三基因模型）"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter, KaplanMeierFitter

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(_ROOT, 'data')
THREE=['SLC16A3','SPP2','MMP7']
COEF={'SLC16A3':0.0920072109058105,'SPP2':-0.03765527210177806,'MMP7':0.023524515638499957}

logcpm=pd.read_pickle(DATA+'/logcpm_matrix.pkl')
surv=pd.read_pickle(DATA+'/survival_data.pkl').set_index('sample_id')
surv=surv[surv['OS_time_days']>0]
samples=[s for s in surv.index if s in logcpm.index]
X3=logcpm.loc[samples,THREE].astype(float)
ydf=surv.loc[samples,['OS_event','OS_time_days']]

rs=sum(COEF[g]*X3[g] for g in THREE)
cdf=pd.DataFrame({'rs':rs.values,'T':ydf['OS_time_days'].values,
                  'E':ydf['OS_event'].astype(int).values},index=X3.index)
cph=CoxPHFitter(); cph.fit(cdf,duration_col='T',event_col='E')
print('三基因Cox: HR=%.3f (95%%CI %.3f-%.3f), p=%.3g'%(
    cph.hazard_ratios_['rs'],np.exp(cph.confidence_intervals_.loc['rs'].iloc[0]),
    np.exp(cph.confidence_intervals_.loc['rs'].iloc[1]),cph.summary.loc['rs','p']))
print('样本: %d, 事件: %d'%(len(cdf),cdf['E'].sum()))

# 校准曲线
calib={}
for t in [365,1095,1825]:
    ps=cph.predict_survival_function(cdf,times=[t]).iloc[0].values
    grp=pd.qcut(ps,5,labels=False,duplicates='drop'); obs,pre=[],[]
    for g in sorted(pd.unique(grp)):
        m=grp==g; kmf=KaplanMeierFitter(); kmf.fit(cdf['T'][m],cdf['E'][m])
        s=kmf.predict(t)
        obs.append(round(float(s),4) if np.isfinite(s) else None)
        pre.append(round(float(ps[m].mean()),4))
    calib[str(t)]={'predicted':pre,'observed':obs}
    print('  %d年  预测: %s'%(t//365,' '.join('%.2f'%v for v in pre)))
    print('        实际: %s'%(' '.join('%.2f'%v for v in obs if v)))

# DCA（修复：去掉错误比较，用 IPCW 权重）
def dca(cdf,sp,t,ths):
    T,E=cdf['T'].values.astype(float),cdf['E'].values.astype(int); n=len(T)
    kc=KaplanMeierFitter(); kc.fit(T,1-E)              # 删失分布
    tmax=T.max()-1
    G_T=np.clip(np.array([kc.predict(min(x,tmax)) or 1e-6 for x in T]),1e-6,1)
    G_t=max(float(kc.predict(min(t,tmax))) or 1e-6,1e-6)
    p=1-sp; nb,nb_all=[],[]
    for pt in ths:
        hi=p>=pt
        tp=np.sum(((T<=t)&(E==1)&hi)/G_T)/n
        fp=np.sum(((T>t)&hi)/G_t)/n
        nb.append(round(float(tp-fp*(pt/(1-pt))),4))
        prev=float(np.sum(((T<=t)&(E==1))/G_T)/n)
        nb_all.append(round(float(prev-(1-prev)*(pt/(1-pt))),4))
    return nb,nb_all

ths=np.round(np.arange(0.05,0.95,0.05),2); d={}
for t in [365,1095,1825]:
    sp=cph.predict_survival_function(cdf,times=[t]).iloc[0].values
    m,a=dca(cdf,sp,t,ths); d[str(t)]={'thresholds':ths.tolist(),'model':m,'treat_all':a,'treat_none':[0.0]*len(ths)}
    # 报告净获益峰值
    print('  %d年 DCA: 模型最大净获益=%.4f @阈值%.2f | treat-all=%.4f'%(
        t//365,max(m),ths[int(np.argmax(m))],a[int(np.argmax(m))]))

json.dump({'calibration':calib,'dca':d},open(DATA+'/calibration_dca_results.json','w'),indent=2)
print(); print('✓ 已保存 calibration_dca_results.json')
