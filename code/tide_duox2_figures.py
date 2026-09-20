# -*- coding: utf-8 -*-
"""TIDE 与 DUOX2 关联 + 生成全部增强分析图表"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['STHeiti','Heiti SC','PingFang SC','Heiti TC','STSong','SimHei','sans-serif']
plt.rcParams['axes.unicode_minus'] = False

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE=_ROOT
DATA=BASE+'/data'; FIG=BASE+'/figures'

# ---------- 1. TIDE × DUOX2 ----------
logcpm=pd.read_pickle(DATA+'/logcpm_matrix.pkl')
tide=pd.read_csv(DATA+'/tide_output.txt', sep='\t', index_col=0)
common=[s for s in tide.index if s in logcpm.index]
tide=tide.loc[common]
duox2=logcpm.loc[common,'DUOX2']
print('TIDE 样本: %d | 与表达矩阵交集: %d' % (len(tide), len(common)))

rows=[]
for col in ['TIDE','Dysfunction','Exclusion','MDSC','CAF','TAM M2','IFNG','CD8','CTL','MSI Score','CD274']:
    if col not in tide.columns: continue
    x=duox2.values; y=pd.to_numeric(tide[col],errors='coerce').values
    m=np.isfinite(x)&np.isfinite(y)
    rho,p=stats.spearmanr(x[m],y[m])
    rows.append({'TIDE_metric':col,'Spearman_rho':round(float(rho),4),'P_value':float(p),
                 'n':int(m.sum())})
td=pd.DataFrame(rows).sort_values('Spearman_rho')
print()
print('=== DUOX2 表达与 TIDE 免疫指标的相关性 (Spearman) ===')
for _,r in td.iterrows():
    star='***' if r.P_value<0.001 else '**' if r.P_value<0.01 else '*' if r.P_value<0.05 else ''
    print('  %-14s rho=%+.4f  p=%.3g %s' % (r.TIDE_metric, r.Spearman_rho, r.P_value, star))
td.to_csv(DATA+'/DUOX2_TIDE_correlation.csv', index=False)

# 高低 DUOX2 组的 TIDE 差异
med=duox2.median(); grp=(duox2>med).map({True:'DUOX2-high',False:'DUOX2-low'})
print()
print('=== DUOX2 高/低表达组的 TIDE 指标差异 (Mann-Whitney) ===')
comp=[]
for col in ['TIDE','Dysfunction','Exclusion','MDSC','CAF','TAM M2','IFNG','CD8','CTL','CD274']:
    if col not in tide.columns: continue
    a=pd.to_numeric(tide.loc[grp=='DUOX2-high',col],errors='coerce').dropna()
    b=pd.to_numeric(tide.loc[grp=='DUOX2-low',col],errors='coerce').dropna()
    if len(a)<10 or len(b)<10: continue
    u,p=stats.mannwhitneyu(a,b,alternative='two-sided')
    comp.append({'metric':col,'median_high':round(float(a.median()),4),
                 'median_low':round(float(b.median()),4),'P':float(p)})
cdf=pd.DataFrame(comp)
for _,r in cdf.iterrows():
    star='***' if r.P<0.001 else '**' if r.P<0.01 else '*' if r.P<0.05 else ''
    print('  %-12s 高=%8.4f 低=%8.4f  p=%.3g %s' % (r.metric,r.median_high,r.median_low,r.P,star))
cdf.to_csv(DATA+'/DUOX2_TIDE_group_comparison.csv', index=False)

# ---------- 2. 图1：ML 模型对比 ----------
ml=pd.read_csv(DATA+'/ml_model_comparison_final.csv')
fig,ax=plt.subplots(figsize=(9,5))
ml_s=ml.sort_values('C_index')
colors=['#c0392b' if '三基因' in m else '#5b8ff9' for m in ml_s['Model']]
bars=ax.barh(range(len(ml_s)), ml_s['C_index'], xerr=ml_s['SD'],
             color=colors, alpha=0.85, capsize=4, error_kw={'ecolor':'#333','elinewidth':1})
ax.set_yticks(range(len(ml_s)))
ax.set_yticklabels([m.replace(' (Coxnet, alpha tuned)','') for m in ml_s['Model']], fontsize=10)
ax.set_xlabel('Cross-validated C-index (5 repeats × 5 folds)', fontsize=11)
ax.set_title('Prognostic model comparison: the 3-gene signature ranks first', fontsize=12, weight='bold')
ax.axvline(0.5, ls='--', c='gray', lw=1, label='Random (0.5)')
for i,(v,s) in enumerate(zip(ml_s['C_index'], ml_s['SD'])):
    ax.text(v+s+0.008, i, '%.3f'%v, va='center', fontsize=9.5, weight='bold')
ax.set_xlim(0.45, 0.80); ax.legend(loc='lower right'); ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig(FIG+'/Enh_Fig1_ML_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close(); print('\n✓ Enh_Fig1_ML_model_comparison.png')

# ---------- 3. 图2：校准曲线 ----------
cal=json.load(open(DATA+'/calibration_dca_results.json'))
fig,axes=plt.subplots(1,3,figsize=(13,4.3))
for ax,t in zip(axes,['365','1095','1825']):
    c=cal['calibration'][t]; pre,obs=c['predicted'],c['observed']
    ax.plot([0,1],[0,1],'k--',lw=1,label='Perfect calibration')
    ax.plot(pre,obs,'o-',color='#c0392b',lw=2,ms=7,label='3-gene model')
    ax.set_xlabel('Predicted survival probability',fontsize=9.5)
    ax.set_ylabel('Observed (Kaplan-Meier)',fontsize=9.5)
    ax.set_title('%s-year overall survival'%('1' if t=='365' else '3' if t=='1095' else '5'),
                 fontsize=11,weight='bold')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(fontsize=8,loc='upper left'); ax.grid(alpha=0.3)
plt.suptitle('Calibration curves of the 3-gene prognostic model', fontsize=12, weight='bold')
plt.tight_layout(); plt.savefig(FIG+'/Enh_Fig2_Calibration.png', dpi=300, bbox_inches='tight')
plt.close(); print('✓ Enh_Fig2_Calibration.png')

# ---------- 4. 图3：DCA ----------
fig,axes=plt.subplots(1,3,figsize=(13,4.3))
for ax,t in zip(axes,['365','1095','1825']):
    d=cal['dca'][t]; th,m,a,n=d['thresholds'],d['model'],d['treat_all'],d['treat_none']
    ax.plot(th,m,'-',color='#c0392b',lw=2.2,label='3-gene model')
    ax.plot(th,a,'--',color='#5b8ff9',lw=1.8,label='Treat all')
    ax.plot(th,n,':',color='gray',lw=1.8,label='Treat none')
    ax.set_xlabel('Threshold probability',fontsize=9.5)
    ax.set_ylabel('Net benefit',fontsize=9.5)
    ax.set_title('%s-year DCA'%('1' if t=='365' else '3' if t=='1095' else '5'),fontsize=11,weight='bold')
    ax.set_ylim(-0.6,0.6); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Decision curve analysis: net benefit of the 3-gene model', fontsize=12, weight='bold')
plt.tight_layout(); plt.savefig(FIG+'/Enh_Fig3_DCA.png', dpi=300, bbox_inches='tight')
plt.close(); print('✓ Enh_Fig3_DCA.png')

# ---------- 5. 图4：DUOX2 vs TIDE ----------
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
key=[k for k in ['TIDE','Dysfunction','Exclusion','MDSC','CAF','TAM M2','IFNG','CD8','CTL','CD274'] if k in tide.columns]
ax=axes[0]
sub=td[td.TIDE_metric.isin(key)].sort_values('Spearman_rho')
cols=['#c0392b' if v>0 else '#2e86ab' for v in sub.Spearman_rho]
ax.barh(sub.TIDE_metric, sub.Spearman_rho, color=cols, alpha=0.85)
ax.axvline(0,c='k',lw=1)
ax.set_xlabel('Spearman rho (DUOX2 vs TIDE metric)',fontsize=9.5)
ax.set_title('DUOX2 vs TIDE immune metrics',fontsize=11,weight='bold')
for i,v in enumerate(sub.Spearman_rho):
    ax.text(v+(0.012 if v>0 else -0.012), i, '%+.3f'%v, va='center',
            ha='left' if v>0 else 'right', fontsize=8.5)
ax.set_xlim(-0.45,0.45); ax.grid(axis='x',alpha=0.3)
ax=axes[1]
hi=pd.to_numeric(tide.loc[grp=='DUOX2-high','TIDE'],errors='coerce').dropna()
lo=pd.to_numeric(tide.loc[grp=='DUOX2-low','TIDE'],errors='coerce').dropna()
bp=ax.boxplot([lo,hi],patch_artist=True,widths=0.55)
ax.set_xticks([1,2]); ax.set_xticklabels(['DUOX2-low','DUOX2-high'])
for p,c in zip(bp['boxes'],['#2e86ab','#c0392b']): p.set_facecolor(c); p.set_alpha(0.65)
u,pv=stats.mannwhitneyu(hi,lo,alternative='two-sided')
ax.set_ylabel('TIDE score',fontsize=9.5)
ax.set_title('TIDE score by DUOX2 expression\n(Mann-Whitney p = %.3g)'%pv,fontsize=11,weight='bold')
ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(FIG+'/Enh_Fig4_DUOX2_TIDE.png', dpi=300, bbox_inches='tight')
plt.close(); print('✓ Enh_Fig4_DUOX2_TIDE.png')
print()
print('所有图表已保存到 figures/')
