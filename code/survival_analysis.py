# -*- coding: utf-8 -*-
"""生存分析：KM + 单/多因素 Cox（修正原文 P0-2 方向性矛盾）
- 用修正后的分期分组
- 检查分期与生存的真实关系
- 输出 KM 图数据、Cox 结果表
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

meta = pd.read_csv(f'{DATA}/meta_clinical_cleaned.tsv', sep='\t')

# 分析集：原发肿瘤 + 有分期 + 有生存
sample_meta = pd.read_csv(f'{DATA}/sample_meta.tsv', sep='\t')
tumor = sample_meta[sample_meta['sample_type'] == 'Primary Tumor']
analy = tumor.merge(meta[['case_id', 'stage_group', 'stage_derived', 'OS_time_days', 'OS_event',
                           'gender', 'AGE', 'race']], on='case_id', how='inner')
analy = analy[analy['stage_group'].notna() & analy['OS_time_days'].notna() & analy['OS_event'].notna()]

print(f'生存分析样本: {len(analy)}')
print(analy['stage_group'].value_counts().to_dict())

# 保留每个 case 一个肿瘤样本（去重）
analy = analy.drop_duplicates(subset='case_id', keep='first')
print(f'去重后: {len(analy)}')

# ===== 1. KM 分析（按分期组）=====
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

print('\n=== 1. KM 生存分析（按分期组）===')
kmf = KaplanMeierFitter()
groups = ['early', 'intermediate', 'advanced']
labels = {'early': '早期(I期)', 'intermediate': '中期(II期)', 'advanced': '晚期(III/IV期)'}

surv_summary = {}
for g in groups:
    d = analy[analy['stage_group'] == g]
    kmf.fit(d['OS_time_days'], event_observed=d['OS_event'], label=labels[g])
    # 提取 1/3/5 年生存率
    s = kmf.survival_function_
    times = {'1年': 365, '3年': 1095, '5年': 1825}
    rates = {}
    for k, t in times.items():
        sub = s[s.index <= t]
        rate = sub.iloc[-1].values[0] if len(sub) else np.nan
        rates[k] = f'{rate:.1%}' if not np.isnan(rate) else 'NA'
    surv_summary[g] = rates
    print(f'{labels[g]}: n={len(d)}, 死亡={d["OS_event"].sum()} | ' +
          ' | '.join(f'{k} {v}' for k, v in rates.items()))

# log-rank 检验（两两比较）
print('\nlog-rank 检验（两两比较）:')
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        a = analy[analy['stage_group'] == groups[i]]
        b = analy[analy['stage_group'] == groups[j]]
        result = logrank_test(a['OS_time_days'], b['OS_time_days'],
                              event_observed_A=a['OS_event'], event_observed_B=b['OS_event'])
        print(f'{labels[groups[i]]} vs {labels[groups[j]]}: p={result.p_value:.4f}')

# 3 组整体 log-rank
from lifelines.statistics import multivariate_logrank_test
result = multivariate_logrank_test(analy['OS_time_days'], analy['stage_group'], analy['OS_event'])
print(f'三组整体 log-rank: p={result.p_value:.2e}')

# ===== 2. Cox 单因素 =====
print('\n=== 2. 单因素 Cox ===')
from lifelines import CoxPHFitter

cph = CoxPHFitter()
# 分期作为分类变量，参考=早期
df_cox = analy[['OS_time_days', 'OS_event', 'stage_group']].copy()
df_cox['stage_group'] = pd.Categorical(df_cox['stage_group'], categories=['early', 'intermediate', 'advanced'])

cph.fit(df_cox, duration_col='OS_time_days', event_col='OS_event', formula='stage_group')
print(cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].to_string())
print(f"C-index: {cph.concordance_index_:.3f}")

# ===== 3. 多因素 Cox（年龄+性别+分期）=====
print('\n=== 3. 多因素 Cox（年龄+性别+分期，参考=早期）===')
df_cox2 = analy[['OS_time_days', 'OS_event', 'stage_group', 'AGE', 'gender']].copy()
df_cox2['stage_group'] = pd.Categorical(df_cox2['stage_group'], categories=['early', 'intermediate', 'advanced'])
df_cox2['AGE'] = df_cox2['AGE'].fillna(df_cox2['AGE'].median())
df_cox2['gender_bin'] = (df_cox2['gender'] == 'Male').astype(int)
df_cox2 = df_cox2.dropna(subset=['gender_bin']).reset_index(drop=True)
# 显式 dummy 编码，规避 lifelines formula 的兼容问题
df_cox2 = pd.get_dummies(df_cox2, columns=['stage_group'], prefix='stage', drop_first=True)
df_cox2 = df_cox2.drop(columns=[c for c in ['gender'] if c in df_cox2.columns])
print(f'多因素模型样本: {len(df_cox2)}')
print(df_cox2[['OS_time_days', 'OS_event', 'AGE', 'gender_bin', 'stage_intermediate', 'stage_advanced']].head())
cph2 = CoxPHFitter()
cph2.fit(df_cox2, duration_col='OS_time_days', event_col='OS_event')
print(cph2.summary[['coef', 'exp(coef)', 'se(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']].to_string())
print(f"C-index: {cph2.concordance_index_:.3f}")

# ===== 4. Schoenfeld 残差检验（PH 假设）=====
print('\n=== 4. PH 假设检验（Schoenfeld 残差）===')
from lifelines.statistics import proportional_hazard_test
try:
    ph_test = proportional_hazard_test(cph2, df_cox2)
    print(ph_test.summary.to_string())
except Exception as e:
    print('PH test err:', e)

# 保存结果
analy.to_csv(f'{DATA}/survival_analysis_ready.tsv', sep='\t', index=False)
print('\n保存: survival_analysis_ready.tsv')
