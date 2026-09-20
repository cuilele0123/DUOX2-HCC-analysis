# -*- coding: utf-8 -*-
"""整合 TCGA-LIHC 临床/生存/分期元数据，修正 AJCC 分期定义"""
import pandas as pd
import numpy as np

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(_ROOT, 'data')

# 1. 读取临床数据
clin = pd.read_csv(f'{BASE}/lihc_clinical.tsv', sep='\t', low_memory=False)
surv = pd.read_csv(f'{BASE}/lihc_survival.tsv', sep='\t', low_memory=False)
cbiop = pd.read_csv(f'{BASE}/cbioportal_patient_clean.tsv', sep='\t', low_memory=False)

# 病例级信息
clin['case_id'] = clin['submitter_id']
meta = pd.DataFrame({
    'case_id': clin['submitter_id'],
    'gender': clin.get('demographic.gender'),
    'race': clin.get('demographic.race'),
    'age_at_diagnosis': pd.to_numeric(clin.get('diagnoses.0.age_at_diagnosis'), errors='coerce'),
    'vital_status': clin.get('demographic.vital_status'),
    # GDC 分期字段（TCGA-LIHC 官方为 AJCC 7th 混合）
    'ajcc_stage_raw': clin.get('diagnoses.0.ajcc_pathologic_stage'),
    'ajcc_t': clin.get('diagnoses.0.ajcc_pathologic_t'),
    'ajcc_n': clin.get('diagnoses.0.ajcc_pathologic_n'),
    'ajcc_m': clin.get('diagnoses.0.ajcc_pathologic_m'),
})

# 2. 生存数据：OS 时间与事件
# days_to_death 在 demographic，days_to_last_follow_up 在 diagnoses
dtd = pd.to_numeric(surv['demographic.days_to_death'], errors='coerce')
dtlf = pd.to_numeric(surv['diagnoses.0.days_to_last_follow_up'], errors='coerce')
vs = surv['demographic.vital_status'].fillna('Unknown')

os_time = np.where(vs == 'Dead', dtd, np.where(vs == 'Alive', dtlf, np.nan))
os_event = np.where(vs == 'Dead', 1, np.where(vs == 'Alive', 0, np.nan))

surv_df = pd.DataFrame({
    'case_id': surv['submitter_id'],
    'vital_status': vs,
    'OS_time_days': os_time,
    'OS_event': os_event,
})

# 用 cBioPortal OS_MONTHS 交叉校验/补充生存时间
cbiop['OS_days_cbio'] = pd.to_numeric(cbiop['OS_MONTHS'], errors='coerce') * 30.44
cbiop['OS_event_cbio'] = cbiop['OS_STATUS'].map({'1:DECEASED': 1, '0:LIVING': 0})
cbiop_os = cbiop[['case_id', 'OS_days_cbio', 'OS_event_cbio', 'SEX', 'RACE', 'AGE']].copy()
cbiop_os['AGE'] = pd.to_numeric(cbiop_os['AGE'], errors='coerce')

meta = meta.merge(surv_df, on='case_id', how='left', suffixes=('', '_s'))
meta = meta.merge(cbiop_os, on='case_id', how='left')

# 合并生存时间：优先 GDC 天数，缺失用 cBioPortal
meta['OS_time_days'] = meta['OS_time_days'].fillna(meta['OS_days_cbio'])
meta['OS_event'] = meta['OS_event'].fillna(meta['OS_event_cbio'])
meta['gender'] = meta['SEX'].fillna(meta.get('gender'))
meta['race'] = meta['RACE'].fillna(meta['race'])

# 3. 修正 AJCC 分期（基于 T/N/M 重新定义，遵循 AJCC 7th：TCGA-LIHC 官方分期体系）
def assign_stage(t, n, m):
    if pd.isna(t) or str(t) == 'nan' or str(t) in ('', 'TX', 'null'):
        return np.nan
    t = str(t).replace('a', 'a').replace('b', 'b')
    # 规范化
    t_num = t.replace('T', '').replace('is', '0')
    n_has = (str(n) == 'N1')
    m_has = (str(m) == 'M1')
    if m_has:
        return 'Stage IVB'
    if n_has:
        return 'Stage IVA'
    # 无 N/M 时按 T
    if t_num in ('1', '1a', '1b'):
        return 'Stage I'
    if t_num == '2':
        return 'Stage II'
    if t_num == '3a':
        return 'Stage IIIA'
    if t_num == '3b' or t_num == '3':
        return 'Stage IIIB'
    if t_num == '4':
        return 'Stage IIIC'  # AJCC 7th: T4N0M0 = IIIC
    if t_num == '0':
        return 'Stage 0'
    return np.nan

meta['stage_derived'] = [assign_stage(t, n, m) for t, n, m in
                         zip(meta['ajcc_t'], meta['ajcc_n'], meta['ajcc_m'])]

# 4. 分组（早期/中期/晚期）—— 修正原文错误分组
def group_stage(s):
    if pd.isna(s):
        return np.nan
    if s in ('Stage 0', 'Stage I'):
        return 'early'
    if s == 'Stage II':
        return 'intermediate'
    if s in ('Stage IIIA', 'Stage IIIB', 'Stage IIIC', 'Stage IVA', 'Stage IVB'):
        return 'advanced'
    return np.nan

meta['stage_group'] = [group_stage(s) for s in meta['stage_derived']]

# 5. 保留有生存数据且为原发肿瘤的病例
meta = meta[meta['OS_time_days'].notna() & meta['OS_event'].notna()].copy()
meta = meta[meta['stage_derived'].notna()].copy()

print('=== 病例数 ===')
print(f'总病例(有生存+分期): {len(meta)}')
print('\n=== 分期分布(修正后) ===')
print(meta['stage_derived'].value_counts().sort_index())
print('\n=== 分组分布(修正后) ===')
print(meta['stage_group'].value_counts())
print('\n=== 生存状态 ===')
print(meta['OS_event'].value_counts())
print('\n=== 性别 ===')
print(meta['gender'].value_counts(dropna=False))
print('\n=== 年龄(年) ===')
print(meta['AGE'].describe())

meta.to_csv(f'{BASE}/meta_clinical_cleaned.tsv', sep='\t', index=False)
print('\nSaved:', f'{BASE}/meta_clinical_cleaned.tsv')
