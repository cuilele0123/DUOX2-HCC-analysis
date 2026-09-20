# -*- coding: utf-8 -*-
"""外部验证：cBioPortal Firehose Legacy 数据集（lihc_tcga，379例，TCGA同一队列的另一处理版本）"""
import urllib.request, json, time, os
import pandas as pd
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

def api_get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == retries-1: raise
            time.sleep(2)

# Firehose Legacy 队列（与 PanCancer Atlas 同一批样本但处理版本不同）
sid = 'lihc_tcga'
# 查找 mRNA 谱
mprofiles = api_get(f'https://www.cbioportal.org/api/studies/{sid}/molecular-profiles')
mrna = None
for m in mprofiles:
    if m['molecularAlterationType'] == 'MRNA_EXPRESSION' and m['datatype'] == 'CONTINUOUS':
        mrna = m['molecularProfileId']; break
print(f'mRNA 谱: {mrna}')

# 模型基因：DUOX2(50506), SLC16A3(9123), SPP2(6691), MMP7(4316)
genes_ent = {'DUOX2':50506, 'SLC16A3':9123, 'SPP2':6691, 'MMP7':4316}

# 下载表达
all_data = []
for g, eid in genes_ent.items():
    d = api_get(f'https://www.cbioportal.org/api/molecular-profiles/{mrna}/molecular-data?sampleListId={sid}_all&entrezGeneId={eid}')
    print(f'{g}: {len(d)} 样本')
    for x in d:
        all_data.append({'sample_id': x['sampleId'], 'patient_id': x['patientId'], 'gene': g, 'value': x['value']})
expr_df = pd.DataFrame(all_data)
expr_df.to_csv(f'{DATA}/firehose_LIHC_expr.tsv', sep='\t', index=False)

# 临床生存
pc = api_get(f'https://www.cbioportal.org/api/studies/{sid}/clinical-data?clinicalDataType=PATIENT&projection=DETAILED')
os_m = {}; os_st = {}
for c in pc:
    if c.get('clinicalAttributeId') == 'OS_MONTHS':
        try: os_m[c['patientId']] = float(c['value'])
        except: pass
    elif c.get('clinicalAttributeId') == 'OS_STATUS':
        os_st[c['patientId']] = c.get('value','')
clin_df = pd.DataFrame([{'patient_id':p, 'OS_months':os_m[p], 'OS_status':os_st[p]} for p in os_m])
clin_df.to_csv(f'{DATA}/firehose_LIHC_clinical.tsv', sep='\t', index=False)

# 分期
stg = {}
for c in pc:
    if c.get('clinicalAttributeId') == 'AJCC_PATHOLOGIC_TUMOR_STAGE':
        stg[c['patientId']] = c.get('value','')
pd.DataFrame([{'patient_id':p, 'stage':stg[p]} for p in stg]).to_csv(f'{DATA}/firehose_LIHC_stage.tsv', sep='\t', index=False)

print(f'\n总: 表达 {len(expr_df["sample_id"].unique())} 样本, 生存 {len(clin_df)} 患者, 分期 {len(stg)}')
