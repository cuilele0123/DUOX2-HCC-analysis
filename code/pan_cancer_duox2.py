# -*- coding: utf-8 -*-
"""泛癌分析 v2：DUOX2 在多个 TCGA 癌种中的表达与预后（修正映射）
分子数据自带 patientId，直接使用；患者临床数据含 OS_MONTHS/OS_STATUS
"""
import urllib.request, json, time
import pandas as pd
import numpy as np

import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = f'{BASE}/data'

studies = {
    'LIHC': 'lihc_tcga_pan_can_atlas_2018',
    'CHOL': 'chol_tcga_pan_can_atlas_2018',
    'PAAD': 'paad_tcga_pan_can_atlas_2018',
    'STAD': 'stad_tcga_pan_can_atlas_2018',
    'COADREAD': 'coadread_tcga_pan_can_atlas_2018',
    'ESCA': 'esca_tcga_pan_can_atlas_2018',
    'BLCA': 'blca_tcga_pan_can_atlas_2018',
    'KIRC': 'kirc_tcga_pan_can_atlas_2018',
    'KIRP': 'kirp_tcga_pan_can_atlas_2018',
    'KICH': 'kich_tcga_pan_can_atlas_2018',
    'BRCA': 'brca_tcga_pan_can_atlas_2018',
    'LUAD': 'luad_tcga_pan_can_atlas_2018',
    'LUSC': 'lusc_tcga_pan_can_atlas_2018',
    'HNSC': 'hnsc_tcga_pan_can_atlas_2018',
    'UCEC': 'ucec_tcga_pan_can_atlas_2018',
    'THCA': 'thca_tcga_pan_can_atlas_2018',
    'PRAD': 'prad_tcga_pan_can_atlas_2018',
    'GBM': 'gbm_tcga_pan_can_atlas_2018',
    'CESC': 'cesc_tcga_pan_can_atlas_2018',
    'OV': 'ov_tcga_pan_can_atlas_2018',
}

def api_get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == retries-1: raise
            time.sleep(2)

def api_post(url, payload, retries=3):
    for i in range(retries):
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data, headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == retries-1: raise
            time.sleep(2)

DUOX2_ENTREZ = 50506
results = []
for cancer, sid in studies.items():
    try:
        # 分子谱
        mprofiles = api_get(f'https://www.cbioportal.org/api/studies/{sid}/molecular-profiles')
        mrna_profile = None
        for m in mprofiles:
            if m['molecularAlterationType'] == 'MRNA_EXPRESSION' and m['datatype'] == 'CONTINUOUS':
                mrna_profile = m['molecularProfileId']; break
        if not mrna_profile:
            print(f'{cancer}: 无 mRNA profile'); continue

        # DUOX2 表达（GET endpoint 简单可靠）
        mol_data = api_get(f'https://www.cbioportal.org/api/molecular-profiles/{mrna_profile}/molecular-data?sampleListId={sid}_all&entrezGeneId={DUOX2_ENTREZ}')

        # 患者 OS（DETAILED 获取属性，构建 patientId → {os, status}）
        pat_clin = api_get(f'https://www.cbioportal.org/api/studies/{sid}/clinical-data?clinicalDataType=PATIENT&projection=DETAILED')
        pat_os, pat_status = {}, {}
        for c in pat_clin:
            aid = c.get('clinicalAttributeId')
            if aid == 'OS_MONTHS':
                try: pat_os[c['patientId']] = float(c['value'])
                except: pass
            elif aid == 'OS_STATUS':
                pat_status[c['patientId']] = c.get('value','')

        # 样本类型（SAMPLE 临床）
        samp_clin = api_get(f'https://www.cbioportal.org/api/studies/{sid}/clinical-data?clinicalDataType=SAMPLE&projection=DETAILED')
        samp_type = {}
        for c in samp_clin:
            if c.get('clinicalAttributeId') == 'SAMPLE_TYPE':
                samp_type[c['sampleId']] = c.get('value','')

        rows = []
        for d in mol_data:
            val = d.get('value')
            if val is None: continue
            pid = d.get('patientId')
            sid_ = d.get('sampleId')
            os_m = pat_os.get(pid, np.nan)
            st = pat_status.get(pid, '')
            event = 1 if 'DECEASED' in st.upper() else (0 if 'LIVING' in st.upper() else np.nan)
            rows.append({'cancer': cancer, 'sample_id': sid_, 'patient_id': pid,
                         'DUOX2_expr': float(val), 'OS_months': os_m, 'OS_event': event,
                         'sample_type': samp_type.get(sid_, '')})
        df = pd.DataFrame(rows)
        results.append(df)
        n_expr = len(df); n_surv = df['OS_months'].notna().sum()
        print(f'{cancer}: 表达 {n_expr}, 有生存 {n_surv}', flush=True)
    except Exception as e:
        print(f'{cancer}: ERR {e}', flush=True)
    time.sleep(0.5)

all_data = pd.concat(results, ignore_index=True)
all_data.to_csv(f'{DATA}/pan_cancer_DUOX2.csv', index=False)
print(f'\n总样本: {len(all_data)}, 癌种: {all_data["cancer"].nunique()}')
