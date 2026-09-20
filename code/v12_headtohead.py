#!/usr/bin/env python
"""
v12 additions (part 1 - local data only)
========================================
(A) Head-to-head comparison of the three-gene signature against AJCC pathologic
    stage (and age) in TCGA-LIHC, using out-of-fold cross-validated risk scores
    so that the signature is not credited with in-sample optimism.
(B) Multivariable Cox: does the signature add information beyond AJCC stage?
(C) Bootstrap confidence intervals for the difference in C-index.
(D) Firehose Legacy overlap with TCGA-LIHC, and validation restricted to the
    genuinely non-overlapping patients.
"""
import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, 'data')
OUT = {}
rng = np.random.default_rng(20250913)

COEF = {'SLC16A3': 0.092, 'SPP2': -0.038, 'MMP7': 0.024}
STAGE_MAP = {'Stage 0': 0, 'Stage I': 1, 'Stage II': 2, 'Stage III': 3,
             'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
             'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4}


def cidx(t, e, s):
    return concordance_index_censored(np.asarray(e, bool),
                                      np.asarray(t, float),
                                      np.asarray(s, float))[0]


def boot_diff(t, e, s_new, s_ref, n_boot=4000):
    t, e = np.asarray(t, float), np.asarray(e, bool)
    s_new, s_ref = np.asarray(s_new, float), np.asarray(s_ref, float)
    n = len(t)
    obs = cidx(t, e, s_new) - cidx(t, e, s_ref)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if e[i].sum() < 5:
            continue
        d.append(cidx(t[i], e[i], s_new[i]) - cidx(t[i], e[i], s_ref[i]))
    d = np.array(d)
    return (round(float(obs), 3), [round(float(np.percentile(d, 2.5)), 3),
                                   round(float(np.percentile(d, 97.5)), 3)],
            round(float((d <= 0).mean()), 4))


def main():
    lc = pd.read_pickle(os.path.join(D, 'logcpm_matrix.pkl'))
    lass = pd.read_csv(os.path.join(D, 'lasso_risk_analysis.csv'))
    met = pd.read_csv(os.path.join(D, 'meta_clinical_cleaned.tsv'), sep='\t')

    d = lass.merge(met[['case_id', 'ajcc_stage_raw', 'age_at_diagnosis', 'gender']],
                   on='case_id', how='left')
    d['sample_key'] = d['sample_id']
    # expression lookup: logcpm index are sample barcodes TCGA-XX-XXXX-01A
    key2i = {str(i): i for i in lc.index}
    d['lc_idx'] = [key2i.get(str(s), None) for s in d['sample_id']]
    miss = d['lc_idx'].isna().sum()
    print('samples without expression:', miss, '/', len(d))

    X = lc.loc[[i for i in d['lc_idx'] if i is not None],
               ['SLC16A3', 'SPP2', 'MMP7']].astype(float)
    X.index = [s for s, i in zip(d['sample_id'], d['lc_idx']) if i is not None]
    d = d[d['lc_idx'].notna()].set_index('sample_id').join(X, how='inner')
    d = d.dropna(subset=['OS_time_days', 'OS_event', 'ajcc_stage_raw'])
    d = d[d['OS_time_days'] > 0]
    d['ajcc_ord'] = d['ajcc_stage_raw'].map(STAGE_MAP)
    d = d.dropna(subset=['ajcc_ord'])
    print('analysis n =', len(d), 'events =', int(d['OS_event'].sum()))

    d['risk_fixed'] = sum(COEF[g] * d[g] for g in COEF)

    # ---------- out-of-fold cross-validated risk score ----------
    d['risk_oof'] = np.nan
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ev = d['OS_event'].astype(int).values
    for tr, te in skf.split(d, ev):
        trd, ted = d.iloc[tr], d.iloc[te]
        try:
            cph = CoxPHFitter()
            cph.fit(trd[['OS_time_days', 'OS_event', 'SLC16A3', 'SPP2', 'MMP7']],
                    'OS_time_days', 'event_col' if False else 'OS_event')
            d.iloc[te, d.columns.get_loc('risk_oof')] = cph.predict_partial_hazard(
                ted[['SLC16A3', 'SPP2', 'MMP7']]).values
        except Exception as e:
            print('fold failed', e)
    d = d.dropna(subset=['risk_oof'])
    print('OOF complete n =', len(d))

    tt, ee = d['OS_time_days'], d['OS_event'].astype(bool)
    h2h = {}
    for name, col in [('3-gene signature (out-of-fold)', 'risk_oof'),
                      ('3-gene signature (fixed TCGA coefficients)', 'risk_fixed'),
                      ('AJCC pathologic stage', 'ajcc_ord')]:
        h2h[name] = round(cidx(tt, ee, d[col]), 3)
    sub = d.dropna(subset=['age_at_diagnosis'])
    h2h['Age'] = round(cidx(sub['OS_time_days'], sub['OS_event'].astype(bool),
                            sub['age_at_diagnosis']), 3)
    OUT['tcga_h2h_cindex'] = h2h
    OUT['tcga_h2h_n'] = int(len(d))
    OUT['tcga_h2h_events'] = int(d['OS_event'].sum())

    # ---------- combined model ----------
    cph = CoxPHFitter()
    dd = d[['OS_time_days', 'OS_event', 'risk_oof', 'ajcc_ord']].astype(float)
    cph.fit(dd, 'OS_time_days', 'OS_event')
    comb = cph.predict_partial_hazard(d[['risk_oof', 'ajcc_ord']].astype(float)).values
    OUT['tcga_h2h_cindex']['3-gene + AJCC (combined)'] = round(cidx(tt, ee, comb), 3)
    OUT['tcga_multivariable_cox'] = {
        'n': int(len(d)),
        'HR_risk_per_SD': round(float(cph.hazard_ratios_['risk_oof']), 3),
        'p_risk': float(cph.summary.loc['risk_oof', 'p']),
        'HR_ajcc_per_stage': round(float(cph.hazard_ratios_['ajcc_ord']), 3),
        'p_ajcc': float(cph.summary.loc['ajcc_ord', 'p'])}

    # standardise the OOF risk for an interpretable per-SD HR
    sd = d['risk_oof'].std()
    cph2 = CoxPHFitter()
    d['risk_oof_z'] = (d['risk_oof'] - d['risk_oof'].mean()) / sd
    cph2.fit(d[['OS_time_days', 'OS_event', 'risk_oof_z', 'ajcc_ord']].astype(float),
             'OS_time_days', 'OS_event')
    OUT['tcga_multivariable_cox']['HR_risk_per_SD'] = round(
        float(cph2.hazard_ratios_['risk_oof_z']), 3)
    OUT['tcga_multivariable_cox']['CI_risk_per_SD'] = [
        round(float(np.exp(cph2.confidence_intervals_.loc['risk_oof_z'].iloc[0])), 3),
        round(float(np.exp(cph2.confidence_intervals_.loc['risk_oof_z'].iloc[1])), 3)]
    OUT['tcga_multivariable_cox']['HR_ajcc'] = round(
        float(cph2.hazard_ratios_['ajcc_ord']), 3)
    OUT['tcga_multivariable_cox']['CI_ajcc'] = [
        round(float(np.exp(cph2.confidence_intervals_.loc['ajcc_ord'].iloc[0])), 3),
        round(float(np.exp(cph2.confidence_intervals_.loc['ajcc_ord'].iloc[1])), 3)]
    OUT['tcga_multivariable_cox']['p_risk'] = float(cph2.summary.loc['risk_oof_z', 'p'])
    OUT['tcga_multivariable_cox']['p_ajcc'] = float(cph2.summary.loc['ajcc_ord', 'p'])

    # ---------- bootstrap differences ----------
    o, ci, p1 = boot_diff(tt, ee, d['risk_oof'], d['ajcc_ord'])
    OUT['tcga_dC_signature_vs_AJCC'] = dict(delta=o, ci=ci, p_one_sided=p1)
    o, ci, p1 = boot_diff(tt, ee, comb, d['ajcc_ord'])
    OUT['tcga_dC_combined_vs_AJCC'] = dict(delta=o, ci=ci, p_one_sided=p1)

    # ---------- Firehose overlap ----------
    fh = pd.read_csv(os.path.join(D, 'firehose_validation_combined.tsv'), sep='\t')
    tcga_ids = set(met['case_id'].astype(str))
    fh['short'] = fh['patient_id'].astype(str).str.extract(
        r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', expand=False)
    fh['short'] = fh['short'].fillna(fh['patient_id'].astype(str))
    fh['in_tcga'] = fh['short'].isin(tcga_ids)
    OUT['firehose'] = dict(n=int(len(fh)),
                           overlapping=int(fh['in_tcga'].sum()),
                           non_overlapping=int((~fh['in_tcga']).sum()))

    res = {}
    for label, sub_ in [('all_348', fh), ('non_overlapping', fh[~fh['in_tcga']])]:
        s = sub_.dropna(subset=['OS_months', 'OS_status'])
        s = s.copy()
        s['OS_status'] = (s['OS_status'].astype(str)
                          .str.extract(r'([01])', expand=False).astype(float))
        s = s.dropna(subset=['OS_status'])
        s = s[s['OS_months'] > 0]
        if len(s) < 20:
            continue
        cols = {g: (g + '_log2' if (g + '_log2') in s.columns else g)
                for g in COEF}
        risk = sum(COEF[g] * s[cols[g]] for g in COEF)
        md = risk.median()
        from lifelines.statistics import logrank_test
        hi_, lo_ = s[risk >= md], s[risk < md]
        lr = logrank_test(hi_['OS_months'], lo_['OS_months'],
                          hi_['OS_status'], lo_['OS_status'])
        res[label] = dict(n=int(len(s)), events=int(s['OS_status'].sum()),
                          c_index=round(cidx(s['OS_months'], s['OS_status'], risk), 3),
                          logrank_p=float(lr.p_value),
                          HR_high_vs_low=None)
        try:
            s2 = s.assign(risk_z=(risk - risk.mean()) / risk.std())
            c3 = CoxPHFitter()
            c3.fit(s2[['OS_months', 'OS_status', 'risk_z']].astype(float),
                   'OS_months', 'OS_status')
            res[label]['HR_high_vs_low'] = round(float(c3.hazard_ratios_['risk_z']), 3)
            res[label]['HR_p'] = float(c3.summary.loc['risk_z', 'p'])
        except Exception as e:
            print('cox failed', label, e)
    OUT['firehose_validation'] = res

    with open(os.path.join(D, 'v12_headtohead.json'), 'w') as f:
        json.dump(OUT, f, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == '__main__':
    main()
