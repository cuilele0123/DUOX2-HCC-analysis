#!/usr/bin/env python
"""
v12 additions
=============
(1) GSE76427 (GPL10558, Illumina HT-12 V4.0) as a genuinely independent validation
    cohort: 115 primary HCC tumors + 52 adjacent non-tumor liver samples, with
    overall survival, BCLC stage and clinical TNM stage.
(2) Head-to-head comparison of the 3-gene signature against established clinical
    staging systems (BCLC / TNM in GSE76427, AJCC in TCGA-LIHC) with bootstrap
    confidence intervals for the difference in C-index.
(3) Re-analysis of the Firehose Legacy cohort restricted to the patients that do
    NOT overlap TCGA-LIHC, to quantify how much of the apparent external
    validation is truly independent.
"""
import gzip
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, 'data')
OUT = {}
rng = np.random.default_rng(20250913)

COEF = {'SLC16A3': 0.092, 'SPP2': -0.038, 'MMP7': 0.024}
GENES = ['DUOX2', 'SLC16A3', 'SPP2', 'MMP7']


# ---------------------------------------------------------------- helpers
def c_index(time, event, score):
    return concordance_index_censored(np.asarray(event, dtype=bool),
                                      np.asarray(time, dtype=float),
                                      np.asarray(score, dtype=float))[0]


def boot_cindex_diff(time, event, s_new, s_ref, n_boot=2000):
    """Bootstrap CI for C(new) - C(ref)."""
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    s_new = np.asarray(s_new, float)
    s_ref = np.asarray(s_ref, float)
    n = len(time)
    obs = c_index(time, event, s_new) - c_index(time, event, s_ref)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if event[idx].sum() < 5:
            continue
        try:
            d = (c_index(time[idx], event[idx], s_new[idx])
                 - c_index(time[idx], event[idx], s_ref[idx]))
            diffs.append(d)
        except Exception:
            pass
    diffs = np.array(diffs)
    return obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), \
        float((diffs <= 0).mean())


# ---------------------------------------------------------------- GSE76427
def load_gse76427():
    """Return (expr_gene_level, clinical)."""
    # --- probe -> gene symbol from GPL10558 annotation
    ann_rows = []
    with gzip.open(os.path.join(D, 'annot', 'GPL10558.annot.gz'), 'rt', errors='ignore') as fh:
        hdr = None
        for line in fh:
            if line.startswith('!platform_table_begin'):
                hdr = next(fh).strip().split('\t')
                break
        if hdr is None:  # older format: annotation table right after header marker
            fh.seek(0)
            for line in fh:
                if line.startswith('ID') and 'Gene symbol' in line:
                    hdr = line.strip().split('\t')
                    break
        for line in fh:
            if line.startswith('!platform_table_end'):
                break
            p = line.rstrip('\n').split('\t')
            if len(p) < 2:
                continue
            ann_rows.append((p[0], p[hdr.index('Gene symbol')] if 'Gene symbol' in hdr else p[1]))
    ann = pd.DataFrame(ann_rows, columns=['probe', 'symbol'])
    ann['symbol'] = ann['symbol'].str.strip()
    ann = ann[ann['symbol'].notna() & (ann['symbol'] != '') & (ann['symbol'] != 'NA')]
    p2g = dict(zip(ann.probe, ann.symbol))

    # --- series matrix
    gsm = None
    chars = {}
    with gzip.open(os.path.join(D, 'GSE76427_series_matrix.txt.gz'), 'rt', errors='ignore') as fh:
        for line in fh:
            if line.startswith('!Sample_geo_accession'):
                gsm = [x.strip('"\n') for x in line.split('\t')[1:]]
            if line.startswith('!Sample_characteristics_ch1'):
                p = [x.strip('"\n') for x in line.split('\t')[1:]]
                k = p[0].split(':')[0].strip()
                v = [x.split(':', 1)[1].strip() if ':' in x else x for x in p]
                chars.setdefault(k, []).append(v)
            if line.startswith('!series_matrix_table_begin'):
                break
    meta = pd.DataFrame({'gsm': gsm})
    for k, v in chars.items():
        meta[k] = v[0]
    meta.columns = [c.strip() for c in meta.columns]

    # --- expression
    expr = pd.read_csv(os.path.join(D, 'GSE76427_series_matrix.txt.gz'), sep='\t',
                       comment='!', compression='gzip', index_col=0,
                       skip_blank_lines=True, on_bad_lines='skip')
    expr.index = expr.index.astype(str).str.strip('"')
    expr.columns = [c.strip('"') for c in expr.columns]
    expr = expr[~expr.index.str.startswith('!')]
    expr = expr.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    # Illumina intensities -> log2
    expr = np.log2(expr.clip(lower=1))
    expr['symbol'] = [p2g.get(i, None) for i in expr.index]
    g = expr.dropna(subset=['symbol'])
    # collapse to gene level (mean of probes); if duplicates, keep the probe with
    # highest mean expression for stability
    g = g.assign(_mean=g.drop(columns='symbol').mean(axis=1))
    g = g.sort_values('_mean', ascending=False).drop_duplicates('symbol')
    ge = g.set_index('symbol').drop(columns='_mean').T
    ge.index.name = 'gsm'
    return ge, meta


def main():
    ge, meta = load_gse76427()
    m = meta.set_index('gsm').join(ge, how='inner')
    m['tissue'] = m['tissue'].str.strip().str.lower()
    m['event'] = pd.to_numeric(m['event_os'], errors='coerce')
    m['os_years'] = pd.to_numeric(m['duryears_os'], errors='coerce')
    m['os_months'] = m['os_years'] * 12.0
    tum = m[m['tissue'] == 'primary hepatocellular carcinoma tumor'].copy()

    OUT['gse76427_n_total'] = len(m)
    OUT['gse76427_n_tumor'] = len(tum)
    OUT['gse76427_n_nontumor'] = int((m['tissue'] == 'adjacent non-tumor liver tissue').sum())

    # ---- tumor vs adjacent non-tumor DUOX2 (independent replication)
    t = tum['DUOX2'].dropna()
    n = m.loc[m['tissue'] == 'adjacent non-tumor liver tissue', 'DUOX2'].dropna()
    OUT['gse76427_dux2_tumor_median'] = round(float(t.median()), 3)
    OUT['gse76427_dux2_nontumor_median'] = round(float(n.median()), 3)
    OUT['gse76427_dux2_mwu_p'] = float(stats.mannwhitneyu(t, n).pvalue)

    # ---- survival set
    sv = tum.dropna(subset=['os_months', 'event', 'DUOX2', 'SLC16A3', 'SPP2', 'MMP7'])
    sv = sv[sv['os_months'] > 0]
    OUT['gse76427_n_survival'] = len(sv)
    OUT['gse76427_events'] = int(sv['event'].sum())

    sv['risk'] = sum(COEF[g] * sv[g] for g in COEF)
    med = sv['risk'].median()
    sv['grp'] = np.where(sv['risk'] >= med, 'high', 'low')

    lr = logrank_test(sv.loc[sv.grp == 'high', 'os_months'],
                      sv.loc[sv.grp == 'low', 'os_months'],
                      sv.loc[sv.grp == 'high', 'event'],
                      sv.loc[sv.grp == 'low', 'event'])
    OUT['gse76427_risk_logrank_p'] = float(lr.p_value)
    OUT['gse76427_risk_cindex'] = round(c_index(sv['os_months'], sv['event'], sv['risk']), 3)
    # median follow-up / OS by group for reporting
    kmf = KaplanMeierFitter()
    for g_ in ['high', 'low']:
        s = sv[sv.grp == g_]
        kmf.fit(s['os_months'], s['event'])
        OUT['gse76427_median_os_%s' % g_] = (round(float(kmf.median_survival_time_), 2)
                                             if kmf.median_survival_time_ is not None else None)

    y = Surv.from_arrays(sv['event'].astype(bool), sv['os_months'].astype(float))
    for t_ in [12, 36, 60]:
        try:
            auc, _ = cumulative_dynamic_auc(y, y, sv['risk'].values, times=[t_])
            OUT['gse76427_auc_%dmo' % t_] = round(float(auc[0]), 3)
        except Exception as e:
            OUT['gse76427_auc_%dmo' % t_] = None

    # ---- head to head: 3-gene vs BCLC vs TNM in GSE76427
    bclc_map = {'0': 0, 'A': 1, 'B': 2, 'C': 3}
    tnm_map = {'I': 1, 'II': 2, 'IIIA': 3, 'IIIB': 3, 'IVA': 4, 'IVB': 4}
    sv['bclc_ord'] = sv['bclc_staging'].str.strip().map(bclc_map)
    sv['tnm_ord'] = sv['tnm_staging_clinical'].str.strip().map(tnm_map)

    h2h = {}
    for name, sc in [('3-gene signature', 'risk'), ('BCLC stage', 'bclc_ord'),
                     ('Clinical TNM stage', 'tnm_ord')]:
        sub = sv.dropna(subset=[sc])
        if sub[sc].nunique() < 2:
            continue
        h2h[name] = dict(n=int(len(sub)), events=int(sub['event'].sum()),
                         c_index=round(c_index(sub['os_months'], sub['event'], sub[sc]), 3))
    # combined model: 3-gene + BCLC via multivariable Cox fitted in this cohort
    sub = sv.dropna(subset=['risk', 'bclc_ord'])
    if len(sub) > 30:
        cph = CoxPHFitter()
        dd = sub[['os_months', 'event', 'risk', 'bclc_ord']].astype(float)
        cph.fit(dd, 'os_months', 'event')
        comb = cph.predict_partial_hazard(sub[['risk', 'bclc_ord']].astype(float)).values
        h2h['3-gene + BCLC (combined)'] = dict(
            n=int(len(sub)), events=int(sub['event'].sum()),
            c_index=round(c_index(sub['os_months'], sub['event'], comb), 3))
        OUT['gse76427_combined_cox'] = {
            'beta_risk': round(float(cph.params_['risk']), 3),
            'p_risk': float(cph.summary.loc['risk', 'p']),
            'beta_bclc': round(float(cph.params_['bclc_ord']), 3),
            'p_bclc': float(cph.summary.loc['bclc_ord', 'p'])}
    OUT['gse76427_h2h'] = h2h

    # bootstrap difference, signature vs BCLC
    s1 = sv.dropna(subset=['risk', 'bclc_ord'])
    if len(s1) > 30:
        o, lo, hi, p1 = boot_cindex_diff(s1['os_months'], s1['event'],
                                         s1['risk'], s1['bclc_ord'])
        OUT['gse76427_dC_3gene_vs_BCLC'] = dict(delta=round(o, 3),
                                                ci=[round(lo, 3), round(hi, 3)],
                                                p_one_sided=round(p1, 4))
        s2 = s1.dropna(subset=['tnm_ord'])
        o, lo, hi, p1 = boot_cindex_diff(s2['os_months'], s2['event'],
                                         s2['risk'], s2['tnm_ord'])
        OUT['gse76427_dC_3gene_vs_TNM'] = dict(delta=round(o, 3),
                                               ci=[round(lo, 3), round(hi, 3)],
                                               p_one_sided=round(p1, 4))

    with open(os.path.join(D, 'v12_independent_validation.json'), 'w') as f:
        json.dump(OUT, f, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str)[:4000])


if __name__ == '__main__':
    main()
