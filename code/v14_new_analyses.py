# -*- coding: utf-8 -*-
"""
v14_new_analyses.py —— 回应第三轮审稿意见的三项补充分析
  1. GSE76427 独立队列中 DUOX2 的分期梯度（BCLC / TNM）—— 回应"补充独立验证"
  2. 免疫细胞(9)与检查点基因(10)相关性的 BH-FDR 校正 —— 回应"多重比较未校正"
  3. LASSO-Cox 与 RSF 的配对 bootstrap C-index 差异检验 —— 回应"模型比较缺统计检验"
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RNG = np.random.default_rng(20260914)

OUT = {}


def bh_fdr(p):
    """Benjamini-Hochberg q values."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(o[::-1]):
        i = n - rank              # 1-based descending rank
        val = p[idx] * n / i
        prev = min(prev, val)
        q[idx] = prev
    return np.minimum(q, 1.0)


# ============================================================ 1. GSE76427 分期梯度
def stage_gradient():
    from v12_independent_validation import load_gse76427
    ge, meta = load_gse76427()
    m = meta.set_index('gsm').join(ge, how='inner')
    m['tissue'] = m['tissue'].astype(str).str.strip().str.lower()
    tum = m[m['tissue'].str.contains('tumor', case=False, na=False)].copy()
    res = {'n_tumor': int(len(tum))}

    BCLC_ORD = {'0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
    TNM_ORD = {'I': 1, 'II': 2, 'III': 3, 'IIIA': 3, 'IIIB': 3, 'IIIC': 3,
               'IV': 4, 'IVA': 4, 'IVB': 4}

    for label, col, mapping in [
            ('BCLC stage', 'bclc_staging', BCLC_ORD),
            ('Clinical TNM stage', 'tnm_staging_clinical', TNM_ORD)]:
        s = tum[['DUOX2', col]].dropna().copy()
        s['ord'] = s[col].astype(str).str.strip().str.upper().map(mapping)
        s = s.dropna(subset=['ord'])
        s['ord'] = s['ord'].astype(int)
        groups = [g['DUOX2'].values for _, g in s.groupby('ord') if len(g) >= 5]
        kw = stats.kruskal(*groups) if len(groups) >= 2 else None
        rho, prho = stats.spearmanr(s['ord'].values, s['DUOX2'].values)
        res[label] = {
            'n': int(len(s)),
            'groups': {k: int(v) for k, v in s.groupby('ord').size().items()},
            'medians': {k: round(float(g['DUOX2'].median()), 3)
                        for k, g in s.groupby('ord')},
            'kruskal_p': float(kw.pvalue) if kw else None,
            'spearman_rho': float(rho),
            'spearman_p': float(prho),
        }
    # 与 TCGA 对照
    l = pd.read_csv(os.path.join(D, 'lasso_risk_analysis.csv'))
    lc = pd.read_pickle(os.path.join(D, 'logcpm_matrix.pkl'))
    key = {str(i): i for i in lc.index}
    vals, stages = [], []
    for sid, sg in zip(l.sample_id, l.stage_group):
        k = key.get(str(sid))
        if k is not None:
            vals.append(lc.loc[k, 'DUOX2']); stages.append(sg)
    df = pd.DataFrame({'x': vals, 'g': stages}).dropna()
    order = {'early': 1, 'intermediate': 2, 'advanced': 3}
    df['o'] = df['g'].map(order)
    df = df.dropna(subset=['o'])
    gg = [g['x'].values for _, g in df.groupby('o')]
    res['TCGA-LIHC stage group'] = {
        'n': int(len(df)),
        'groups': df['g'].value_counts().to_dict(),
        'medians': df.groupby('g')['x'].median().round(3).to_dict(),
        'kruskal_p': float(stats.kruskal(*gg).pvalue),
        'spearman_rho': float(stats.spearmanr(df['o'], df['x'])[0]),
        'spearman_p': float(stats.spearmanr(df['o'], df['x'])[1]),
    }
    return res


# ============================================================ 2. 相关性 FDR
def corr_fdr():
    out = {}
    for name, f in [('immune_cell', 'DUOX2_immune_corr.csv'),
                    ('checkpoint', 'DUOX2_checkpoint_corr.csv')]:
        p = os.path.join(D, f)
        if not os.path.exists(p):
            out[name] = 'FILE NOT FOUND'
            continue
        d = pd.read_csv(p)
        cols = {c.lower(): c for c in d.columns}
        gene_c = cols.get('gene') or cols.get('cell') or cols.get('cell_type') or d.columns[0]
        rho_c = next((cols[k] for k in ('rho', 'spearman_rho', 'cor', 'r') if k in cols), None)
        p_c = next((cols[k] for k in ('p', 'pvalue', 'p_value', 'pval') if k in cols), None)
        rows = []
        if rho_c and p_c:
            q = bh_fdr(d[p_c].values)
            for i, (_, r) in enumerate(d.iterrows()):
                rows.append({'feature': r[gene_c], 'rho': round(float(r[rho_c]), 3),
                             'p': float(r[p_c]), 'q_BH': round(float(q[i]), 4)})
        out[name] = {'n_tests': len(d), 'columns': list(d.columns), 'results': rows}
    # 泛癌 20 瘤种复核
    pc = pd.read_csv(os.path.join(D, 'pan_cancer_DUOX2_survival.csv'))
    if 'p_value' in pc.columns or 'p' in pc.columns:
        pc_col = 'p_value' if 'p_value' in pc.columns else 'p'
        q = bh_fdr(pc[pc_col].values)
        out['pancancer'] = {
            'n_tests': len(pc),
            'significant_p05': [str(r.get('cancer', r.get('tumor', i)))
                                for i, r in pc.iterrows() if r[pc_col] < 0.05],
            'significant_q05': [str(r.get('cancer', r.get('tumor', i)))
                                for i, r, qq in zip(range(len(pc)), pc.iterrows(), q) if qq < 0.05],
        }
    return out


# ============================================================ 3. 模型比较
def model_test():
    p = os.path.join(D, 'ml_model_comparison_results.csv')
    if not os.path.exists(p):
        return 'FILE NOT FOUND'
    d = pd.read_csv(p)
    return {'columns': list(d.columns), 'shape': list(d.shape), 'head': d.head(12).to_dict('records')}


if __name__ == '__main__':
    print('=' * 70)
    print('1. GSE76427 / TCGA 分期梯度')
    print('=' * 70)
    sg = stage_gradient()
    OUT['stage_gradient'] = sg
    for k, v in sg.items():
        if k == 'n_tumor':
            continue
        print('\n[%s] n=%s' % (k, v['n']))
        print('  分组: %s' % v['groups'])
        print('  中位: %s' % v['medians'])
        print('  Kruskal-Wallis P = %s' % (round(v['kruskal_p'], 4) if v['kruskal_p'] is not None else 'NA'))
        print('  Spearman rho = %+.3f (P = %s)' % (v['spearman_rho'], round(v['spearman_p'], 4)))

    print()
    print('=' * 70)
    print('2. 相关性多重比较校正')
    print('=' * 70)
    cf = corr_fdr()
    OUT['corr_fdr'] = cf
    for k, v in cf.items():
        if isinstance(v, str):
            print(k, v); continue
        print('\n[%s] 检验数 = %s' % (k, v['n_tests']))
        if 'results' in v:
            for r in v['results']:
                flag = '  <-- q<0.05' if r['q_BH'] < 0.05 else ''
                print('   %-22s rho=%+.3f  P=%.2e  q=%.4f%s'
                      % (r['feature'], r['rho'], r['p'], r['q_BH'], flag))
        else:
            print('   原始 P<0.05:', v['significant_p05'])
            print('   BH q<0.05  :', v['significant_q05'])

    print()
    print('=' * 70)
    print('3. 模型比较数据可用性')
    print('=' * 70)
    mt = model_test()
    OUT['model_test_probe'] = mt
    print(json.dumps(mt, indent=1, default=str)[:1500])

    with open(os.path.join(D, 'v14_new_analyses.json'), 'w') as f:
        json.dump(OUT, f, indent=1, default=str)
    print('\nsaved -> data/v14_new_analyses.json')
