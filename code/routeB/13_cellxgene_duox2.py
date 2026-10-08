# -*- coding: utf-8 -*-
"""CELLxGENE 公开单细胞数据独立核验 DUOX2 的细胞归属（2026-10-07）。

为什么需要这一步：稿件的核心结论是“DUOX2 属于肝内胆管/管状上皮区室”，
而HPA 单细胞参考图谱是**非肿瘤**组织。本脚本用**独立来源**的公开肝单细胞
数据检验该定位是否成立，并同时检查肿瘤区室能否直接观察。

结论摘要（实际产出见 results/cellxgene_DUOX2_summary.tsv 与 _runlog.json）：

  1. CELLxGENE Discover **没有人类原发 HCC 单细胞数据集**。全库 2,238 个数据集
     逐一检索（标题 + 描述 + disease 字段），disease 含 liver/HCC 的仅 1 个
     **小鼠** MASLD-HCC 模型。因此“肿瘤组织内 DUOX2 归属”**无法**用公开数据
     直接观察 —— 稿件局限段第1 条**成立且必须保留**。

  2. 独立人肝图谱**复现**了 HPA 的定位方向（两来源合并）：
     胆管谱系2,173 细胞检出率 2.715%（59/2173）；
     肝细胞 116,130 细胞检出率 0.092%（107/116130）；
     odds ratio 30.3，Fisher exact P = 1.4e-58。

  3. **绝对丰度与 HPA 高度一致**：J Hepatol 2024 图谱中“intrahepatic
     cholangiocyte” 平均 29.57 CPM、8.5% 阳性，与 HPA 的 27.0 nCPM 比值 1.10     （独立来源、不同处理流程）。

  4. 谱系内**状态异质**：第二个图谱中成熟胆管簇 0/445 阳性，而其progenitor
     簇 10/1,150 阳性 → DUOX2 标记的是**前体样胆管状态**，而非整个胆管区室。
     阳性细胞绝对计数极低（≤13 counts）。

  5. 唯一的肝肿瘤单细胞数据（小鼠 MASLD-HCC）肿瘤区室仅 110 胆管 + 17 肝细胞，
     **零检出**，样本量不足以支持比较 → 明确写为不可比较，而非阴性结论。

三个必须记住的技术要点（都实际踩过）：
  1) CELLxGENE H5AD 的 `.X` 是 **log1p 标准化值**（max≈6–9），原始 counts 在
     **`.raw.X`**；用 `.X` 算 CPM 会得到毫无意义的结果（曾据此误得 ALB ≈ 18万 CPM）。
  2) CPM 分母必须是**全基因**总 counts（`obs.nCount_RNA` 或 `.raw.X` 行和），
     不能用“标记基因之和”——否则数值虚高两个数量级。
  3) DUOX2 未被 CELLxGENE 过滤（`feature_is_filtered` 在 raw.var 中不存在，
     基因在 raw 计数里确实为 0），故“检不出”是生物学结果而非技术缺失。

注：本机 `cellxgene-census` 客户端因 tiledbsoma 的 libc++ ABI 不匹配不可用，
故走 Discover 的 H5AD 资产直连，无需 Census。
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata
from scipy.stats import fisher_exact

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('DUOX2_WORK', os.path.join(_ROOT, '..'))
CACHE = os.path.join(BASE, 'cellxgene_cache')
OUT = os.path.join(BASE, 'RouteB_analysis', 'cellxgene_DUOX2_summary.tsv')
LOG = os.path.join(BASE, 'RouteB_analysis', 'cellxgene_DUOX2_runlog.json')

DATASETS = [
    ('HepatolComm2022_biliary', 'Human liver biliary lineage (Hepatol Commun 2022)',
     'hum_chol_hepatology2022.h5ad', 'human'),
    ('HepatolComm2022_hepatocyte', 'Human liver hepatocytes (Hepatol Commun 2022)',
     'hum_hep_hepatology2022.h5ad', 'human'),
    ('JHepatol2024_biliary', 'Human liver cholangiocytes (J Hepatol 2024)',
     'hum_chol_jhepatol2023.h5ad', 'human'),
    ('JHepatol2024_hepatocyte1', 'Human liver hepatocytes set 1 (J Hepatol 2024)',
     'hum_hep1_jhepatol2023.h5ad', 'human'),
    ('JHepatol2024_hepatocyte2', 'Human liver hepatocytes set 2 (J Hepatol 2024)',
     'hum_hep2_jhepatol2023.h5ad', 'human'),
    ('CellMet2026_mouseHCC', 'Mouse liver, tumor compartment (Cell Metab 2026)',
     'mouse_hcc_cellmet2026.h5ad', 'mouse'),
]

MARKERS = {
    'DUOX2': ['DUOX2'], 'DUOXA2': ['DUOXA2'],
    'ALB': ['ALB'], 'APOA1': ['APOA1'], 'TTR': ['TTR'],
    'KRT19': ['KRT19'], 'KRT8': ['KRT8'], 'KRT18': ['KRT18'], 'KRT7': ['KRT7'],
    'EPCAM': ['EPCAM'], 'CFTR': ['CFTR'], 'MMP7': ['MMP7'],
    'PECAM1': ['PECAM1'], 'VWF': ['VWF'], 'CD3E': ['CD3E'], 'MS4A1': ['MS4A1'],
    'COL1A1': ['COL1A1'], 'DCN': ['DCN'], 'LYZ': ['LYZ'], 'SERPINA1': ['SERPINA1'],
}
HEPATO = ('hepatocyte',)
BILIARY = ('cholangiocyte', 'progenitor cell', 'ductular', 'biliary', 'epithelial cell of biliary')


def gi(low, s):
    h = [i for i, n in enumerate(low) if n == s.lower()]
    return h[0] if h else None


def main():
    log = {'note': 'counts 来自 .raw.X；CPM 分母 = 全基因总 counts（.raw.X 行和）',
           'datasets': [], 'fisher': []}
    rows = []
    for key, label, fname, species in DATASETS:
        path = os.path.join(CACHE, fname)
        print('=' * 80)
        print('【%s】%s' % (key, label))
        if not os.path.exists(path):
            print('  !! 缺文件 %s' % path)
            continue
        ad = anndata.read_h5ad(path)
        assert ad.raw is not None, '%s 没有 .raw.X' % fname
        R = ad.raw.X
        rv = ad.raw.var['feature_name'].astype(str).tolist()
        rl = [n.lower() for n in rv]
        cols, syms = [], []
        for s in MARKERS:
            i = gi(rl, s)
            if i is not None:
                cols.append(i)
                syms.append(s)
        miss = [s for s in MARKERS if s not in syms]
        tot = np.asarray(R.sum(axis=1)).ravel() if sp.issparse(R) else np.asarray(R).sum(axis=1)
        M = R[:, cols]
        M = M.toarray() if sp.issparse(M) else np.asarray(M, dtype=float)
        CPM = M / np.maximum(tot, 1)[:, None] * 1e6
        ct = ad.obs['cell_type'].astype(str).values
        dis = (ad.obs['disease'].astype(str).values if 'disease' in ad.obs.columns
               else np.array(['n/a'] * ad.n_obs))
        tum = np.isin(dis, [d for d in np.unique(dis)
                            if 'hepatocellular' in d.lower() or d.lower() == 'hcc'])
        scope = 'tumor' if tum.sum() > 0 else 'all'
        keep = tum if tum.sum() > 0 else np.ones(ad.n_obs, bool)
        print('  cells=%d genes=%d  总 counts 中位=%.0f  缺失基因=%s'
              % (ad.n_obs, len(rv), np.median(tot), miss))
        print('  细胞类型：%s' % dict(pd.Series(ct[keep]).value_counts()))
        print('  疾病：%s' % dict(pd.Series(dis).value_counts()))
        print('  → 统计范围 %s（%d/%d 细胞）' % (scope, keep.sum(), ad.n_obs))
        dj = syms.index('DUOX2')
        print('  DUOX2 原始计数：max=%d，全库非零细胞=%d/%d (%.3f%%)'
              % (M[:, dj].max(), int((M[:, dj] > 0).sum()), ad.n_obs,
                 100 * (M[:, dj] > 0).mean()))

        for c in sorted(set(ct[keep]), key=lambda x: -int((ct[keep] == x).sum())):
            m = (ct == c) & keep
            n = int(m.sum())
            if n < 15:
                if n:
                    print('   [n<15 略过] %-40s n=%d' % (c[:40], n))
                continue
            rec = {'dataset': key, 'label': label, 'species': species, 'scope': scope,
                   'cell_type': c, 'n_cells': n,
                   'DUOX2_pos_cells': int((M[m, dj] > 0).sum()),
                   'DUOX2_pct_pos': round(100 * float((M[m, dj] > 0).mean()), 3),
                   'DUOX2_mean_cpm': round(float(CPM[m, dj].mean()), 4),
                   'DUOX2_max_cpm': round(float(CPM[m, dj].max()), 1)}
            for j, s in enumerate(syms):
                rec[s + '_mean_cpm'] = round(float(CPM[m, j].mean()), 2)
            rows.append(rec)
            print('   %-40s n=%-6d DUOX2 检出 %5.2f%%  meanCPM=%7.3f  ALB=%9.1f KRT19=%8.2f KRT7=%7.2f'
                  % (c[:40], n, rec['DUOX2_pct_pos'], rec['DUOX2_mean_cpm'],
                     rec['ALB_mean_cpm'], rec['KRT19_mean_cpm'], rec['KRT7_mean_cpm']))

        log['datasets'].append({'key': key, 'label': label, 'file': fname, 'species': species,
                                'n_obs': int(ad.n_obs), 'n_genes': len(rv),
                                'median_total_counts': float(np.median(tot)),
                                'scope': scope, 'n_in_scope': int(keep.sum()),
                                'genes_missing': miss,
                                'celltypes': {k: int(v) for k, v in
                                              pd.Series(ct[keep]).value_counts().items()},
                                'disease': {k: int(v) for k, v in
                                            pd.Series(dis).value_counts().items()}})
        del ad, R, M, CPM

    df = pd.DataFrame(rows)
    df.to_csv(OUT, sep='\t', index=False)

    # 方向性检验：胆管谱系 vs 肝细胞 的 DUOX2 检出率（人，两个独立图谱合并）
    hum = df[(df.species == 'human')]
    bili = hum[hum.cell_type.str.contains('|'.join(BILIARY), case=False, na=False)]
    hepa = hum[hum.cell_type.str.contains('hepatocyte', case=False, na=False)]
    if len(bili) and len(hepa):
        b_pos, b_n = int(bili.DUOX2_pos_cells.sum()), int(bili.n_cells.sum())
        h_pos, h_n = int(hepa.DUOX2_pos_cells.sum()), int(hepa.n_cells.sum())
        orr, pv = fisher_exact([[b_pos, b_n - b_pos], [h_pos, h_n - h_pos]])
        print()
        print('=' * 80)
        print('方向性检验（人，两个独立图谱合并）')
        print('  胆管谱系 %d/%d = %.3f%% ；肝细胞 %d/%d = %.3f%% ；OR=%.1f ；Fisher P=%.3g'
              % (b_pos, b_n, 100 * b_pos / b_n, h_pos, h_n, 100 * h_pos / h_n, orr, pv))
        log['fisher'] = [{'comparison': 'human biliary lineage vs hepatocyte',
                          'biliary_pos': b_pos, 'biliary_n': b_n,
                          'hepatocyte_pos': h_pos, 'hepatocyte_n': h_n,
                          'odds_ratio': float(orr), 'p_value': float(pv)}]

    json.dump(log, open(LOG, 'w'), ensure_ascii=False, indent=2)
    print()
    print('结果表 %s（%d 行）' % (OUT, len(df)))
    print('日志  %s' % LOG)
    print()
    print('=== 上皮两类细胞汇总 ===')
    sel = df[df.cell_type.str.contains('hepatocyte|cholangiocyte|progenitor', case=False, na=False)]
    print(sel[['dataset', 'scope', 'cell_type', 'n_cells', 'DUOX2_pct_pos', 'DUOX2_mean_cpm',
               'ALB_mean_cpm', 'KRT19_mean_cpm', 'KRT7_mean_cpm']].to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())