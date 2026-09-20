# -*- coding: utf-8 -*-
"""合并 424 个 STAR-Counts 文件为表达矩阵"""
import pandas as pd
import numpy as np
import os, time

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(_ROOT, 'data')
CNT_DIR = os.path.join(BASE, 'counts')

# 样本→文件映射（样本级 ID）
samples = {}
with open(os.path.join(BASE, 'lihc_files_list2.tsv')) as f:
    f.readline()
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 5:
            samples[parts[3]] = {'file_id': parts[0], 'case_id': parts[2],
                                 'sample_type': parts[4]}

# 逐个读取 unstranded counts
matrices = {}
gene_ids = None
t0 = time.time()
for i, (sid, info) in enumerate(samples.items()):
    fp = os.path.join(CNT_DIR, info['file_id'] + '.tsv')
    try:
        df = pd.read_csv(fp, sep='\t', comment='#', usecols=['gene_id', 'unstranded'])
        df = df.set_index('gene_id')
        # 只保留基因行（去掉 N_* 汇总行）
        df = df[~df.index.str.startswith('N_')]
        if gene_ids is None:
            gene_ids = df.index.tolist()
        matrices[sid] = df['unstranded']
    except Exception as e:
        print(f'ERR {sid}: {e}', flush=True)
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{len(samples)} 读取完成 ({time.time()-t0:.0f}s)', flush=True)

# 合并为 DataFrame
expr = pd.DataFrame(matrices).T  # 样本 × 基因
expr.index.name = 'sample_id'

# 基因 ID 规范化：ENSGxxxxx
expr = expr.loc[:, ~expr.columns.duplicated()]

# 样本标注
sample_meta = pd.DataFrame([
    {'sample_id': sid, 'case_id': info['case_id'], 'barcode': sid, 'sample_type': info['sample_type']}
    for sid, info in samples.items()
])

print(f'表达矩阵: {expr.shape[0]} 样本 × {expr.shape[1]} 基因')
print(f'内存: {expr.memory_usage().sum()/1024/1024:.0f} MB')
print('样本类型分布:', sample_meta['sample_type'].value_counts().to_dict())

# 保存
expr.to_pickle(os.path.join(BASE, 'expr_counts.pkl'))
sample_meta.to_csv(os.path.join(BASE, 'sample_meta.tsv'), sep='\t', index=False)
print('Saved expr_counts.pkl')
