# -*- coding: utf-8 -*-
"""组装 Zenodo v2.0.0 归档（2026-10-08）。

背景：v1.0.0 归档的题名是路线 A（旧命题为"DUOX2 indicates immune activation
and metabolic suppression but not prognosis…"这一否定式标题，已随路线 B
重构作废），而 Zenodo 记录的题名已手工改成
路线 B 论文的题名。若不修仓库，日后任何一次 GitHub Release 都会用旧题名
**覆盖**网页端的修正（Zenodo 元数据优先级 `.zenodo.json` > `CITATION.cff`）。

本脚本做四件事：
  1. `.zenodo.json`：题名/描述/关键词/版本改为路线 B，version → 2.0.0；
  2. `CITATION.cff`：同步题名与版本（GitHub 网页"Cite"与仓库首页会显示它）；
  3. `README.md`：换题名、去掉已拒稿的刊名、修正英式拼写、补route B 内容清单；
  4. 把 route B 的分析脚本与派生结果入库到 `code/routeB/` 与 `results/routeB/`。

**不改** v1.0.0 已有内容（results/ 与 figures/ 下的路线 A 表与图原样保留）——
它们仍是论文中 TCGA/GSE14520/泛癌/TIDE 各节的数据来源。
"""
import hashlib
import json
import os
import re
import shutil

# 本脚本在**归档前**于作者工作副本上运行，故此处为本地绝对路径；
# 它产出的 `.zenodo.json` / `CITATION.cff` / `README.md` 不含任何绝对路径。
REPO = os.environ.get('DUOX2_REPO', os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..')))
WORK = os.environ.get('DUOX2_WORK', os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
ROUTE_B = os.environ.get('DUOX2_ROUTEB', os.path.join(WORK, 'RouteB_analysis'))

TITLE = ('DUOX2 marks biliary/ductular epithelium in hepatocellular carcinoma: '
         'reported immune and prognostic associations reflect tissue composition')

DESCRIPTION = (
    'Analysis code, derived result tables and publication figures for a multi-cohort '
    'study of DUOX2 in hepatocellular carcinoma (HCC). The study first localizes the '
    'transcript: in a single-cell reference map of human tissues and, independently, in '
    'two human liver single-cell atlases distributed through CELLxGENE, DUOX2 is carried '
    'by the biliary/ductular epithelial compartment rather than by malignant hepatocytes '
    'or by immune cells. Compartment-adjusted analysis then shows that the reported '
    'associations with interferon-gamma signaling disappear once hepatocyte, stromal and '
    'immune content are accounted for, while the cholangiocyte-lineage association is '
    'retained in two independent bulk cohorts (TCGA-LIHC, 371 tumors; GSE14520, 221 '
    'tumors from an HBV-dominant Chinese cohort). Neither cohort recovers the prognostic '
    'associations previously attributed to DUOX2, and a program derived from its '
    'co-expression network is unrelated to benefit from checkpoint blockade. The pipeline '
    'covers single-cell reference extraction and compartment analysis, TCGA-LIHC data '
    'acquisition and AJCC stage reconstruction, differential expression (pydeseq2), GO/KEGG '
    'and GSEA enrichment, LASSO-Cox prognostic modelling with calibration and '
    'decision-curve analysis, benchmark comparison against random survival forests, '
    'gradient boosting and survival SVM, ssGSEA immune infiltration, TIDE scoring with '
    'epithelial/stromal compartment adjustment for tumor purity, external validation in '
    'GSE14520 and GSE76427, head-to-head comparison with AJCC and BCLC stage, a '
    '20-tumor-type pan-cancer analysis, immune checkpoint inhibitor analysis in '
    'GSE140901, and the independent single-cell atlas verification reported in the '
    'manuscript. Raw public data are not redistributed; download scripts are included.'
)

KEYWORDS = [
    'hepatocellular carcinoma',
    'DUOX2',
    'cholangiocyte',
    'ductular reaction',
    'biliary epithelium',
    'tumor microenvironment',
    'tissue composition',
    'cell of origin',
    'TCGA-LIHC',
    'single-cell atlas',
    'bioinformatics',
]


def read(p):
    return open(p, encoding='utf-8').read()


def write(p, s):
    open(p, 'w', encoding='utf-8').write(s)


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def main():
    log = []

    # ---------------------------------------------------------------- 1. .zenodo.json
    p = os.path.join(REPO, '.zenodo.json')
    z = json.loads(read(p))
    old_title = z['title']
    z['title'] = 'Analysis code for: ' + TITLE
    z['description'] = DESCRIPTION
    z['keywords'] = KEYWORDS
    z['version'] = '2.0.0'
    write(p, json.dumps(z, indent=2, ensure_ascii=False) + '\n')
    log.append(('.zenodo.json', 'title/description/keywords/version → 2.0.0'))
    log.append(('', '  旧题名长度 %d → 新 %d' % (len(old_title), len(z['title']))))
    log.append(('', '  keywords %d → %d 个' % (8, len(KEYWORDS))))

    # ---------------------------------------------------------------- 2. CITATION.cff
    p = os.path.join(REPO, 'CITATION.cff')
    c = read(p)
    c = re.sub(r'^title: ".*"$',
               'title: "Analysis code for: %s"' % TITLE,
               c, count=1, flags=re.M)
    c = re.sub(r'^version: ".*"$', 'version: "2.0.0"', c, count=1, flags=re.M)
    c = re.sub(r'^date-released: ".*"$', 'date-released: "2026-10-08"', c, count=1, flags=re.M)
    c = re.sub(r'^abstract: ".*"$',
               'abstract: "Analysis code, derived result tables and publication figures '
               'for a multi-cohort study of DUOX2 in hepatocellular carcinoma. Raw public '
               'data are not redistributed; download scripts are included."',
               c, count=1, flags=re.M)
    write(p, c)
    log.append(('CITATION.cff', 'title/version/date-released/abstract 已同步'))

    # ---------------------------------------------------------------- 3. README.md
    p = os.path.join(REPO, 'README.md')
    md = read(p)
    old_h1 = md.split('\n')[0]
    md = md.replace(
        old_h1,
        '# Analysis code for: ' + TITLE, 1)
    md = md.replace(
        'Version 1.0.0 · released 2026-09-20',
        'Version 2.0.0 · released 2026-10-08', 1)
    # 去掉已拒稿期刊的具名（改刊时不必再动）
    md = md.replace(
        'publication figures for a multi-cohort study of DUOX2 in hepatocellular carcinoma\n'
        '(HCC). It accompanies the manuscript submitted to *Scientific Reports*.',
        'publication figures for a multi-cohort study of DUOX2 in hepatocellular carcinoma\n'
        '(HCC). It accompanies the manuscript on the biliary/ductular localization of DUOX2;\n'
        'the manuscript identifier and the journal are given in the archival record of this\n'
        'release.', 1)
    write(p, md)
    log.append(('README.md', '题名/版本已换；去掉已拒稿刊名，改为中性表述'))

    # ---------------------------------------------------------------- 4. 拷入 route B
    dst_code = os.path.join(REPO, 'code', 'routeB')
    dst_res = os.path.join(REPO, 'results', 'routeB')
    os.makedirs(dst_code, exist_ok=True)
    os.makedirs(dst_res, exist_ok=True)

    n_code = 0
    for f in sorted(os.listdir(ROUTE_B)):
        if not f.endswith('.py') or f.startswith('_bak'):
            continue
        if f == '00_normalize_for_archive.py':
            continue        # 一次性规范化工具，不属于分析流程
        if f.startswith(('06_', '07_', '08_', '09_', '10_', '11_', '12_', '14b_', '15_')):
            continue        # 稿件装配/改稿/校验脚本，非分析流程
        shutil.copy2(os.path.join(ROUTE_B, f), os.path.join(dst_code, f))
        n_code += 1
    log.append(('code/routeB/', '拷入 %d 个分析脚本' % n_code))

    n_res = 0
    res_files = ['cellxgene_DUOX2_summary.tsv', 'cellxgene_DUOX2_runlog.json',
                 'HPA_DUOX2_single_cell_type.tsv', 'partial_correlation_controlled.tsv',
                 'GSE140901_scores.tsv', 'GSE140901_survival.tsv',
                 'GSE140901_ICI_benefit.tsv']
    for f in res_files:
        s = os.path.join(ROUTE_B, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst_res, f))
            n_res += 1
    log.append(('results/routeB/', '拷入 %d 个派生结果表' % n_res))

    # 图件：主图 6 幅 + 补充图 S9
    dst_fig = os.path.join(REPO, 'figures', 'routeB')
    os.makedirs(dst_fig, exist_ok=True)
    n_fig = 0
    for src_dir, prefix in ((os.path.join(WORK, 'v21_Figures'), 'Figure'),
                            (os.path.join(WORK, 'v21_Supplementary_Figures'),
                             'SupplementaryFigure')):
        if not os.path.isdir(src_dir):
            continue
        for f in sorted(os.listdir(src_dir)):
            if not f.endswith('.png'):
                continue
            tag = ('S9' if f.startswith('FigureS') else
                   re.sub(r'^Figure(\d+)_.*$', r'\1', f))
            shutil.copy2(os.path.join(src_dir, f),
                         os.path.join(dst_fig, '%s_%s' % (prefix, tag)))
            n_fig += 1
    log.append(('figures/routeB/', '拷入 %d 幅图（主图 + S9）' % n_fig))

    # ---------------------------------------------------------------- 5. 复查
    print('=' * 74)
    for a, b in log:
        print('  %-20s %s' % (a, b))
    print('=' * 74)

    print()
    print('=== 复查 1：三处题名是否逐字一致 ===')
    zt = json.loads(read(os.path.join(REPO, '.zenodo.json')))['title']
    ct = re.search(r'^title: "(.*)"$', read(os.path.join(REPO, 'CITATION.cff')),
                   re.M).group(1)
    rt = read(os.path.join(REPO, 'README.md')).split('\n')[0].lstrip('# ')
    want = 'Analysis code for: ' + TITLE
    for nm, v in (('.zenodo.json', zt), ('CITATION.cff', ct), ('README.md', rt)):
        print('   %-16s %s  (%d 字符)' % (nm, '一致' if v == want else '不一致！', len(v)))
    assert zt == ct == rt == want

    print()
    print('=== 复查 2：仓库内是否还残留旧题名 / 英式拼写 / 真实姓名 ===')
    # 注意：以下模式串本身必然命中自己（它们就是被查找的目标），
    # 故扫描时跳过本脚本，否则永远无法清零。
    bad = [
        ('旧题名', 'DUOX2 indicates immune activation'),
        ('英式 tumour', r'\btumour\b'),
        ('真实姓名路径', r'/Users/weishengying'),
        ('已拒稿刊名', 'Scientific Reports'),
    ]
    for label, pat in bad:
        hits = []
        rx = re.compile(pat)
        for root, dirs, fs in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__')]
            for f in fs:
                if f == os.path.basename(__file__):
                    continue
                fp = os.path.join(root, f)
                if os.path.splitext(f)[1] in ('.png', '.npy', '.pkl'):
                    continue
                try:
                    s = read(fp)
                except Exception:
                    continue
                for i, l in enumerate(s.split('\n'), 1):
                    if rx.search(l):
                        hits.append('%s:L%d' % (os.path.relpath(fp, REPO), i))
        print('   %-14s 命中 %d 处%s' % (label, len(hits),
                                       ('  → ' + ', '.join(hits[:5])) if hits else ''))

    print()
    print('=== 复查 3：JSON 合法性 ===')
    json.loads(read(os.path.join(REPO, '.zenodo.json')))
    print('   .zenodo.json 解析通过')

    print()
    print('=== 复查 4：新增文件清单 ===')
    for d in ('code/routeB', 'results/routeB', 'figures/routeB'):
        p = os.path.join(REPO, d)
        fs = sorted(os.listdir(p))
        tot = sum(os.path.getsize(os.path.join(p, f)) for f in fs)
        print('   %-16s %2d 个文件  %8.1f KB' % (d, len(fs), tot / 1024))
        for f in fs:
            print('        %s' % f)


if __name__ == '__main__':
    main()
