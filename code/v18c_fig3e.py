# -*- coding: utf-8 -*-
"""
v18c_fig3e.py —— 产出 Figure 3 的 e 面板（跨队列 C-index 森林图）panels/f24b_flat.png。

方案B（按最终印刷尺寸出图）后：
  · 该面板为通栏（合成后 1938 px = 6.46 in @300 dpi），故设计宽度就取 ~6.5 in；
  · 图内字号 = 最终印刷字号（标签 5.4 pt / 数值 5.0 pt / 标题 6.9 pt），不再是
    在 6.6 in 画布上画 8–10 pt 再被合成缩小；
  · 本脚本**只产出面板**，Figure 3 的合成由 v17_compose.py 统一负责（旧版在此自行
    合成，用的是已废弃的面板组合，会覆盖成品图）。
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = os.path.join(BASE, 'data')
PAN = os.path.join(BASE, 'panels')
MEDIA = os.path.join(BASE, '_media')
OUT = os.path.join(BASE, 'SR_submission', 'figures')
BACKUP = os.path.join(BASE, 'SR_submission', 'figures_backup')
DESK = os.environ.get('DUOX2_FIG_OUT', OUT)
os.makedirs(OUT, exist_ok=True)

# ---------- 字体：Scientific Reports 图片规范 —— clear sans-serif ----------
for p in ['/System/Library/Fonts/Helvetica.ttc',
          '/System/Library/Fonts/Supplemental/Arial.ttf',
          '/System/Library/Fonts/Supplemental/Arial Bold.ttf']:
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.unicode_minus'] = False

J = json.load(open(os.path.join(DATA, 'v16_gse14520_check.json')))

LABELS = ['TCGA-LIHC\n(5-fold CV, training)', 'GSE14520\n(refit, in-sample)',
          'GSE14520\n(refit, 5-fold CV)', 'GSE14520\n(fixed coefficients)',
          'GSE76427\n(fixed coefficients)']
VALS = [0.690, 0.634, J['gse14520_refit']['c_index_cv5'],
        J['gse14520_fixed']['c_index'], 0.482]
CIS = [(0.620, 0.760), (0.573, 0.700), tuple(J['gse14520_refit']['cv5_ci95']),
       tuple(J['gse14520_fixed']['ci95']), (0.335, 0.642)]
COLS = ['#1f4e79', '#7f7f7f', '#7f7f7f', '#c00000', '#c00000']


def render_e(w_in, h_in, out_png, fs_lab=5.4, fs_txt=5.0, fs_title=6.9):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ys = np.arange(len(LABELS))[::-1]
    # 首行是 5 折 CV 的折间均值±SD，不是 95% CI，标注需与正文口径一致
    TXT = ['%.3f \u00b1 %.3f (mean \u00b1 SD)' % (VALS[0], (CIS[0][1] - CIS[0][0]) / 2.0)]
    TXT += ['%.3f (95%% CI %.3f\u2013%.3f)' % (v, ci[0], ci[1])
            for v, ci in list(zip(VALS, CIS))[1:]]
    for y, v, ci, c, t in zip(ys, VALS, CIS, COLS, TXT):
        ax.plot([ci[0], ci[1]], [y, y], color=c, lw=1.12, alpha=0.85,
                solid_capstyle='round')
        ax.plot(v, y, 'o', color=c, ms=4.09)
        ax.text(0.795, y + 0.13, t, fontsize=fs_txt, va='center', ha='left')
    ax.axvline(0.5, color='#000000', ls='--', lw=0.66, zorder=1)
    ax.text(0.5, -0.62, '0.5', fontsize=fs_txt, ha='center', va='center', zorder=4,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.2))
    ax.set_yticks(ys)
    ax.set_yticklabels(LABELS, fontsize=fs_lab)
    ax.set_xlabel('C-index', fontsize=6.3, labelpad=2)
    ax.set_xlim(0.22, 0.86)
    ax.set_ylim(-0.85, len(LABELS) - 0.15)
    ax.set_title('Discrimination of the three-gene model across evaluations',
                 loc='left', fontsize=fs_title, pad=5)
    ax.grid(axis='x', alpha=0.25, ls=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=0.4, labelsize=5.4)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.06,
                facecolor='white')
    plt.close(fig)
    im = Image.open(out_png)
    return im.size


# ---------- 1. 渲染 e 面板（方案B：按最终印刷尺寸出图）----------
# 面板最终宽度 = 通栏 1938 px @300dpi = 6.46 in，故设计宽度就取 6.46/ρ ≈ 6.5 in，
# 图内字号直接等于最终印刷字号（fs_lab/fs_txt/fs_title = 5.4/5.0/6.9 pt）。
# 合成（Figure 3 a,b / c,d / e 通栏）由 v17_compose.py 统一负责，本脚本不再自行合成，
# 以免用旧的面板组合覆盖成品图。
FULL_W = 1938          # 合成后 e 面板宽度（通栏 165.1 mm @300 dpi）
TARGET_H = (735, 790)  # 期望合成行高 -> aspect ≈ 0.38-0.40
CENTER_H = 756

best, best_png = None, None
for w_in in (6.1, 6.3, 6.5, 6.7):
    h_in = round(0.390 * w_in, 2)
    tmp = os.path.join(MEDIA, 'f24b_try_%.2f_%.2f.png') % (w_in, h_in)
    w, h = render_e(w_in, h_in, tmp)
    row_h = round(h * FULL_W / w)
    print('figsize=(%.2f, %.2f) -> panel %dx%d  aspect %.3f  composite row_h=%d'
          % (w_in, h_in, w, h, h / w, row_h))
    if best is None or abs(row_h - CENTER_H) < best[0]:
        best = (abs(row_h - CENTER_H), w_in, h_in, tmp)
    if TARGET_H[0] <= row_h <= TARGET_H[1]:
        break

print('选用 figsize = (%.2f, %.2f)' % (best[1], best[2]))
E_PNG = os.path.join(PAN, 'f24b_flat.png')
w, h = render_e(best[1], best[2], E_PNG)
print('e 面板 ->', E_PNG, '%dx%d' % (w, h), 'aspect %.3f' % (h / w),
      '| 合成行高', round(h * FULL_W / w), '| 设计宽 %.2f in' % (w / 300.))
if not os.path.exists(E_PNG):
    raise SystemExit('e 面板未生成')
print('\n注意：Figure 3 的合成请运行 v17_compose.py（本脚本只产出 e 面板）。')
