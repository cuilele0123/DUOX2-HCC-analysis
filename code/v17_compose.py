# -*- coding: utf-8 -*-
"""
v17_compose.py —— 把面板合成为 8 幅 SR 主图 + 补充材料图
输出: SR_submission/figures/Figure1..8.png, Supplementary Fig S1-S4
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
PAN = os.path.join(BASE, 'panels')
MEDIA = os.path.join(BASE, '_media')
OUT = os.path.join(BASE, 'SR_submission', 'figures')
os.makedirs(OUT, exist_ok=True)

# 画布 = Word 正文栏宽 165.1 mm @300 dpi = 1950 px。
# 方案B 的关键：面板按最终印刷尺寸出图，此处 1:1 放置，**图内字号即最终字号**。
CW = 1950
GAP = 40           # 行/列间距（3.4 mm）
BAND = 44          # 面板字母带高
MARGIN = 6
LETTER_PX = 28     # 面板字母 7 pt（300 dpi 下文字带高 ≈ 4.0 × pt）

def _panel_font(px):
    """面板字母字体：SR 要求小写粗体 + 无衬线；Helvetica 粗体面在 .ttc 的 index 1。"""
    for _p, _i in [('/System/Library/Fonts/Helvetica.ttc', 1),
                   ('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 0)]:
        try:
            return ImageFont.truetype(_p, px, index=_i)
        except Exception:
            continue
    return ImageFont.truetype(
        fm.findfont(fm.FontProperties(family='sans-serif', weight='bold')), px)


FONT = _panel_font(LETTER_PX)


def load(key):
    if key.startswith('img'):
        return Image.open(os.path.join(MEDIA, 'image%s.png' % key[3:]))
    return Image.open(os.path.join(PAN, key + '.png'))


def compose(name, rows, note=''):
    """rows: list of rows; each row = list of (key, span). span=1 半宽, 2 全宽"""
    n_cells = 2
    inner = CW - 2 * MARGIN
    cell_w = (inner - GAP) // 2
    full_w = inner
    # 预缩放
    scaled_rows = []
    for row in rows:
        items = []
        for key, span in row:
            im = load(key)
            w = full_w if span == 2 else cell_w
            h = round(im.size[1] * w / im.size[0])
            items.append((im.resize((w, h), Image.LANCZOS), key, span))
        scaled_rows.append(items)
    heights = [max(it[0].size[1] for it in row) for row in scaled_rows]
    H = MARGIN + sum(BAND + h + GAP for h in heights) - GAP + MARGIN
    canvas = Image.new('RGB', (CW, H), 'white')
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    for row, rh in zip(scaled_rows, heights):
        # 行内布局：span==2 占满；否则按列排
        span2 = [it for it in row if it[2] == 2]
        span1 = [it for it in row if it[2] == 1]
        x = MARGIN
        if span1:
            n1 = len(span1)
            positions = []
            if n1 == 1:
                positions = [MARGIN + (inner - cell_w) // 2]  # 单面板居中
            else:
                total = n1 * cell_w + (n1 - 1) * GAP
                x0 = MARGIN + max(0, (inner - total) // 2)
                positions = [x0 + i * (cell_w + GAP) for i in range(n1)]
            for (im, key, _), px in zip(span1, positions):
                letter, idx = next_letters(len(scaled_rows), scaled_rows, row)
            # 字母在下方统一编号，这里先画面板
        # 统一面板顶对齐，画字母
        x_iter = []
        all_items = [(it, 'full' if it[2] == 2 else 'cell') for it in row]
        # 计算每个面板 x
        xs = []
        cells1 = [it for it in row if it[2] == 1]
        if len(cells1) == 1 and not any(it[2] == 2 for it in row):
            xs = [MARGIN + (inner - cell_w) // 2]
        elif cells1:
            x0 = MARGIN
            xs = [x0 + i * (cell_w + GAP) for i in range(len(cells1))]
        fulls = [it for it in row if it[2] == 2]
        fx = MARGIN
        for it in row:
            im, key, span = it
            if span == 2:
                px, pw = fx, full_w
            else:
                px, pw = xs[cells1.index(it)], cell_w
            letter = LETTERS[global_counter[0]]
            global_counter[0] += 1
            draw.text((px + 4, y + 2), letter, font=FONT, fill='black')
            canvas.paste(im, (px, y + BAND))
        y += BAND + rh + GAP
    out = os.path.join(OUT, name + '.png')
    canvas.save(out, dpi=(300, 300))
    print('%-14s %dx%d  %s' % (name, CW, H, note))


LETTERS = 'abcdefgh'

# 用两阶段：先定义带自动字母的简化版
_counter = [0]

# 重新实现（简洁、字母自动编号，按行读取顺序）
def compose2(name, rows, note=''):
    inner = CW - 2 * MARGIN
    cell_w = (inner - GAP) // 2
    full_w = inner
    scaled, heights = [], []
    for row in rows:
        items = []
        for key, span in row:
            im = load(key)
            w = full_w if span == 2 else cell_w
            h = round(im.size[1] * w / im.size[0])
            items.append([im.resize((w, h), Image.LANCZOS), span])
        scaled.append(items)
        heights.append(max(it[0].size[1] for it in items))
    H = MARGIN + sum(BAND + h + GAP for h in heights) - GAP + MARGIN
    canvas = Image.new('RGB', (CW, H), 'white')
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    k = 0
    for row, rh in zip(scaled, heights):
        cells1 = [it for it in row if it[1] == 1]
        xs = {}
        if len(cells1) == 1:
            xs[id(cells1[0])] = MARGIN + (inner - cell_w) // 2
        else:
            for i, it in enumerate(cells1):
                xs[id(it)] = MARGIN + i * (cell_w + GAP)
        for it in row:
            im, span = it
            px = full_w_x = MARGIN if span == 2 else xs[id(it)]
            letter = LETTERS[k]
            k += 1
            draw.text((px + 4, y + 2), letter, font=FONT, fill='black')
            canvas.paste(im, (px, y + BAND))
        y += BAND + rh + GAP
    out = os.path.join(OUT, name + '.png')
    canvas.save(out, dpi=(300, 300))
    print('%-16s %dx%d (x%.2f full-page)  %s' % (name, CW, H, H / 1950., note))


# ---------------- 8 幅主图 ----------------
compose2('Figure1', [
    [('img1', 1), ('f2a', 1)],
    [('f2b', 1), ('img4', 1)],
    [('img5', 1), ('f3a', 1)],
], 'expression & stage')

compose2('Figure2', [
    [('f8auc', 1), ('f7b', 1)],
    [('f8coefa', 1), ('img17', 1)],
], 'model internal')

compose2('Figure3', [
    [('f7a', 1), ('f3b', 1)],
    [('img9', 1), ('f15a', 1)],
    [('f24b_flat', 2)],
], 'external validation')

compose2('Figure4', [
    [('img11', 1), ('img12', 1)],
], 'coexpression & GSEA')

compose2('Figure5', [
    [('f13a', 1), ('f13b', 1)],
    [('f20a', 1), ('f20b', 1)],
], 'immune & TIDE')

compose2('Figure6', [
    [('img14', 2)],
    [('img15', 1), ('f16a', 1)],
    [('f16b', 1)],
], 'pan-cancer')

compose2('Figure7', [
    [('f22a', 1), ('f22b', 1)],
    [('f22d', 1)],
], 'staging & independence')

compose2('Figure8', [
    [('f23row', 2)],
], 'power & CI')

# ---------------- 补充材料图 ----------------
compose2('FigureS1', [[('ga', 1)]], 'GSE14520 TNM')       # gb 备用(与Fig3b重复)
compose2('FigureS2', [[('img18', 2)]], 'calibration')
compose2('FigureS3', [[('img19', 2)]], 'DCA')
compose2('FigureS4', [
    [('f21a', 1), ('f21b', 1)],
    [('f21c', 1), ('f21d', 1)],
], 'purity robustness')
print('DONE ->', OUT)
