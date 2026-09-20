# -*- coding: utf-8 -*-
"""
v25_final.py —— 方案B 全套重绘管线。

与 v22_sans.py 相比的变化：
  · 绘图脚本已由 v24_finalsize.py 改为"最终印刷尺寸 + 5–7 pt 字号"
  · 合成画布 1950 px（165.1 mm @300 dpi），面板 1:1 放置（半栏 949 / 通栏 1938）
  · 加入 v18c_fig3e.py 产出 Figure 3e 面板（该脚本不再自行合成 Figure 3）
  · 末尾统一调用 v17_compose.py 合成 12 幅图并同步到桌面投稿目录

依赖顺序（保持原管线语义）：
  1) make_all_figures_sci.py 的 20 幅图
  2) 立即裁 ga（来自 make_all 的 Fig14_GSE14520_validation，须在 gse14520_figures 覆盖前）
  3) fix_v11_figures / gse14520_figures / make_fig21 / make_fig22 /
     v16_power_analysis / v16b_gse14520_check / v18c_fig3e
  4) 裁 f15a（gse14520 的 Fig15_cohort_summary）与其余主面板
  5) 重建 _media
  6) v17_compose.py 合成 Figure1-8 + S1-S4 并同步 Desktop
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import os
import shutil

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
PAN = os.path.join(BASE, 'panels')
MEDIA = os.path.join(BASE, '_media')
FIGD = os.path.join(BASE, 'figures')
OUT = os.path.join(BASE, 'SR_submission', 'figures')
DESK = os.environ.get('DUOX2_FIG_OUT', OUT)
for d in (PAN, MEDIA, OUT):
    os.makedirs(d, exist_ok=True)

captured = {}
_orig = Figure.savefig


def _patched(self, fname, *a, **k):
    captured[os.path.basename(str(fname)).rsplit('.', 1)[0]] = self
    try:
        return _orig(self, fname, *a, **k)
    except Exception as e:
        print('  savefig 跳过:', fname, e)
        return None


Figure.savefig = _patched


def _buf(fig, dpi=300):
    """原始画布渲染（不套 tight bbox）。仅用于确实需要整幅画布的场合。"""
    FigureCanvasAgg(fig)
    fig.dpi = dpi
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


def _tight_buf(fig, dpi=300):
    """按 tight bbox 渲染，返回 (图像数组, ox, oy_top)。

    为什么必须这样：之前的 _buf 直接用画布渲染再按 axes 的 tight bbox 裁切。
    当 figsize 偏小（方案B 把尺寸收到最终印刷宽后尤其明显）时，x 轴标题会被挤出
    画布之外，画布渲染里它就是残缺的 —— 实拍发现 Figure 1d/1e 的 "Time (days)"、
    Figure 2a 的 "Time (years)" 被底边切成半个字。改成 tight 渲染后，图外内容
    也一并进入画布，再按平移量裁切，就不会丢。
    ox = tight 图左边界（英寸，自画布左）；oy_top = tight 图上边界（英寸，自画布底）。
    """
    import io
    from PIL import Image
    FigureCanvasAgg(fig)
    fig.dpi = dpi
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(renderer)                 # 单位：英寸
    fi_w, fi_h = fig.get_size_inches()
    pos = plt.rcParams['savefig.pad_inches'] or 0.0
    Figure.savefig = _orig
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor='white')
    finally:
        Figure.savefig = _patched
    buf.seek(0)
    arr = np.asarray(Image.open(buf).convert('RGB'))
    # tight 图的原点：左边界 ox（英寸，自画布左），上边界 oy_top（英寸，自画布底）
    ox = tb.x0 - pos
    oy_top = tb.y1 + pos
    return arr, ox, oy_top


def _crop_axes(arr, ox, oy_top, bb, pad=8, dpi=300):
    """在 tight 图上按 bb（原画布显示坐标，y 向上）裁切。

    原画布显示坐标 -> tight 图像素：
      x_img = bb.x - ox * dpi
      y_img = oy_top * dpi - bb.y      （图像 y 向下，故用上边界减）
    """
    from PIL import Image
    H, W = arr.shape[:2]
    x0 = int(round(bb.x0 - ox * dpi)) - pad
    x1 = int(round(bb.x1 - ox * dpi)) + pad
    y0 = int(round(oy_top * dpi - bb.y1)) - pad      # bb.y1 = 上边缘
    y1 = int(round(oy_top * dpi - bb.y0)) + pad
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError('裁切区域为空: x %d-%d y %d-%d (图像 %dx%d)'
                           % (x0, x1, y0, y1, W, H))
    return Image.fromarray(arr[y0:y1, x0:x1])


def crop_fig(key, prefix, naxes):
    from PIL import Image
    fig = captured[key]
    arr, ox, oy_top = _tight_buf(fig)
    assert len(fig.axes) == naxes, '%s: 期望 %d axes，实得 %d' % (key, naxes, len(fig.axes))
    letters = 'abcdefgh'
    out = []
    for i, ax in enumerate(fig.axes):
        bb = ax.get_tightbbox(fig.canvas.get_renderer())
        im = _crop_axes(arr, ox, oy_top, bb)
        p = os.path.join(PAN, '%s%s.png' % (prefix, letters[i]))
        im.save(p)
        out.append((os.path.basename(p), im.size))
    print('  %-8s <- %s' % (prefix, out))


def crop_row(key, out_name, naxes):
    """把 1xN 网格的两个面板按**同一坐标框**裁成一张图。

    必要性：tight bbox 使各面板裁出的宽度不等（f23a 815 px vs f23b 1036 px），
    合成时若各自缩放到同宽，则同一张图内两个面板的实际字号会差 1.9 倍
    （实测 f23a 10.8 pt vs f23b 5.7 pt）。合并成整行并以通栏放置后，
    两面板共享同一缩放系数，字号自然一致。
    """
    from PIL import Image
    import matplotlib.transforms as _mt
    fig = captured[key]
    arr, ox, oy_top = _tight_buf(fig)
    assert len(fig.axes) == naxes, '%s: 期望 %d axes，实得 %d' % (key, naxes, len(fig.axes))
    bbs = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in fig.axes]
    from matplotlib.transforms import Bbox
    union = Bbox.union([Bbox.from_extents(b.x0, b.y0, b.x1, b.y1) for b in bbs])
    im = _crop_axes(arr, ox, oy_top, union)
    p = os.path.join(PAN, out_name)
    im.save(p)
    print('  %-10s <- %s (整行合并裁切)' % (out_name, im.size))


def render_full(key, fname):
    from PIL import Image
    buf = _buf(captured[key])
    out = os.path.join(MEDIA, fname)
    Image.fromarray(buf).save(out)
    print('  %-20s <- %s' % (fname, Image.open(out).size))


def render_tight(key, fname):
    from PIL import Image
    fig = captured[key]
    out = os.path.join(MEDIA, fname)
    Figure.savefig = _orig
    try:
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    finally:
        Figure.savefig = _patched
    print('  %-20s <- %s' % (fname, Image.open(out).size))


def copy_fig(src, fname):
    out = os.path.join(MEDIA, fname)
    shutil.copy2(os.path.join(FIGD, src), out)
    print('  %-20s <- %s (copy)' % (fname, src))


def run_script(script):
    path = os.path.join(BASE, script)
    g = {'__name__': '__main__', '__file__': path}
    exec(compile(open(path).read(), script, 'exec'), g)


print('=' * 70)
print('1) make_all_figures_sci：20 幅图')
print('=' * 70)
import make_all_figures_sci as M
print('  font.family =', plt.rcParams['font.family'],
      '| font.size =', plt.rcParams['font.size'])
for fn in [M.fig1, M.fig2, M.fig3, M.fig4, M.fig5, M.fig6, M.fig7, M.fig8,
           M.fig9, M.fig10, M.fig11, M.fig12, M.fig13, M.fig14, M.fig15,
           M.fig16, M.fig17, M.fig18, M.fig19, M.fig20]:
    try:
        fn()
        print('   ✓', fn.__name__)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('   ✗', fn.__name__, e)

print('\n2) 裁 ga（fig14，必须在 gse14520_figures 覆盖前）')
crop_fig('Fig14_GSE14520_validation', 'g', 2)

print('\n3) 其余生成脚本')
for s in ['fix_v11_figures.py', 'gse14520_figures.py', 'make_fig21.py',
          'make_fig22.py', 'v16_power_analysis.py', 'v16b_gse14520_check.py',
          'v18c_fig3e.py']:
    print('  ->', s)
    run_script(s)

print('\n4) 裁 f15a 与主面板')
crop_fig('Fig15_cohort_summary', 'f15', 1)
crop_fig('Fig2_DUOX2_expr', 'f2', 2)
crop_fig('Fig12_external_validation_KM', 'f7', 2)
crop_fig('Fig7_immune', 'f13', 2)
crop_fig('Fig11_pancancer_KM', 'f16', 2)
crop_fig('Enh_Fig4_DUOX2_TIDE', 'f20', 2)
crop_fig('Fig13_coef_comparison', 'f8coef', 1)
crop_fig('Fig3_GSE14520_v11', 'f3', 2)
crop_fig('Fig21_purity_adjusted_TIDE', 'f21', 4)
crop_fig('Fig22_headtohead_v12', 'f22', 4)
crop_row('fig23_power_ci', 'f23row.png', 2)   # Figure 8：两面板同一坐标框
crop_fig('fig24_direction_ci', 'f24', 2)
# f8auc：单 axes 图，直接复制 tight 落盘版（画布渲染会切掉 "Time (years)"）
shutil.copy2(os.path.join(FIGD, 'Fig8_AUC.png'), os.path.join(PAN, 'f8auc.png'))
from PIL import Image as _PILImage
print('  %-10s <- %s (copy, tight)' % ('f8auc.png',
      _PILImage.open(os.path.join(PAN, 'f8auc.png')).size))

print('\n5) 重建 _media')
copy_fig('Fig1_volcano_v11.png', 'image1.png')
copy_fig('Fig16_GSE14520_refit_AUC.png', 'image9.png')
copy_fig('Fig5_coexpression.png', 'image11.png')
copy_fig('Fig6_GSEA.png', 'image12.png')
copy_fig('Fig10_pancancer_expr.png', 'image14.png')
copy_fig('Fig9_pancancer_forest.png', 'image15.png')
copy_fig('Enh_Fig2_Calibration.png', 'image18.png')
copy_fig('Enh_Fig3_DCA.png', 'image19.png')
# image4/5 原先用画布渲染，字号放大后 x 轴标题被底边切掉；
# 直接采用 savefig(bbox_inches='tight') 落盘的成品，保证标签完整。
copy_fig('Fig3_stage_KM.png', 'image4.png')
copy_fig('Fig5_KM_duox2_v11.png', 'image5.png')
render_tight('Enh_Fig1_ML_model_comparison', 'image17.png')

Figure.savefig = _orig
print('\n6) 合成 12 幅成品图')
run_script('v17_compose.py')

print('\n7) 同步桌面投稿目录')
if os.path.realpath(DESK) != os.path.realpath(OUT):
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(OUT, f), os.path.join(DESK, f))
else:
    print('  未设置 DUOX2_FIG_OUT，跳过外部同步')
print('  已同步 %d 个 png -> %s' % (len([f for f in os.listdir(OUT) if f.endswith('.png')]), DESK))

print('\nDONE')
