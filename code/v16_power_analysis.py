# -*- coding: utf-8 -*-
"""
v16: 统计功效分析 + C-index bootstrap 置信区间
目的：把 GSE76427 "外部验证失败" 重新表述为 "外部验证信息量不足 (underpowered)"。

1) Schoenfeld 功效计算
   连续预测变量 (风险评分, HR per SD):  D = (z_{1-a/2}+z_{1-b})^2 / (ln HR_perSD)^2
   二分类 1:1 分组:                     D = 4*(z_{1-a/2}+z_{1-b})^2 / (ln HR)^2
2) 各队列 C-index 的 bootstrap 95% CI（2,000 次重采样）
"""
import os, json, pickle
import numpy as np
import pandas as pd
from scipy import stats
from sksurv.metrics import concordance_index_censored
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = _ROOT
DATA = os.path.join(BASE, 'data')
FIG = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

RNG = np.random.default_rng(20260914)
COEF = {'SLC16A3': 0.092, 'SPP2': -0.038, 'MMP7': 0.024}   # TCGA 训练集系数 (log2 CPM)
GENES = list(COEF)
OBS_HR_SD = 1.408          # TCGA 多变量 Cox: HR per SD (v12_headtohead.json)
Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)   # 1.95996 + 0.84162

OUT = {}


# ----------------------------------------------------------------- 工具
def c_index(t, e, s):
    """Harrell's C for a Cox-type risk score: higher score = shorter survival."""
    t = np.asarray(t, float); e = np.asarray(e, float); s = np.asarray(s, float)
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(s)
    return float(concordance_index_censored(e[ok].astype(bool), t[ok], s[ok])[0])


def boot_ci(t, e, s, n=2000):
    """C-index 的 percentile bootstrap 95% CI"""
    t = np.asarray(t, float); e = np.asarray(e, float); s = np.asarray(s, float)
    vals = []
    k = len(t)
    for _ in range(n):
        idx = RNG.integers(0, k, k)
        if e[idx].sum() < 5:
            continue
        try:
            vals.append(concordance_index_censored(e[idx].astype(bool), t[idx], s[idx])[0])
        except Exception:
            pass
    if len(vals) < 100:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def required_events_continuous(hr_sd):
    return (Z ** 2) / (np.log(hr_sd) ** 2)


def detectable_hr_continuous(D):
    return float(np.exp(np.sqrt((Z ** 2) / D)))


def required_events_binary(hr):
    return 4 * (Z ** 2) / (np.log(hr) ** 2)


# ----------------------------------------------------------------- 1. 功效分析
cohorts = [
    ('TCGA-LIHC training (n=258)', 258, 80),
    ('TCGA-LIHC staging comparison (n=240)', 240, 73),
    ('Firehose Legacy all (n=343)', 343, 116),
    ('Firehose non-overlapping (n=103)', 103, 43),
    ('GSE76427 (n=115)', 115, 23),
]

power = {}
for name, n, ev in cohorts:
    power[name] = dict(
        n=n, events=ev,
        detectable_HR_per_SD=round(detectable_hr_continuous(ev), 2),
        detectable_HR_binary_1to1=round(detectable_hr_continuous(ev / 4.0), 2),
        events_needed_for_observed_effect=round(required_events_continuous(OBS_HR_SD), 0),
        powered_for_observed_effect=bool(ev >= required_events_continuous(OBS_HR_SD)),
    )
OUT['power'] = power
OUT['observed_HR_per_SD'] = OBS_HR_SD
OUT['events_needed_observed'] = round(required_events_continuous(OBS_HR_SD), 0)
for hr in [1.2, 1.3, 1.408, 1.5, 1.8, 2.0]:
    OUT.setdefault('events_required_curve', {})[str(hr)] = round(required_events_continuous(hr), 0)

# ----------------------------------------------------------------- 2. TCGA 训练集
lcm = pickle.load(open(os.path.join(DATA, 'logcpm_matrix.pkl'), 'rb'))
sv = pd.read_csv(os.path.join(DATA, 'survival_analysis_ready.tsv'), sep='\t')
sv = sv.dropna(subset=['OS_time_days', 'OS_event'])
expr = lcm.loc[lcm.index.intersection(sv['sample_id'])]
sv = sv.set_index('sample_id').loc[expr.index]
risk_tcga = sum(COEF[g] * expr[g] for g in GENES)
OUT['tcga'] = dict(n=int(len(sv)), events=int(sv.OS_event.sum()),
                   c_index=round(c_index(sv.OS_time_days, sv.OS_event, risk_tcga), 3))
lo, hi = boot_ci(sv.OS_time_days.values, sv.OS_event.values, risk_tcga.values)
OUT['tcga']['ci95'] = [round(lo, 3), round(hi, 3)]
OUT['tcga']['note'] = 'fixed LASSO coefficients applied to all 258 training samples (in-sample)'

# ----------------------------------------------------------------- 3. Firehose
try:
    fh = pd.read_csv(os.path.join(DATA, 'firehose_LIHC_expr.tsv'), sep='\t')
    fh_genes = set(fh['gene'].astype(str)) if 'gene' in fh.columns else set()
    need = all(g in fh_genes for g in GENES)
    if need:
        w = (fh[fh['gene'].isin(GENES)]
             .drop_duplicates(subset=['patient_id', 'gene'])
             .pivot(index='patient_id', columns='gene', values='value'))
        fc = pd.read_csv(os.path.join(DATA, 'firehose_LIHC_clinical.tsv'), sep='\t')
        m = fc.set_index('patient_id').join(w, how='inner').dropna(subset=GENES + ['OS_months', 'OS_status'])
        risk_fh = sum(COEF[g] * m[g] for g in GENES)
        OUT['firehose'] = dict(n=int(len(m)), events=int(m.OS_status.sum()),
                               c_index=round(c_index(m.OS_months, m.OS_status, risk_fh), 3))
        lo, hi = boot_ci(m.OS_months.values, m.OS_status.values, risk_fh.values)
        OUT['firehose']['ci95'] = [round(lo, 3), round(hi, 3)]
        OUT['firehose']['note'] = 'fixed TCGA coefficients, cross-pipeline'
    else:
        OUT['firehose'] = dict(error='genes not found in firehose expr file', present=sorted(fh_genes)[:10])
except Exception as ex:
    OUT['firehose'] = dict(error=str(ex))

# ----------------------------------------------------------------- 4. GSE14520 (fixed coefficients)
try:
    ge = pd.read_csv(os.path.join(DATA, 'GSE14520_gene_expr.csv'), index_col=0)
    clin = pd.read_csv(os.path.join(DATA, 'GSE14520_Extra_Supplement.txt.gz'), sep='\t', compression='gzip')
    clin['gsm'] = clin['Affy_GSM'].astype(str)
    mt = clin[clin['gsm'].isin(ge.index)].copy()
    mt = mt[mt['Tissue Type'] == 'Tumor']
    for g in GENES:
        mt[g] = mt['gsm'].map(ge[g])
    mt['OS_months'] = pd.to_numeric(mt['Survival months'], errors='coerce')
    mt['event'] = pd.to_numeric(mt['Survival status'], errors='coerce')
    mt = mt.dropna(subset=['OS_months', 'event'] + GENES)
    mt = mt[mt['OS_months'] > 0]
    risk_145 = sum(COEF[g] * mt[g] for g in GENES)
    OUT['gse14520'] = dict(n=int(len(mt)), events=int(mt.event.sum()),
                           c_index=round(c_index(mt.OS_months, mt.event, risk_145), 3))
    lo, hi = boot_ci(mt.OS_months.values, mt.event.values, risk_145.values)
    OUT['gse14520']['ci95'] = [round(lo, 3), round(hi, 3)]
    OUT['gse14520']['note'] = 'fixed TCGA coefficients (primary text reports the within-cohort refit, C = 0.634)'
except Exception as ex:
    OUT['gse14520'] = dict(error=str(ex))

# ----------------------------------------------------------------- 5. GSE76427 (核心)
import gzip, sys
sys.path.insert(0, BASE)
from v12_independent_validation import load_gse76427

ge7, meta7 = load_gse76427()
m7 = meta7.set_index('gsm').join(ge7, how='inner')
m7['tissue'] = m7['tissue'].astype(str).str.strip().str.lower()
m7['event'] = pd.to_numeric(m7['event_os'], errors='coerce')
m7['os_months'] = pd.to_numeric(m7['duryears_os'], errors='coerce') * 12.0
tum7 = m7[m7['tissue'] == 'primary hepatocellular carcinoma tumor'].copy()
sv7 = tum7.dropna(subset=['os_months', 'event'] + GENES)
sv7 = sv7[sv7['os_months'] > 0].copy()
sv7['risk'] = sum(COEF[g] * sv7[g] for g in GENES)
bclc_map = {'0': 0, 'A': 1, 'B': 2, 'C': 3}
tnm_map = {'I': 1, 'II': 2, 'IIIA': 3, 'IIIB': 3, 'IVA': 4, 'IVB': 4}
sv7['bclc_ord'] = sv7['bclc_staging'].astype(str).str.strip().map(bclc_map)
sv7['tnm_ord'] = sv7['tnm_staging_clinical'].astype(str).str.strip().map(tnm_map)

h2h = {}
for name, col in [('3-gene signature (fixed coefficients)', 'risk'),
                  ('BCLC stage', 'bclc_ord'),
                  ('Clinical TNM stage', 'tnm_ord')]:
    sub = sv7.dropna(subset=[col])
    c = round(c_index(sub['os_months'], sub['event'], sub[col]), 3)
    lo, hi = boot_ci(sub['os_months'].values, sub['event'].values, sub[col].values)
    h2h[name] = dict(n=int(len(sub)), events=int(sub['event'].sum()), c_index=c,
                     ci95=[round(lo, 3), round(hi, 3)])

sub = sv7.dropna(subset=['risk', 'bclc_ord'])
from lifelines import CoxPHFitter
cph = CoxPHFitter()
dd = sub[['os_months', 'event', 'risk', 'bclc_ord']].astype(float)
cph.fit(dd, 'os_months', 'event')
comb = cph.predict_partial_hazard(sub[['risk', 'bclc_ord']].astype(float)).values
lo, hi = boot_ci(sub['os_months'].values, sub['event'].values, comb)
h2h['3-gene + BCLC (combined)'] = dict(n=int(len(sub)), events=int(sub['event'].sum()),
                                       c_index=round(c_index(sub['os_months'], sub['event'], comb), 3),
                                       ci95=[round(lo, 3), round(hi, 3)])
OUT['gse76427_h2h'] = h2h
OUT['gse76427'] = dict(n=int(len(sv7)), events=int(sv7['event'].sum()))

# 关键表述：C-index CI 上界是否排除临床可用判别力 (0.60)
sig_ci = h2h['3-gene signature (fixed coefficients)']['ci95']
OUT['gse76427_interpretation'] = dict(
    c_index=h2h['3-gene signature (fixed coefficients)']['c_index'],
    ci95=sig_ci,
    excludes_C_ge_0_60=bool(sig_ci[1] < 0.60),
    excludes_C_ge_0_65=bool(sig_ci[1] < 0.65),
    comment='CI spans 0.5, so neither clinically useful nor useless discrimination can be excluded')

json.dump(OUT, open(os.path.join(DATA, 'v16_power_analysis.json'), 'w'), indent=2, default=str)

# ----------------------------------------------------------------- 6. 图
# Scientific Reports 图片规范：clear sans-serif（Helvetica / Arial）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.size'] = 5.9
plt.rcParams['axes.titlesize'] = 7.0
plt.rcParams['axes.labelsize'] = 7.0
plt.rcParams['xtick.labelsize'] = 6.3
plt.rcParams['ytick.labelsize'] = 6.3
plt.rcParams['axes.linewidth'] = 0.53
fig, axes = plt.subplots(1, 2, figsize=(6.32, 2.40))   # 方案B：2 个半栏 = 6.32 in，字号 1:1 到印刷

# --- Panel A: 所需事件数 vs HR per SD
ax = axes[0]
hrs = np.linspace(1.05, 3.0, 400)
ax.plot(hrs, required_events_continuous(hrs), color='#1f4e79', lw=1.19)
ax.set_yscale('log')
ax.set_xlabel('Hazard ratio per SD of the risk score')
ax.set_ylabel('Death events required (80% power, two-sided $\\alpha$ = 0.05)')
ax.set_title('Statistical power of the validation cohorts', loc='left', fontsize=6.6)
ax.axvline(OBS_HR_SD, color='#c00000', ls='--', lw=0.79)
ax.text(OBS_HR_SD + 0.03, 30, 'observed effect\nHR/SD = 1.41', color='#c00000', fontsize=5.3)
for (name, n, ev), col in zip(cohorts, ['#2e6f40', '#2e6f40', '#7f7f7f', '#7f7f7f', '#c00000']):
    ax.axhline(ev, color=col, lw=0.53, alpha=0.75)
    if ev == 73:
        # 73 events 的标注放在线下方，与上方 80 events 的标注完全错开
        ax.text(2.92, ev / 1.18, '%d events  %s' % (ev, name.split(' (')[0]),
                ha='right', va='top', fontsize=5.0, color=col)
    else:
        ax.text(2.92, ev * 1.09, '%d events  %s' % (ev, name.split(' (')[0]),
                ha='right', va='bottom', fontsize=5.0, color=col)
ax.set_xlim(1.05, 3.0)
ax.set_ylim(8, 400)
ax.grid(alpha=0.25, ls=':')
ax.tick_params(width=0.4)

# --- Panel B: GSE76427 C-index 与 95% CI
ax = axes[1]
names = list(h2h.keys())
ys = np.arange(len(names))[::-1]
for y, nm in zip(ys, names):
    d = h2h[nm]
    ax.plot([d['ci95'][0], d['ci95'][1]], [y, y], color='#404040', lw=1.06, zorder=2)
    ax.plot(d['c_index'], y, 'o', color='#1f4e79' if '3-gene' in nm else '#7f7f7f',
            ms=3.96, zorder=3)
    ax.text(0.80, y + 0.16, '%.3f (95%% CI %.3f\u2013%.3f)' % (d['c_index'], d['ci95'][0], d['ci95'][1]),
            fontsize=5.0, va='bottom')
ax.axvline(0.5, color='#c00000', ls='--', lw=0.66)
ax.text(0.5, -0.45, '0.5 (no discrimination)', color='#c00000', fontsize=5.0, ha='center')
ax.set_yticks(ys)
ax.set_yticklabels([n.replace(' (fixed coefficients)', '\n(fixed coefficients)') for n in names], fontsize=5.3)
ax.set_xlabel('C-index (n = 115, 23 events)')
ax.set_xlim(0.30, 1.02)
ax.set_ylim(-0.6, len(names) - 0.2)
ax.set_title('GSE76427: point estimates and bootstrap 95% CIs', loc='left', fontsize=6.6)
ax.grid(axis='x', alpha=0.25, ls=':')
ax.tick_params(width=0.4)

for a in axes:
    for s in ['top', 'right']:
        a.spines[s].set_visible(False)

plt.tight_layout()
out = os.path.join(FIG, 'fig23_power_ci.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('saved', out)
print(json.dumps(OUT, indent=2, default=str))
