# Analysis code for: DUOX2 marks biliary/ductular epithelium in hepatocellular carcinoma: reported immune and prognostic associations reflect tissue composition

Version 2.0.0 · released 2026-10-08

This archive contains the complete analysis code, the derived result tables and the
publication figures for a multi-cohort study of DUOX2 in hepatocellular carcinoma
(HCC). It accompanies the manuscript on the biliary/ductular localization of DUOX2;
the manuscript identifier and the journal are given in the archival record of this
release.

## 1. What is included

| Path | Contents |
|---|---|
| `code/` | Python analysis scripts (data acquisition through final figure composition) |
| `results/` | Derived tables and JSON summaries underlying every reported statistic |
| `results/routeB/` | Derived tables for the compartment and single-cell atlas analyses |
| `code/routeB/` | Analysis scripts for the localization, compartment-adjustment and single-cell atlas steps |
| `figures/` | Publication figures of the first release: `Figure1-8` and `FigureS1-4` (PNG, 1950 px = 165.1 mm at 300 dpi) |
| `figures/routeB/` | Publication figures of the present release: `Figure_1-6` and `SupplementaryFigure_S9` (PNG, 1950-2160 px at 300 dpi; S9 at 600 dpi) |
| `figures/intermediate/` | Single-analysis figures and panels before composition |
| `requirements.txt` | Exact package versions used |
| `CITATION.cff`, `.zenodo.json` | Citation and archival metadata |
| `LICENSE` | MIT license |

The figure scripts use a clear sans-serif face (Helvetica/Arial) throughout, as required by most journals in this field; no journal-specific styling is assumed.

Raw public data are **not** redistributed here. They are freely available from the
sources listed in section 3 and can be re-downloaded with `code/download_*.py`.

## 2. Environment

Python 3.13. Tested with the versions in `requirements.txt`:

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Scripts are run from inside `code/`. Output directories (`data/`, `figures/`,
`panels/`, `SR_submission/figures/`) are created relative to the script directory;
the absolute paths used on the development machine have been replaced with
script-relative paths, so nothing outside this archive is required.

## 3. Data sources

| Dataset | Role | Accession / portal |
|---|---|---|
| TCGA-LIHC | discovery cohort, 424 samples | GDC Data Portal (project TCGA-LIHC) |
| TCGA PanCancer Atlas | pan-cancer cohort, 20 types / 7,892 samples | cBioPortal |
| Firehose Legacy LIHC | cross-pipeline comparison, 379 samples | cBioPortal, `lihc_tcga` |
| GSE14520 | external validation, 221 tumors | GEO |
| GSE76427 | external validation, 167 samples (115 tumor / 52 adjacent) | GEO, GPL10558 |
| HPA v25.1 | cell-type expression reference | proteinatlas.org |

## 4. Script map

### Data preparation
- `download_counts.py`, `download_resume*.py`, `download_parallel.py` - fetch STAR-Counts
  and clinical files from the GDC API with retry/resume.
- `build_metadata.py` - stage reconstruction from AJCC 7th edition T/N/M fields,
  survival-time construction and cBioPortal cross-checking.
- `build_expr_matrix.py` - log2(CPM+1) matrix construction and gene filtering.

### Differential expression and enrichment
- `deseq_analysis.py` - pydeseq2 contrast of advanced- versus early-stage tumors
  (201 patients: 139 early, 62 advanced; 44,240 genes after filtering).
- `enrichment_analysis.py` - GO/KEGG over-representation (Enrichr API via gseapy)
  and pre-ranked GSEA against genome-wide Pearson correlation with DUOX2.

### DUOX2-specific analyses
- `duox2_deep_analysis.py`, `duox2_core_verify.py` - stage association, adjacent
  non-tumor comparison, and the co-expression network.

### Prognostic model
- `lasso_cox_model.py` - LASSO-Cox selection over the 500 most stage-associated
  genes, with five-fold cross-validation along the alpha path.
- `ml_model_comparison.py`, `ml_final.py` - benchmarking of the three-gene signature
  against random survival forest, gradient boosting and Cox alternatives under
  identical folds.
- `ml_calibration_dca.py`, `dca_fix.py` - time-dependent calibration and decision
  curve analysis.

### Immune microenvironment
- `tide_duox2_figures.py` - TIDE scoring, ssGSEA infiltration and checkpoint
  correlations.
- `purity_adjusted_tide.py` - rank-based epithelial/stromal/immune compartment
  scores, the E/S ratio adjustment of DUOX2-TIDE association, over-adjustment
  diagnostics, stratified analysis, multivariate regression and the bootstrap test
  of compartment origin.

### External validation and staging comparison
- `firehose_ext_validation.py` - cross-pipeline comparison, including the
  patient-level overlap analysis with the discovery cohort.
- `gse14520_parse.py`, `gse14520_validation.py` - independent cohort validation and
  within-cohort coefficient refitting.
- `v12_independent_validation.py` - the fully independent GSE76427 cohort
  (GPL10558 annotation, tumor/non-tumor comparison, signature transfer under
  three scaling strategies, and head-to-head comparison with BCLC/TNM stage).
- `v12_headtohead.py` - out-of-fold risk scores for the head-to-head comparison of
  the signature against AJCC stage, bootstrap confidence intervals on the C-index
  difference, and the multivariable Cox model.
- `v16b_gse14520_check.py` - direction-of-effect check for the transferred
  signature in GSE14520.
- `v16_power_analysis.py` - Schoenfeld power analysis and the CI widths shown in
  Figure 8.
- `v14_new_analyses.py` - additional analyses added in response to review.

### Pan-cancer
- `pan_cancer_duox2.py`, `pan_cancer_figures.py` - 20 tumor types, 7,892 samples,
  Benjamini-Hochberg correction of the survival associations.

### Figures
- `make_all_figures_sci.py` - 20 main analysis figures at 300 dpi (Figure 1-4 and
  Figure 6 sources among them).
- `gse14520_figures.py`, `make_fig21.py`, `make_fig22.py`,
  `ext_validation_figures.py`, `fix_v11_figures.py`, `make_fig17_v15.py` -
  cohort-specific and panel-specific figures.
- `v18c_fig3e.py` - the flat direction-of-effect panel of Figure 3e.
- `v17_compose.py` - composes the individual panels into `Figure1-8` and
  `FigureS1-4` on a 1950 px (165.1 mm at 300 dpi) canvas, one panel-width per
  journal column.
- `v25_final.py` - **the master pipeline**: runs the scripts above in dependency
  order, crops every panel from a tight-bbox render, and calls `v17_compose.py`.

## 5. Regenerating the publication figures

```
cd code
python v25_final.py
```

This writes the 12 composite figures into `SR_submission/figures/`. Set the
environment variable `DUOX2_FIG_OUT` to also copy them to another directory.

The figures are drawn at their final print size, so the nominal font size inside
each panel equals the final printed size (5-7 pt at 300 dpi, sans-serif
throughout), as required by the Nature Portfolio artwork guidelines.

## 6. Figure composition

| Figure | Panels | Source script |
|---|---|---|
| Figure 1 | volcano; DUOX2 by stage and tissue; stage KM; DUOX2 KM; GSE14520 | `make_all_figures_sci.py`, `fix_v11_figures.py` |
| Figure 2 | time-dependent AUC; external KM; coefficients; ML comparison | `make_all_figures_sci.py` |
| Figure 3 | external KM; GSE14520 refit AUC; cohort summary; direction of effect | `gse14520_figures.py`, `v18c_fig3e.py` |
| Figure 4 | co-expression network; GSEA | `make_all_figures_sci.py` |
| Figure 5 | immune infiltration; checkpoints; TIDE | `make_all_figures_sci.py` |
| Figure 6 | pan-cancer expression; forest plot; pancancer KM | `pan_cancer_figures.py` |
| Figure 7 | staging head-to-head; cohort independence | `v12_headtohead.py` |
| Figure 8 | statistical power and CI widths | `v16_power_analysis.py` |
| Figure S1 | GSE14520 TNM | `make_all_figures_sci.py` |
| Figure S2 | calibration | `ml_calibration_dca.py` |
| Figure S3 | decision curve | `ml_calibration_dca.py` |
| Figure S4 | tumor-purity robustness | `make_fig21.py` |

## 7. Reproducing the published numbers

Run the scripts in the order given in section 4. Each writes its output into
`data/` (relative to the script directory); `results/` in this archive holds the
final versions of those files as used in the manuscript. Every statistic quoted in
the text is present in `results/` - for example:

- `results/deseq2_early_vs_advanced_DEG.csv` - stage-associated genes
- `results/lasso_risk_analysis.csv` - per-patient risk scores (n = 258)
- `results/v12_headtohead.json` - signature C-index 0.662 versus AJCC 0.582
- `results/v12_independent_validation.json` - GSE76427 transfer results
- `results/purity_adjusted_summary.json` - purity-adjusted TIDE correlations

## 8. Citation

If you use this code, please cite the archived release (see `CITATION.cff`) and the
manuscript. The release is deposited on Zenodo; the version DOI is given in the
Code availability section of the manuscript.

## 9. License

MIT (see `LICENSE`).
