# Changelog

All notable changes to EpiProfile-Plants Dashboard.

---

## v3.14 (2026-10-30)

### Added
- **Upload accepts CSV, TSV, real Excel (.xlsx) and EpiProfile's tab-separated .xls**; delimiter is sniffed automatically. Phenodata column aliases (`Sample`, `Group`, `Condition`...) are mapped onto `Sample_Name` / `Sample_Group`.
- **Excel export** (`.xlsx`, one sheet per table) in Export to R.
- Screenshots of the demo dataset in `docs/screenshots/`.
- **Portable configuration**: experiments are declared in `experiments.json` (template: `experiments.example.json`), located via `--config`, `$EPIPROFILE_CONFIG` or next to the script. Relative paths resolve against the config file.
- **CLI `NAME=PATH` syntax** and automatic inference of `ndebug` / `dataset` from names, paths and ratios filenames, so the ndebug Compare and Paper & Design tabs work for directories passed on the command line.
- **Bundled demo dataset** (`examples/demo_experiment`, synthetic) loaded when nothing is configured; generator in `scripts/make_demo_data.py`.
- **Smoke tests** (`tests/test_smoke.py`): load the demo and render all 15 tabs.
- `EPIPROFILE_PORT` / `EPIPROFILE_HOST` environment variables, `--no-demo` flag.

### Changed
- Hard-coded Windows default paths removed from the source (moved to `experiments.example.json`).
- Default bind host is now `127.0.0.1` (use `--host 0.0.0.0` to expose on the network).
- Bare `except:` clauses narrowed to `except Exception:`.
- README / WHITEPAPER updated to the current tab count and layout; `epiprofile.db` and `experiments.json` ignored by git.

### Fixed
- **Dash 4 compatibility**: header Experiment / Palette dropdowns rendered white-on-white (new `.dash-dropdown-*` markup inherited the header text colour); PCA, Statistics and ndebug Compare stayed empty until a control was touched (`prevent_initial_call` on their content callbacks).
- Missing experiment directories now exit with status 1 and an actionable message.

---

## v3.13 (2026-02-26)

### Changed
- Sidebar navigation panel replaces the horizontal tab strip; compact header; fixed experiment selector overlap.

---

## v3.12 (2026-02-22)

### Added
- **Paper & Design tab** with publication information for PXD046788, PXD014739 and PXD046034.

---

## v3.11 (2026-02-22)

### Added
- **H3K79 Validation tab** aggregating ratios, areas, RT, PSMs and per-sample detail files for the H3 73-83 region.
- Tissue filter in Statistics and Comparisons.
- Green SVG icon set for all sections.

### Fixed
- PUB layout template conflicts; PXD046034 phenodata path; duplicate legend kwarg in H3K79 box plot.

---

## v3.10 (2026-02-21)

### Added
- **7 Experiments**: All 4 datasets with ndebug_0 and ndebug_2 variants (PXD046788 x2, PXD014739 x2, PXD046034 x1, Ontogeny x2)
- **ndebug Comparison Tab** (13th tab): Compare detection efficacy between ndebug modes
  - Detection overlap (Venn + pie chart)
  - Ratio concordance scatter with Spearman correlation
  - Per-group concordance bar chart
  - Detection rate distribution + CV comparison
  - Differential detection table (top 50 features)
  - MA plot for normalized area comparison
  - Summary statistics table
- **Extended EXPERIMENTS config**: Rich dict structure with dataset, ndebug, ratios_file, singleptm_dir per experiment
- `load_experiment()` now accepts custom `ratios_file` and `singleptm_dir` parameters

### Changed
- **Header redesign**: Premium multi-stop gradient, CSS molecular dot pattern overlay, layered glow effects with blur, split title typography ("EpiProfile" + "-Plants"), gradient accent underline, gradient version badge with double shadow, emoji icons on data tags and stats ribbon
- **Footer**: Matching darker gradient, updated feature list with ndebug Compare
- Experiment dropdown widened to 440px for longer ndebug names
- Glass card panels: enhanced with inset highlight, 20px blur, 16px border-radius
- Stats ribbon: 13 Analysis Tabs, icons on all items
- DEFAULTS dict replaced by EXPERIMENTS dict (richer structure)
- ALT_RATIOS removed (superseded by EXPERIMENTS)

---

## v3.9 (2026-02-21)

### Added
- **PCA Score Plot**: Proper covariance-based 95% confidence ellipses (chi-squared, 2 dof)
- **PCA Biplot**: Publication-quality with gradient-opacity arrows, white-background labels
- **Silhouette Analysis**: Automatic optimal K selection (K=2..6), silhouette plot, scores bar chart
- **K-Means on PCA**: Cluster assignments projected onto PCA score space
- **Region Map - Sequence Coverage Map**: Histone sequence tracks showing peptide positions
- **Region Map - PTM Landscape**: Bubble chart of modification diversity per residue
- **Region Map - Sequence Context Cards**: Annotated peptide sequences with modified residues highlighted
- Scree plot extended to all computed components with cumulative line
- Ward dendrogram with group-colored tick labels

### Changed
- PCA marker styling: larger markers (12px) with white borders, 0.9 opacity
- Biplot: sample points subtle (0.35 opacity) behind arrows, dark red gradient arrows
- Region Map completely rewritten with 7 sections and histone sequence annotation
- Hierarchical clustering: colored tick labels by experimental group

### Fixed
- PCA ellipses now use eigendecomposition of covariance matrix (not axis-aligned)

---

## v3.8.2 (2026-02-21)

### Changed
- Header redesign: dark green/black gradient with elegant typography
- Title 44px with text shadow, version badge with green glow
- Matching dark footer with green accents
- Tabs: larger font (14.5px), green selected underline
- Description bar: neutral grey with darker badges

---

## v3.8.1 (2026-02-21)

### Changed
- Lighter header gradient, removed SVG logo
- Modernized upload section with glass card styling
- Wider content area (1800px max)

---

## v3.8 (2026-02-21)

### Fixed
- Biclustering x/y axis swap in phm() calls
- K-Means heatmap x/y swap (groups as columns, clusters as rows)
- SpectralBiclustering: added positive shift and log method for numerical stability
- Cluster boundary lines on biclustering heatmap

### Added
- Co-occurrence / mutual exclusivity analysis (Jaccard + log2 odds ratio)
- New experiment upload via web browser (Replace or New mode)
- Group color bar annotations on cluster heatmaps
- Adaptive axis sizing with smoother font curve (7-18pt)
- Auto-height for heatmaps based on feature count

### Changed
- Improved header/footer aesthetics with glass effects
- Adaptive bottom margins for rotated axis labels

---

## v3.7 (2026-02-21)

### Fixed
- `_db()` name collision: callback function shadowed SQLite helper, causing crash
- `n_up` bitwise AND: scalar & Series bug in statistics summary
- `_stats_export`: now respects user filter selections (source, design, FDR)
- `_enrich_stats`: added `is_log` parameter to prevent double-log FC
- `tab_log` DataFrame truthiness: `if df:` -> `if not df.empty`

### Added
- **Export to R tab**: user-defined filters, R script generation, ZIP bundle export
- Preview button for export data
- CSV/TSV format options

---

## v3.6 (2026-02-20)

### Added
- Areas as primary data source (log2 + quantile normalization)
- Data Source dropdowns in PCA, Statistics, and Comparisons tabs
- Separator bug fix: correctly parses all 3 blocks in histone_ratios.xls
- RT (retention time) extraction and storage
- QC tab: before/after quantile normalization box plots

---

## v3.5 (2026-02-20)

### Added
- SQLite database for analysis logging and session tracking
- Statistics/Comparisons: classification filters (histone, PTM type, direction)
- Spectral biclustering in PCA tab
- CSV export from statistics and comparisons
- 3-slot file upload with validation
- Adaptive font sizing for axis labels
- Analysis Log tab

---

## v3.4 (2026-02-20)

### Added
- 12 color palettes (ggsci-inspired) with dropdown selector
- Increased axis label sizes across all plots
- Enriched phenodata with additional metadata columns

---

## v3.3 (2026-02-20)

### Added
- Green plant gradient theme
- File upload support (phenodata + ratios + single PTMs)
- 5th experiment (Ontogeny RawData)
- Design filter for multi-design experiments (PXD046034)

---

## v3.2 (2026-02-19)

### Added
- Region Map tab (hDP-level aggregated heatmap)
- Phenodata tab (sample metadata viewer)
- Faceted violin plots

### Fixed
- PCA variance labeling

---

## v3.1 (2026-02-19)

### Added
- Proper hDP/hPF/hPTM data hierarchy classification
- 6 interactive filters on Peptidoforms tab
- PCA with 95% confidence ellipses, biplots, 3D
- UpSet plots for PTM co-occurrence
- Kruskal-Wallis + Mann-Whitney U statistics with BH-FDR
- Editable DataTables with CSV export
- 4 pre-configured experiments
- Recursive file finder for nested directories
