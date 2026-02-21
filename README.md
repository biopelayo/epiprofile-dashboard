# EpiProfile-Plants Dashboard

Interactive, publication-quality visualization dashboard for **EpiProfile-Plants** histone PTM quantification output. Built with Dash 4.0 and Plotly 6.0.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-4.0+-14532d?logo=plotly&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.0+-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)
![Version](https://img.shields.io/badge/Version-3.8-0f1f13)

---

## Overview

EpiProfile-Plants Dashboard provides **12 interconnected analysis modules** for exploring histone post-translational modification (PTM) data from mass spectrometry experiments. It correctly handles the three-level hierarchical structure of EpiProfile output and provides non-parametric statistics, PCA, biclustering, co-occurrence analysis, and R-ready data export -- all from a web browser with no programming required.

> See [WHITEPAPER.md](WHITEPAPER.md) for the full technical documentation.

---

## Data Hierarchy

The dashboard classifies EpiProfile output into three biological levels:

| Level | Name | Description | Source |
|-------|------|-------------|--------|
| **hDP** | Derivatized Peptide | Peptide region headers (e.g., `TKQTAR(H3_3_8)`) | `histone_ratios.xls` headers |
| **hPF** | Peptidoform | Combinatorial modifications (e.g., `H3_9_17 K9me2K14ac`) | `histone_ratios.xls` data rows |
| **hPTM** | Individual PTM | Single marks (e.g., `H3K4me1`) | `histone_ratios_single_PTMs.xls` |
| **Areas** | MS1 Intensities | Raw peak areas, log2 + quantile normalized | `histone_ratios.xls` area block |
| **RT** | Retention Time | Chromatographic retention times (minutes) | `histone_ratios.xls` RT block |

---

## 12 Analysis Tabs

| Tab | Key Features |
|-----|-------------|
| **Peptidoforms (hPF)** | Clustered heatmap, 6 interactive filters, violin plots, editable DataTable |
| **Single PTMs (hPTM)** | Z-score heatmap, grouped bars, violin/box plots |
| **QC Dashboard** | Missingness, completeness, area distributions, before/after QN |
| **PCA & Clustering** | 2D/3D PCA, biplots, scree, dendrogram, correlation, biclustering, K-Means |
| **Statistics** | Kruskal-Wallis + BH-FDR, volcano, enrichment, data source selector |
| **UpSet / Co-occurrence** | Detection patterns, Jaccard index, log2 odds ratio, mutual exclusivity |
| **Region Map** | Mean ratios at derivatized peptide level |
| **Comparisons** | Mann-Whitney U pairwise, volcano, FC bars, MA plots |
| **Phenodata** | Sample metadata viewer |
| **Sample Browser** | PDF chromatograms, per-sample profiles |
| **Export to R** | Filtered data + R script bundle (ZIP) |
| **Analysis Log** | SQLite-backed audit trail |

---

## Quick Start

```bash
# Clone
git clone https://github.com/biopelayo/epiprofile-dashboard.git
cd epiprofile-dashboard

# Install dependencies
pip install -r requirements.txt

# Run with default experiments
python epiprofile_dashboard.py

# Or with your own EpiProfile output directory
python epiprofile_dashboard.py /path/to/epiprofile/output

# Multiple experiments
python epiprofile_dashboard.py /path/to/exp1 /path/to/exp2

# Custom port
python epiprofile_dashboard.py /path/to/output --port 8080
```

Open **http://localhost:8050** in your browser.

---

## Key Features

### Normalization Pipeline
Raw MS1 areas undergo: **zeros to NaN** (non-detects) -> **log2 transform** -> **quantile normalization** (Bolstad 2003). This corrects systematic run-to-run biases while preserving non-detected features as NaN.

### Non-parametric Statistics
- **Kruskal-Wallis H-test** for multi-group comparisons
- **Mann-Whitney U test** for pairwise comparisons
- **Benjamini-Hochberg FDR** correction
- Log-scale aware fold-change computation

### Interactive Analysis
- Switch between **ratios** (compositional) and **areas** (absolute) as data source
- 12 ggsci-inspired color palettes
- Experiment selector with live upload support
- Design filters for complex experiments
- CSV/TSV/R-bundle export with user-defined filters

### Biclustering
Spectral biclustering (Kluger 2003) simultaneously clusters features and samples, with cluster boundary visualization and group color annotations.

### Co-occurrence Analysis
Jaccard similarity + log2 odds ratio matrices reveal PTM pairs that co-occur or are mutually exclusive, with hierarchical clustering visualization.

---

## Supported Experiments

Ships with 5 pre-configured *Arabidopsis thaliana* experiments:

| Experiment | Samples | Groups | Data Available |
|-----------|---------|--------|----------------|
| PXD046788 | 58 | 5 | Ratios + Areas + RT |
| PXD014739 | 114 | 7 | Ratios only |
| PXD046034 | 48 | 8 | Ratios + Areas + RT |
| Ontogeny 1exp | 34 | 4 | Ratios only |
| Ontogeny RawData | 34 | 4 | Ratios + Areas + RT |

New experiments can be uploaded directly via the web interface (Replace or New mode).

---

## Input File Formats

EpiProfile-Plants generates `.xls` files that are actually **tab-separated** (TSV, MATLAB convention):

| File | Content |
|------|---------|
| `histone_ratios.xls` | Three blocks: Ratios, Areas, RT (separated by unnamed columns) |
| `histone_ratios_single_PTMs.xls` | 45 individual PTM marks |
| `phenodata_arabidopsis_project.tsv` | Sample metadata (Sample, Group, Design) |
| `histone_layouts/` | Per-sample PDFs, detail files, PSMs |

---

## Directory Structure

```
epiprofile-dashboard/
|-- epiprofile_dashboard.py   # Main application (~3,700 lines)
|-- WHITEPAPER.md             # Technical white paper
|-- requirements.txt          # Python dependencies
|-- .gitignore
|-- README.md
```

---

## Requirements

- **Python** 3.10+
- **Memory** ~200 MB per loaded experiment
- **Browser** Chrome, Firefox, or Edge

---

## Version History

| Version | Highlights |
|---------|-----------|
| v3.8 | Biclustering fix, co-occurrence analysis, web upload, dark theme header |
| v3.7 | Statistics bug fixes, Export to R tab |
| v3.6 | Areas as primary data source, log2+QN normalization |
| v3.5 | SQLite logging, biclustering, adaptive sizing, 3-slot upload |
| v3.4 | 12 ggsci color palettes, enriched phenodata |
| v3.3 | Green theme, upload support, 5th experiment |
| v3.2 | Region Map, Phenodata tab, faceted violins |
| v3.1 | hDP/hPF/hPTM hierarchy, PCA biplots, UpSet, statistics |

---

## Citation

If you use this dashboard in your research, please cite:

```
EpiProfile-Plants Dashboard v3.8
https://github.com/biopelayo/epiprofile-dashboard
```

---

## What is EpiProfile-Plants?

[EpiProfile](https://github.com/zfyuan/EpiProfile2.0) is a MATLAB-based tool for quantifying histone post-translational modifications from mass spectrometry data. The `-Plants` variant is optimized for plant histones (H3.1, H3.3 variants). This dashboard provides interactive visualization and statistical analysis that goes beyond the static MATLAB PDFs.

---

## License

MIT
