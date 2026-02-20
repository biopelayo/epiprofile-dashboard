# EpiProfile-Plants Dashboard

Interactive visualization dashboard for **EpiProfile-Plants** histone PTM quantification output. Built with Dash and Plotly.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Dash](https://img.shields.io/badge/Dash-4.0+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Data Hierarchy

The dashboard correctly classifies EpiProfile output into three biological levels:

| Level | Name | Description | Source |
|-------|------|-------------|--------|
| **hDP** | Derivatized Peptide | Peptide region headers (e.g., `TKQTAR(H3_3_8)`) | `histone_ratios.xls` headers |
| **hPF** | Peptidoform | Combinatorial modifications on a peptide (e.g., `H3_9_17 K9me2K14ac`) | `histone_ratios.xls` data rows |
| **hPTM** | Individual PTM | Single modification marks (e.g., `H3K4me1`) | `histone_ratios_single_PTMs.xls` |
| **SeqVar** | Sequence Variant | Amino acid variants per region (e.g., `TKQTAR` vs `TKQSAR`) | `histone_ratios.xls` variant block |

## Features

| Tab | Description |
|-----|-------------|
| **Peptidoforms (hPF)** | Full hPF ratio heatmap with working filters (histone type, region, group, modification type, min threshold, top N by variance), top variable peptidoforms, group distribution violin plots, editable DataTable |
| **Single PTMs (hPTM)** | Clustered heatmap, Z-score normalization, violin plots, grouped bars, editable DataTable |
| **QC Dashboard** | Missingness heatmap, peptide completeness, area distributions, noise analysis, summary stat cards |
| **PCA & Clustering** | 2D PCA with 95% confidence ellipses, scree plot, PCA biplot with top 15 loading arrows, 3D PCA, hierarchical dendrogram, Spearman correlation heatmap |
| **Statistics** | Kruskal-Wallis test with Benjamini-Hochberg FDR correction, volcano plot, significant features bar chart |
| **UpSet / Co-occurrence** | PTM co-occurrence from combinatorial peptidoforms, detection patterns across groups, modification complexity analysis |
| **Comparisons** | Mann-Whitney U test + FDR, volcano, fold-change bars, MA plots; data level selector (hPTM vs hPF) |
| **Sample Browser** | PDF chromatogram viewer, per-sample PTM profile bar charts |

### v3.1 Highlights

- **Proper hDP/hPF/hPTM hierarchy** — correctly classifies peptide regions, peptidoforms, individual marks, and sequence variants
- **Working filter callbacks** — 6 interactive filters on the Peptidoforms tab that actually filter data
- **Publication PCA** — biplots with loading arrows, 95% confidence ellipses, scree plots, 3D PCA
- **UpSet plots** — PTM co-occurrence analysis from combinatorial peptidoforms (pure Plotly)
- **Robust statistics** — Kruskal-Wallis + Mann-Whitney U with Benjamini-Hochberg FDR correction
- **Clean white professional theme** with publication-quality styling
- **4 experiment support** out of the box (PXD046788, PXD014739, PXD046034, Ontogeny)
- **Editable DataTables** — sort, filter, edit cells, delete rows/columns, export CSV on the fly
- **Recursive file finder** — auto-discovers data files in nested directory structures

### Supported EpiProfile Output

The dashboard auto-detects and parses:

- `histone_ratios.xls` / `.tsv` — Full ratio + area matrices (TSV format)
- `histone_ratios_single_PTMs.xls` / `.tsv` — 45 individual PTM marks
- `histone_logs.txt` — Processing log with RT warnings
- `histone_layouts/` — Per-sample folders with:
  - PDF chromatograms (XIC plots)
  - `detail/*.xls` — Area and RT data per peptide region
  - `detail/psm/identification_list.xls` — Peptide-spectrum matches
  - `detail/psm/*.plabel` — Spectrum-to-modification mappings
  - `H3_Snapshot.xls`, `H4_Snapshot.xls` — Amino acid modification maps
- `phenodata_arabidopsis_project.tsv` — Sample metadata (optional)

## Quick Start

```bash
# Clone
git clone https://github.com/biopelayo/epiprofile-dashboard.git
cd epiprofile-dashboard

# Install dependencies
pip install -r requirements.txt

# Run with your EpiProfile output directory
python epiprofile_dashboard.py /path/to/epiprofile/output

# Or multiple experiments
python epiprofile_dashboard.py /path/to/exp1 /path/to/exp2

# Custom port
python epiprofile_dashboard.py /path/to/output --port 8080
```

Then open **http://localhost:8050** in your browser.

## What is EpiProfile-Plants?

[EpiProfile](https://github.com/zfyuan/EpiProfile2.0) is a MATLAB-based tool for quantifying histone post-translational modifications (PTMs) from mass spectrometry data. The `-Plants` variant is optimized for plant histones (H3.1, H3.3 variants).

This dashboard provides interactive visualization of EpiProfile output that goes beyond the static MATLAB PDFs, enabling:

- Multi-experiment comparison with dropdown selector
- Proper hDP/hPF/hPTM data level classification
- Clustered heatmaps with hierarchical ordering
- PCA biplots with confidence ellipses for sample quality
- Kruskal-Wallis and Mann-Whitney statistical testing with FDR
- UpSet-style PTM co-occurrence analysis
- PSM-level mass accuracy quality control
- Publication-ready plots with consistent styling
- On-the-fly data editing and CSV export
- Advanced PTM filtering with multiple criteria

## Directory Structure

```
epiprofile-dashboard/
├── epiprofile_dashboard.py   # Main application (single file, ~1440 lines)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+
- ~200 MB RAM per loaded experiment
- Modern browser (Chrome, Firefox, Edge)

## License

MIT
