# EpiProfile-Plants Dashboard

Interactive visualization dashboard for **EpiProfile-Plants** histone PTM quantification output. Built with Dash and Plotly.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Dash](https://img.shields.io/badge/Dash-4.0+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

| Tab | Description |
|-----|-------------|
| **Histone Ratios** | Full PTM ratio heatmap, top variable PTMs, group distributions |
| **Single PTMs** | Clustered heatmap, Z-score normalization, PCA, violin plots, grouped bars |
| **QC Dashboard** | Missingness heatmap, peptide completeness, area distributions, noise analysis |
| **PSM Explorer** | Mass accuracy (ppm), measured vs calculated m/z, modifications, charge states, RT |
| **Sample Browser** | PDF chromatogram viewer, per-sample PTM profile bar charts |
| **Comparisons** | Log2 fold change, scatter plots, MA plots between groups |
| **Correlations** | Spearman sample/peptide correlation heatmaps, hierarchical dendrogram, H3 vs H4 |

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

- Multi-experiment comparison with a dropdown selector
- Clustered heatmaps with hierarchical ordering
- PCA for sample quality assessment
- PSM-level mass accuracy quality control
- Publication-ready plots with consistent styling

## Directory Structure

```
epiprofile-dashboard/
├── epiprofile_dashboard.py   # Main application (single file)
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
