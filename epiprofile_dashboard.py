"""
EpiProfile-Plants Dashboard v2.0 — Publication-Quality Visualization
=====================================================================
Interactive Dash/Plotly dashboard for EpiProfile-Plants output.

Usage:
  python epiprofile_dashboard.py <dir1> [dir2] [dir3] ...

  Each argument is a path to an EpiProfile output directory containing
  histone_layouts/, histone_ratios.xls, histone_logs.txt, etc.

  If no arguments given, tries default paths (edit DEFAULTS below).

Examples:
  python epiprofile_dashboard.py /data/experiment1
  python epiprofile_dashboard.py /data/exp1 /data/exp2 --port 8080

Access:  http://localhost:8050  (or --port N)
"""

import os, re, sys, base64, math, textwrap, configparser, argparse
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from dash import Dash, html, dcc, callback, Input, Output, State, dash_table, ctx
import dash

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Default experiment paths (used when no CLI arguments are given).
# Edit these to point to your EpiProfile output directories:
DEFAULTS = {
    "PXD046788 (Arabidopsis treatments)": r"D:\epiprofile_data\PXD046788\MS1_MS2\RawData",
    "Ontogeny 1exp (Arabidopsis stages)": r"E:\EpiProfile_Proyecto\EpiProfile_20_AT\histone_layouts_ontogeny_1exp",
}

# Parse CLI arguments
parser = argparse.ArgumentParser(description="EpiProfile-Plants Dashboard v2.0")
parser.add_argument("dirs", nargs="*", help="Paths to EpiProfile output directories")
parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
args, _ = parser.parse_known_args()

if args.dirs:
    EXPERIMENTS = {}
    for p in args.dirs:
        p = os.path.abspath(p)
        name = os.path.basename(p) or p
        EXPERIMENTS[name] = p
else:
    EXPERIMENTS = {k: v for k, v in DEFAULTS.items() if os.path.isdir(v)}
    if not EXPERIMENTS:
        print("ERROR: No valid experiment directories found.")
        print("Usage:  python epiprofile_dashboard.py <dir1> [dir2] ...")
        print("Or edit the DEFAULTS dict in the script.")
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# PUBLICATION PLOT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

FONT_FAMILY = "Arial, Helvetica, sans-serif"

COLORS = {
    "bg": "#0f1117", "card": "#1a1d26", "border": "#2d3140",
    "text": "#e0e0e6", "accent": "#6ea8fe", "accent2": "#ff8a65",
    "accent3": "#66bb6a", "muted": "#8e95a4", "warn": "#ffd54f",
    "h3": "#7e57c2", "h4": "#26a69a",
}

GROUP_COLORS = px.colors.qualitative.Set2

PUB_TEMPLATE = go.layout.Template()
PUB_TEMPLATE.layout = go.Layout(
    font=dict(family=FONT_FAMILY, size=12, color=COLORS["text"]),
    paper_bgcolor=COLORS["card"], plot_bgcolor="#12141c",
    xaxis=dict(gridcolor="#252838", gridwidth=1, zerolinecolor="#252838",
               linecolor=COLORS["border"], linewidth=1, mirror=True,
               title_font=dict(size=13, family=FONT_FAMILY),
               tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#252838", gridwidth=1, zerolinecolor="#252838",
               linecolor=COLORS["border"], linewidth=1, mirror=True,
               title_font=dict(size=13, family=FONT_FAMILY),
               tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    margin=dict(t=40, b=50, l=60, r=20),
    colorway=GROUP_COLORS,
)

card_style = {
    "backgroundColor": COLORS["card"], "borderRadius": "10px",
    "border": f"1px solid {COLORS['border']}", "padding": "20px",
    "marginBottom": "16px", "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
}

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def try_read_tsv(path):
    """Read a file that could be TSV disguised as .xls or actual .tsv"""
    return pd.read_csv(path, sep="\t")


def load_experiment(base_dir):
    """Load all data for a given experiment directory."""
    data = {}
    data["base_dir"] = base_dir
    layouts_dir = os.path.join(base_dir, "histone_layouts")

    # Detect if layouts are in base dir itself (ontogeny) or in histone_layouts/ (PXD)
    if os.path.isdir(layouts_dir):
        data["layouts_dir"] = layouts_dir
    else:
        data["layouts_dir"] = base_dir

    ld = data["layouts_dir"]

    # ── Load single PTMs ──
    for fname in ["histone_ratios_single_PTMs.tsv", "histone_ratios_single_PTMs.xls"]:
        fpath = os.path.join(ld, fname)
        if not os.path.exists(fpath):
            fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            df = try_read_tsv(fpath)
            df.columns = ["PTM"] + [c.split(",", 1)[1] if "," in c else c for c in df.columns[1:]]
            df = df.set_index("PTM")
            df = df.apply(pd.to_numeric, errors="coerce")
            data["single_ptms"] = df
            break

    # ── Load histone_ratios.xls (PXD format with ratio+area blocks) ──
    for fname in ["histone_ratios.xls", "histone_ratios.tsv"]:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            raw = try_read_tsv(fpath)
            sep_idx = None
            for i, col in enumerate(raw.columns):
                if "Unnamed" in str(col) and i > 0:
                    sep_idx = i
                    break
            if sep_idx:
                ratios = raw.iloc[:, :sep_idx].copy()
                ratios.columns = ["PTM"] + [c.split(",", 1)[1] if "," in c else c for c in ratios.columns[1:]]
                ratios = ratios.iloc[1:].reset_index(drop=True).set_index("PTM")
                ratios = ratios.apply(pd.to_numeric, errors="coerce")
                data["ratios"] = ratios

                areas = raw.iloc[:, sep_idx + 1:].copy()
                areas.insert(0, "PTM", raw.iloc[:, 0])
                areas.columns = ["PTM"] + [re.sub(r"\.\d+$", "", c.split(",", 1)[1]) if "," in str(c) else str(c) for c in areas.columns[1:]]
                areas = areas.iloc[1:].reset_index(drop=True).set_index("PTM")
                areas = areas.apply(pd.to_numeric, errors="coerce")
                data["areas"] = areas
            break

    # ── Load phenodata if present ──
    for fname in ["phenodata_arabidopsis_project.tsv", "phenodata.tsv"]:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            fpath = os.path.join(ld, fname)
        if os.path.exists(fpath):
            data["phenodata"] = pd.read_csv(fpath, sep="\t")
            break

    # ── Build sample metadata ──
    ref_df = data.get("single_ptms", data.get("ratios"))
    if ref_df is not None:
        sample_names = list(ref_df.columns)
        data["sample_names"] = sample_names
        data["metadata"] = build_metadata(sample_names, data.get("phenodata"))

    # ── Sample folders ──
    folders = []
    for dname in sorted(os.listdir(ld)):
        full = os.path.join(ld, dname)
        if os.path.isdir(full) and re.match(r"\d+[_-]", dname):
            folders.append(dname)
    data["sample_folders"] = folders

    # ── Parse logs ──
    log_path = os.path.join(base_dir, "histone_logs.txt")
    if os.path.exists(log_path):
        data["logs"] = parse_logs(log_path)
    else:
        data["logs"] = []

    # ── Parse ALL identification lists ──
    data["all_psm"] = load_all_psm(ld, folders)

    # ── Parse all detail XLS for area/RT data ──
    data["all_detail"] = load_all_detail(ld, folders)

    return data


def build_metadata(sample_names, phenodata=None):
    """Build sample metadata merging phenodata when available."""
    records = []
    for s in sample_names:
        name = s.strip()
        treatment, tissue, rep = parse_sample_name(name)
        records.append({"Sample": s, "Treatment": treatment, "Tissue": tissue, "Replicate": rep})
    meta = pd.DataFrame(records)

    if phenodata is not None and "Sample_Name" in phenodata.columns:
        # Try to merge on sample name suffix
        pheno = phenodata.copy()
        # Extract number prefix from Sample_Name for matching
        pheno["_num"] = pheno["Sample_Name"].str.extract(r"^(\d+)-").astype(float)
        meta["_num"] = meta["Sample"].str.extract(r"^(\d+)").astype(float) if meta["Sample"].str.match(r"^\d").any() else range(len(meta))

        if "Sample_Group" in pheno.columns:
            merge_map = dict(zip(pheno["Sample_Name"], pheno["Sample_Group"]))
            # Also try matching by number
            num_map = dict(zip(pheno["_num"].dropna(), pheno.loc[pheno["_num"].notna(), "Sample_Group"]))

            def get_group(row):
                if row["Sample"] in merge_map:
                    return merge_map[row["Sample"]]
                # Try by extracting the sample part
                for k, v in merge_map.items():
                    if str(k).split("-", 1)[-1] if "-" in str(k) else str(k) in row["Sample"]:
                        return v
                if row["_num"] in num_map:
                    return num_map[row["_num"]]
                return row["Treatment"]

            meta["Group"] = meta.apply(get_group, axis=1)
            if "Batch" in pheno.columns:
                batch_map = dict(zip(pheno["_num"].dropna(), pheno.loc[pheno["_num"].notna(), "Batch"]))
                meta["Batch"] = meta["_num"].map(batch_map).fillna("A")
        meta = meta.drop(columns=["_num"], errors="ignore")

    if "Group" not in meta.columns:
        meta["Group"] = meta["Treatment"]
    if "Batch" not in meta.columns:
        meta["Batch"] = "A"

    return meta


def parse_sample_name(name):
    """Parse treatment, tissue, replicate from sample name."""
    if name.startswith("1y-CTR"):
        treatment = "1y-CTR"
        rest = name.replace("1y-CTR_", "")
    elif "-" in name and name[0].isdigit():
        # Ontogeny format: 20250506-05-2025017-1
        parts = name.split("-")
        treatment = parts[-1] if len(parts) >= 4 else name
        rest = name
    else:
        parts = name.split("_", 1)
        treatment = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

    tissue = "Root" if any(t in rest.lower() for t in ["rh", "root"]) else \
             "Shoot" if any(t in rest.lower() for t in ["sh", "shoot"]) else "Whole"
    rep_match = re.findall(r"(\d+)$", rest)
    rep = int(rep_match[0]) if rep_match else 0
    return treatment, tissue, rep


def parse_logs(path):
    """Parse histone_logs.txt."""
    with open(path, "r") as f:
        content = f.read()
    samples = []
    current_sample = None
    regions, warnings = [], []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            if current_sample:
                samples.append({"sample": current_sample, "regions": regions[:],
                                "warnings": warnings[:], "n_warnings": len(warnings)})
            current_sample = None
            regions, warnings = [], []
            continue
        if line.startswith("elapsed time"):
            continue
        if ".." not in line and "=" not in line and "hno=" not in line and "MS1" not in line and "pep_" not in line and "n rt_ref" not in line:
            current_sample = line
            continue
        if ".." in line:
            for r in line.split(".."):
                r = r.strip()
                if r:
                    regions.append(r)
            continue
        if any(k in line for k in ["rt_ref", "hno=", "MS1", "pep_", "n rt_ref"]):
            warnings.append(line)
    if current_sample:
        samples.append({"sample": current_sample, "regions": regions[:],
                        "warnings": warnings[:], "n_warnings": len(warnings)})
    return samples


def load_all_psm(layouts_dir, folders):
    """Load all identification_list.xls from PSM folders."""
    all_rows = []
    for folder in folders:
        psm_dir = os.path.join(layouts_dir, folder, "detail", "psm")
        idl_path = os.path.join(psm_dir, "identification_list.xls")
        if os.path.exists(idl_path):
            try:
                df = pd.read_csv(idl_path, sep="\t")
                df["_sample_folder"] = folder
                all_rows.append(df)
            except Exception:
                pass
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


def load_all_detail(layouts_dir, folders):
    """Load detail XLS files (area + rt sections) for all samples."""
    all_records = []
    for folder in folders:
        detail_dir = os.path.join(layouts_dir, folder, "detail")
        if not os.path.isdir(detail_dir):
            continue
        for f in os.listdir(detail_dir):
            if f.endswith(".xls") and not f.startswith("Iso_"):
                fpath = os.path.join(detail_dir, f)
                try:
                    record = parse_detail_xls(fpath)
                    if record:
                        record["_sample_folder"] = folder
                        record["_region"] = f.replace(".xls", "")
                        all_records.append(record)
                except Exception:
                    pass
    return all_records


def parse_detail_xls(path):
    """Parse a detail XLS file with [area] and [rt] sections."""
    with open(path, "r") as f:
        lines = f.readlines()
    if not lines:
        return None

    peptide_seq = lines[0].strip()
    area_rows, rt_rows = [], []
    section = None
    area_header, rt_header = None, None

    for line in lines[1:]:
        line = line.rstrip()
        if line == "[area]":
            section = "area"
            continue
        elif line == "[rt]":
            section = "rt"
            continue
        elif not line:
            continue

        if section == "area":
            parts = line.split("\t")
            if parts[0] == "peptide":
                area_header = parts
            else:
                area_rows.append(parts)
        elif section == "rt":
            parts = line.split("\t")
            if parts[0] == "peptide":
                rt_header = parts
            else:
                rt_rows.append(parts)

    # Extract total area and ratio for each modification
    mods = []
    for row in area_rows:
        mod_name = row[0]
        # Find total column (second to last pair)
        total_area = 0
        ratio = 0
        try:
            # Total is at -2, fraction at -1
            total_area = float(row[-2]) if len(row) >= 2 else 0
            ratio = float(row[-1]) if len(row) >= 1 else 0
        except (ValueError, IndexError):
            pass
        # Find RT
        rt_val = 0
        for rrow in rt_rows:
            if rrow[0] == mod_name and len(rrow) > 1:
                try:
                    rt_val = float(rrow[1])
                except ValueError:
                    pass
                break
        mods.append({"peptide": peptide_seq, "modification": mod_name,
                     "total_area": total_area, "ratio": ratio, "rt": rt_val})

    return {"peptide": peptide_seq, "modifications": mods}


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DEFAULT EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

print("Loading experiments...")
EXP_DATA = {}
for name, path in EXPERIMENTS.items():
    if os.path.isdir(path):
        print(f"  Loading: {name}...")
        EXP_DATA[name] = load_experiment(path)
        sp = EXP_DATA[name].get("single_ptms")
        if sp is not None:
            print(f"    Single PTMs: {sp.shape}")
        r = EXP_DATA[name].get("ratios")
        if r is not None:
            print(f"    Full Ratios: {r.shape}")
        print(f"    Sample folders: {len(EXP_DATA[name]['sample_folders'])}")
        psm = EXP_DATA[name].get("all_psm")
        if psm is not None and not psm.empty:
            print(f"    PSM identifications: {len(psm)}")
        print(f"    Detail records: {len(EXP_DATA[name].get('all_detail', []))}")

DEFAULT_EXP = list(EXP_DATA.keys())[0] if EXP_DATA else None

# ═══════════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════════

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "EpiProfile-Plants v2.0"

tab_style = {"color": COLORS["muted"], "backgroundColor": COLORS["card"],
             "borderColor": COLORS["border"], "padding": "8px 16px", "fontSize": "13px"}
tab_selected = {"color": COLORS["accent"], "backgroundColor": COLORS["bg"],
                "borderTop": f"2px solid {COLORS['accent']}", "padding": "8px 16px", "fontSize": "13px"}

app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "fontFamily": FONT_FAMILY,
           "color": COLORS["text"]},
    children=[
        # ── HEADER ──
        html.Div(style={"backgroundColor": COLORS["card"], "borderBottom": f"1px solid {COLORS['border']}",
                         "padding": "12px 32px", "display": "flex", "alignItems": "center", "gap": "20px",
                         "flexWrap": "wrap"},
                 children=[
                     html.Div([
                         html.H1("EpiProfile-Plants", style={"margin": "0", "fontSize": "22px",
                                                              "color": COLORS["accent"], "letterSpacing": "1px"}),
                         html.Span("Interactive PTM Dashboard v2.0", style={"color": COLORS["muted"], "fontSize": "11px"}),
                     ]),
                     html.Div(style={"flex": "1"}),
                     html.Div([
                         html.Label("Experiment:", style={"color": COLORS["muted"], "fontSize": "11px", "marginRight": "8px"}),
                         dcc.Dropdown(id="exp-selector",
                                      options=[{"label": k, "value": k} for k in EXP_DATA.keys()],
                                      value=DEFAULT_EXP, clearable=False,
                                      style={"width": "340px", "backgroundColor": COLORS["bg"], "fontSize": "12px"}),
                     ], style={"display": "flex", "alignItems": "center"}),
                 ]),

        # ── TABS ──
        dcc.Tabs(id="main-tabs", value="tab-ratios",
                 style={"backgroundColor": COLORS["card"]},
                 colors={"border": COLORS["border"], "primary": COLORS["accent"], "background": COLORS["card"]},
                 children=[
                     dcc.Tab(label="Histone Ratios", value="tab-ratios", style=tab_style, selected_style=tab_selected),
                     dcc.Tab(label="Single PTMs", value="tab-single", style=tab_style, selected_style=tab_selected),
                     dcc.Tab(label="QC Dashboard", value="tab-qc", style=tab_style, selected_style=tab_selected),
                     dcc.Tab(label="PSM Explorer", value="tab-psm", style=tab_style, selected_style=tab_selected),
                     dcc.Tab(label="Sample Browser", value="tab-browser", style=tab_style, selected_style=tab_selected),
                     dcc.Tab(label="Comparisons", value="tab-compare", style=tab_style, selected_style=tab_selected),
                     dcc.Tab(label="Correlations", value="tab-corr", style=tab_style, selected_style=tab_selected),
                 ]),

        html.Div(id="tab-content", style={"padding": "20px 32px"}),

        # ── Hidden store for current experiment data ──
        dcc.Store(id="current-exp", data=DEFAULT_EXP),
    ]
)


# Sync experiment selector with store
@callback(Output("current-exp", "data"), Input("exp-selector", "value"))
def update_exp_store(exp):
    return exp


# ── Tab router ──
@callback(Output("tab-content", "children"), Input("main-tabs", "value"), Input("current-exp", "data"))
def render_tab(tab, exp):
    if not exp or exp not in EXP_DATA:
        return html.Div("No experiment loaded", style={"color": COLORS["accent2"], "textAlign": "center", "padding": "60px"})
    d = EXP_DATA[exp]
    if tab == "tab-ratios":
        return build_ratios_tab(d)
    elif tab == "tab-single":
        return build_single_tab(d)
    elif tab == "tab-qc":
        return build_qc_tab(d)
    elif tab == "tab-psm":
        return build_psm_tab(d)
    elif tab == "tab-browser":
        return build_browser_tab(d)
    elif tab == "tab-compare":
        return build_compare_tab(d)
    elif tab == "tab-corr":
        return build_corr_tab(d)
    return html.Div("Select a tab")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: publication-quality figure defaults
# ═══════════════════════════════════════════════════════════════════════════

def pub_fig(fig, height=500):
    fig.update_layout(template=PUB_TEMPLATE, height=height)
    return fig


def pub_heatmap(z, x, y, colorscale="Viridis", title="", zmin=None, zmax=None, height=600):
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y, colorscale=colorscale,
        colorbar=dict(thickness=15, len=0.9, title=dict(text=title, side="right", font=dict(size=10))),
        hoverongaps=False, zmin=zmin, zmax=zmax,
    ))
    fig.update_layout(
        template=PUB_TEMPLATE, height=height,
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        margin=dict(l=180, b=120, t=30, r=30),
    )
    return fig


def cluster_order(df, axis=0, method="ward"):
    """Return reordered index using hierarchical clustering."""
    try:
        data = df.fillna(0).values if axis == 0 else df.fillna(0).values.T
        if data.shape[0] < 3:
            return list(df.index) if axis == 0 else list(df.columns)
        dist = pdist(data, metric="euclidean")
        link = linkage(dist, method=method)
        order = leaves_list(link)
        src = list(df.index) if axis == 0 else list(df.columns)
        return [src[i] for i in order]
    except Exception:
        return list(df.index) if axis == 0 else list(df.columns)


def get_group_colors(groups):
    """Consistent color mapping for groups."""
    unique = sorted(set(groups))
    cmap = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(unique)}
    return cmap


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: HISTONE RATIOS
# ═══════════════════════════════════════════════════════════════════════════

def build_ratios_tab(d):
    has_ratios = "ratios" in d
    if not has_ratios:
        return html.Div(style=card_style, children=[
            html.H3("Histone Ratios", style={"color": COLORS["accent"]}),
            html.P("No histone_ratios.xls found for this experiment. Use the Single PTMs tab instead.",
                   style={"color": COLORS["muted"]}),
        ])

    ratios = d["ratios"]
    meta = d["metadata"]
    groups = sorted(meta["Group"].unique())

    # Filter PTMs (remove peptide header rows)
    ptm_idx = [i for i in ratios.index if not (i.endswith(")") and "(" in i)]
    df = ratios.loc[ptm_idx].dropna(how="all")
    df = df[(df != 0).any(axis=1)]

    # Order columns by group
    meta_sorted = meta.sort_values(["Group", "Tissue", "Replicate"])
    col_order = [s for s in meta_sorted["Sample"] if s in df.columns]
    df = df[col_order]

    # ── Heatmap ──
    heatmap = pub_heatmap(df.values, df.columns.tolist(), df.index.tolist(),
                          colorscale="Viridis", title="Ratio", height=max(500, len(df) * 7))

    # ── Top variable PTMs ──
    var_s = df.var(axis=1).dropna().sort_values(ascending=False).head(25)
    var_fig = go.Figure(go.Bar(x=var_s.values, y=var_s.index.tolist(), orientation="h",
                               marker=dict(color=var_s.values, colorscale="Plasma")))
    pub_fig(var_fig, 450)
    var_fig.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
                          margin=dict(l=200), xaxis_title="Variance")

    # ── Box per group for top PTM ──
    top_ptm = var_s.index[0]
    melt = df.loc[[top_ptm]].T.reset_index()
    melt.columns = ["Sample", "Ratio"]
    melt = melt.merge(meta, on="Sample")
    box_fig = px.box(melt, x="Group", y="Ratio", color="Tissue", points="all",
                     title=f"{top_ptm}", color_discrete_map={"Root": COLORS["accent2"],
                                                              "Shoot": COLORS["accent3"], "Whole": COLORS["accent"]})
    pub_fig(box_fig, 380)

    # ── Stacked bar of peptide regions ──
    # Group PTMs by H3/H4 region
    region_counts = {}
    for idx in df.index:
        prefix = idx.split(" ")[0] if " " in idx else idx
        histone = "H3" if prefix.startswith("H3") else "H4" if prefix.startswith("H4") else "Other"
        region_counts[histone] = region_counts.get(histone, 0) + 1
    pie_fig = px.pie(values=list(region_counts.values()), names=list(region_counts.keys()),
                     title="PTMs by Histone", color_discrete_sequence=[COLORS["h3"], COLORS["h4"], COLORS["muted"]])
    pub_fig(pie_fig, 300)

    return html.Div([
        html.Div(style=card_style, children=[
            html.H3("Full Histone PTM Ratios", style={"color": COLORS["accent"], "marginTop": "0", "fontSize": "16px"}),
            html.P(f"{df.shape[0]} modifications × {df.shape[1]} samples", style={"color": COLORS["muted"], "fontSize": "12px"}),
            dcc.Graph(figure=heatmap),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "2"}, children=[
                html.H3("Top 25 Most Variable PTMs", style={"color": COLORS["accent"], "marginTop": "0", "fontSize": "15px"}),
                dcc.Graph(figure=var_fig),
            ]),
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.H3(f"Distribution: {top_ptm}", style={"color": COLORS["accent"], "marginTop": "0", "fontSize": "15px"}),
                dcc.Graph(figure=box_fig),
                dcc.Graph(figure=pie_fig),
            ]),
        ]),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: SINGLE PTMs
# ═══════════════════════════════════════════════════════════════════════════

def build_single_tab(d):
    if "single_ptms" not in d:
        return html.Div(style=card_style, children=[html.P("No single PTM data found.")])

    df = d["single_ptms"].copy()
    meta = d["metadata"]
    groups = sorted(meta["Group"].unique())
    gcmap = get_group_colors(groups)

    # Order columns
    meta_sorted = meta.sort_values(["Group", "Tissue", "Replicate"])
    col_order = [s for s in meta_sorted["Sample"] if s in df.columns]
    df = df[col_order]

    # ── Clustered heatmap ──
    row_order = cluster_order(df, axis=0)
    df_clust = df.loc[row_order]

    hm_fig = pub_heatmap(df_clust.values, df_clust.columns.tolist(), df_clust.index.tolist(),
                         colorscale="RdBu_r", title="Ratio", height=max(500, len(df) * 11))
    # Add group annotation bars
    group_labels = [meta.set_index("Sample").loc[s, "Group"] if s in meta["Sample"].values else "?" for s in df_clust.columns]

    # ── Z-score heatmap ──
    zscored = df_clust.apply(lambda row: (row - row.mean()) / (row.std() + 1e-10), axis=1)
    zscore_hm = pub_heatmap(zscored.values, zscored.columns.tolist(), zscored.index.tolist(),
                            colorscale="RdBu_r", title="Z-score", zmin=-3, zmax=3,
                            height=max(500, len(df) * 11))

    # ── PCA ──
    from sklearn.decomposition import PCA as skPCA
    pca_data = df.T.fillna(0)
    try:
        pca = skPCA(n_components=2)
        coords = pca.fit_transform(pca_data.values)
        pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Sample": pca_data.index})
        pca_df = pca_df.merge(meta, on="Sample")
        pca_fig = px.scatter(pca_df, x="PC1", y="PC2", color="Group", hover_name="Sample",
                             symbol="Batch" if "Batch" in pca_df.columns else None,
                             title="PCA — Single PTMs")
        pca_fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        pub_fig(pca_fig, 420)
        pca_fig.update_layout(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    except Exception:
        pca_fig = go.Figure()
        pub_fig(pca_fig, 420)

    # ── Violin / Box per group for selected marks ──
    key_marks = [m for m in ["H3K9me2", "H3K14ac", "H3K27me3", "H3K4me1", "H4K16ac", "H3K9ac"]
                 if m in df.index][:6]
    if key_marks:
        melt_list = []
        for m in key_marks:
            vals = df.loc[m]
            tmp = pd.DataFrame({"Sample": vals.index, "Ratio": vals.values, "PTM": m})
            tmp = tmp.merge(meta, on="Sample")
            melt_list.append(tmp)
        melt_all = pd.concat(melt_list)
        violin_fig = px.violin(melt_all, x="PTM", y="Ratio", color="Group", box=True, points="all")
        pub_fig(violin_fig, 420)
        violin_fig.update_layout(xaxis_title="", yaxis_title="Ratio")
    else:
        violin_fig = go.Figure()
        pub_fig(violin_fig, 420)

    # ── Grouped bar: mean ± SD ──
    bar_data = []
    for ptm in df.index:
        for grp in groups:
            samples = meta[meta["Group"] == grp]["Sample"].tolist()
            vals = df.loc[ptm, [s for s in samples if s in df.columns]].dropna()
            bar_data.append({"PTM": ptm, "Group": grp, "Mean": vals.mean(), "SD": vals.std()})
    bar_df = pd.DataFrame(bar_data)
    # Top 15 by max mean
    top_ptms = bar_df.groupby("PTM")["Mean"].max().sort_values(ascending=False).head(15).index
    bar_df_top = bar_df[bar_df["PTM"].isin(top_ptms)]
    bar_fig = px.bar(bar_df_top, x="PTM", y="Mean", color="Group", barmode="group", error_y="SD")
    pub_fig(bar_fig, 420)
    bar_fig.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=9)), yaxis_title="Mean Ratio")

    return html.Div([
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.H3("Clustered Heatmap — Ratios", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
                dcc.Graph(figure=hm_fig),
            ]),
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.H3("Z-score Heatmap", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
                dcc.Graph(figure=zscore_hm),
            ]),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.H3("PCA — Sample Clustering", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
                dcc.Graph(figure=pca_fig),
            ]),
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.H3("Key PTM Distributions", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
                dcc.Graph(figure=violin_fig),
            ]),
        ]),
        html.Div(style=card_style, children=[
            html.H3("Top 15 PTMs — Group Means ± SD", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
            dcc.Graph(figure=bar_fig),
        ]),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: QC DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def build_qc_tab(d):
    meta = d.get("metadata", pd.DataFrame())
    df = d.get("single_ptms", d.get("ratios"))
    if df is None:
        return html.Div(style=card_style, children=[html.P("No data for QC.")])

    # ── Missingness heatmap ──
    binary = (~df.isna() & (df != 0)).astype(int)
    miss_hm = pub_heatmap(binary.values, binary.columns.tolist(), binary.index.tolist(),
                          colorscale=[[0, "#1a1d26"], [1, COLORS["accent3"]]],
                          title="Detected", height=max(400, len(binary) * 9))
    miss_hm.update_layout(title=dict(text="Peptide Detection (1=Detected, 0=Missing)", font=dict(size=13)))

    # ── Missing count per sample ──
    miss_count = (df.isna() | (df == 0)).sum(axis=0)
    miss_bar_df = pd.DataFrame({"Sample": miss_count.index, "Missing": miss_count.values})
    miss_bar_df = miss_bar_df.merge(meta, on="Sample", how="left")
    miss_bar = px.bar(miss_bar_df, x="Sample", y="Missing", color="Group",
                      title="Missing Peptides per Sample")
    pub_fig(miss_bar, 350)
    miss_bar.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=8)))

    # ── Peptide completeness histogram ──
    detected_per_ptm = binary.sum(axis=1)
    comp_hist = px.histogram(x=detected_per_ptm.values, nbins=20,
                             labels={"x": "# Samples Detected", "y": "# Peptides"},
                             title="Peptide Completeness")
    pub_fig(comp_hist, 300)

    # ── Area distribution (if available) ──
    areas = d.get("areas")
    area_box = go.Figure()
    if areas is not None:
        # Log10 area per sample
        log_areas = np.log10(areas.replace(0, np.nan))
        melt_area = log_areas.stack().reset_index()
        melt_area.columns = ["PTM", "Sample", "Log10Area"]
        melt_area = melt_area.merge(meta, on="Sample", how="left")
        area_box = px.box(melt_area, x="Sample", y="Log10Area", color="Group",
                          title="Log₁₀(Area) Distribution per Sample")
        pub_fig(area_box, 380)
        area_box.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=7)), showlegend=False)

    # ── Low ratio percentage ──
    low_pct = ((df < 0.01) & (df > 0)).sum(axis=0) / (df > 0).sum(axis=0) * 100
    low_df = pd.DataFrame({"Sample": low_pct.index, "LowRatio%": low_pct.values})
    low_df = low_df.merge(meta, on="Sample", how="left")
    low_bar = px.bar(low_df, x="Sample", y="LowRatio%", color="Group",
                     title="% Peptides with Ratio < 1% (Noise)")
    pub_fig(low_bar, 300)
    low_bar.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=7)))

    # ── Log warnings ──
    logs = d.get("logs", [])
    n_clean = sum(1 for e in logs if e["n_warnings"] == 0)
    n_warn = sum(1 for e in logs if e["n_warnings"] > 0)
    total_w = sum(e["n_warnings"] for e in logs)

    # ── Summary cards ──
    n_samples = len(meta) if not meta.empty else df.shape[1]
    n_ptms = df.shape[0]
    total_detected = int(binary.sum().sum())
    total_possible = binary.shape[0] * binary.shape[1]
    completeness = total_detected / total_possible * 100 if total_possible > 0 else 0

    summary_cards = html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "16px", "flexWrap": "wrap"}, children=[
        _stat_card("Samples", str(n_samples), COLORS["accent"]),
        _stat_card("PTMs", str(n_ptms), COLORS["accent"]),
        _stat_card("Completeness", f"{completeness:.1f}%", COLORS["accent3"]),
        _stat_card("Clean Runs", str(n_clean), COLORS["accent3"]),
        _stat_card("Warnings", str(n_warn), COLORS["accent2"] if n_warn > 0 else COLORS["accent3"]),
        _stat_card("Total Warn Lines", str(total_w), COLORS["warn"] if total_w > 0 else COLORS["accent3"]),
    ])

    return html.Div([
        summary_cards,
        html.Div(style=card_style, children=[
            html.H3("Missingness Heatmap", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
            dcc.Graph(figure=miss_hm),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.H3("Missing Peptides per Sample", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
                dcc.Graph(figure=miss_bar),
            ]),
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.H3("Peptide Completeness", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
                dcc.Graph(figure=comp_hist),
            ]),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.H3("Area Distribution", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
                dcc.Graph(figure=area_box),
            ]),
            html.Div(style={**card_style, "flex": "1"}, children=[
                html.H3("Noise Ratio (%)", style={"color": COLORS["accent"], "marginTop": 0, "fontSize": "15px"}),
                dcc.Graph(figure=low_bar),
            ]),
        ]),
    ])


def _stat_card(label, value, color):
    return html.Div(style={**card_style, "flex": "1", "minWidth": "120px", "textAlign": "center",
                           "padding": "12px"}, children=[
        html.H2(value, style={"color": color, "margin": "0", "fontSize": "28px"}),
        html.P(label, style={"color": COLORS["muted"], "margin": "0", "fontSize": "11px"}),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: PSM EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

def build_psm_tab(d):
    psm = d.get("all_psm", pd.DataFrame())
    if psm.empty:
        return html.Div(style=card_style, children=[
            html.H3("PSM Explorer", style={"color": COLORS["accent"]}),
            html.P("No identification_list.xls files found in PSM folders.", style={"color": COLORS["muted"]}),
        ])

    # Standardize columns
    cols = psm.columns.tolist()
    col_map = {}
    for c in cols:
        cl = c.lower().strip()
        if "measured" in cl and "m/z" in cl:
            col_map[c] = "measured_mz"
        elif "calculated" in cl and "m/z" in cl:
            col_map[c] = "calculated_mz"
        elif cl == "charge":
            col_map[c] = "charge"
        elif cl == "ppm":
            col_map[c] = "ppm"
        elif cl == "sequence":
            col_map[c] = "sequence"
        elif "modification1" in cl:
            col_map[c] = "modification"
        elif "histone" in cl:
            col_map[c] = "histone_type"
        elif "retention" in cl:
            col_map[c] = "rt"
        elif cl == "filename":
            col_map[c] = "filename"
    psm = psm.rename(columns=col_map)

    n_spectra = len(psm)
    n_unique_pep = psm["sequence"].nunique() if "sequence" in psm.columns else 0
    n_samples = psm["_sample_folder"].nunique()

    # ── Mass accuracy histogram ──
    if "ppm" in psm.columns:
        ppm_vals = pd.to_numeric(psm["ppm"], errors="coerce").dropna()
        ppm_fig = px.histogram(ppm_vals, nbins=100, title="Mass Accuracy Distribution",
                               labels={"value": "Mass Error (ppm)", "count": "Count"},
                               color_discrete_sequence=[COLORS["accent"]])
        pub_fig(ppm_fig, 350)
        ppm_fig.add_vline(x=0, line_dash="dash", line_color=COLORS["accent2"])
        ppm_fig.add_vline(x=ppm_vals.median(), line_dash="dot", line_color=COLORS["accent3"],
                          annotation_text=f"Median: {ppm_vals.median():.2f} ppm")
    else:
        ppm_fig = go.Figure()
        pub_fig(ppm_fig, 350)

    # ── PSMs per sample ──
    psm_per_sample = psm.groupby("_sample_folder").size().reset_index(name="PSMs")
    psm_per_sample = psm_per_sample.sort_values("PSMs", ascending=False)
    psm_bar = px.bar(psm_per_sample, x="_sample_folder", y="PSMs", title="PSMs per Sample",
                     color_discrete_sequence=[COLORS["accent"]])
    pub_fig(psm_bar, 350)
    psm_bar.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=7)), xaxis_title="Sample")

    # ── Modification distribution ──
    if "modification" in psm.columns:
        mod_counts = psm["modification"].value_counts().head(20)
        mod_bar = px.bar(x=mod_counts.index, y=mod_counts.values, title="Top 20 Modifications Identified",
                         labels={"x": "Modification", "y": "Count"},
                         color_discrete_sequence=[COLORS["h3"]])
        pub_fig(mod_bar, 350)
        mod_bar.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=9)))
    else:
        mod_bar = go.Figure()
        pub_fig(mod_bar, 350)

    # ── Peptide frequency ──
    if "sequence" in psm.columns:
        pep_counts = psm["sequence"].value_counts().head(15)
        pep_bar = px.bar(x=pep_counts.index, y=pep_counts.values, title="Most Identified Peptides",
                         labels={"x": "Peptide", "y": "# PSMs"},
                         color_discrete_sequence=[COLORS["h4"]])
        pub_fig(pep_bar, 320)
        pep_bar.update_layout(xaxis=dict(tickangle=45))
    else:
        pep_bar = go.Figure()
        pub_fig(pep_bar, 320)

    # ── RT distribution ──
    if "rt" in psm.columns:
        rt_vals = pd.to_numeric(psm["rt"], errors="coerce").dropna()
        rt_fig = px.histogram(rt_vals, nbins=80, title="Retention Time Distribution",
                              labels={"value": "RT (min)", "count": "Count"},
                              color_discrete_sequence=[COLORS["accent3"]])
        pub_fig(rt_fig, 320)
    else:
        rt_fig = go.Figure()
        pub_fig(rt_fig, 320)

    # ── Charge state distribution ──
    if "charge" in psm.columns:
        charge_counts = psm["charge"].value_counts().sort_index()
        charge_fig = px.pie(values=charge_counts.values, names=[f"+{c}" for c in charge_counts.index],
                            title="Charge State Distribution", color_discrete_sequence=GROUP_COLORS)
        pub_fig(charge_fig, 300)
    else:
        charge_fig = go.Figure()
        pub_fig(charge_fig, 300)

    # ── Scatter: measured vs calculated m/z ──
    if "measured_mz" in psm.columns and "calculated_mz" in psm.columns:
        scatter_df = psm[["measured_mz", "calculated_mz"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(scatter_df) > 5000:
            scatter_df = scatter_df.sample(5000, random_state=42)
        mz_scatter = px.scatter(scatter_df, x="calculated_mz", y="measured_mz",
                                title="Measured vs Calculated m/z", opacity=0.4,
                                color_discrete_sequence=[COLORS["accent"]])
        pub_fig(mz_scatter, 350)
        max_mz = max(scatter_df["calculated_mz"].max(), scatter_df["measured_mz"].max())
        min_mz = min(scatter_df["calculated_mz"].min(), scatter_df["measured_mz"].min())
        mz_scatter.add_shape(type="line", x0=min_mz, y0=min_mz, x1=max_mz, y1=max_mz,
                             line=dict(color=COLORS["accent2"], dash="dash", width=1))
    else:
        mz_scatter = go.Figure()
        pub_fig(mz_scatter, 350)

    return html.Div([
        # Summary
        html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "16px"}, children=[
            _stat_card("Total PSMs", f"{n_spectra:,}", COLORS["accent"]),
            _stat_card("Unique Peptides", str(n_unique_pep), COLORS["h3"]),
            _stat_card("Samples", str(n_samples), COLORS["accent3"]),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=ppm_fig)]),
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=mz_scatter)]),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=psm_bar)]),
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=mod_bar)]),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=pep_bar)]),
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=rt_fig)]),
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=charge_fig)]),
        ]),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: SAMPLE BROWSER
# ═══════════════════════════════════════════════════════════════════════════

def build_browser_tab(d):
    folders = d.get("sample_folders", [])
    if not folders:
        return html.Div(style=card_style, children=[html.P("No sample folders found.")])

    return html.Div([
        html.Div(style={**card_style, "display": "flex", "gap": "16px"}, children=[
            html.Div(style={"flex": "1"}, children=[
                html.Label("Sample:", style={"color": COLORS["muted"], "fontSize": "11px"}),
                dcc.Dropdown(id="br-sample", options=[{"label": f, "value": f} for f in folders],
                             value=folders[0], clearable=False, style={"backgroundColor": COLORS["bg"]}),
            ]),
            html.Div(style={"flex": "1"}, children=[
                html.Label("PDF:", style={"color": COLORS["muted"], "fontSize": "11px"}),
                dcc.Dropdown(id="br-pdf", style={"backgroundColor": COLORS["bg"]}),
            ]),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "2"}, id="br-pdf-view"),
            html.Div(style={**card_style, "flex": "1"}, id="br-info"),
        ]),
    ])


@callback(Output("br-pdf", "options"), Output("br-pdf", "value"),
          Input("br-sample", "value"), Input("current-exp", "data"))
def update_br_pdfs(folder, exp):
    if not folder or not exp or exp not in EXP_DATA:
        return [], None
    ld = EXP_DATA[exp]["layouts_dir"]
    path = os.path.join(ld, folder)
    pdfs = sorted([f for f in os.listdir(path) if f.endswith(".pdf")]) if os.path.isdir(path) else []
    opts = [{"label": p, "value": p} for p in pdfs]
    return opts, pdfs[0] if pdfs else None


@callback(Output("br-pdf-view", "children"),
          Input("br-sample", "value"), Input("br-pdf", "value"), Input("current-exp", "data"))
def update_br_pdf_view(folder, pdf_name, exp):
    if not folder or not pdf_name or not exp or exp not in EXP_DATA:
        return html.P("Select a sample and PDF", style={"color": COLORS["muted"]})
    ld = EXP_DATA[exp]["layouts_dir"]
    pdf_path = os.path.join(ld, folder, pdf_name)
    if not os.path.exists(pdf_path):
        return html.P("PDF not found", style={"color": COLORS["accent2"]})
    with open(pdf_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return html.Div([
        html.H3(pdf_name.replace(".pdf", ""), style={"color": COLORS["accent"], "fontSize": "15px", "marginTop": 0}),
        html.Iframe(src=f"data:application/pdf;base64,{encoded}",
                    style={"width": "100%", "height": "550px", "border": "none", "borderRadius": "6px"}),
    ])


@callback(Output("br-info", "children"),
          Input("br-sample", "value"), Input("current-exp", "data"))
def update_br_info(folder, exp):
    if not folder or not exp or exp not in EXP_DATA:
        return html.P("Select a sample")
    d = EXP_DATA[exp]
    sample_name = "_".join(folder.split("_")[1:]) if "_" in folder else folder

    # Find matching sample in ratios/single_ptms
    ref_df = d.get("single_ptms", d.get("ratios"))
    info_items = []

    if ref_df is not None:
        # Try exact and partial match
        matching_col = None
        for col in ref_df.columns:
            if sample_name in col or col in sample_name:
                matching_col = col
                break
        if matching_col:
            vals = ref_df[matching_col].dropna()
            vals = vals[vals != 0]
            # Show as horizontal bar chart
            if len(vals) > 0:
                fig = go.Figure(go.Bar(x=vals.values, y=vals.index.tolist(), orientation="h",
                                       marker=dict(color=vals.values, colorscale="Viridis")))
                pub_fig(fig, max(300, len(vals) * 16))
                fig.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
                                  margin=dict(l=120, t=10), xaxis_title="Ratio")
                info_items.append(dcc.Graph(figure=fig))
            else:
                info_items.append(html.P("No non-zero ratios", style={"color": COLORS["muted"]}))
        else:
            info_items.append(html.P(f"Sample '{sample_name}' not matched", style={"color": COLORS["accent2"]}))

    return html.Div([
        html.H3("PTM Profile", style={"color": COLORS["accent"], "fontSize": "15px", "marginTop": 0}),
        *info_items,
    ])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════

def build_compare_tab(d):
    meta = d.get("metadata", pd.DataFrame())
    if meta.empty:
        return html.Div(style=card_style, children=[html.P("No metadata.")])
    groups = sorted(meta["Group"].unique())
    df = d.get("single_ptms", d.get("ratios"))
    if df is None:
        return html.Div(style=card_style, children=[html.P("No ratio data.")])

    if len(groups) < 2:
        return html.Div(style=card_style, children=[html.P("Need at least 2 groups for comparison.")])

    # Pre-compute all pairwise comparisons
    epsilon = 1e-6
    pair_figs = []

    g1, g2 = groups[0], groups[1] if len(groups) > 1 else groups[0]

    return html.Div([
        html.Div(style={**card_style, "display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            html.Div(style={"flex": "1"}, children=[
                html.Label("Group A:", style={"color": COLORS["muted"], "fontSize": "11px"}),
                dcc.Dropdown(id="cmp-a", options=[{"label": g, "value": g} for g in groups],
                             value=g1, clearable=False, style={"backgroundColor": COLORS["bg"]}),
            ]),
            html.Div(style={"flex": "1"}, children=[
                html.Label("Group B:", style={"color": COLORS["muted"], "fontSize": "11px"}),
                dcc.Dropdown(id="cmp-b", options=[{"label": g, "value": g} for g in groups],
                             value=g2, clearable=False, style={"backgroundColor": COLORS["bg"]}),
            ]),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, id="cmp-fc"),
            html.Div(style={**card_style, "flex": "1"}, id="cmp-scatter"),
        ]),
        html.Div(style={**card_style}, id="cmp-ma"),
    ])


@callback(Output("cmp-fc", "children"), Output("cmp-scatter", "children"), Output("cmp-ma", "children"),
          Input("cmp-a", "value"), Input("cmp-b", "value"), Input("current-exp", "data"))
def update_cmp(ga, gb, exp):
    if not exp or exp not in EXP_DATA:
        return [html.P("N/A")] * 3
    d = EXP_DATA[exp]
    df = d.get("single_ptms", d.get("ratios"))
    meta = d["metadata"]
    epsilon = 1e-6

    sa = meta[meta["Group"] == ga]["Sample"].tolist()
    sb = meta[meta["Group"] == gb]["Sample"].tolist()
    ca = [c for c in df.columns if c in sa]
    cb = [c for c in df.columns if c in sb]

    if not ca or not cb:
        empty_msg = html.P("No samples for selection", style={"color": COLORS["muted"]})
        return empty_msg, empty_msg, empty_msg

    mean_a = df[ca].mean(axis=1)
    mean_b = df[cb].mean(axis=1)

    # ── Fold change ──
    fc = np.log2((mean_b + epsilon) / (mean_a + epsilon))
    fc = fc.dropna()
    fc = fc[np.isfinite(fc)].sort_values()
    colors = [COLORS["accent3"] if v > 0.5 else COLORS["accent2"] if v < -0.5 else COLORS["muted"] for v in fc.values]
    fc_fig = go.Figure(go.Bar(x=fc.values, y=fc.index.tolist(), orientation="h", marker_color=colors))
    pub_fig(fc_fig, max(400, len(fc) * 14))
    fc_fig.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
                         margin=dict(l=140), xaxis_title=f"log₂(FC) {gb}/{ga}")
    fc_fig.add_vline(x=0, line_color=COLORS["muted"], line_dash="dash")

    # ── Scatter ──
    sc_df = pd.DataFrame({"A": mean_a, "B": mean_b}).dropna()
    sc_fig = px.scatter(sc_df, x="A", y="B", text=sc_df.index, hover_name=sc_df.index,
                        labels={"A": f"Mean {ga}", "B": f"Mean {gb}"},
                        color_discrete_sequence=[COLORS["accent"]])
    sc_fig.update_traces(textposition="top center", textfont=dict(size=7), marker=dict(size=8))
    pub_fig(sc_fig, 450)
    mx = max(sc_df.max().max(), 0.01)
    sc_fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx,
                     line=dict(color=COLORS["muted"], dash="dash"))

    # ── MA plot ──
    M = np.log2((mean_b + epsilon) / (mean_a + epsilon))
    A_vals = 0.5 * (np.log2(mean_a + epsilon) + np.log2(mean_b + epsilon))
    ma_df = pd.DataFrame({"M": M, "A": A_vals, "PTM": M.index}).dropna()
    ma_df = ma_df[np.isfinite(ma_df["M"]) & np.isfinite(ma_df["A"])]
    ma_fig = px.scatter(ma_df, x="A", y="M", hover_name="PTM", text="PTM",
                        title="MA Plot (Ratio vs Intensity)",
                        labels={"A": "Average Intensity (A)", "M": f"log₂(FC) {gb}/{ga} (M)"},
                        color_discrete_sequence=[COLORS["accent"]])
    ma_fig.update_traces(textposition="top center", textfont=dict(size=7), marker=dict(size=7))
    pub_fig(ma_fig, 400)
    ma_fig.add_hline(y=0, line_color=COLORS["muted"], line_dash="dash")

    return (
        html.Div([html.H3(f"Fold Change: {gb} vs {ga}", style={"color": COLORS["accent"], "fontSize": "15px", "marginTop": 0}),
                  dcc.Graph(figure=fc_fig)]),
        html.Div([html.H3("Scatter", style={"color": COLORS["accent"], "fontSize": "15px", "marginTop": 0}),
                  dcc.Graph(figure=sc_fig)]),
        html.Div([dcc.Graph(figure=ma_fig)]),
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7: CORRELATIONS & CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

def build_corr_tab(d):
    df = d.get("single_ptms", d.get("ratios"))
    meta = d.get("metadata", pd.DataFrame())
    if df is None:
        return html.Div(style=card_style, children=[html.P("No data.")])

    # ── Sample-sample Spearman correlation ──
    corr_matrix = df.corr(method="spearman")
    # Cluster samples
    col_order = cluster_order(corr_matrix, axis=0)
    corr_matrix = corr_matrix.loc[col_order, col_order]

    corr_hm = pub_heatmap(corr_matrix.values, corr_matrix.columns.tolist(), corr_matrix.index.tolist(),
                          colorscale="RdBu_r", title="Spearman ρ", zmin=-1, zmax=1,
                          height=max(500, len(corr_matrix) * 12))
    corr_hm.update_layout(title=dict(text="Sample-Sample Correlation", font=dict(size=14)))

    # ── Peptide-peptide correlation ──
    pep_corr = df.T.corr(method="spearman")
    pep_col_order = cluster_order(pep_corr, axis=0) if len(pep_corr) < 100 else list(pep_corr.index)
    pep_corr = pep_corr.loc[pep_col_order, pep_col_order]
    pep_hm = pub_heatmap(pep_corr.values, pep_corr.columns.tolist(), pep_corr.index.tolist(),
                         colorscale="RdBu_r", title="Spearman ρ", zmin=-1, zmax=1,
                         height=max(500, len(pep_corr) * 10))
    pep_hm.update_layout(title=dict(text="Peptide-Peptide Correlation", font=dict(size=14)))

    # ── Dendrogram (text-based via plotly) ──
    try:
        data_for_dend = df.T.fillna(0).values
        dist = pdist(data_for_dend, metric="euclidean")
        link = linkage(dist, method="ward")
        from scipy.cluster.hierarchy import dendrogram as scipy_dend
        dend_result = scipy_dend(link, labels=df.columns.tolist(), no_plot=True)

        # Build plotly dendrogram
        dend_fig = go.Figure()
        for i, (x_coords, y_coords) in enumerate(zip(dend_result["icoord"], dend_result["dcoord"])):
            dend_fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode="lines",
                                          line=dict(color=COLORS["accent"], width=1.5),
                                          showlegend=False))
        # Add sample labels
        tick_positions = [5 + 10 * i for i in range(len(dend_result["ivl"]))]
        dend_fig.update_layout(
            template=PUB_TEMPLATE, height=350,
            xaxis=dict(tickmode="array", tickvals=tick_positions, ticktext=dend_result["ivl"],
                       tickangle=45, tickfont=dict(size=7)),
            yaxis_title="Distance (Ward)",
            title=dict(text="Hierarchical Clustering Dendrogram", font=dict(size=14)),
            margin=dict(b=120),
        )
    except Exception:
        dend_fig = go.Figure()
        pub_fig(dend_fig, 350)

    # ── H3 vs H4 scatter ──
    h3_ptms = [p for p in df.index if p.startswith("H3")]
    h4_ptms = [p for p in df.index if p.startswith("H4")]
    if h3_ptms and h4_ptms:
        h3_mean = df.loc[h3_ptms].mean(axis=0)
        h4_mean = df.loc[h4_ptms].mean(axis=0)
        hh_df = pd.DataFrame({"H3_mean": h3_mean, "H4_mean": h4_mean, "Sample": h3_mean.index})
        hh_df = hh_df.merge(meta, on="Sample", how="left")
        hh_scatter = px.scatter(hh_df, x="H3_mean", y="H4_mean", color="Group", hover_name="Sample",
                                title="Avg H3 vs H4 Ratio per Sample")
        hh_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        pub_fig(hh_scatter, 380)
    else:
        hh_scatter = go.Figure()
        pub_fig(hh_scatter, 380)

    return html.Div([
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=corr_hm)]),
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=pep_hm)]),
        ]),
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            html.Div(style={**card_style, "flex": "2"}, children=[dcc.Graph(figure=dend_fig)]),
            html.Div(style={**card_style, "flex": "1"}, children=[dcc.Graph(figure=hh_scatter)]),
        ]),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = args.port
    host = args.host
    print("\n" + "=" * 60)
    print("  EpiProfile-Plants Dashboard v2.0")
    print(f"  http://localhost:{port}")
    print("=" * 60 + "\n")
    app.run(debug=False, port=port, host=host)
