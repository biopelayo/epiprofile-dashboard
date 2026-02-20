"""
EpiProfile-Plants Dashboard v3.5 -- Publication-Quality Visualization
=====================================================================
Interactive Dash/Plotly dashboard for EpiProfile-Plants output.

Properly classifies histone data into three levels:
  hPTM  = individual PTM marks         (from histone_ratios_single_PTMs.xls)
  hPF   = peptidoforms (combinatorial) (from histone_ratios.xls, data rows)
  hDP   = derivatized peptide regions  (from histone_ratios.xls, headers)
  SeqVar = sequence variants per region (from histone_ratios.xls, variant block)

Features: SQLite analysis tracking, biclustering, data export, adaptive sizing,
          full upload validation, analysis logging, classification filters.

Usage:
  python epiprofile_dashboard.py <dir1> [dir2] ...
  python epiprofile_dashboard.py                    # uses DEFAULTS
Access:  http://localhost:8050
"""

import os, re, sys, base64, math, textwrap, configparser, argparse, warnings, json, time
import sqlite3
from pathlib import Path
from io import StringIO, BytesIO
from itertools import combinations
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram as scipy_dend
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from sklearn.decomposition import PCA as skPCA
from sklearn.cluster import SpectralBiclustering, KMeans
from statsmodels.stats.multitest import multipletests
from dash import Dash, html, dcc, callback, Input, Output, State, dash_table, ctx, no_update
import dash

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================
# DATABASE & LOGGING
# ======================================================================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epiprofile.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
        base_dir TEXT NOT NULL, n_samples INTEGER, n_groups INTEGER,
        n_hptm INTEGER, n_hpf INTEGER, n_regions INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_accessed TIMESTAMP);
    CREATE TABLE IF NOT EXISTS analysis_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment TEXT NOT NULL, analysis_type TEXT NOT NULL,
        parameters TEXT, n_features INTEGER, n_significant INTEGER,
        summary TEXT, duration_ms INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS saved_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER REFERENCES analysis_log(id),
        result_type TEXT NOT NULL, file_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment TEXT, filename TEXT NOT NULL, file_type TEXT,
        n_rows INTEGER, n_cols INTEGER, status TEXT, message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_activity TIMESTAMP, n_actions INTEGER DEFAULT 0,
        experiments_viewed TEXT);
    """)
    con.commit(); con.close()

init_db()

def _db():
    return sqlite3.connect(DB_PATH)

SESSION_ID = None
def start_session():
    global SESSION_ID
    con = _db(); cur = con.cursor()
    cur.execute("INSERT INTO sessions (started_at, last_activity, n_actions, experiments_viewed) VALUES (?,?,0,'[]')",
                (datetime.now().isoformat(), datetime.now().isoformat()))
    SESSION_ID = cur.lastrowid; con.commit(); con.close()

start_session()

def log_analysis(experiment, analysis_type, parameters=None, n_features=0, n_significant=0, summary="", duration_ms=0):
    try:
        con = _db(); cur = con.cursor()
        cur.execute("INSERT INTO analysis_log (experiment, analysis_type, parameters, n_features, n_significant, summary, duration_ms) VALUES (?,?,?,?,?,?,?)",
                    (experiment, analysis_type, json.dumps(parameters) if parameters else None, n_features, n_significant, summary, duration_ms))
        aid = cur.lastrowid
        if SESSION_ID:
            cur.execute("UPDATE sessions SET n_actions=n_actions+1, last_activity=? WHERE id=?",
                        (datetime.now().isoformat(), SESSION_ID))
        con.commit(); con.close(); return aid
    except Exception as e:
        print(f"  DB log error: {e}"); return None

def log_upload(experiment, filename, file_type, n_rows=0, n_cols=0, status="success", message=""):
    try:
        con = _db(); cur = con.cursor()
        cur.execute("INSERT INTO uploads (experiment, filename, file_type, n_rows, n_cols, status, message) VALUES (?,?,?,?,?,?,?)",
                    (experiment, filename, file_type, n_rows, n_cols, status, message))
        con.commit(); con.close()
    except Exception as e:
        print(f"  DB upload log error: {e}")

def log_experiment(name, base_dir, n_samples=0, n_groups=0, n_hptm=0, n_hpf=0, n_regions=0):
    try:
        con = _db(); cur = con.cursor()
        cur.execute("INSERT OR REPLACE INTO experiments (name, base_dir, n_samples, n_groups, n_hptm, n_hpf, n_regions, last_accessed) VALUES (?,?,?,?,?,?,?,?)",
                    (name, base_dir, n_samples, n_groups, n_hptm, n_hpf, n_regions, datetime.now().isoformat()))
        con.commit(); con.close()
    except Exception as e:
        print(f"  DB experiment log error: {e}")

def get_analysis_history(experiment=None, limit=100):
    con = _db(); q = "SELECT * FROM analysis_log"
    params = []
    if experiment: q += " WHERE experiment=?"; params.append(experiment)
    q += " ORDER BY created_at DESC LIMIT ?"; params.append(limit)
    df = pd.read_sql_query(q, con, params=params); con.close(); return df

def get_upload_history(limit=50):
    con = _db(); df = pd.read_sql_query("SELECT * FROM uploads ORDER BY created_at DESC LIMIT ?", con, params=(limit,)); con.close(); return df

def get_session_info():
    if not SESSION_ID: return {}
    con = _db(); cur = con.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (SESSION_ID,))
    row = cur.fetchone(); con.close()
    if row: return {"id":row[0],"started":row[1],"last_activity":row[2],"n_actions":row[3],"experiments":row[4]}
    return {}

# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULTS = {
    "PXD046788 (Arabidopsis treatments)": r"D:\epiprofile_data\PXD046788\MS1_MS2\RawData",
    "PXD014739 (Arabidopsis histone)": r"D:\epiprofile_data\PXD014739\RawData",
    "PXD046034 (Arabidopsis FAS/NAP)": r"E:\EpiProfile_AT_PXD046034_raw\PXD046034\PXD046034",
    "Ontogeny 1exp (MS1+MS2)": r"E:\EpiProfile_Proyecto\EpiProfile_20_AT\histone_layouts_ontogeny_1exp",
    "Ontogeny RawData (ndebug_2)": r"E:\EpiProfile_Proyecto\EpiProfile_20_AT\RawData",
}

# Alternative histone_ratios files for Ontogeny (ndebug/rt_ref variants)
ALT_RATIOS = {
    "Ontogeny RawData (ndebug_2)": {
        "ndebug_2 (default)": "histone_ratios.xls",
        "ndebug_0": "histone_ratios_ndebug0_ontogenia.xls",
        "original (Nov 2024)": "histone_ratios_ontogenia.xls",
    },
}

parser = argparse.ArgumentParser(description="EpiProfile-Plants Dashboard v3.5")
parser.add_argument("dirs", nargs="*", help="EpiProfile output directories")
parser.add_argument("--port", type=int, default=8050)
parser.add_argument("--host", default="0.0.0.0")
args, _ = parser.parse_known_args()

if args.dirs:
    EXPERIMENTS = {}
    for p in args.dirs:
        p = os.path.abspath(p)
        EXPERIMENTS[os.path.basename(p) or p] = p
else:
    EXPERIMENTS = {k: v for k, v in DEFAULTS.items() if os.path.isdir(v)}
    if not EXPERIMENTS:
        print("ERROR: No valid experiment directories."); sys.exit(1)

# ======================================================================
# DESIGN SYSTEM
# ======================================================================

FONT = "'Inter','Segoe UI',-apple-system,BlinkMacSystemFont,Arial,sans-serif"
C = {
    "bg":"#f0fdf4","card":"#fff","border":"#d1d5db","text":"#1e293b","text2":"#475569",
    "accent":"#16a34a","accent_l":"#dcfce7","accent_d":"#15803d","red":"#dc2626","green":"#059669",
    "muted":"#94a3b8","warn":"#d97706","h3":"#7c3aed","h4":"#0891b2",
    "header_bg":"linear-gradient(135deg, #166534 0%, #15803d 40%, #22c55e 100%)",
}
# ---------- COLOR PALETTES (ggsci-inspired) ----------
PALETTES = {
    "EpiProfile (default)": ["#16a34a","#059669","#d97706","#dc2626","#0891b2","#7c3aed","#db2777",
                              "#0d9488","#ea580c","#4338ca","#4f46e5","#ca8a04","#9333ea","#2563eb","#c026d3"],
    "Simpsons":   ["#FED439","#709AE1","#8A9197","#D2AF81","#FD7446","#D5E4A2","#197EC0",
                   "#F05C3B","#46732E","#71D0F5","#370335","#075149","#C80813","#91331F","#1A9993"],
    "Futurama":   ["#FF6F00","#C71000","#008EA0","#8A4198","#5A9599","#FF6348","#84D7E1",
                   "#FF95A8","#3D3B25","#ADE2D0","#1A5354","#3F4041","#543005","#B5E0D4","#C9992D"],
    "Lancet":     ["#00468B","#ED0000","#42B540","#0099B4","#925E9F","#FDAF91","#AD002A",
                   "#ADB6B6","#1B1919","#5B7E9A","#C4A35A","#7D5A44","#3B3B3B","#006B38","#A50026"],
    "NEJM":       ["#BC3C29","#0072B5","#E18727","#20854E","#7876B1","#6F99AD","#FFDC91",
                   "#EE4C97","#8C564B","#5BB5A2","#D4A168","#9B2335","#2C73D2","#845B97","#B5651D"],
    "Nature":     ["#E64B35","#4DBBD5","#00A087","#3C5488","#F39B7F","#8491B4","#91D1C2",
                   "#DC0000","#7E6148","#B09C85","#1B7837","#999999","#5E4FA2","#D95F02","#636363"],
    "AAAS":       ["#3B4992","#EE0000","#008B45","#631879","#008280","#BB0021","#5F559B",
                   "#A20056","#808180","#1B1919","#E69F00","#56B4E9","#F0E442","#009E73","#CC79A7"],
    "JCO":        ["#0073C2","#EFC000","#868686","#CD534C","#7AA6DC","#003C67","#8F7700",
                   "#3B3B3B","#A73030","#4A6990","#C4A35A","#7D5A44","#006B38","#A50026","#D4A168"],
    "D3":         ["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2",
                   "#7F7F7F","#BCBD22","#17BECF","#AEC7E8","#FFBB78","#98DF8A","#FF9896","#C5B0D5"],
    "UCSCGB":     ["#FF0000","#FF9900","#FFCC00","#00FF00","#6699FF","#CC33FF","#99991E",
                   "#999999","#FF00CC","#CC0000","#FFCCCC","#FFFF00","#CCFF00","#358000","#0000CC"],
    "Dark2":      ["#1B9E77","#D95F02","#7570B3","#E7298A","#66A61E","#E6AB02","#A6761D",
                   "#666666","#1F78B4","#33A02C","#FB9A99","#E31A1C","#FDBF6F","#FF7F00","#CAB2D6"],
    "Set1":       ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00","#FFFF33","#A65628",
                   "#F781BF","#999999","#66C2A5","#FC8D62","#8DA0CB","#E78AC3","#A6D854","#FFD92F"],
}
GC = PALETTES["EpiProfile (default)"]

# Plant leaf SVG logo
LOGO = "data:image/svg+xml;base64," + base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">'
    b'<defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">'
    b'<stop offset="0%" style="stop-color:#22c55e"/>'
    b'<stop offset="100%" style="stop-color:#15803d"/></linearGradient></defs>'
    b'<rect width="48" height="48" rx="12" fill="url(#g)"/>'
    b'<text x="24" y="20" text-anchor="middle" fill="white" font-family="Arial" '
    b'font-weight="bold" font-size="10">EPI</text>'
    b'<text x="24" y="35" text-anchor="middle" fill="#bbf7d0" font-family="Arial" '
    b'font-weight="bold" font-size="11">PLANT</text>'
    b'<circle cx="10" cy="10" r="4" fill="#4ade80" opacity="0.6"/>'
    b'<circle cx="38" cy="38" r="3" fill="#4ade80" opacity="0.4"/>'
    b'</svg>').decode()

PUB = go.layout.Template()
PUB.layout = go.Layout(
    font=dict(family=FONT, size=14, color=C["text"]),
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    xaxis=dict(gridcolor="#f1f5f9",linecolor="#cbd5e1",linewidth=1,mirror=True,
               title_font=dict(size=18,color="#1e293b",weight=700),
               tickfont=dict(size=13,color="#334155")),
    yaxis=dict(gridcolor="#f1f5f9",linecolor="#cbd5e1",linewidth=1,mirror=True,
               title_font=dict(size=18,color="#1e293b",weight=700),
               tickfont=dict(size=13,color="#334155")),
    legend=dict(bgcolor="rgba(255,255,255,0.95)",font=dict(size=13,color="#334155"),
                bordercolor="#e5e7eb",borderwidth=1),
    margin=dict(t=50,b=70,l=80,r=25), colorway=GC)

CS = {"backgroundColor":"#fff","borderRadius":"14px","border":"1px solid #d1d5db",
      "padding":"28px","marginBottom":"20px",
      "boxShadow":"0 2px 6px rgba(0,0,0,0.05),0 1px 3px rgba(0,0,0,0.04)"}
DS = {"fontSize":"14px","borderRadius":"10px"}
TC = {"backgroundColor":"#fff","color":C["text"],"border":"1px solid #e5e7eb",
      "fontSize":"13px","textAlign":"left","padding":"10px 14px","fontFamily":FONT,
      "minWidth":"90px","maxWidth":"300px","overflow":"hidden","textOverflow":"ellipsis"}
TH = {"backgroundColor":"#f0fdf4","fontWeight":"600","color":"#166534",
      "borderBottom":f"2px solid {C['accent']}","fontSize":"12px",
      "textTransform":"uppercase","letterSpacing":"0.5px"}
TDC = [{"if":{"state":"active"},"backgroundColor":"#eef2ff","border":f"1px solid {C['accent']}"},
       {"if":{"row_index":"odd"},"backgroundColor":"#fafbfe"}]

# ======================================================================
# DATA LOADERS
# ======================================================================

def try_read_tsv(path):
    return pd.read_csv(path, sep="\t")

def find_file(base_dir, filenames):
    if isinstance(filenames, str): filenames = [filenames]
    for fn in filenames:
        fp = os.path.join(base_dir, fn)
        if os.path.exists(fp): return fp
    for root, dirs, files in os.walk(base_dir):
        if root.replace(base_dir,"").count(os.sep) > 3: dirs.clear(); continue
        for fn in filenames:
            if fn in files: return os.path.join(root, fn)
    return None


def parse_ratios_hierarchy(raw_df):
    """Parse histone_ratios.xls into structured hDP/hPF/variant levels."""
    first_col = raw_df.iloc[:, 0].astype(str)

    # Find separator column for ratio/area blocks
    sep_idx = None
    for i, col in enumerate(raw_df.columns):
        if "Unnamed" in str(col) and i > 0:
            sep_idx = i; break

    sample_cols_r = raw_df.columns[1:sep_idx] if sep_idx else raw_df.columns[1:]
    sample_names = [c.split(",",1)[1] if "," in c else c for c in sample_cols_r]

    # Build ratio matrix
    ratio_block = raw_df.iloc[:, :sep_idx] if sep_idx else raw_df.copy()
    ratio_block.columns = ["PTM"] + list(sample_names)
    ratio_block = ratio_block.iloc[1:].reset_index(drop=True)  # skip sub-header row

    # Build area matrix if present
    area_block = None
    if sep_idx and sep_idx + 1 < len(raw_df.columns):
        ab = raw_df.iloc[:, sep_idx+1:].copy()
        ab.insert(0, "PTM", raw_df.iloc[:, 0])
        acols = [re.sub(r"\.\d+$","",c.split(",",1)[1]) if "," in str(c) else str(c) for c in ab.columns[1:]]
        ab.columns = ["PTM"] + acols
        ab = ab.iloc[1:].reset_index(drop=True)
        area_block = ab

    # Classify rows
    hdp_list = []     # peptide region headers
    hpf_list = []     # peptidoforms
    variant_list = [] # sequence variants
    current_hdp = None
    current_type = "hpf"  # hpf or variant

    for _, row in ratio_block.iterrows():
        name = str(row["PTM"]).strip()
        if not name or name == "nan":
            continue

        # Peptide header: SEQUENCE(REGION) like TKQTAR(H3_3_8)
        m_header = re.match(r'^([A-Z]+)\(([^)]+)\)$', name)
        # Variant header: unmod(REGION) like unmod(H3_3_8)
        m_variant = re.match(r'^unmod\(([^)]+)\)$', name)

        if m_header:
            seq = m_header.group(1)
            region = m_header.group(2)
            histone = "H3.3" if region.startswith("H33") else \
                      "H3" if region.startswith("H3") else \
                      "H4" if region.startswith("H4") else "Other"
            current_hdp = {"name": name, "sequence": seq, "region": region,
                           "histone": histone}
            hdp_list.append(current_hdp)
            current_type = "hpf"
            # Store the header row ratios as well (sum-level)
            vals = pd.to_numeric(row.iloc[1:], errors="coerce")
            current_hdp["ratios"] = vals.values
            continue

        if m_variant:
            region = m_variant.group(1)
            current_hdp = {"name": name, "region": region, "histone": "",
                           "sequence": "unmod_header"}
            current_type = "variant"
            continue

        if current_type == "variant":
            vals = pd.to_numeric(row.iloc[1:], errors="coerce")
            # variant rows: "H3_3_8 TKQTAR"
            parts = name.split(" ", 1)
            region = parts[0] if parts else name
            seq_var = parts[1] if len(parts) > 1 else name
            variant_list.append({
                "name": name, "region": region, "sequence": seq_var,
                "values": vals.values
            })
            continue

        # Regular peptidoform row
        vals = pd.to_numeric(row.iloc[1:], errors="coerce")
        parts = name.split(" ", 1)
        region = parts[0]
        mod = parts[1] if len(parts) > 1 else "unmod"

        # Parse individual PTMs from mod string
        individual_ptms = []
        if mod != "unmod":
            # e.g. "K9me2K14ac" -> ["K9me2", "K14ac"]
            individual_ptms = re.findall(r'[KRSTkrst]\d+[a-z0-9]+', mod)

        histone = "H3.3" if region.startswith("H33") else \
                  "H3" if region.startswith("H3") else \
                  "H4" if region.startswith("H4") else "Other"

        is_combo = len(individual_ptms) > 1

        hpf_list.append({
            "name": name, "region": region, "modification": mod,
            "histone": histone, "is_combo": is_combo,
            "individual_ptms": individual_ptms, "n_mods": len(individual_ptms),
            "values": vals.values,
        })

    # Build DataFrames
    if hpf_list:
        hpf_df = pd.DataFrame([h["values"] for h in hpf_list],
                               columns=sample_names,
                               index=[h["name"] for h in hpf_list])
        hpf_meta = pd.DataFrame([{k: v for k, v in h.items() if k != "values"} for h in hpf_list])
    else:
        hpf_df = pd.DataFrame()
        hpf_meta = pd.DataFrame()

    if variant_list:
        var_df = pd.DataFrame([v["values"] for v in variant_list],
                               columns=sample_names,
                               index=[v["name"] for v in variant_list])
        var_meta = pd.DataFrame([{k: v for k, v in v.items() if k != "values"} for v in variant_list])
    else:
        var_df = pd.DataFrame()
        var_meta = pd.DataFrame()

    # Area matrix
    areas_df = None
    if area_block is not None:
        a_filtered = area_block[area_block["PTM"].isin(hpf_df.index)].copy()
        if not a_filtered.empty:
            a_filtered = a_filtered.set_index("PTM")
            a_filtered = a_filtered.apply(pd.to_numeric, errors="coerce")
            # Fix duplicate column names
            acn = []
            for cn in a_filtered.columns:
                acn.append(cn if cn not in acn else cn + "_area")
            a_filtered.columns = acn
            areas_df = a_filtered

    return {
        "sample_names": sample_names,
        "hdp_list": hdp_list,
        "hpf_df": hpf_df,
        "hpf_meta": hpf_meta,
        "var_df": var_df,
        "var_meta": var_meta,
        "areas": areas_df,
    }


def load_experiment(base_dir):
    data = {"base_dir": base_dir}
    layouts_dir = os.path.join(base_dir, "histone_layouts")
    data["layouts_dir"] = layouts_dir if os.path.isdir(layouts_dir) else base_dir
    ld = data["layouts_dir"]

    # -- single PTMs (hPTM level) --
    for fn in ["histone_ratios_single_PTMs.tsv","histone_ratios_single_PTMs.xls"]:
        fp = find_file(base_dir, fn)
        if not fp: fp = os.path.join(ld, fn) if os.path.exists(os.path.join(ld, fn)) else None
        if fp and os.path.exists(fp):
            try:
                df = try_read_tsv(fp)
                df.columns = ["PTM"] + [c.split(",",1)[1] if "," in c else c for c in df.columns[1:]]
                df = df.set_index("PTM").apply(pd.to_numeric, errors="coerce")
                data["hptm"] = df
                break
            except Exception: pass

    # -- histone_ratios.xls (hDP/hPF hierarchy) --
    fp = find_file(base_dir, ["histone_ratios.xls","histone_ratios.tsv"])
    if fp:
        try:
            raw = try_read_tsv(fp)
            parsed = parse_ratios_hierarchy(raw)
            data["hpf"] = parsed["hpf_df"]
            data["hpf_meta"] = parsed["hpf_meta"]
            data["hdp_list"] = parsed["hdp_list"]
            data["var_df"] = parsed["var_df"]
            data["var_meta"] = parsed["var_meta"]
            if parsed["areas"] is not None:
                data["areas"] = parsed["areas"]
        except Exception as e:
            print(f"    WARN parsing ratios: {e}")

    # -- phenodata --
    pheno_names = ["phenodata_arabidopsis_project.tsv","phenodata.tsv","phenodata_PXD046034.tsv",
                   "phenodata_ontogeny.tsv"]
    for fn in pheno_names:
        fp = find_file(base_dir, fn)
        if not fp: fp = os.path.join(ld, fn) if os.path.exists(os.path.join(ld, fn)) else None
        if fp and os.path.exists(fp):
            try: data["phenodata"] = pd.read_csv(fp, sep="\t"); break
            except: pass

    # -- metadata --
    ref = data.get("hptm", data.get("hpf"))
    if ref is not None:
        data["sample_names"] = list(ref.columns)
        data["metadata"] = build_metadata(list(ref.columns), data.get("phenodata"))

    # -- sample folders --
    folders = []
    for dn in sorted(os.listdir(ld)):
        if os.path.isdir(os.path.join(ld, dn)) and re.match(r"\d+[_-]", dn):
            folders.append(dn)
    data["sample_folders"] = folders

    # -- logs --
    lp = find_file(base_dir, "histone_logs.txt")
    data["logs"] = parse_logs(lp) if lp else []

    # -- PSM --
    data["all_psm"] = load_all_psm(ld, folders)

    # -- description --
    data["description"] = build_desc(data)
    return data


def build_desc(d):
    parts = []
    m = d.get("metadata")
    if m is not None and not m.empty:
        parts.append(f"{len(m)} samples")
        parts.append(f"{m['Group'].nunique()} groups")
    if "hptm" in d: parts.append(f"{d['hptm'].shape[0]} hPTMs")
    if "hpf" in d and not d["hpf"].empty: parts.append(f"{d['hpf'].shape[0]} peptidoforms")
    if "hdp_list" in d: parts.append(f"{len(d['hdp_list'])} peptide regions")
    psm = d.get("all_psm", pd.DataFrame())
    if not psm.empty: parts.append(f"{len(psm):,} PSMs")
    return " | ".join(parts) if parts else ""


def _match_sample(sample, pheno_names):
    """Robust matching: try exact, then stripped prefixes, then substring."""
    if sample in pheno_names:
        return sample
    # Strip leading number-dash from both sides
    s_stripped = re.sub(r'^\d+-', '', sample)
    for pn in pheno_names:
        pn_stripped = re.sub(r'^\d+-', '', pn)
        if s_stripped == pn_stripped or s_stripped == pn or sample == pn_stripped:
            return pn
    # Substring match (longest common suffix)
    for pn in pheno_names:
        if sample in pn or pn in sample:
            return pn
    return None


def build_metadata(sample_names, phenodata=None):
    records = []
    for s in sample_names:
        t, ti, r = parse_sample_name(s.strip())
        records.append({"Sample":s,"Treatment":t,"Tissue":ti,"Replicate":r})
    meta = pd.DataFrame(records)

    if phenodata is not None and "Sample_Name" in phenodata.columns:
        pheno = phenodata.copy()
        pn_list = pheno["Sample_Name"].tolist()

        # Build robust match map: data column -> phenodata row
        match_map = {}
        for s in sample_names:
            m = _match_sample(s, pn_list)
            if m is not None:
                match_map[s] = m

        if match_map:
            # Map all phenodata columns to meta using match
            pheno_indexed = pheno.set_index("Sample_Name")
            for col in pheno.columns:
                if col == "Sample_Name":
                    continue
                meta_col = "Group" if col == "Sample_Group" else col
                vals = []
                for s in meta["Sample"]:
                    pn = match_map.get(s)
                    if pn is not None and pn in pheno_indexed.index:
                        vals.append(pheno_indexed.loc[pn, col])
                    else:
                        vals.append(np.nan)
                meta[meta_col] = vals

            n_matched = sum(1 for s in sample_names if s in match_map)
            print(f"    Phenodata: matched {n_matched}/{len(sample_names)} samples")

    if "Group" not in meta.columns: meta["Group"] = meta["Treatment"]
    if "Batch" not in meta.columns: meta["Batch"] = "A"
    return meta


def parse_sample_name(name):
    if name.startswith("1y-CTR"):
        treatment = "1y-CTR"; rest = name.replace("1y-CTR_","")
    elif "-" in name and name[0].isdigit():
        parts = name.split("-"); treatment = parts[-1] if len(parts)>=4 else name; rest = name
    else:
        parts = name.split("_",1); treatment = parts[0]; rest = parts[1] if len(parts)>1 else ""
    tissue = "Root" if any(t in rest.lower() for t in ["rh","root"]) else \
             "Shoot" if any(t in rest.lower() for t in ["sh","shoot"]) else "Whole"
    rm = re.findall(r"(\d+)$", rest)
    return treatment, tissue, int(rm[0]) if rm else 0


def parse_logs(path):
    with open(path,"r",errors="replace") as f: content = f.read()
    samples = []; cur = None; regs = []; warns = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            if cur: samples.append({"sample":cur,"regions":regs[:],"warnings":warns[:],"n_warnings":len(warns)})
            cur = None; regs = []; warns = []; continue
        if line.startswith("elapsed"): continue
        if ".." not in line and "=" not in line and "hno=" not in line and "MS1" not in line and "pep_" not in line and "n rt_ref" not in line:
            cur = line; continue
        if ".." in line:
            for r in line.split(".."): r=r.strip(); regs.append(r) if r else None; continue
        if any(k in line for k in ["rt_ref","hno=","MS1","pep_","n rt_ref"]): warns.append(line)
    if cur: samples.append({"sample":cur,"regions":regs[:],"warnings":warns[:],"n_warnings":len(warns)})
    return samples


def load_all_psm(ld, folders):
    rows = []
    for f in folders:
        fp = os.path.join(ld, f, "detail", "psm", "identification_list.xls")
        if os.path.exists(fp):
            try:
                df = pd.read_csv(fp, sep="\t"); df["_sample_folder"] = f; rows.append(df)
            except: pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ======================================================================
# STATISTICAL HELPERS
# ======================================================================

def robust_group_test(df, meta, groups):
    """Kruskal-Wallis + pairwise Mann-Whitney with FDR correction."""
    results = []
    for ptm in df.index:
        group_vals = {}
        for g in groups:
            samps = meta[meta["Group"]==g]["Sample"].tolist()
            vals = df.loc[ptm, [s for s in samps if s in df.columns]].dropna()
            vals = vals[vals != 0]
            if len(vals) > 0:
                group_vals[g] = vals.values

        if len(group_vals) < 2:
            continue

        # Kruskal-Wallis
        try:
            arrays = list(group_vals.values())
            if all(len(a) >= 2 for a in arrays):
                kw_stat, kw_p = kruskal(*arrays)
            else:
                kw_stat, kw_p = np.nan, np.nan
        except:
            kw_stat, kw_p = np.nan, np.nan

        # Means and medians
        all_means = {g: np.mean(v) for g, v in group_vals.items()}
        all_medians = {g: np.median(v) for g, v in group_vals.items()}

        results.append({
            "PTM": ptm,
            "KW_stat": kw_stat, "KW_pval": kw_p,
            "n_groups": len(group_vals),
            **{f"mean_{g}": all_means.get(g, np.nan) for g in groups},
            **{f"median_{g}": all_medians.get(g, np.nan) for g in groups},
            **{f"n_{g}": len(group_vals.get(g, [])) for g in groups},
        })

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)
    # FDR correction
    valid_p = res_df["KW_pval"].dropna()
    if len(valid_p) > 0:
        _, fdr, _, _ = multipletests(valid_p.values, method="fdr_bh")
        res_df.loc[valid_p.index, "KW_FDR"] = fdr
    else:
        res_df["KW_FDR"] = np.nan

    res_df = res_df.sort_values("KW_pval")
    return res_df


def pairwise_mw(df, meta, g1, g2):
    """Mann-Whitney U between two groups for all PTMs."""
    sa = meta[meta["Group"]==g1]["Sample"].tolist()
    sb = meta[meta["Group"]==g2]["Sample"].tolist()
    ca = [c for c in df.columns if c in sa]
    cb = [c for c in df.columns if c in sb]
    results = []
    for ptm in df.index:
        va = df.loc[ptm, ca].dropna().values.astype(float)
        vb = df.loc[ptm, cb].dropna().values.astype(float)
        va = va[va != 0]; vb = vb[vb != 0]
        if len(va) >= 2 and len(vb) >= 2:
            try:
                stat, p = mannwhitneyu(va, vb, alternative="two-sided")
                fc = np.log2((np.mean(vb)+1e-8)/(np.mean(va)+1e-8))
                results.append({"PTM":ptm,"U_stat":stat,"pval":p,"log2FC":fc,
                                "mean_A":np.mean(va),"mean_B":np.mean(vb),
                                "median_A":np.median(va),"median_B":np.median(vb)})
            except: pass
    if not results: return pd.DataFrame()
    rdf = pd.DataFrame(results)
    if len(rdf) > 1:
        _, fdr, _, _ = multipletests(rdf["pval"].values, method="fdr_bh")
        rdf["FDR"] = fdr
    else:
        rdf["FDR"] = rdf["pval"]
    return rdf.sort_values("pval")


# ======================================================================
# LOAD ALL EXPERIMENTS
# ======================================================================

print("Loading experiments...")
EXP_DATA = {}
for name, path in EXPERIMENTS.items():
    if os.path.isdir(path):
        print(f"  {name}...")
        EXP_DATA[name] = load_experiment(path)
        d = EXP_DATA[name]
        n_hptm = d["hptm"].shape[0] if "hptm" in d else 0
        n_hpf = d["hpf"].shape[0] if "hpf" in d and not d["hpf"].empty else 0
        n_reg = len(d.get("hdp_list", []))
        meta = d.get("metadata", pd.DataFrame())
        n_samp = len(meta) if not meta.empty else 0
        n_grp = meta["Group"].nunique() if not meta.empty and "Group" in meta.columns else 0
        if "hptm" in d: print(f"    hPTM: {d['hptm'].shape}")
        if n_hpf: print(f"    hPF:  {d['hpf'].shape}")
        if n_reg: print(f"    hDP:  {n_reg} regions")
        print(f"    Folders: {len(d['sample_folders'])}")
        psm = d.get("all_psm", pd.DataFrame())
        if not psm.empty: print(f"    PSMs: {len(psm)}")
        print(f"    >> {d.get('description','')}")
        # Log to database
        log_experiment(name, path, n_samp, n_grp, n_hptm, n_hpf, n_reg)

DEFAULT_EXP = list(EXP_DATA.keys())[0] if EXP_DATA else None

# ======================================================================
# APP
# ======================================================================

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "EpiProfile-Plants Dashboard"

ts = {"color":"#475569","backgroundColor":"#fff","border":"none",
      "borderBottom":"3px solid transparent","padding":"12px 22px",
      "fontSize":"14px","fontWeight":"500","fontFamily":FONT}
tss = {**ts,"color":C["accent_d"],"borderBottom":f"3px solid {C['accent']}","fontWeight":"700",
       "backgroundColor":"#f0fdf4"}

# ======================================================================
# LAYOUT
# ======================================================================

app.layout = html.Div(style={"backgroundColor":C["bg"],"minHeight":"100vh","fontFamily":FONT,"color":C["text"]}, children=[
    # ---- HERO HEADER (big, green gradient) ----
    html.Div(style={"background":"linear-gradient(135deg, #166534 0%, #15803d 40%, #22c55e 100%)",
                     "padding":"28px 40px 20px","color":"white","position":"relative","overflow":"hidden"}, children=[
        # Decorative circles
        html.Div(style={"position":"absolute","top":"-30px","right":"-30px","width":"160px","height":"160px",
                         "borderRadius":"50%","background":"rgba(255,255,255,0.08)"}),
        html.Div(style={"position":"absolute","bottom":"-20px","left":"10%","width":"100px","height":"100px",
                         "borderRadius":"50%","background":"rgba(255,255,255,0.05)"}),
        # Top row: logo + title + experiment selector
        html.Div(style={"display":"flex","alignItems":"center","gap":"20px","flexWrap":"wrap","position":"relative","zIndex":"1"}, children=[
            html.Img(src=LOGO, style={"height":"56px","width":"56px","borderRadius":"14px",
                                       "boxShadow":"0 4px 12px rgba(0,0,0,0.2)"}),
            html.Div([
                html.H1("EpiProfile-Plants", style={"margin":"0","fontSize":"32px","fontWeight":"800",
                         "letterSpacing":"-0.5px","color":"white","lineHeight":"1.1"}),
                html.Div(style={"display":"flex","gap":"10px","alignItems":"center","marginTop":"4px"}, children=[
                    html.Span("PTM Dashboard v3.5", style={"color":"#bbf7d0","fontSize":"14px","fontWeight":"500"}),
                    html.Span("|", style={"color":"rgba(255,255,255,0.4)"}),
                    html.Span("hPTM", style={"background":"rgba(255,255,255,0.15)","padding":"2px 8px",
                              "borderRadius":"4px","fontSize":"12px","fontWeight":"600"}),
                    html.Span("hPF", style={"background":"rgba(255,255,255,0.15)","padding":"2px 8px",
                              "borderRadius":"4px","fontSize":"12px","fontWeight":"600"}),
                    html.Span("hDP", style={"background":"rgba(255,255,255,0.15)","padding":"2px 8px",
                              "borderRadius":"4px","fontSize":"12px","fontWeight":"600"}),
                ]),
            ]),
            html.Div(style={"flex":"1"}),
            # Experiment selector
            html.Div(style={"display":"flex","flexDirection":"column","gap":"4px"}, children=[
                html.Span("EXPERIMENT",style={"color":"#bbf7d0","fontSize":"11px","fontWeight":"700",
                           "letterSpacing":"1.5px","textTransform":"uppercase"}),
                dcc.Dropdown(id="exp-sel", options=[{"label":k,"value":k} for k in EXP_DATA],
                             value=DEFAULT_EXP, clearable=False,
                             style={"width":"400px","fontSize":"14px","borderRadius":"10px"}),
            ]),
            # Color palette selector
            html.Div(style={"display":"flex","flexDirection":"column","gap":"4px"}, children=[
                html.Span("COLOR PALETTE",style={"color":"#bbf7d0","fontSize":"11px","fontWeight":"700",
                           "letterSpacing":"1.5px","textTransform":"uppercase"}),
                dcc.Dropdown(id="palette-sel",
                             options=[{"label":k,"value":k} for k in PALETTES],
                             value="EpiProfile (default)", clearable=False,
                             style={"width":"220px","fontSize":"14px","borderRadius":"10px"}),
            ]),
        ]),
        # Upload area (collapsible, 3-slot)
        html.Details(style={"marginTop":"16px","position":"relative","zIndex":"1"}, children=[
            html.Summary("Upload phenodata / histone_ratios / single_PTMs files",
                         style={"cursor":"pointer","color":"#bbf7d0","fontSize":"13px","fontWeight":"500"}),
            html.Div(style={"display":"flex","gap":"16px","marginTop":"12px","flexWrap":"wrap"}, children=[
                html.Div(style={"flex":"1","minWidth":"250px"}, children=[
                    html.Label("1. Phenodata TSV", style={"color":"#bbf7d0","fontSize":"11px","fontWeight":"600"}),
                    dcc.Upload(id="upload-pheno",
                        children=html.Div(["Drop or ", html.A("select phenodata.tsv",style={"color":"#4ade80","fontWeight":"600"})]),
                        style={"border":"2px dashed rgba(255,255,255,0.3)","borderRadius":"10px","padding":"14px",
                               "textAlign":"center","color":"rgba(255,255,255,0.7)","fontSize":"13px",
                               "cursor":"pointer","backgroundColor":"rgba(0,0,0,0.1)"},
                        multiple=False),
                ]),
                html.Div(style={"flex":"1","minWidth":"250px"}, children=[
                    html.Label("2. histone_ratios.xls (TSV)", style={"color":"#bbf7d0","fontSize":"11px","fontWeight":"600"}),
                    dcc.Upload(id="upload-ratios",
                        children=html.Div(["Drop or ", html.A("select histone_ratios",style={"color":"#4ade80","fontWeight":"600"})]),
                        style={"border":"2px dashed rgba(255,255,255,0.3)","borderRadius":"10px","padding":"14px",
                               "textAlign":"center","color":"rgba(255,255,255,0.7)","fontSize":"13px",
                               "cursor":"pointer","backgroundColor":"rgba(0,0,0,0.1)"},
                        multiple=False),
                ]),
                html.Div(style={"flex":"1","minWidth":"250px"}, children=[
                    html.Label("3. Single PTMs TSV", style={"color":"#bbf7d0","fontSize":"11px","fontWeight":"600"}),
                    dcc.Upload(id="upload-singleptm",
                        children=html.Div(["Drop or ", html.A("select single_PTMs",style={"color":"#4ade80","fontWeight":"600"})]),
                        style={"border":"2px dashed rgba(255,255,255,0.3)","borderRadius":"10px","padding":"14px",
                               "textAlign":"center","color":"rgba(255,255,255,0.7)","fontSize":"13px",
                               "cursor":"pointer","backgroundColor":"rgba(0,0,0,0.1)"},
                        multiple=False),
                ]),
            ]),
            html.Div(id="upload-status", style={"color":"#4ade80","fontSize":"12px","marginTop":"8px"}),
        ]),
    ]),
    # ---- Description bar ----
    html.Div(id="desc-bar", style={"backgroundColor":"#dcfce7","padding":"8px 40px",
                                     "fontSize":"13px","color":"#166534","fontWeight":"500",
                                     "borderBottom":"1px solid #bbf7d0"}),
    # ---- Tabs ----
    dcc.Tabs(id="tabs", value="tab-hpf", style={"backgroundColor":"#fff","borderBottom":"2px solid #d1d5db"},
             colors={"border":"transparent","primary":C["accent"],"background":"#fff"}, children=[
        dcc.Tab(label="Peptidoforms (hPF)", value="tab-hpf", style=ts, selected_style=tss),
        dcc.Tab(label="Single PTMs (hPTM)", value="tab-hptm", style=ts, selected_style=tss),
        dcc.Tab(label="QC Dashboard", value="tab-qc", style=ts, selected_style=tss),
        dcc.Tab(label="PCA & Clustering", value="tab-pca", style=ts, selected_style=tss),
        dcc.Tab(label="Statistics", value="tab-stats", style=ts, selected_style=tss),
        dcc.Tab(label="UpSet / Co-occurrence", value="tab-upset", style=ts, selected_style=tss),
        dcc.Tab(label="Region Map", value="tab-region", style=ts, selected_style=tss),
        dcc.Tab(label="Comparisons", value="tab-cmp", style=ts, selected_style=tss),
        dcc.Tab(label="Phenodata", value="tab-pheno", style=ts, selected_style=tss),
        dcc.Tab(label="Sample Browser", value="tab-browse", style=ts, selected_style=tss),
        dcc.Tab(label="Analysis Log", value="tab-log", style=ts, selected_style=tss),
    ]),
    html.Div(id="tab-out", style={"padding":"28px 40px","maxWidth":"1700px","margin":"0 auto"}),
    # ---- Download component (hidden, triggered by export callbacks) ----
    dcc.Download(id="download-data"),
    # ---- Footer ----
    html.Div(style={"textAlign":"center","padding":"24px","color":"#166534","fontSize":"13px",
                     "background":"linear-gradient(135deg, #dcfce7 0%, #f0fdf4 100%)",
                     "borderTop":"2px solid #bbf7d0","marginTop":"40px"}, children=[
        html.Span("EpiProfile-Plants v3.5 | Histone PTM Quantification Dashboard | ", style={"fontWeight":"500"}),
        html.A("GitHub", href="https://github.com/biopelayo/epiprofile-dashboard",
               style={"color":"#15803d","textDecoration":"none","fontWeight":"700"}, target="_blank"),
    ]),
    dcc.Store(id="cur-exp", data=DEFAULT_EXP),
    dcc.Store(id="cur-palette", data="EpiProfile (default)"),
])


# ======================================================================
# CALLBACKS - ROUTING
# ======================================================================

@callback(Output("cur-exp","data"), Input("exp-sel","value"))
def _se(e): return e

@callback(Output("cur-palette","data"), Input("palette-sel","value"))
def _sp(p):
    global GC
    if p and p in PALETTES:
        GC = PALETTES[p]
        PUB.layout.colorway = GC
    return p

@callback(Output("desc-bar","children"), Input("cur-exp","data"), Input("cur-palette","data"))
def _db(e, pal):
    desc = EXP_DATA[e].get("description","") if e and e in EXP_DATA else ""
    pal_name = pal if pal else "EpiProfile (default)"
    return f"{desc}  |  Palette: {pal_name}"

@callback(Output("upload-status","children"),
          Input("upload-pheno","contents"), Input("upload-ratios","contents"),
          Input("upload-singleptm","contents"),
          State("upload-pheno","filename"), State("upload-ratios","filename"),
          State("upload-singleptm","filename"),
          State("cur-exp","data"), prevent_initial_call=True)
def _upload(pheno_content, ratios_content, sptm_content, pheno_name, ratios_name, sptm_name, exp):
    if not exp or exp not in EXP_DATA:
        return html.Div("No experiment selected.", style={"color":"#fca5a5"})
    d = EXP_DATA[exp]
    results = []

    if pheno_content:
        try:
            _, content_string = pheno_content.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            pheno_df = pd.read_csv(StringIO(decoded), sep="\t")
            # Validate required columns
            missing = [c for c in ["Sample_Name","Sample_Group"] if c not in pheno_df.columns]
            if missing:
                results.append(html.Span(f"Phenodata WARN: missing columns {missing}. ", style={"color":"#fcd34d"}))
            else:
                d["phenodata"] = pheno_df
                ref = d.get("hptm", d.get("hpf"))
                if ref is not None:
                    d["metadata"] = build_metadata(list(ref.columns), pheno_df)
                    d["description"] = build_desc(d)
                    # Match report
                    matched = len([s for s in ref.columns if s in pheno_df["Sample_Name"].values])
                    results.append(html.Span(
                        f"Phenodata OK: {pheno_name} ({len(pheno_df)} rows, {matched}/{len(ref.columns)} samples matched, "
                        f"groups: {', '.join(sorted(pheno_df['Sample_Group'].unique()))}). ",
                        style={"color":"#4ade80"}))
                log_upload(exp, pheno_name, "phenodata", len(pheno_df), len(pheno_df.columns), "success",
                           f"{len(pheno_df)} rows")
        except Exception as e:
            results.append(html.Span(f"Phenodata ERROR: {e}. ", style={"color":"#fca5a5"}))
            log_upload(exp, pheno_name or "?", "phenodata", 0, 0, "error", str(e))

    if ratios_content:
        try:
            _, content_string = ratios_content.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            raw = pd.read_csv(StringIO(decoded), sep="\t")
            parsed = parse_ratios_hierarchy(raw)
            d["hpf"] = parsed["hpf_df"]
            d["hpf_meta"] = parsed["hpf_meta"]
            d["hdp_list"] = parsed["hdp_list"]
            d["var_df"] = parsed["var_df"]
            d["var_meta"] = parsed["var_meta"]
            if parsed["areas"] is not None:
                d["areas"] = parsed["areas"]
            d["description"] = build_desc(d)
            results.append(html.Span(
                f"Ratios OK: {ratios_name} ({parsed['hpf_df'].shape[0]} hPF, {len(parsed['hdp_list'])} regions). ",
                style={"color":"#4ade80"}))
            log_upload(exp, ratios_name, "histone_ratios", parsed['hpf_df'].shape[0], parsed['hpf_df'].shape[1],
                       "success", f"{parsed['hpf_df'].shape[0]} hPF")
        except Exception as e:
            results.append(html.Span(f"Ratios ERROR: {e}. ", style={"color":"#fca5a5"}))
            log_upload(exp, ratios_name or "?", "histone_ratios", 0, 0, "error", str(e))

    if sptm_content:
        try:
            _, content_string = sptm_content.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            sptm_df = pd.read_csv(StringIO(decoded), sep="\t", index_col=0)
            sptm_df = sptm_df.apply(pd.to_numeric, errors="coerce")
            d["hptm"] = sptm_df
            d["description"] = build_desc(d)
            results.append(html.Span(
                f"Single PTMs OK: {sptm_name} ({sptm_df.shape[0]} PTMs, {sptm_df.shape[1]} samples). ",
                style={"color":"#4ade80"}))
            log_upload(exp, sptm_name, "single_ptms", sptm_df.shape[0], sptm_df.shape[1],
                       "success", f"{sptm_df.shape[0]} hPTMs")
        except Exception as e:
            results.append(html.Span(f"Single PTMs ERROR: {e}. ", style={"color":"#fca5a5"}))
            log_upload(exp, sptm_name or "?", "single_ptms", 0, 0, "error", str(e))

    return html.Div(results) if results else ""

@callback(Output("tab-out","children"), Input("tabs","value"), Input("cur-exp","data"), Input("cur-palette","data"))
def _rt(tab, exp, pal):
    if not exp or exp not in EXP_DATA:
        return html.Div("No experiment loaded", style={"color":C["red"],"textAlign":"center","padding":"80px"})
    d = EXP_DATA[exp]
    try:
        if tab == "tab-log": return tab_log(d, exp)
        return {"tab-hpf":tab_hpf,"tab-hptm":tab_hptm,"tab-qc":tab_qc,
                "tab-pca":tab_pca,"tab-stats":tab_stats,"tab-upset":tab_upset,
                "tab-region":tab_region,"tab-cmp":tab_cmp,"tab-pheno":tab_pheno,
                "tab-browse":tab_browse}.get(tab, lambda x: html.Div("?"))(d)
    except Exception as e:
        import traceback
        return html.Div([html.H3("Error",style={"color":C["red"]}),
                         html.Pre(traceback.format_exc(),style={"fontSize":"11px","color":C["text2"],"whiteSpace":"pre-wrap"})])

# ======================================================================
# HELPERS
# ======================================================================

def adaptive_font(n_items, base=14, min_size=8, max_size=18):
    """Compute font size based on number of items to display."""
    if n_items <= 15: return max_size
    if n_items <= 30: return base
    if n_items <= 60: return max(min_size + 2, base - 2)
    if n_items <= 100: return max(min_size + 1, base - 3)
    return min_size

def adaptive_margin_l(labels, base=80, per_char=6.5, min_m=60, max_m=300):
    """Compute left margin based on longest label length."""
    if not labels: return base
    longest = max(len(str(l)) for l in labels)
    return int(min(max_m, max(min_m, base + longest * per_char)))

def adaptive_legend(n_groups):
    """Return legend layout dict based on number of groups."""
    if n_groups <= 8:
        return dict(font=dict(size=13), x=1.02, y=1, xanchor="left")
    if n_groups <= 15:
        return dict(font=dict(size=11), x=1.02, y=1, xanchor="left")
    return dict(font=dict(size=9), orientation="h", y=-0.15, x=0.5, xanchor="center")

def classify_ptm_name(ptm_name):
    """Classify a PTM by histone and modification type."""
    histone = "H3.3" if ptm_name.startswith("H33") else \
              "H3" if ptm_name.startswith("H3") else \
              "H4" if ptm_name.startswith("H4") else "Other"
    name_l = ptm_name.lower()
    if "me" in name_l: ptm_type = "Methylation"
    elif "ac" in name_l: ptm_type = "Acetylation"
    elif "ph" in name_l: ptm_type = "Phosphorylation"
    elif "ub" in name_l: ptm_type = "Ubiquitination"
    elif "unmod" in name_l: ptm_type = "Unmodified"
    else: ptm_type = "Other"
    return histone, ptm_type

def pfig(fig, h=500, n_x=None, n_y=None, n_groups=None):
    """Publication figure with adaptive sizing."""
    fig.update_layout(template=PUB, height=h, colorway=GC)
    xsz = adaptive_font(n_x) if n_x else 13
    ysz = adaptive_font(n_y) if n_y else 13
    fig.update_xaxes(title_font=dict(size=18, color="#1e293b"),
                     tickfont=dict(size=xsz, color="#334155"))
    fig.update_yaxes(title_font=dict(size=18, color="#1e293b"),
                     tickfont=dict(size=ysz, color="#334155"))
    if n_groups:
        fig.update_layout(legend=adaptive_legend(n_groups))
    return fig

def phm(z, x, y, cs="Viridis", title="", zmin=None, zmax=None, h=600, meta=None):
    """Publication heatmap with adaptive font sizes. If meta is provided, adds a group color bar."""
    xsz = adaptive_font(len(x))
    ysz = adaptive_font(len(y))
    ml = adaptive_margin_l(y)
    if meta is not None and not meta.empty:
        sample_groups = {}
        for _, row in meta.iterrows():
            sample_groups[row["Sample"]] = row["Group"]
        groups_unique = sorted(set(sample_groups.values()))
        gc_map = {g: GC[i % len(GC)] for i, g in enumerate(groups_unique)}
        group_labels = [sample_groups.get(s, "?") for s in x]

        fig = make_subplots(rows=2, cols=1, row_heights=[0.03, 0.97],
                            vertical_spacing=0.005, shared_xaxes=True)
        fig.add_trace(go.Heatmap(
            z=[[groups_unique.index(g) if g in groups_unique else 0 for g in group_labels]],
            x=x, y=["Group"], colorscale=[[i/(max(len(groups_unique)-1,1)),gc_map[g]] for i,g in enumerate(groups_unique)],
            showscale=False, hovertext=[[g for g in group_labels]], hoverinfo="text",
            zmin=0, zmax=max(len(groups_unique)-1,1)), row=1, col=1)
        fig.add_trace(go.Heatmap(z=z,x=x,y=y,colorscale=cs,
            colorbar=dict(thickness=14,len=0.85,title=dict(text=title,side="right",font=dict(size=13)),tickfont=dict(size=12)),
            hoverongaps=False, zmin=zmin, zmax=zmax), row=2, col=1)
        fig.update_layout(template=PUB,height=h,margin=dict(l=ml,b=130,t=35,r=40))
        fig.update_xaxes(tickangle=45,tickfont=dict(size=xsz), row=2, col=1)
        fig.update_yaxes(tickfont=dict(size=ysz),autorange="reversed", row=2, col=1)
        fig.update_yaxes(tickfont=dict(size=13), row=1, col=1)
        return fig
    else:
        fig = go.Figure(go.Heatmap(z=z,x=x,y=y,colorscale=cs,
            colorbar=dict(thickness=14,len=0.9,title=dict(text=title,side="right",font=dict(size=13)),tickfont=dict(size=12)),
            hoverongaps=False, zmin=zmin, zmax=zmax))
        fig.update_layout(template=PUB,height=h,xaxis=dict(tickangle=45,tickfont=dict(size=xsz)),
                          yaxis=dict(tickfont=dict(size=ysz),autorange="reversed"),margin=dict(l=ml,b=130,t=35,r=40))
        return fig

def cluster_order(df, axis=0):
    try:
        data = df.fillna(0).values if axis==0 else df.fillna(0).values.T
        if data.shape[0] < 3: return list(df.index) if axis==0 else list(df.columns)
        return [list(df.index if axis==0 else df.columns)[i] for i in leaves_list(linkage(pdist(data),method="ward"))]
    except: return list(df.index) if axis==0 else list(df.columns)

def _sc(label, val, color):
    return html.Div(style={**CS,"flex":"1","minWidth":"145px","textAlign":"center","padding":"20px 14px"}, children=[
        html.H2(str(val),style={"color":color,"margin":"0","fontSize":"34px","fontWeight":"800"}),
        html.P(label,style={"color":C["muted"],"margin":"6px 0 0","fontSize":"12px","textTransform":"uppercase",
                             "letterSpacing":"0.8px","fontWeight":"600"})])

def _st(text, sub=""):
    ch = [html.H3(text,style={"color":"#166534","marginTop":"0","marginBottom":"6px","fontSize":"18px","fontWeight":"700"})]
    if sub: ch.append(html.P(sub,style={"color":C["muted"],"margin":"0 0 14px","fontSize":"13px"}))
    return html.Div(ch)

def _lbl(text):
    return html.Label(text, style={"fontSize":"12px","color":"#166534","fontWeight":"600",
                                    "textTransform":"uppercase","letterSpacing":"0.8px"})

def make_table(df, tid, max_r=200):
    dd = df.head(max_r).reset_index() if df.index.name or df.index.dtype != "int64" else df.head(max_r)
    for col in dd.select_dtypes(include=[np.number]).columns: dd[col] = dd[col].round(6)
    return dash_table.DataTable(id=tid, data=dd.to_dict("records"),
        columns=[{"name":str(c),"id":str(c),"deletable":True,"renamable":True,
                  "type":"numeric" if pd.api.types.is_numeric_dtype(dd[c]) else "text"} for c in dd.columns],
        editable=True, row_deletable=True, filter_action="native", sort_action="native",
        sort_mode="multi", page_action="native", page_size=25, export_format="csv",
        style_cell=TC, style_header=TH, style_data_conditional=TDC,
        style_table={"overflowX":"auto","borderRadius":"8px"},
        style_filter={"backgroundColor":"#f8fafc","fontSize":"11px"})


# ======================================================================
# TAB: PEPTIDOFORMS (hPF) -- with working filters
# ======================================================================

def tab_hpf(d):
    hpf = d.get("hpf", pd.DataFrame())
    hpf_meta = d.get("hpf_meta", pd.DataFrame())
    if hpf.empty:
        return html.Div(style=CS, children=[_st("Peptidoforms (hPF)"),
            html.P("No histone_ratios.xls found.",style={"color":C["muted"]})])

    meta = d["metadata"]
    groups = sorted(meta["Group"].unique())
    histones = sorted(hpf_meta["histone"].unique()) if "histone" in hpf_meta.columns else []
    regions = sorted(hpf_meta["region"].unique()) if "region" in hpf_meta.columns else []

    # Summary stats
    n_total = len(hpf)
    n_combo = int(hpf_meta["is_combo"].sum()) if "is_combo" in hpf_meta.columns else 0
    n_single = n_total - n_combo
    n_regions = len(d.get("hdp_list",[]))

    return html.Div([
        # Stats row
        html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}, children=[
            _sc("Peptidoforms", n_total, C["accent"]),
            _sc("Single-mod hPF", n_single, C["green"]),
            _sc("Combo hPF", n_combo, C["warn"]),
            _sc("Peptide Regions", n_regions, C["h3"]),
            _sc("Groups", len(groups), C["h4"]),
        ]),
        # Filter bar
        html.Div(style={**CS,"display":"flex","gap":"16px","alignItems":"flex-end","flexWrap":"wrap"}, children=[
            html.Div(style={"flex":"1","minWidth":"140px"}, children=[
                _lbl("Histone"),
                dcc.Dropdown(id="hpf-hist", options=[{"label":"All","value":"All"}]+[{"label":h,"value":h} for h in histones],
                             value="All", clearable=False, style=DS),
            ]),
            html.Div(style={"flex":"1","minWidth":"140px"}, children=[
                _lbl("Region"),
                dcc.Dropdown(id="hpf-reg", options=[{"label":"All","value":"All"}]+[{"label":r,"value":r} for r in regions],
                             value="All", clearable=False, style=DS),
            ]),
            html.Div(style={"flex":"1","minWidth":"140px"}, children=[
                _lbl("Group"),
                dcc.Dropdown(id="hpf-grp", options=[{"label":"All","value":"All"}]+[{"label":g,"value":g} for g in groups],
                             value="All", clearable=False, style=DS),
            ]),
            html.Div(style={"flex":"1","minWidth":"120px"}, children=[
                _lbl("Type"),
                dcc.Dropdown(id="hpf-type", options=[{"label":"All","value":"All"},{"label":"Single mod","value":"single"},
                             {"label":"Combinatorial","value":"combo"},{"label":"Unmodified","value":"unmod"}],
                             value="All", clearable=False, style=DS),
            ]),
            html.Div(style={"flex":"1","minWidth":"180px"}, children=[
                _lbl("Min mean ratio"),
                dcc.Slider(id="hpf-min", min=0, max=0.5, step=0.01, value=0,
                           marks={0:"0",0.1:"0.1",0.25:"0.25",0.5:"0.5"},
                           tooltip={"placement":"bottom","always_visible":False}),
            ]),
            html.Div(style={"flex":"1","minWidth":"180px"}, children=[
                _lbl("Top N by variance"),
                dcc.Slider(id="hpf-topn", min=10, max=200, step=10, value=50,
                           marks={10:"10",50:"50",100:"100",200:"All"},
                           tooltip={"placement":"bottom","always_visible":False}),
            ]),
            html.Div(style={"flex":"0","minWidth":"100px","display":"flex","alignItems":"flex-end"}, children=[
                html.Button("Export CSV", id="hpf-export", n_clicks=0,
                            style={"padding":"10px 16px","borderRadius":"8px","border":"none",
                                   "backgroundColor":C["accent"],"color":"white","fontWeight":"600",
                                   "cursor":"pointer","fontSize":"13px"})]),
        ]),
        # Dynamic content
        html.Div(id="hpf-content"),
    ])


@callback(Output("hpf-content","children"),
          Input("hpf-hist","value"), Input("hpf-reg","value"), Input("hpf-grp","value"),
          Input("hpf-type","value"), Input("hpf-min","value"), Input("hpf-topn","value"),
          Input("cur-exp","data"))
def update_hpf(hist, reg, grp, htype, min_val, topn, exp):
    if not exp or exp not in EXP_DATA: return html.P("No data")
    d = EXP_DATA[exp]
    hpf = d.get("hpf", pd.DataFrame())
    hpf_meta = d.get("hpf_meta", pd.DataFrame())
    meta = d["metadata"]
    if hpf.empty: return html.P("No hPF data")

    # Apply filters
    mask = pd.Series(True, index=hpf_meta.index)
    if hist != "All" and "histone" in hpf_meta.columns:
        mask &= hpf_meta["histone"] == hist
    if reg != "All" and "region" in hpf_meta.columns:
        mask &= hpf_meta["region"] == reg
    if htype == "single":
        mask &= (~hpf_meta["is_combo"]) & (hpf_meta["modification"] != "unmod")
    elif htype == "combo":
        mask &= hpf_meta["is_combo"]
    elif htype == "unmod":
        mask &= hpf_meta["modification"] == "unmod"

    filtered_names = hpf_meta.loc[mask, "name"].tolist()
    df = hpf.loc[[n for n in filtered_names if n in hpf.index]].copy()

    # Group filter
    if grp != "All":
        samps = meta[meta["Group"]==grp]["Sample"].tolist()
        cols = [c for c in df.columns if c in samps]
        if cols: df = df[cols]

    df = df.dropna(how="all")
    df = df[(df != 0).any(axis=1)]

    # Min mean filter
    if min_val > 0:
        df = df[df.mean(axis=1) >= min_val]

    # Top N by variance
    if topn < 200 and len(df) > topn:
        var_s = df.var(axis=1).sort_values(ascending=False)
        df = df.loc[var_s.head(topn).index]

    if df.empty:
        return html.P(f"No peptidoforms match filters (0/{len(hpf)})", style={"color":C["muted"],"padding":"40px","textAlign":"center"})

    n_shown = len(df)
    n_total = len(hpf)

    # Heatmap with group color bar
    hm = phm(df.values, df.columns.tolist(), df.index.tolist(),
             cs="Viridis", title="Ratio", h=max(400, len(df)*7), meta=meta)

    # Top variable
    var_s = df.var(axis=1).dropna().sort_values(ascending=False).head(20)
    vf = go.Figure(go.Bar(x=var_s.values, y=var_s.index.tolist(), orientation="h",
                           marker=dict(color=var_s.values, colorscale="Viridis",line=dict(width=0))))
    pfig(vf, 400, n_y=len(var_s)); vf.update_layout(yaxis=dict(autorange="reversed"),
                                      margin=dict(l=adaptive_margin_l(var_s.index.tolist())),xaxis_title="Variance")

    # Faceted violin by group for top 6 most variable hPF
    top_for_violin = var_s.head(6).index.tolist()
    vml = []
    for ptm in top_for_violin:
        if ptm in df.index:
            v = df.loc[ptm].T.reset_index(); v.columns = ["Sample","Ratio"]
            v["hPF"] = ptm; v = v.merge(meta, on="Sample"); vml.append(v)
    if vml:
        vmdf = pd.concat(vml, ignore_index=True)
        viol_fig = px.violin(vmdf, x="Group", y="Ratio", color="Group", facet_col="hPF",
                             facet_col_wrap=3, box=True, points="all", color_discrete_sequence=GC)
        pfig(viol_fig, 550)
        viol_fig.update_layout(showlegend=False)
        viol_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1][:30]))
    else:
        viol_fig = go.Figure(); pfig(viol_fig, 300)

    # Box for top PTM
    top = var_s.index[0] if len(var_s) > 0 else df.index[0]
    melt = df.loc[[top]].T.reset_index(); melt.columns = ["Sample","Ratio"]
    melt = melt.merge(meta, on="Sample")
    bf = px.box(melt, x="Group", y="Ratio", color="Group", points="all",
                title=top, color_discrete_sequence=GC)
    pfig(bf, 350)

    return html.Div([
        html.Div(style=CS, children=[
            _st(f"Peptidoform Heatmap", f"Showing {n_shown} of {n_total} hPF | {df.shape[1]} samples | Group color bar on top"),
            dcc.Graph(figure=hm),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"2","minWidth":"400px"}, children=[
                _st("Top Variable Peptidoforms"), dcc.Graph(figure=vf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"300px"}, children=[
                _st(f"Distribution: {top}"), dcc.Graph(figure=bf)]),
        ]),
        html.Div(style=CS, children=[
            _st("Faceted Violin: Top Variable hPF by Group","Top 6 most variable peptidoforms"),
            dcc.Graph(figure=viol_fig),
        ]),
        html.Div(style=CS, children=[
            _st("Filtered Data Table","Editable | Sortable | Filterable | Export CSV"),
            make_table(df, "hpf-table"),
        ]),
    ])


@callback(Output("download-data","data", allow_duplicate=True),
          Input("hpf-export","n_clicks"),
          State("hpf-hist","value"),State("hpf-reg","value"),State("hpf-grp","value"),
          State("hpf-type","value"),State("hpf-min","value"),State("hpf-topn","value"),
          State("cur-exp","data"), prevent_initial_call=True)
def _hpf_export(n, hist, reg, grp, htype, min_val, topn, exp):
    if not n or not exp or exp not in EXP_DATA: return no_update
    d = EXP_DATA[exp]; hpf = d.get("hpf", pd.DataFrame()); hpf_meta = d.get("hpf_meta", pd.DataFrame())
    meta = d["metadata"]
    if hpf.empty: return no_update
    mask = pd.Series(True, index=hpf_meta.index)
    if hist != "All" and "histone" in hpf_meta.columns: mask &= hpf_meta["histone"] == hist
    if reg != "All" and "region" in hpf_meta.columns: mask &= hpf_meta["region"] == reg
    if htype == "single": mask &= (~hpf_meta["is_combo"]) & (hpf_meta["modification"] != "unmod")
    elif htype == "combo": mask &= hpf_meta["is_combo"]
    elif htype == "unmod": mask &= hpf_meta["modification"] == "unmod"
    filtered_names = hpf_meta.loc[mask, "name"].tolist()
    df = hpf.loc[[n_ for n_ in filtered_names if n_ in hpf.index]].copy()
    if grp != "All":
        samps = meta[meta["Group"]==grp]["Sample"].tolist()
        cols = [c for c in df.columns if c in samps]
        if cols: df = df[cols]
    df = df.dropna(how="all"); df = df[(df != 0).any(axis=1)]
    if min_val > 0: df = df[df.mean(axis=1) >= min_val]
    if topn < 200 and len(df) > topn: df = df.loc[df.var(axis=1).sort_values(ascending=False).head(topn).index]
    fname = f"{exp.split('(')[0].strip()}_hPF_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    log_analysis(exp, "export_hpf", {"hist":hist,"reg":reg,"grp":grp,"type":htype}, len(df), 0, f"Exported {len(df)} hPF")
    return dcc.send_data_frame(df.to_csv, fname)


# ======================================================================
# TAB: SINGLE PTMs (hPTM)
# ======================================================================

def tab_hptm(d):
    if "hptm" not in d:
        return html.Div(style=CS, children=[html.P("No single PTM data.",style={"color":C["muted"]})])

    df = d["hptm"].copy()
    meta = d["metadata"]
    groups = sorted(meta["Group"].unique())

    meta_s = meta.sort_values(["Group","Tissue","Replicate"])
    co = [s for s in meta_s["Sample"] if s in df.columns]; df = df[co]

    # Clustered heatmap with group color bar
    ro = cluster_order(df, 0); df_c = df.loc[ro]
    hm = phm(df_c.values, df_c.columns.tolist(), df_c.index.tolist(),
             cs="RdBu_r", title="Ratio", h=max(500, len(df)*11), meta=meta)

    # Z-score with group color bar
    zs = df_c.apply(lambda r: (r-r.mean())/(r.std()+1e-10), axis=1)
    zhm = phm(zs.values, zs.columns.tolist(), zs.index.tolist(),
              cs="RdBu_r", title="Z-score", zmin=-3, zmax=3, h=max(500, len(df)*11), meta=meta)

    # Faceted violin: key marks by group
    km = [m for m in ["H3K9me2","H3K14ac","H3K27me3","H3K4me1","H4K16ac","H3K9ac",
                       "H3K36me1","H3K27me1","H4K20me1","H3K4me3"] if m in df.index][:8]
    if km:
        ml = []
        for m in km:
            v = df.loc[m]; t = pd.DataFrame({"Sample":v.index,"Ratio":v.values,"PTM":m})
            t = t.merge(meta, on="Sample"); ml.append(t)
        vdata = pd.concat(ml)
        vf = px.violin(vdata, x="Group", y="Ratio", color="Group", facet_col="PTM",
                        facet_col_wrap=4, box=True, points="all", color_discrete_sequence=GC)
        pfig(vf, 600); vf.update_layout(showlegend=False)
        vf.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    else:
        vf = go.Figure(); pfig(vf, 420)

    # Grouped bar
    bd = []
    for ptm in df.index:
        for g in groups:
            samps = meta[meta["Group"]==g]["Sample"].tolist()
            vals = df.loc[ptm, [s for s in samps if s in df.columns]].dropna()
            if len(vals)>0: bd.append({"PTM":ptm,"Group":g,"Mean":vals.mean(),"SD":vals.std()})
    bdf = pd.DataFrame(bd)
    tp = bdf.groupby("PTM")["Mean"].max().sort_values(ascending=False).head(15).index
    bf = px.bar(bdf[bdf["PTM"].isin(tp)], x="PTM", y="Mean", color="Group", barmode="group",
                error_y="SD", color_discrete_sequence=GC)
    pfig(bf, 420, n_x=len(tp), n_groups=len(groups)); bf.update_layout(xaxis=dict(tickangle=45),yaxis_title="Mean Ratio")

    return html.Div([
        html.Div(style={**CS,"display":"flex","gap":"12px","alignItems":"center","justifyContent":"flex-end"}, children=[
            html.Button("Export hPTM CSV", id="hptm-export", n_clicks=0,
                        style={"padding":"10px 16px","borderRadius":"8px","border":"none",
                               "backgroundColor":C["accent"],"color":"white","fontWeight":"600",
                               "cursor":"pointer","fontSize":"13px"})]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"450px"}, children=[
                _st("Clustered Heatmap","Ward linkage | hPTM ratios | Group color bar"), dcc.Graph(figure=hm)]),
            html.Div(style={**CS,"flex":"1","minWidth":"450px"}, children=[
                _st("Z-score Heatmap","Row-wise normalization | Group color bar"), dcc.Graph(figure=zhm)]),
        ]),
        html.Div(style=CS, children=[
            _st("Key hPTM Faceted Violins","Faceted by PTM, colored by group"), dcc.Graph(figure=vf)]),
        html.Div(style=CS, children=[
            _st("Top 15 hPTMs by Group Mean +/- SD"), dcc.Graph(figure=bf)]),
        html.Div(style=CS, children=[
            _st("hPTM Data","Editable | Sortable | Filterable | Export CSV"),
            make_table(df, "hptm-table")]),
    ])


@callback(Output("download-data","data", allow_duplicate=True),
          Input("hptm-export","n_clicks"), State("cur-exp","data"), prevent_initial_call=True)
def _hptm_export(n, exp):
    if not n or not exp or exp not in EXP_DATA: return no_update
    d = EXP_DATA[exp]; df = d.get("hptm")
    if df is None: return no_update
    fname = f"{exp.split('(')[0].strip()}_hPTM_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    log_analysis(exp, "export_hptm", {}, len(df), 0, f"Exported {len(df)} hPTMs")
    return dcc.send_data_frame(df.to_csv, fname)


# ======================================================================
# TAB: QC
# ======================================================================

def tab_qc(d):
    meta = d.get("metadata", pd.DataFrame())
    df = d.get("hptm", d.get("hpf"))
    if df is None or df.empty:
        return html.Div(style=CS, children=[html.P("No data for QC.")])

    binary = (~df.isna() & (df!=0)).astype(int)
    mhm = phm(binary.values, binary.columns.tolist(), binary.index.tolist(),
              cs=[[0,"#fef2f2"],[1,"#059669"]], title="Detected", h=max(400,len(binary)*9))

    mc = (df.isna()|(df==0)).sum(axis=0)
    mdf = pd.DataFrame({"Sample":mc.index,"Missing":mc.values}).merge(meta, on="Sample", how="left")
    mb = px.bar(mdf, x="Sample", y="Missing", color="Group", title="Missing per Sample", color_discrete_sequence=GC)
    pfig(mb, 350); mb.update_layout(xaxis=dict(tickangle=45,tickfont=dict(size=12)))

    dpp = binary.sum(axis=1)
    ch = px.histogram(x=dpp.values, nbins=20, labels={"x":"# Samples Detected","y":"# Features"},
                      title="Completeness", color_discrete_sequence=[C["accent"]])
    pfig(ch, 300)

    areas = d.get("areas")
    ab = go.Figure()
    if areas is not None:
        la = np.log10(areas.replace(0,np.nan))
        ma = la.stack().reset_index(); ma.columns = ["PTM","Sample","Log10Area"]
        ma = ma.merge(meta, on="Sample", how="left")
        ab = px.box(ma, x="Sample", y="Log10Area", color="Group",
                    title="Log10(Area) per Sample", color_discrete_sequence=GC)
        pfig(ab, 380); ab.update_layout(xaxis=dict(tickangle=45,tickfont=dict(size=11)),showlegend=False)

    logs = d.get("logs",[]); nc = sum(1 for e in logs if e["n_warnings"]==0)
    nw = sum(1 for e in logs if e["n_warnings"]>0); tw = sum(e["n_warnings"] for e in logs)
    ns = len(meta) if not meta.empty else df.shape[1]; np_ = df.shape[0]
    td = int(binary.sum().sum()); tp = binary.shape[0]*binary.shape[1]
    comp = td/tp*100 if tp>0 else 0

    return html.Div([
        html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}, children=[
            _sc("Samples",str(ns),C["accent"]), _sc("Features",str(np_),C["accent"]),
            _sc("Completeness",f"{comp:.1f}%",C["green"]),
            _sc("Clean Runs",str(nc),C["green"]),
            _sc("Warnings",str(nw),C["red"] if nw>0 else C["green"]),
        ]),
        html.Div(style=CS, children=[_st("Missingness Heatmap","Green=detected | Red=missing"), dcc.Graph(figure=mhm)]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=mb)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=ch)]),
        ]),
        html.Div(style=CS, children=[_st("Area Distribution"), dcc.Graph(figure=ab)]) if areas is not None else html.Div(),
    ])


# ======================================================================
# TAB: PCA & CLUSTERING (publication quality)
# ======================================================================

def tab_pca(d):
    df = d.get("hptm", d.get("hpf"))
    meta = d.get("metadata", pd.DataFrame())
    if df is None or df.empty:
        return html.Div(style=CS, children=[html.P("No data.")])

    X = df.T.fillna(0)
    n_comp = min(3, X.shape[1], X.shape[0])
    if n_comp < 2:
        return html.Div(style=CS, children=[html.P("Not enough features for PCA.")])

    pca = skPCA(n_components=n_comp)
    coords = pca.fit_transform(X.values)
    ev = pca.explained_variance_ratio_ * 100

    # PCA scatter
    pc_df = pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"Sample":X.index})
    if n_comp>=3: pc_df["PC3"] = coords[:,2]
    pc_df = pc_df.merge(meta, on="Sample")

    fig1 = px.scatter(pc_df, x="PC1", y="PC2", color="Group", hover_name="Sample",
                      symbol="Tissue" if pc_df["Tissue"].nunique()>1 else None,
                      color_discrete_sequence=GC)
    fig1.update_traces(marker=dict(size=11,line=dict(width=1.5,color="white")))
    pfig(fig1, 500)
    fig1.update_layout(xaxis_title=f"PC1 ({ev[0]:.1f}%)", yaxis_title=f"PC2 ({ev[1]:.1f}%)",
                       title=dict(text="PCA - Sample Space",font=dict(size=18)))

    # Add 95% confidence ellipses per group
    for grp in sorted(pc_df["Group"].unique()):
        gd = pc_df[pc_df["Group"]==grp]
        if len(gd) >= 3:
            mx, my = gd["PC1"].mean(), gd["PC2"].mean()
            sx, sy = gd["PC1"].std(), gd["PC2"].std()
            theta = np.linspace(0,2*np.pi,50)
            x_ell = mx + 1.96*sx*np.cos(theta)
            y_ell = my + 1.96*sy*np.sin(theta)
            fig1.add_trace(go.Scatter(x=x_ell,y=y_ell,mode="lines",
                                       line=dict(width=1,dash="dash"),showlegend=False,
                                       opacity=0.4,hoverinfo="skip"))

    # Scree plot
    all_ev = pca.explained_variance_ratio_
    sfig = go.Figure()
    sfig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(all_ev))],
                          y=all_ev*100, marker_color=C["accent"],name="Individual"))
    sfig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(all_ev))],
                              y=np.cumsum(all_ev)*100, mode="lines+markers",
                              marker_color=C["red"],name="Cumulative",yaxis="y2"))
    pfig(sfig, 350)
    sfig.update_layout(yaxis_title="Variance Explained (%)",
                       yaxis2=dict(title="Cumulative %",overlaying="y",side="right",range=[0,105]),
                       title=dict(text="Scree Plot",font=dict(size=18)))

    # Loadings biplot
    loadings = pca.components_[:2].T  # features x 2
    load_df = pd.DataFrame({"Feature":df.index,"PC1":loadings[:,0],"PC2":loadings[:,1]})
    # Top 15 by loading magnitude
    load_df["mag"] = np.sqrt(load_df["PC1"]**2 + load_df["PC2"]**2)
    top_load = load_df.nlargest(15, "mag")

    bfig = go.Figure()
    # Plot samples (faded)
    for grp in sorted(pc_df["Group"].unique()):
        gd = pc_df[pc_df["Group"]==grp]
        bfig.add_trace(go.Scatter(x=gd["PC1"],y=gd["PC2"],mode="markers",name=grp,
                                   marker=dict(size=8,opacity=0.4),showlegend=True))
    # Plot loading arrows
    scale = np.abs(coords[:,:2]).max() / (np.abs(loadings).max()+1e-10) * 0.8
    for _, row in top_load.iterrows():
        bfig.add_annotation(ax=0,ay=0,x=row["PC1"]*scale,y=row["PC2"]*scale,
                            xref="x",yref="y",axref="x",ayref="y",
                            showarrow=True,arrowhead=2,arrowsize=1.5,arrowwidth=1.5,
                            arrowcolor=C["red"])
        bfig.add_annotation(x=row["PC1"]*scale*1.1,y=row["PC2"]*scale*1.1,
                            text=row["Feature"],showarrow=False,font=dict(size=12,color=C["red"]))
    pfig(bfig, 500)
    bfig.update_layout(xaxis_title=f"PC1 ({ev[0]:.1f}%)", yaxis_title=f"PC2 ({ev[1]:.1f}%)",
                       title=dict(text="PCA Biplot - Samples + Top Loadings",font=dict(size=18)))

    # 3D PCA if available
    fig3d = go.Figure()
    if n_comp >= 3:
        fig3d = px.scatter_3d(pc_df, x="PC1", y="PC2", z="PC3", color="Group",
                               hover_name="Sample", color_discrete_sequence=GC)
        fig3d.update_traces(marker=dict(size=6))
        pfig(fig3d, 500)
        fig3d.update_layout(scene=dict(
            xaxis_title=f"PC1 ({ev[0]:.1f}%)", yaxis_title=f"PC2 ({ev[1]:.1f}%)",
            zaxis_title=f"PC3 ({ev[2]:.1f}%)"),
            title=dict(text="3D PCA",font=dict(size=18)))

    # Dendrogram
    try:
        dd = df.T.fillna(0).values
        dist_ = pdist(dd, metric="euclidean")
        link_ = linkage(dist_, method="ward")
        dr = scipy_dend(link_, labels=df.columns.tolist(), no_plot=True)
        dfig = go.Figure()
        for xc, yc in zip(dr["icoord"],dr["dcoord"]):
            dfig.add_trace(go.Scatter(x=xc,y=yc,mode="lines",line=dict(color=C["accent"],width=1.5),showlegend=False))
        tp_ = [5+10*i for i in range(len(dr["ivl"]))]
        dfig.update_layout(template=PUB,height=350,
            xaxis=dict(tickmode="array",tickvals=tp_,ticktext=dr["ivl"],tickangle=45,tickfont=dict(size=11)),
            yaxis_title="Distance (Ward)",title=dict(text="Hierarchical Clustering",font=dict(size=18)),margin=dict(b=120))
    except:
        dfig = go.Figure(); pfig(dfig, 350)

    # Correlation heatmap
    corr = df.corr(method="spearman")
    co_ = cluster_order(corr, 0); corr = corr.loc[co_,co_]
    chm = phm(corr.values, corr.columns.tolist(), corr.index.tolist(),
              cs="RdBu_r", title="Spearman", zmin=-1, zmax=1, h=max(500,len(corr)*12))

    # --- Feature clustering (K-means + dendrogram) ---
    feat_X = df.fillna(0).values  # features x samples matrix
    n_feats = feat_X.shape[0]

    # Feature dendrogram (cluster features by their ratio profiles)
    try:
        feat_dist = pdist(feat_X, metric="euclidean")
        feat_link = linkage(feat_dist, method="ward")
        feat_dr = scipy_dend(feat_link, labels=df.index.tolist(), no_plot=True)
        feat_dfig = go.Figure()
        for xc, yc in zip(feat_dr["icoord"], feat_dr["dcoord"]):
            feat_dfig.add_trace(go.Scatter(x=yc, y=xc, mode="lines",
                line=dict(color=C["green"], width=1.2), showlegend=False))
        tp_f = [5+10*i for i in range(len(feat_dr["ivl"]))]
        feat_dfig.update_layout(template=PUB, height=max(400, n_feats*14),
            yaxis=dict(tickmode="array", tickvals=tp_f, ticktext=feat_dr["ivl"],
                       tickfont=dict(size=adaptive_font(n_feats))),
            xaxis_title="Distance (Ward)",
            title=dict(text="Feature Dendrogram (PTM Clustering)", font=dict(size=18)),
            margin=dict(l=adaptive_margin_l(feat_dr["ivl"])))
    except Exception:
        feat_dfig = go.Figure(); pfig(feat_dfig, 350)

    # K-means clustering of features
    k_vals = min(6, max(2, n_feats // 5))  # auto K
    try:
        km = KMeans(n_clusters=k_vals, random_state=42, n_init=10).fit(feat_X)
        km_labels = km.labels_
        clust_df = pd.DataFrame({"Feature": df.index, "Cluster": km_labels})
        # Feature cluster heatmap: mean ratio per cluster x group
        groups_list = sorted(meta["Group"].unique())
        group_means = pd.DataFrame(index=df.index)
        for g in groups_list:
            samps_g = meta[meta["Group"]==g]["Sample"].tolist()
            cols_g = [c for c in df.columns if c in samps_g]
            if cols_g: group_means[g] = df[cols_g].mean(axis=1)
        group_means["Cluster"] = km_labels
        cluster_means = group_means.groupby("Cluster")[groups_list].mean()

        km_hm = phm(cluster_means.values,
                     [f"Cluster {i}" for i in cluster_means.index],
                     groups_list, cs="Greens",
                     title=f"Feature Clusters (K={k_vals}) x Group Means",
                     h=max(250, k_vals*50))
    except Exception:
        km_hm = go.Figure(); pfig(km_hm, 300)
        clust_df = pd.DataFrame()

    # Biclustering heatmap (features x samples, reordered)
    try:
        Z = df.fillna(0).values
        n_bic = min(4, n_feats, X.shape[0])
        if n_bic >= 2 and Z.shape[0] >= 4 and Z.shape[1] >= 4:
            bic = SpectralBiclustering(n_clusters=n_bic, random_state=42)
            bic.fit(Z)
            row_order = np.argsort(bic.row_labels_)
            col_order = np.argsort(bic.column_labels_)
            Z_reord = Z[row_order][:, col_order]
            feats_reord = [df.index[i] for i in row_order]
            samps_reord = [df.columns[i] for i in col_order]
            bic_hm = phm(Z_reord, feats_reord, samps_reord, cs="YlGnBu",
                         title=f"Biclustering (n={n_bic}) — Features x Samples",
                         h=max(500, n_feats*12))
        else:
            bic_hm = go.Figure(); pfig(bic_hm, 300)
    except Exception:
        bic_hm = go.Figure(); pfig(bic_hm, 300)

    children = [
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=fig1)]),
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=bfig)]),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"350px"}, children=[dcc.Graph(figure=sfig)]),
            html.Div(style={**CS,"flex":"2","minWidth":"500px"}, children=[dcc.Graph(figure=dfig)]),
        ]),
    ]
    if n_comp >= 3:
        children.append(html.Div(style=CS, children=[dcc.Graph(figure=fig3d)]))
    children.append(html.Div(style=CS, children=[_st("Sample Correlation"), dcc.Graph(figure=chm)]))

    # Feature clustering section
    children.append(html.H3("Feature Clustering", style={"color":C["accent"],"marginTop":"32px","marginBottom":"8px","fontSize":"20px"}))
    children.append(html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
        html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=feat_dfig)]),
        html.Div(style={**CS,"flex":"1","minWidth":"350px"}, children=[dcc.Graph(figure=km_hm)]),
    ]))
    children.append(html.Div(style=CS, children=[
        _st("Biclustering","Spectral biclustering — features and samples reordered together"),
        dcc.Graph(figure=bic_hm)]))

    return html.Div(children)


# ======================================================================
# TAB: STATISTICS
# ======================================================================

def _enrich_stats(res, groups):
    """Add classification columns to statistics results."""
    histone_col, ptm_type_col, direction_col = [], [], []
    for _, row in res.iterrows():
        h, t = classify_ptm_name(row["PTM"])
        histone_col.append(h); ptm_type_col.append(t)
        means = [row.get(f"mean_{g}", np.nan) for g in groups]
        means_valid = [(g, m) for g, m in zip(groups, means) if not np.isnan(m) and m > 0]
        if len(means_valid) >= 2:
            max_g = max(means_valid, key=lambda x: x[1])
            min_g = min(means_valid, key=lambda x: x[1])
            fc = np.log2(max_g[1] / (min_g[1] + 1e-10))
            direction_col.append(f"Up in {max_g[0]}" if fc > 0.5 else f"Down in {max_g[0]}" if fc < -0.5 else "Unchanged")
        else:
            direction_col.append("N/A")
    res["Histone"] = histone_col; res["PTM_type"] = ptm_type_col; res["Direction"] = direction_col
    return res


def tab_stats(d):
    df = d.get("hptm", d.get("hpf"))
    meta = d.get("metadata", pd.DataFrame())
    if df is None or meta.empty:
        return html.Div(style=CS, children=[html.P("No data.")])

    # Check if Design column exists (for stratified analysis)
    designs = sorted(meta["Design"].unique()) if "Design" in meta.columns and meta["Design"].nunique() > 1 else []

    # Build filter controls
    filter_children = []
    if designs:
        filter_children.append(html.Div(style={"flex":"1","minWidth":"220px"}, children=[
            _lbl("Design / Strain"),
            dcc.Dropdown(id="stats-design",
                options=[{"label":"All (pooled - CAUTION)","value":"All"}]+[{"label":f"Design {x}","value":str(x)} for x in designs],
                value=str(designs[0]), clearable=False, style=DS)]))
    else:
        filter_children.append(html.Div(dcc.Store(id="stats-design", data="none")))

    filter_children.extend([
        html.Div(style={"flex":"1","minWidth":"160px"}, children=[
            _lbl("Show"),
            dcc.Dropdown(id="stats-show",
                options=[{"label":"All features","value":"all"},{"label":"Significant (FDR < threshold)","value":"sig"},
                         {"label":"Up-regulated","value":"up"},{"label":"Down-regulated / changed","value":"down"}],
                value="all", clearable=False, style=DS)]),
        html.Div(style={"flex":"1","minWidth":"160px"}, children=[
            _lbl("Classify by"),
            dcc.Dropdown(id="stats-classify",
                options=[{"label":"None","value":"none"},{"label":"Histone (H3/H3.3/H4)","value":"histone"},
                         {"label":"PTM type (me/ac/ph)","value":"ptm_type"},{"label":"Direction","value":"direction"}],
                value="none", clearable=False, style=DS)]),
        html.Div(style={"flex":"1","minWidth":"200px"}, children=[
            _lbl("FDR threshold"),
            dcc.Slider(id="stats-fdr", min=-3, max=-1, step=0.1, value=-1.301,
                       marks={-3:"0.001",-2:"0.01",-1.301:"0.05",-1:"0.1"},
                       tooltip={"placement":"bottom","always_visible":False})]),
        html.Div(style={"flex":"0","minWidth":"120px","display":"flex","alignItems":"flex-end"}, children=[
            html.Button("Export CSV", id="stats-export", n_clicks=0,
                        style={"padding":"10px 20px","borderRadius":"8px","border":"none",
                               "backgroundColor":C["accent"],"color":"white","fontWeight":"600",
                               "cursor":"pointer","fontSize":"13px"})]),
    ])

    return html.Div([
        html.Div(style={**CS,"display":"flex","gap":"16px","alignItems":"flex-end","flexWrap":"wrap"},
                 children=filter_children),
        html.Div(id="stats-out"),
    ])


@callback(Output("stats-out","children"),
          Input("stats-design","value"), Input("stats-show","value"),
          Input("stats-classify","value"), Input("stats-fdr","value"),
          Input("cur-exp","data"), prevent_initial_call=True)
def _stats_filtered(design, show, classify, fdr_log, exp):
    t0 = time.time()
    if not exp or exp not in EXP_DATA: return html.P("N/A")
    d = EXP_DATA[exp]
    df = d.get("hptm", d.get("hpf"))
    meta = d.get("metadata", pd.DataFrame())
    if df is None or meta.empty: return html.P("No data.")

    fdr_thresh = 10 ** fdr_log if fdr_log else 0.05

    # Filter to design if applicable
    if design and design not in ("All", "none") and "Design" in meta.columns:
        meta = meta[meta["Design"].astype(str) == str(design)].copy()
        samps = meta["Sample"].tolist()
        cols = [c for c in df.columns if c in samps]; df = df[cols]

    groups = sorted(meta["Group"].unique())
    if len(groups) < 2:
        return html.P(f"Only {len(groups)} group(s). Need >= 2.", style={"color":C["red"],"padding":"20px"})

    res = robust_group_test(df, meta, groups)
    if res.empty: return html.P("Could not compute statistics.", style={"color":C["red"]})

    res = _enrich_stats(res, groups)

    # Compute FC for volcano
    fc_list = []
    for _, row in res.iterrows():
        means = [row.get(f"mean_{g}", np.nan) for g in groups]
        means = [m for m in means if not np.isnan(m) and m > 0]
        if len(means) >= 2:
            fc_list.append(np.log2(max(means) / min(means)))
        else:
            fc_list.append(0)
    res["maxLog2FC"] = fc_list

    n_total = len(res)
    n_sig = int((res["KW_FDR"] < fdr_thresh).sum())
    n_up = int(res["Direction"].str.startswith("Up").sum() & (res["KW_FDR"] < fdr_thresh))
    n_down = n_sig - n_up

    # Apply show filter
    display_res = res.copy()
    if show == "sig":
        display_res = display_res[display_res["KW_FDR"] < fdr_thresh]
    elif show == "up":
        display_res = display_res[(display_res["KW_FDR"] < fdr_thresh) & display_res["Direction"].str.startswith("Up")]
    elif show == "down":
        display_res = display_res[(display_res["KW_FDR"] < fdr_thresh) & ~display_res["Direction"].str.startswith("Up")]

    # Volcano plot colored by classification
    vdf = res[res["maxLog2FC"] > 0].copy()
    vdf["negLog10FDR"] = -np.log10(vdf["KW_FDR"] + 1e-300)
    color_col = classify if classify != "none" else None
    if color_col and color_col in ("histone", "ptm_type", "direction"):
        col_map = {"histone":"Histone","ptm_type":"PTM_type","direction":"Direction"}[color_col]
        vfig = px.scatter(vdf, x="maxLog2FC", y="negLog10FDR", hover_name="PTM",
                          color=col_map, color_discrete_sequence=GC,
                          labels={"maxLog2FC":"Max |log2(FC)|","negLog10FDR":"-log10(FDR)"})
    else:
        vdf["Significant"] = vdf["KW_FDR"] < fdr_thresh
        vfig = px.scatter(vdf, x="maxLog2FC", y="negLog10FDR", hover_name="PTM",
                          color="Significant", color_discrete_map={True:C["red"],False:C["muted"]},
                          labels={"maxLog2FC":"Max |log2(FC)|","negLog10FDR":"-log10(FDR)"})
    pfig(vfig, 480, n_groups=len(groups))
    vfig.add_hline(y=-np.log10(fdr_thresh), line_dash="dash", line_color=C["red"],
                   annotation_text=f"FDR={fdr_thresh:.3f}")
    vfig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    vfig.update_layout(title=dict(text="Volcano Plot (Kruskal-Wallis)", font=dict(size=18)))

    # Top features bar (use display_res to respect show filter)
    bar_res = display_res.head(40)
    if not bar_res.empty:
        n_bar = len(bar_res)
        sbf = go.Figure(go.Bar(
            x=-np.log10(bar_res["KW_FDR"].values + 1e-300),
            y=bar_res["PTM"].tolist(), orientation="h",
            marker=dict(color=-np.log10(bar_res["KW_FDR"].values + 1e-300), colorscale="Reds", line=dict(width=0))))
        pfig(sbf, max(300, n_bar * 20), n_y=n_bar)
        sbf.update_layout(yaxis=dict(autorange="reversed"),
                          margin=dict(l=adaptive_margin_l(bar_res["PTM"].tolist())),
                          xaxis_title="-log10(FDR)",
                          title=dict(text=f"Top Features ({show})", font=dict(size=18)))
    else:
        sbf = go.Figure(); pfig(sbf, 300)

    # Classification summary pie chart
    if classify != "none" and not display_res.empty:
        col_map = {"histone":"Histone","ptm_type":"PTM_type","direction":"Direction"}
        cname = col_map.get(classify, "Histone")
        if cname in display_res.columns:
            vc = display_res[cname].value_counts()
            pief = px.pie(values=vc.values, names=vc.index, color_discrete_sequence=GC,
                          title=f"Distribution by {cname}")
            pfig(pief, 380)
        else:
            pief = go.Figure(); pfig(pief, 100)
    else:
        pief = None

    dur = int((time.time() - t0) * 1000)
    design_label = f"Design {design}" if design not in ("All", "none", None) else "All"
    log_analysis(exp, "kruskal_wallis", {"groups": groups, "design": design_label, "fdr": fdr_thresh, "show": show},
                 n_total, n_sig, f"KW: {n_sig}/{n_total} sig (FDR<{fdr_thresh}), {len(groups)} groups", dur)

    children = [
        html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}, children=[
            _sc("Tested", str(n_total), C["accent"]),
            _sc(f"Significant (FDR<{fdr_thresh:.3f})", str(n_sig), C["red"] if n_sig > 0 else C["green"]),
            _sc("Showing", str(len(display_res)), C["h3"]),
            _sc("Groups", str(len(groups)), C["h4"]),
        ]),
        html.P(f"Analysis: {design_label} | Groups: {', '.join(groups)} | n={len(meta)} samples | {dur}ms",
               style={"color":C["accent"],"fontWeight":"600","fontSize":"14px","marginBottom":"12px"}),
    ]

    row1 = [html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=vfig)])]
    if pief:
        row1.append(html.Div(style={**CS,"flex":"0 0 350px"}, children=[dcc.Graph(figure=pief)]))
    children.append(html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=row1))

    children.append(html.Div(style=CS, children=[
        _st(f"Top Features Bar", f"Showing {len(bar_res)} features | sorted by p-value"),
        dcc.Graph(figure=sbf)]))

    children.append(html.Div(style=CS, children=[
        _st("Statistical Results", f"Kruskal-Wallis + BH FDR | {show} | Editable | Export CSV"),
        make_table(display_res, "stats-table")]))

    return html.Div(children)


@callback(Output("download-data","data"),
          Input("stats-export","n_clicks"), State("cur-exp","data"), prevent_initial_call=True)
def _stats_export(n, exp):
    if not n or not exp or exp not in EXP_DATA: return no_update
    d = EXP_DATA[exp]; df = d.get("hptm", d.get("hpf")); meta = d.get("metadata", pd.DataFrame())
    if df is None: return no_update
    groups = sorted(meta["Group"].unique()) if not meta.empty else []
    if len(groups) < 2: return no_update
    res = robust_group_test(df, meta, groups)
    if res.empty: return no_update
    res = _enrich_stats(res, groups)
    fname = f"{exp.split('(')[0].strip()}_statistics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    log_analysis(exp, "export_stats", {"format":"csv","n_rows":len(res)}, len(res), 0, f"Exported {len(res)} rows")
    return dcc.send_data_frame(res.to_csv, fname, index=False)


# ======================================================================
# TAB: UPSET / CO-OCCURRENCE
# ======================================================================

def tab_upset(d):
    hpf_meta = d.get("hpf_meta", pd.DataFrame())
    hpf = d.get("hpf", pd.DataFrame())
    meta = d.get("metadata", pd.DataFrame())

    if hpf_meta.empty or hpf.empty:
        return html.Div(style=CS, children=[html.P("No peptidoform data for UpSet analysis.")])

    groups = sorted(meta["Group"].unique())

    # ---- UpSet: which PTMs co-occur on the same peptide? ----
    # For each combinatorial hPF, extract the individual PTMs
    combos = hpf_meta[hpf_meta["is_combo"]].copy()

    if combos.empty:
        return html.Div(style=CS, children=[html.P("No combinatorial peptidoforms found.")])

    # Count co-occurrence pairs
    pair_counts = {}
    for _, row in combos.iterrows():
        ptms = row["individual_ptms"]
        if len(ptms) >= 2:
            for p1, p2 in combinations(sorted(ptms), 2):
                key = f"{p1} + {p2}"
                pair_counts[key] = pair_counts.get(key, 0) + 1

    if not pair_counts:
        return html.Div(style=CS, children=[html.P("No PTM co-occurrences found.")])

    # Sort and take top 30
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:30]
    pair_names = [p[0] for p in sorted_pairs]
    pair_vals = [p[1] for p in sorted_pairs]

    # Horizontal bar chart (UpSet-style intersection sizes)
    uf = go.Figure(go.Bar(x=pair_vals, y=pair_names, orientation="h",
                           marker=dict(color=pair_vals,colorscale="Viridis",line=dict(width=0))))
    pfig(uf, max(400, len(pair_names)*20))
    uf.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=13)),
                     margin=dict(l=220),xaxis_title="Co-occurrence Count",
                     title=dict(text="PTM Co-occurrence (from combinatorial hPF)",font=dict(size=18)))

    # ---- Binary detection matrix: which hPTMs are detected per group? ----
    hptm = d.get("hptm")
    if hptm is not None and not meta.empty:
        # For each group, mark PTM as "detected" if mean > threshold
        det_data = {}
        for g in groups:
            samps = meta[meta["Group"]==g]["Sample"].tolist()
            cols = [c for c in hptm.columns if c in samps]
            if cols:
                gm = hptm[cols].mean(axis=1)
                det_data[g] = (gm > 0.001).astype(int)
        det_df = pd.DataFrame(det_data)

        # UpSet-style: count unique detection patterns
        patterns = {}
        for ptm in det_df.index:
            pat = tuple(det_df.loc[ptm].values)
            key = " & ".join([g for g, v in zip(groups, pat) if v == 1])
            if not key: key = "None"
            patterns[key] = patterns.get(key, 0) + 1

        sp = sorted(patterns.items(), key=lambda x: -x[1])[:20]
        pf = go.Figure(go.Bar(x=[p[1] for p in sp], y=[p[0] for p in sp], orientation="h",
                               marker=dict(color=[p[1] for p in sp], colorscale="Plasma",line=dict(width=0))))
        pfig(pf, max(300, len(sp)*22))
        pf.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=12)),
                         margin=dict(l=250),xaxis_title="# PTMs",
                         title=dict(text="PTM Detection Patterns Across Groups",font=dict(size=18)))
    else:
        pf = go.Figure(); pfig(pf, 300)

    # ---- Modification complexity per sample ----
    n_mods_per_sample = []
    for col in hpf.columns:
        vals = hpf[col].dropna()
        n_detected = (vals > 0.001).sum()
        combo_detected = 0
        if not hpf_meta.empty:
            combo_names = hpf_meta[hpf_meta["is_combo"]]["name"].tolist()
            combo_vals = hpf.loc[[n for n in combo_names if n in hpf.index], col]
            combo_detected = (combo_vals > 0.001).sum()
        n_mods_per_sample.append({"Sample":col,"Total_hPF":n_detected,
                                   "Combo_hPF":combo_detected,
                                   "Single_hPF":n_detected - combo_detected})

    ndf = pd.DataFrame(n_mods_per_sample).merge(meta, on="Sample", how="left")
    cf = px.bar(ndf, x="Sample", y=["Single_hPF","Combo_hPF"], color_discrete_sequence=[C["accent"],C["warn"]],
                title="Peptidoform Complexity per Sample", barmode="stack")
    pfig(cf, 380); cf.update_layout(xaxis=dict(tickangle=45,tickfont=dict(size=11)),yaxis_title="# Detected hPF")

    return html.Div([
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=uf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=pf)]),
        ]),
        html.Div(style=CS, children=[dcc.Graph(figure=cf)]),
    ])


# ======================================================================
# TAB: REGION MAP (Lollipop / modification landscape per region)
# ======================================================================

def tab_region(d):
    hpf_meta = d.get("hpf_meta", pd.DataFrame())
    hpf = d.get("hpf", pd.DataFrame())
    hdp_list = d.get("hdp_list", [])
    meta = d.get("metadata", pd.DataFrame())

    if hpf_meta.empty or hpf.empty:
        return html.Div(style=CS, children=[html.P("No peptidoform data for region mapping.")])

    groups = sorted(meta["Group"].unique()) if not meta.empty else []

    # ---- 1. Lollipop: number and type of modifications per region ----
    region_stats = []
    for region in sorted(hpf_meta["region"].unique()):
        rmeta = hpf_meta[hpf_meta["region"] == region]
        n_total = len(rmeta)
        n_combo = int(rmeta["is_combo"].sum())
        n_single = int(((~rmeta["is_combo"]) & (rmeta["modification"] != "unmod")).sum())
        n_unmod = int((rmeta["modification"] == "unmod").sum())
        # Mean ratio across all samples for this region
        rnames = rmeta["name"].tolist()
        rdata = hpf.loc[[n for n in rnames if n in hpf.index]]
        mean_ratio = rdata.values.mean() if not rdata.empty else 0
        # Collect individual PTM types in this region
        all_ptms = []
        for _, row in rmeta.iterrows():
            if isinstance(row.get("individual_ptms"), list):
                all_ptms.extend(row["individual_ptms"])
        unique_ptms = sorted(set(all_ptms))
        # Get histone
        histone = rmeta["histone"].iloc[0] if len(rmeta) > 0 else ""
        # Find hDP header info
        seq = ""
        for hdp in hdp_list:
            if hdp.get("region") == region:
                seq = hdp.get("sequence", ""); break

        region_stats.append({
            "Region": region, "Histone": histone, "Sequence": seq,
            "Total_hPF": n_total, "Single": n_single, "Combo": n_combo,
            "Unmod": n_unmod, "Mean_Ratio": round(mean_ratio, 4),
            "Unique_PTMs": ", ".join(unique_ptms), "n_unique_PTMs": len(unique_ptms),
        })

    rdf = pd.DataFrame(region_stats).sort_values("Region")

    # Lollipop plot: Total modifications per region (stem + dot)
    lf = go.Figure()
    for _, row in rdf.iterrows():
        region = row["Region"]
        # Stem (line from 0 to value)
        lf.add_trace(go.Scatter(
            x=[0, row["Total_hPF"]], y=[region, region],
            mode="lines", line=dict(color=C["accent"], width=2),
            showlegend=False, hoverinfo="skip"))
    # Dots at the end (colored by histone)
    hist_colors = {"H3": C["h3"], "H3.3": C["warn"], "H4": C["h4"], "Other": C["muted"]}
    for hist in rdf["Histone"].unique():
        hd = rdf[rdf["Histone"] == hist]
        lf.add_trace(go.Scatter(
            x=hd["Total_hPF"], y=hd["Region"],
            mode="markers+text", name=hist,
            marker=dict(size=14, color=hist_colors.get(hist, C["accent"]),
                        line=dict(width=2, color="white")),
            text=hd["Total_hPF"].astype(str), textposition="middle right",
            textfont=dict(size=13, color=C["text"])))
    pfig(lf, max(350, len(rdf)*32))
    lf.update_layout(
        xaxis_title="Number of Peptidoforms (hPF)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        margin=dict(l=160, r=40), title=dict(text="Peptidoforms per Region (Lollipop)", font=dict(size=18)))

    # ---- 2. Stacked bar: Single vs Combo vs Unmod per region ----
    sbf = go.Figure()
    sbf.add_trace(go.Bar(x=rdf["Region"], y=rdf["Single"], name="Single-mod",
                          marker_color=C["green"]))
    sbf.add_trace(go.Bar(x=rdf["Region"], y=rdf["Combo"], name="Combinatorial",
                          marker_color=C["warn"]))
    sbf.add_trace(go.Bar(x=rdf["Region"], y=rdf["Unmod"], name="Unmodified",
                          marker_color=C["muted"]))
    pfig(sbf, 400)
    sbf.update_layout(barmode="stack", xaxis=dict(tickangle=45, tickfont=dict(size=12)),
                       yaxis_title="# hPF", title=dict(text="Modification Type per Region", font=dict(size=18)))

    # ---- 3. Lollipop: unique individual PTM marks per region ----
    uf = go.Figure()
    for _, row in rdf.iterrows():
        region = row["Region"]
        uf.add_trace(go.Scatter(
            x=[0, row["n_unique_PTMs"]], y=[region, region],
            mode="lines", line=dict(color=C["h3"], width=2),
            showlegend=False, hoverinfo="skip"))
    uf.add_trace(go.Scatter(
        x=rdf["n_unique_PTMs"], y=rdf["Region"], mode="markers",
        marker=dict(size=12, color=C["h3"], symbol="diamond",
                    line=dict(width=1.5, color="white")),
        text=rdf["Unique_PTMs"], hoverinfo="text+x", name="Unique marks"))
    pfig(uf, max(350, len(rdf)*32))
    uf.update_layout(
        xaxis_title="# Unique PTM Marks",
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        margin=dict(l=160, r=40), title=dict(text="Unique PTM Marks per Region (Lollipop)", font=dict(size=18)))

    # ---- 4. Heatmap: mean ratio per region x group ----
    if groups:
        rg_data = []
        for region in rdf["Region"]:
            rmeta_r = hpf_meta[hpf_meta["region"] == region]
            rnames = rmeta_r["name"].tolist()
            rdata = hpf.loc[[n for n in rnames if n in hpf.index]]
            for g in groups:
                samps = meta[meta["Group"] == g]["Sample"].tolist()
                cols = [c for c in rdata.columns if c in samps]
                if cols:
                    gm = rdata[cols].mean().mean()
                else:
                    gm = np.nan
                rg_data.append({"Region": region, "Group": g, "Mean_Ratio": gm})
        rgdf = pd.DataFrame(rg_data)
        rgp = rgdf.pivot(index="Region", columns="Group", values="Mean_Ratio").fillna(0)
        rghm = phm(rgp.values, rgp.columns.tolist(), rgp.index.tolist(),
                    cs="YlOrRd", title="Mean Ratio", h=max(300, len(rgp)*28))
        rghm.update_layout(title=dict(text="Mean hPF Ratio: Region x Group", font=dict(size=18)))
    else:
        rghm = go.Figure(); pfig(rghm, 300)

    # ---- 5. Faceted box: top regions by group ----
    if groups:
        top_regions = rdf.nlargest(6, "Total_hPF")["Region"].tolist()
        bml = []
        for region in top_regions:
            rmeta_r = hpf_meta[hpf_meta["region"] == region]
            rnames = rmeta_r["name"].tolist()
            rdata = hpf.loc[[n for n in rnames if n in hpf.index]]
            for col in rdata.columns:
                v = rdata[col].mean()
                bml.append({"Region": region, "Sample": col, "Mean_hPF_Ratio": v})
        bmdf = pd.DataFrame(bml).merge(meta, on="Sample", how="left")
        rbf = px.box(bmdf, x="Group", y="Mean_hPF_Ratio", color="Group", facet_col="Region",
                      facet_col_wrap=3, points="all", color_discrete_sequence=GC)
        pfig(rbf, 550)
        rbf.update_layout(showlegend=False)
        rbf.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    else:
        rbf = go.Figure(); pfig(rbf, 300)

    return html.Div([
        html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}, children=[
            _sc("Regions", str(len(rdf)), C["accent"]),
            _sc("Total hPF", str(int(rdf["Total_hPF"].sum())), C["h3"]),
            _sc("Unique PTM Marks", str(int(rdf["n_unique_PTMs"].sum())), C["warn"]),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=lf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=uf)]),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=sbf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=rghm)]),
        ]),
        html.Div(style=CS, children=[
            _st("Top Regions by Group","Faceted box plots for the 6 most diverse regions"),
            dcc.Graph(figure=rbf)]),
        html.Div(style=CS, children=[
            _st("Region Summary Table","Peptide regions with modification counts"),
            make_table(rdf, "region-table")]),
    ])


# ======================================================================
# TAB: COMPARISONS
# ======================================================================

def tab_cmp(d):
    meta = d.get("metadata", pd.DataFrame())
    if meta.empty: return html.Div(style=CS, children=[html.P("No metadata.")])
    groups = sorted(meta["Group"].unique())
    df = d.get("hptm", d.get("hpf"))
    if df is None: return html.Div(style=CS, children=[html.P("No data.")])
    if len(groups)<2: return html.Div(style=CS, children=[html.P("Need >=2 groups.")])

    designs = sorted(meta["Design"].unique()) if "Design" in meta.columns and meta["Design"].nunique() > 1 else []

    filter_children = [
        html.Div(style={"flex":"1","minWidth":"180px"}, children=[
            _lbl("Group A"),
            dcc.Dropdown(id="cmp-a",options=[{"label":g,"value":g} for g in groups],
                         value=groups[0],clearable=False,style=DS)]),
        html.Div(style={"flex":"1","minWidth":"180px"}, children=[
            _lbl("Group B"),
            dcc.Dropdown(id="cmp-b",options=[{"label":g,"value":g} for g in groups],
                         value=groups[1] if len(groups)>1 else groups[0],clearable=False,style=DS)]),
        html.Div(style={"flex":"1","minWidth":"160px"}, children=[
            _lbl("Data Level"),
            dcc.Dropdown(id="cmp-level",options=[{"label":"hPTM (single)","value":"hptm"},
                         {"label":"hPF (peptidoforms)","value":"hpf"}],
                         value="hptm",clearable=False,style=DS)]),
        html.Div(style={"flex":"1","minWidth":"140px"}, children=[
            _lbl("Show"),
            dcc.Dropdown(id="cmp-show",
                options=[{"label":"All features","value":"all"},{"label":"Significant only","value":"sig"},
                         {"label":"Up-regulated","value":"up"},{"label":"Down-regulated","value":"down"}],
                value="all",clearable=False,style=DS)]),
        html.Div(style={"flex":"1","minWidth":"140px"}, children=[
            _lbl("Classify by"),
            dcc.Dropdown(id="cmp-classify",
                options=[{"label":"None","value":"none"},{"label":"Histone","value":"histone"},
                         {"label":"PTM type","value":"ptm_type"},{"label":"Significance","value":"sig"}],
                value="none",clearable=False,style=DS)]),
        html.Div(style={"flex":"1","minWidth":"180px"}, children=[
            _lbl("FDR threshold"),
            dcc.Slider(id="cmp-fdr", min=-3, max=-1, step=0.1, value=-1.301,
                       marks={-3:"0.001",-2:"0.01",-1.301:"0.05",-1:"0.1"},
                       tooltip={"placement":"bottom","always_visible":False})]),
        html.Div(style={"flex":"0","minWidth":"100px","display":"flex","alignItems":"flex-end","gap":"6px"}, children=[
            html.Button("Export CSV", id="cmp-export", n_clicks=0,
                        style={"padding":"10px 16px","borderRadius":"8px","border":"none",
                               "backgroundColor":C["accent"],"color":"white","fontWeight":"600",
                               "cursor":"pointer","fontSize":"13px"})]),
    ]
    if designs:
        filter_children.append(
            html.Div(style={"flex":"1","minWidth":"200px"}, children=[
                _lbl("Design / Strain"),
                html.P("Select groups from same strain!", style={"color":C["red"],"fontSize":"11px","margin":"2px 0","fontWeight":"600"}),
            ])
        )

    return html.Div([
        html.Div(style={**CS,"display":"flex","gap":"16px","flexWrap":"wrap","alignItems":"flex-end"},
                 children=filter_children),
        html.Div(id="cmp-out"),
    ])

@callback(Output("cmp-out","children"),
          Input("cmp-a","value"),Input("cmp-b","value"),Input("cmp-level","value"),
          Input("cmp-show","value"),Input("cmp-classify","value"),Input("cmp-fdr","value"),
          Input("cur-exp","data"))
def _cmp(ga, gb, level, show, classify, fdr_log, exp):
    t0 = time.time()
    if not exp or exp not in EXP_DATA: return html.P("N/A")
    d = EXP_DATA[exp]
    df = d.get(level, d.get("hptm", d.get("hpf")))
    meta = d["metadata"]
    if df is None or df.empty: return html.P("No data for level.")

    fdr_thresh = 10 ** fdr_log if fdr_log else 0.05

    mw = pairwise_mw(df, meta, ga, gb)
    if mw.empty: return html.P("Could not compute comparison.")

    # Add classification columns
    h_col, t_col = [], []
    for _, row in mw.iterrows():
        h, t = classify_ptm_name(row["PTM"])
        h_col.append(h); t_col.append(t)
    mw["Histone"] = h_col; mw["PTM_type"] = t_col

    # Direction based on log2FC
    mw["Direction"] = mw["log2FC"].apply(
        lambda x: "Up" if x > 0.5 else "Down" if x < -0.5 else "Unchanged")
    mw["Significant"] = mw["FDR"] < fdr_thresh

    n_total = len(mw)
    n_sig = int(mw["Significant"].sum())
    n_up = int(((mw["Direction"]=="Up") & mw["Significant"]).sum())
    n_down = int(((mw["Direction"]=="Down") & mw["Significant"]).sum())

    # Apply show filter for display
    display_mw = mw.copy()
    if show == "sig":
        display_mw = display_mw[display_mw["Significant"]]
    elif show == "up":
        display_mw = display_mw[(display_mw["Direction"]=="Up") & display_mw["Significant"]]
    elif show == "down":
        display_mw = display_mw[(display_mw["Direction"]=="Down") & display_mw["Significant"]]

    # Volcano — color by classification or significance
    mw["negLog10FDR"] = -np.log10(mw["FDR"]+1e-300)
    if classify != "none":
        col_map = {"histone":"Histone","ptm_type":"PTM_type","sig":"Direction"}
        cname = col_map.get(classify, "Histone")
        vf = px.scatter(mw, x="log2FC", y="negLog10FDR", hover_name="PTM",
                        color=cname, color_discrete_sequence=GC,
                        labels={"log2FC":f"log2(FC) {gb}/{ga}","negLog10FDR":"-log10(FDR)"},
                        title=f"Volcano: {gb} vs {ga} | Mann-Whitney U + FDR")
    else:
        color_vals = ["Up" if r["Direction"]=="Up" and r["Significant"] else
                      "Down" if r["Direction"]=="Down" and r["Significant"] else "NS"
                      for _, r in mw.iterrows()]
        mw["_Color"] = color_vals
        cmap = {"Up":C["red"],"Down":"#2563eb","NS":C["muted"]}
        vf = px.scatter(mw, x="log2FC", y="negLog10FDR", hover_name="PTM",
                        color="_Color", color_discrete_map=cmap,
                        labels={"log2FC":f"log2(FC) {gb}/{ga}","negLog10FDR":"-log10(FDR)"},
                        title=f"Volcano: {gb} vs {ga} | Mann-Whitney U + FDR")
    pfig(vf, 480, n_groups=2)
    vf.add_hline(y=-np.log10(fdr_thresh),line_dash="dash",line_color=C["red"],
                 annotation_text=f"FDR={fdr_thresh:.3f}")
    vf.add_vline(x=0,line_dash="dash",line_color=C["muted"])
    vf.update_traces(marker=dict(size=9,line=dict(width=0.5,color="white")))

    # FC bar (use display_mw to respect show filter)
    fc_sorted = display_mw.sort_values("log2FC")
    n_fc = len(fc_sorted)
    colors = [C["green"] if v>0.5 else C["red"] if v<-0.5 else C["muted"] for v in fc_sorted["log2FC"]]
    ff = go.Figure(go.Bar(x=fc_sorted["log2FC"].values, y=fc_sorted["PTM"].tolist(),
                           orientation="h", marker_color=colors))
    pfig(ff, max(400, n_fc*16), n_y=n_fc)
    ff.update_layout(yaxis=dict(autorange="reversed"),
                     margin=dict(l=adaptive_margin_l(fc_sorted["PTM"].tolist())),
                     xaxis_title=f"log2(FC) {gb}/{ga}",
                     title=dict(text=f"Fold Change ({show})", font=dict(size=18)))
    ff.add_vline(x=0,line_color=C["muted"],line_dash="dash")

    # MA plot
    mw["A"] = 0.5*(np.log2(mw["mean_A"]+1e-8)+np.log2(mw["mean_B"]+1e-8))
    if classify != "none":
        col_map2 = {"histone":"Histone","ptm_type":"PTM_type","sig":"Direction"}
        cname2 = col_map2.get(classify, "Histone")
        maf = px.scatter(mw, x="A", y="log2FC", hover_name="PTM", color=cname2,
                         color_discrete_sequence=GC,
                         title="MA Plot",labels={"A":"Avg Intensity","log2FC":"log2(FC)"})
    else:
        maf = px.scatter(mw, x="A", y="log2FC", hover_name="PTM", color="_Color",
                         color_discrete_map={"Up":C["red"],"Down":"#2563eb","NS":C["muted"]},
                         title="MA Plot",labels={"A":"Avg Intensity","log2FC":"log2(FC)"})
    pfig(maf, 400); maf.add_hline(y=0,line_color=C["muted"],line_dash="dash")

    dur = int((time.time()-t0)*1000)
    log_analysis(exp, "mann_whitney", {"ga":ga,"gb":gb,"level":level,"fdr":fdr_thresh,"show":show},
                 n_total, n_sig, f"MW: {ga} vs {gb} | {n_sig}/{n_total} sig, {n_up} up, {n_down} down", dur)

    # Table columns to display
    tbl_cols = ["PTM","log2FC","pval","FDR","mean_A","mean_B","median_A","median_B","Histone","PTM_type","Direction"]
    tbl_cols = [c for c in tbl_cols if c in display_mw.columns]

    return html.Div([
        html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}, children=[
            _sc("Tested",str(n_total),C["accent"]),
            _sc(f"Sig (FDR<{fdr_thresh:.3f})",str(n_sig),C["red"] if n_sig>0 else C["green"]),
            _sc("Up",str(n_up),C["red"]),
            _sc("Down",str(n_down),"#2563eb"),
            _sc("Showing",str(len(display_mw)),C["h3"]),
        ]),
        html.P(f"{gb} vs {ga} | {level} | {dur}ms",
               style={"color":C["accent"],"fontWeight":"600","fontSize":"14px","marginBottom":"12px"}),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=vf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=ff)]),
        ]),
        html.Div(style=CS, children=[dcc.Graph(figure=maf)]),
        html.Div(style=CS, children=[
            _st("Mann-Whitney Results",f"FDR-corrected | {show} | Editable | Classified"),
            make_table(display_mw[tbl_cols],"cmp-table")]),
    ])


@callback(Output("download-data","data", allow_duplicate=True),
          Input("cmp-export","n_clicks"),
          State("cmp-a","value"),State("cmp-b","value"),State("cmp-level","value"),
          State("cur-exp","data"), prevent_initial_call=True)
def _cmp_export(n, ga, gb, level, exp):
    if not n or not exp or exp not in EXP_DATA: return no_update
    d = EXP_DATA[exp]; df = d.get(level, d.get("hptm")); meta = d["metadata"]
    if df is None: return no_update
    mw = pairwise_mw(df, meta, ga, gb)
    if mw.empty: return no_update
    h_col, t_col = [], []
    for _, row in mw.iterrows():
        h, t = classify_ptm_name(row["PTM"]); h_col.append(h); t_col.append(t)
    mw["Histone"] = h_col; mw["PTM_type"] = t_col
    mw["Direction"] = mw["log2FC"].apply(lambda x: "Up" if x>0.5 else "Down" if x<-0.5 else "Unchanged")
    fname = f"{exp.split('(')[0].strip()}_{ga}_vs_{gb}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    log_analysis(exp, "export_comparison", {"ga":ga,"gb":gb,"level":level}, len(mw), 0, f"Exported {ga} vs {gb}")
    return dcc.send_data_frame(mw.to_csv, fname, index=False)


# ======================================================================
# TAB: PHENODATA / METADATA
# ======================================================================

def tab_pheno(d):
    meta = d.get("metadata", pd.DataFrame())
    phenodata = d.get("phenodata")
    hptm = d.get("hptm")
    hpf = d.get("hpf", pd.DataFrame())

    if meta.empty:
        return html.Div(style=CS, children=[html.P("No metadata available.")])

    # ---- 1. Phenodata table (editable) ----
    # Enrich metadata with processing info
    enriched = meta.copy()
    # Add processing order (based on position in sample list)
    enriched.insert(0, "Order", range(1, len(enriched)+1))
    # Add batch info if available
    if "Batch" not in enriched.columns:
        enriched["Batch"] = "A"

    # Add data quality columns from hPTM if available
    if hptm is not None:
        det_counts = []
        total_ratios = []
        for s in enriched["Sample"]:
            if s in hptm.columns:
                vals = hptm[s].dropna()
                det_counts.append(int((vals > 0).sum()))
                total_ratios.append(round(vals.sum(), 4))
            else:
                det_counts.append(0)
                total_ratios.append(0)
        enriched["Detected_hPTMs"] = det_counts
        enriched["Sum_Ratios"] = total_ratios

    if not hpf.empty:
        hpf_det = []
        for s in enriched["Sample"]:
            if s in hpf.columns:
                vals = hpf[s].dropna()
                hpf_det.append(int((vals > 0).sum()))
            else:
                hpf_det.append(0)
        enriched["Detected_hPF"] = hpf_det

    # ---- 2. Group distribution plot ----
    gc = meta["Group"].value_counts().reset_index()
    gc.columns = ["Group", "Count"]
    gbar = px.bar(gc, x="Group", y="Count", color="Group", color_discrete_sequence=GC,
                   text="Count", title="Samples per Group")
    pfig(gbar, 350)
    gbar.update_traces(textposition="outside")
    gbar.update_layout(showlegend=False)

    # ---- 3. Tissue distribution ----
    if "Tissue" in meta.columns and meta["Tissue"].nunique() > 1:
        tc = meta.groupby(["Group","Tissue"]).size().reset_index(name="Count")
        tbar = px.bar(tc, x="Group", y="Count", color="Tissue", barmode="stack",
                       color_discrete_sequence=GC[5:], title="Tissue Distribution by Group")
        pfig(tbar, 350)
    else:
        tbar = go.Figure(); pfig(tbar, 100)
        tbar.update_layout(height=100)

    # ---- 4. Batch / processing order plot ----
    if hptm is not None:
        # Sum of ratios per sample in processing order
        order_df = enriched[["Order","Sample","Group","Sum_Ratios"]].copy()
        obf = px.scatter(order_df, x="Order", y="Sum_Ratios", color="Group",
                          hover_name="Sample", color_discrete_sequence=GC,
                          title="Total PTM Signal vs Processing Order (batch effect check)")
        pfig(obf, 380)
        obf.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    else:
        obf = go.Figure(); pfig(obf, 100)

    # ---- 5. Detection quality per group ----
    if "Detected_hPTMs" in enriched.columns:
        dbf = px.box(enriched, x="Group", y="Detected_hPTMs", color="Group", points="all",
                      color_discrete_sequence=GC, title="hPTMs Detected per Sample by Group")
        pfig(dbf, 380)
        dbf.update_layout(showlegend=False)
    else:
        dbf = go.Figure(); pfig(dbf, 100)

    # ---- 6. If raw phenodata file exists, show it too ----
    pheno_section = []
    if phenodata is not None and not phenodata.empty:
        pheno_section = [
            html.Div(style=CS, children=[
                _st("Raw Phenodata File", "Original phenodata TSV as loaded"),
                make_table(phenodata, "raw-pheno-table"),
            ])
        ]

    children = [
        html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}, children=[
            _sc("Samples", str(len(meta)), C["accent"]),
            _sc("Groups", str(meta["Group"].nunique()), C["h4"]),
            _sc("Tissues", str(meta["Tissue"].nunique()) if "Tissue" in meta.columns else "1", C["green"]),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=gbar)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=tbar)]) if meta.get("Tissue", pd.Series()).nunique() > 1 else html.Div(),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=obf)]) if hptm is not None else html.Div(),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=dbf)]) if "Detected_hPTMs" in enriched.columns else html.Div(),
        ]),
        html.Div(style=CS, children=[
            _st("Sample Metadata Table", "Editable | Processing order, groups, batch, data quality metrics"),
            make_table(enriched, "pheno-table"),
        ]),
    ] + pheno_section

    return html.Div(children)


# ======================================================================
# TAB: SAMPLE BROWSER
# ======================================================================

def tab_browse(d):
    folders = d.get("sample_folders",[])
    if not folders: return html.Div(style=CS, children=[html.P("No sample folders.")])
    return html.Div([
        html.Div(style={**CS,"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={"flex":"1","minWidth":"200px"}, children=[
                _lbl("Sample"),
                dcc.Dropdown(id="br-s",options=[{"label":f,"value":f} for f in folders],
                             value=folders[0],clearable=False,style=DS)]),
            html.Div(style={"flex":"1","minWidth":"200px"}, children=[
                _lbl("PDF"),dcc.Dropdown(id="br-p",style=DS)]),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"2","minWidth":"500px"},id="br-v"),
            html.Div(style={**CS,"flex":"1","minWidth":"300px"},id="br-i")]),
    ])

@callback(Output("br-p","options"),Output("br-p","value"),Input("br-s","value"),Input("cur-exp","data"))
def _bp(f, exp):
    if not f or not exp or exp not in EXP_DATA: return [],None
    ld = EXP_DATA[exp]["layouts_dir"]
    p = os.path.join(ld, f)
    pdfs = sorted([x for x in os.listdir(p) if x.endswith(".pdf")]) if os.path.isdir(p) else []
    return [{"label":x,"value":x} for x in pdfs], pdfs[0] if pdfs else None

@callback(Output("br-v","children"),Input("br-s","value"),Input("br-p","value"),Input("cur-exp","data"))
def _bv(f, pn, exp):
    if not f or not pn or not exp or exp not in EXP_DATA:
        return html.P("Select sample and PDF",style={"color":C["muted"]})
    pp = os.path.join(EXP_DATA[exp]["layouts_dir"], f, pn)
    if not os.path.exists(pp): return html.P("Not found",style={"color":C["red"]})
    with open(pp,"rb") as fh: enc = base64.b64encode(fh.read()).decode()
    return html.Div([_st(pn.replace(".pdf",""),"XIC chromatogram"),
        html.Iframe(src=f"data:application/pdf;base64,{enc}",
                    style={"width":"100%","height":"550px","border":"none","borderRadius":"8px"})])

@callback(Output("br-i","children"),Input("br-s","value"),Input("cur-exp","data"))
def _bi(f, exp):
    if not f or not exp or exp not in EXP_DATA: return html.P("Select sample")
    d = EXP_DATA[exp]; sn = "_".join(f.split("_")[1:]) if "_" in f else f
    ref = d.get("hptm", d.get("hpf"))
    if ref is None: return html.P("No data")
    mc = None
    for col in ref.columns:
        if sn in col or col in sn: mc = col; break
    if not mc: return html.P(f"'{sn}' not matched",style={"color":C["muted"]})
    vals = ref[mc].dropna(); vals = vals[vals!=0]
    if vals.empty: return html.P("No non-zero values")
    fig = go.Figure(go.Bar(x=vals.values,y=vals.index.tolist(),orientation="h",
                            marker=dict(color=vals.values,colorscale="Viridis",line=dict(width=0))))
    pfig(fig, max(300,len(vals)*16))
    fig.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=12)),margin=dict(l=150,t=10),xaxis_title="Ratio")
    return html.Div([_st("PTM Profile",f"Sample: {f}"), dcc.Graph(figure=fig)])


# ======================================================================
# TAB: ANALYSIS LOG
# ======================================================================

def tab_log(d, exp):
    """Analysis history, session info, upload log."""
    session = get_session_info()
    history = get_analysis_history(experiment=exp, limit=100)
    uploads = get_upload_history(limit=50)

    # Session info card
    sess_card = html.Div(style={**CS,"display":"flex","gap":"16px","flexWrap":"wrap","marginBottom":"16px"}, children=[
        _sc("Session ID", str(session.get("id","?")), C["accent"]),
        _sc("Actions", str(session.get("n_actions",0)), C["green"]),
        _sc("Started", session.get("started","?")[:19] if session.get("started") else "?", C["h3"]),
        _sc("Last Activity", session.get("last_activity","?")[:19] if session.get("last_activity") else "?", C["h4"]),
    ])

    # Analysis history table
    if history:
        hist_df = pd.DataFrame(history,
            columns=["ID","Experiment","Type","Parameters","Features","Significant","Summary","Duration(ms)","Timestamp"])
        hist_table = html.Div(style=CS, children=[
            _st("Analysis History", f"Last {len(hist_df)} analyses for {exp}"),
            make_table(hist_df, "log-hist-table")])
    else:
        hist_table = html.Div(style=CS, children=[html.P("No analyses recorded yet.", style={"color":C["muted"]})])

    # Upload history table
    if uploads:
        upl_df = pd.DataFrame(uploads,
            columns=["ID","Experiment","Filename","Type","Rows","Cols","Status","Message","Timestamp"])
        upl_table = html.Div(style=CS, children=[
            _st("Upload History", f"Last {len(upl_df)} uploads"),
            make_table(upl_df, "log-upl-table")])
    else:
        upl_table = html.Div(style=CS, children=[html.P("No uploads recorded yet.", style={"color":C["muted"]})])

    # Analysis type breakdown chart
    if history:
        type_counts = hist_df["Type"].value_counts()
        pie_fig = px.pie(values=type_counts.values, names=type_counts.index,
                         color_discrete_sequence=GC, title="Analysis Types")
        pfig(pie_fig, 350)

        # Timeline of analyses
        hist_df["ts"] = pd.to_datetime(hist_df["Timestamp"], errors="coerce")
        timeline = px.scatter(hist_df.dropna(subset=["ts"]), x="ts", y="Type",
                              size="Features", color="Type", hover_data=["Summary","Significant"],
                              color_discrete_sequence=GC, title="Analysis Timeline")
        pfig(timeline, 350)

        charts = html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"350px"}, children=[dcc.Graph(figure=pie_fig)]),
            html.Div(style={**CS,"flex":"2","minWidth":"500px"}, children=[dcc.Graph(figure=timeline)]),
        ])
    else:
        charts = html.Div()

    # Export log button
    export_btn = html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px"}, children=[
        html.Button("Export Analysis Log CSV", id="log-export", n_clicks=0,
                    style={"padding":"10px 20px","borderRadius":"8px","border":"none",
                           "backgroundColor":C["accent"],"color":"white","fontWeight":"600",
                           "cursor":"pointer","fontSize":"13px"}),
    ])

    return html.Div([sess_card, export_btn, charts, hist_table, upl_table])


@callback(Output("download-data","data", allow_duplicate=True),
          Input("log-export","n_clicks"), State("cur-exp","data"), prevent_initial_call=True)
def _log_export(n, exp):
    if not n: return no_update
    history = get_analysis_history(experiment=exp, limit=1000)
    if not history: return no_update
    hist_df = pd.DataFrame(history,
        columns=["ID","Experiment","Type","Parameters","Features","Significant","Summary","Duration(ms)","Timestamp"])
    fname = f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return dcc.send_data_frame(hist_df.to_csv, fname, index=False)


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EpiProfile-Plants Dashboard v3.5")
    print(f"  Experiments: {len(EXP_DATA)}")
    for n in EXP_DATA: print(f"    * {n}")
    print(f"\n  =>  http://localhost:{args.port}")
    print("="*60 + "\n")
    app.run(debug=False, port=args.port, host=args.host)
