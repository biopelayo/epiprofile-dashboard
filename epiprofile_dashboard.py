"""
EpiProfile-Plants Dashboard v3.11 -- Publication-Quality Visualization
=====================================================================
Interactive Dash/Plotly dashboard for EpiProfile-Plants output.

Properly classifies histone data into three levels:
  hPTM  = individual PTM marks         (from histone_ratios_single_PTMs.xls)
  hPF   = peptidoforms (combinatorial) (from histone_ratios.xls, data rows)
  hDP   = derivatized peptide regions  (from histone_ratios.xls, headers)
  SeqVar = sequence variants per region (from histone_ratios.xls, variant block)

Features: SQLite analysis tracking, biclustering, data export, adaptive sizing,
          full upload validation, analysis logging, classification filters,
          areas normalization (log2+QN), R-ready export bundles.

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
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, mannwhitneyu, kruskal, fisher_exact
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
# SVG ICON LIBRARY  -- green line-art scientific icons (data-URI)
# ======================================================================

import urllib.parse as _urlparse

def _svg_icon(key, size=16, extra_style=None):
    """Return html.Img for an SVG icon by key. Falls back to empty span."""
    svg = ICONS.get(key)
    if not svg:
        return html.Span("")
    encoded = _urlparse.quote(svg)
    st = {"width": f"{size}px", "height": f"{size}px",
          "verticalAlign": "middle", "display": "inline-block"}
    if extra_style:
        st.update(extra_style)
    return html.Img(src=f"data:image/svg+xml,{encoded}", style=st)

_S = 'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'

ICONS = {
    # -- Histone / peptidoform: barrel + 3 PTM dots --
    "peptidoform": f'<svg {_S}><rect x="4" y="6" width="16" height="12" rx="4"/><circle cx="8" cy="12" r="1.3" fill="#22c55e"/><circle cx="12" cy="12" r="1.3" fill="#22c55e"/><circle cx="16" cy="12" r="1.3" fill="#22c55e"/></svg>',
    # -- Single PTM mark: circle with radiating dot --
    "ptm": f'<svg {_S}><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5" fill="#22c55e"/><line x1="12" y1="3" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="21"/><line x1="3" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="21" y2="12"/></svg>',
    # -- QC badge: shield + check --
    "qc": f'<svg {_S}><path d="M12 2L3 7v5c0 5.5 3.8 10.3 9 12 5.2-1.7 9-6.5 9-12V7z"/><polyline points="9,12 11,14 15,10"/></svg>',
    # -- PCA scatter: axis + 3 cluster groups --
    "pca": f'<svg {_S}><line x1="3" y1="21" x2="3" y2="3"/><line x1="3" y1="21" x2="21" y2="21"/><circle cx="7" cy="16" r="1.2" fill="#22c55e"/><circle cx="8" cy="14" r="1.2" fill="#22c55e"/><circle cx="9" cy="15" r="1.2" fill="#22c55e"/><circle cx="14" cy="8" r="1.2" fill="#16a34a"/><circle cx="16" cy="9" r="1.2" fill="#16a34a"/><circle cx="15" cy="7" r="1.2" fill="#16a34a"/><circle cx="18" cy="15" r="1.2" fill="#4ade80"/><circle cx="19" cy="13" r="1.2" fill="#4ade80"/></svg>',
    # -- Bar chart + asterisk: statistical bars --
    "bar_chart": f'<svg {_S}><rect x="4" y="14" width="3" height="7" rx="0.5"/><rect x="10" y="8" width="3" height="13" rx="0.5"/><rect x="16" y="4" width="3" height="17" rx="0.5"/><text x="19" y="6" font-size="8" fill="#22c55e" stroke="none" font-weight="bold">*</text></svg>',
    # -- Venn: 2 overlapping circles --
    "venn": f'<svg {_S}><circle cx="9" cy="12" r="6" opacity="0.7"/><circle cx="15" cy="12" r="6" opacity="0.7"/></svg>',
    # -- Region map: horizontal line + 3 ticks --
    "region": f'<svg {_S}><line x1="2" y1="12" x2="22" y2="12"/><line x1="6" y1="8" x2="6" y2="16"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="18" y1="8" x2="18" y2="16"/><circle cx="6" cy="8" r="1" fill="#22c55e"/><circle cx="12" cy="8" r="1" fill="#4ade80"/><circle cx="18" cy="8" r="1" fill="#16a34a"/></svg>',
    # -- Balance/comparisons: scale --
    "balance": f'<svg {_S}><line x1="12" y1="3" x2="12" y2="21"/><line x1="4" y1="7" x2="20" y2="7"/><path d="M4 7l2 7h-4z"/><path d="M20 7l2 7h-4z"/><line x1="8" y1="21" x2="16" y2="21"/></svg>',
    # -- Data table: 3x3 grid --
    "table": f'<svg {_S}><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>',
    # -- Search/magnifier + peak --
    "search": f'<svg {_S}><circle cx="10" cy="10" r="6"/><line x1="14.5" y1="14.5" x2="21" y2="21"/><polyline points="7,12 9,8 11,11 13,9" stroke-width="1.5"/></svg>',
    # -- R logo --
    "r_logo": f'<svg {_S}><circle cx="12" cy="12" r="9"/><text x="8" y="17" font-size="14" fill="#22c55e" stroke="none" font-weight="bold" font-family="serif">R</text></svg>',
    # -- Spectra compare: two overlapping peaks --
    "spectra_compare": f'<svg {_S}><polyline points="2,18 5,18 7,6 9,18 12,18" stroke="#22c55e"/><polyline points="10,18 13,18 15,8 17,18 22,18" stroke="#16a34a" stroke-dasharray="3,2"/></svg>',
    # -- Analysis log: clock + list --
    "log": f'<svg {_S}><circle cx="8" cy="8" r="5"/><polyline points="8,5 8,8 10,9"/><line x1="16" y1="6" x2="22" y2="6"/><line x1="16" y1="10" x2="22" y2="10"/><line x1="14" y1="14" x2="22" y2="14"/><line x1="14" y1="18" x2="22" y2="18"/></svg>',
    # -- Plant/Arabidopsis rosette: 5-petal leaf --
    "plant": f'<svg {_S}><path d="M12 21c0-6-4-9-4-14a4 4 0 018 0c0 5-4 8-4 14z"/><path d="M12 21c-3-2-7-3-9-1" /><path d="M12 21c3-2 7-3 9-1"/><line x1="12" y1="21" x2="12" y2="7"/></svg>',
    # -- DNA helix --
    "dna": f'<svg {_S}><path d="M6 3c0 4 12 4 12 8s-12 4-12 8"/><path d="M18 3c0 4-12 4-12 8s12 4 12 8"/><line x1="7" y1="7" x2="17" y2="7" stroke-width="1.2"/><line x1="7" y1="11" x2="17" y2="11" stroke-width="1.2"/><line x1="7" y1="15" x2="17" y2="15" stroke-width="1.2"/></svg>',
    # -- Microscope/LC-MS --
    "microscope": f'<svg {_S}><circle cx="12" cy="6" r="3"/><line x1="12" y1="9" x2="12" y2="16"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="7" y1="20" x2="17" y2="20"/><line x1="10" y1="16" x2="14" y2="16"/><line x1="12" y1="16" x2="12" y2="20"/></svg>',
    # -- Seedling: growth stages --
    "seedling": f'<svg {_S}><path d="M12 22v-8"/><path d="M12 14c-3-1-5-4-5-7a5 5 0 0110 0c0 3-2 6-5 7z"/><path d="M8 18c-2 0-3-1-3-3"/><path d="M16 18c2 0 3-1 3-3"/></svg>',
    # -- Flask/experiment: Erlenmeyer --
    "flask": f'<svg {_S}><path d="M9 3h6"/><path d="M10 3v6l-5 9a1 1 0 001 1h12a1 1 0 001-1l-5-9V3"/><path d="M7 17h10" stroke-dasharray="2,2"/></svg>',
    # -- Palette: 3 color dots --
    "palette": f'<svg {_S}><circle cx="8" cy="10" r="3" fill="#22c55e" stroke="none"/><circle cx="14" cy="8" r="3" fill="#16a34a" stroke="none"/><circle cx="16" cy="14" r="3" fill="#4ade80" stroke="none"/><path d="M20 12a8 8 0 10-16 0 8 8 0 0016 0z"/></svg>',
    # -- Folder --
    "folder": f'<svg {_S}><path d="M3 6a2 2 0 012-2h4l2 2h8a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2z"/></svg>',
    # -- Dashboard/grid: 4 panels --
    "dashboard": f'<svg {_S}><rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="4" rx="1"/><rect x="13" y="9" width="8" height="12" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/></svg>',
    # -- Fire/trending: upward arrow --
    "fire": f'<svg {_S}><polyline points="3,17 8,12 12,14 21,5"/><polyline points="16,5 21,5 21,10"/></svg>',
    # -- Violin plot shape --
    "violin": f'<svg {_S}><path d="M10 3c-3 3-5 5-5 9s2 6 5 9"/><path d="M14 3c3 3 5 5 5 9s-2 6-5 9"/><line x1="5" y1="12" x2="19" y2="12" stroke-dasharray="2,2"/></svg>',
    # -- Heatmap: 3x3 colored grid --
    "heatmap": f'<svg {_S}><rect x="3" y="3" width="5" height="5" rx="0.5" fill="#22c55e" stroke="none"/><rect x="9.5" y="3" width="5" height="5" rx="0.5" fill="#86efac" stroke="none"/><rect x="16" y="3" width="5" height="5" rx="0.5" fill="#14532d" stroke="none"/><rect x="3" y="9.5" width="5" height="5" rx="0.5" fill="#4ade80" stroke="none"/><rect x="9.5" y="9.5" width="5" height="5" rx="0.5" fill="#166534" stroke="none"/><rect x="16" y="9.5" width="5" height="5" rx="0.5" fill="#22c55e" stroke="none"/><rect x="3" y="16" width="5" height="5" rx="0.5" fill="#14532d" stroke="none"/><rect x="9.5" y="16" width="5" height="5" rx="0.5" fill="#22c55e" stroke="none"/><rect x="16" y="16" width="5" height="5" rx="0.5" fill="#4ade80" stroke="none"/></svg>',
    # -- Chromatogram: MS1 peak line --
    "chromatogram": f'<svg {_S}><polyline points="2,20 5,20 7,16 9,18 11,6 13,18 15,14 17,17 19,20 22,20"/></svg>',
    # -- Clock/RT --
    "clock": f'<svg {_S}><circle cx="12" cy="12" r="9"/><polyline points="12,6 12,12 16,14"/></svg>',
    # -- Leaf (tissue) --
    "leaf": f'<svg {_S}><path d="M17 3c-6 0-13 4-13 13"/><path d="M17 3c0 6-4 13-13 13"/><line x1="4" y1="16" x2="17" y2="3"/></svg>',
    # -- Gear/method --
    "gear": f'<svg {_S}><circle cx="12" cy="12" r="3"/><path d="M12 1v3m0 16v3M1 12h3m16 0h3M4.2 4.2l2.1 2.1m11.4 11.4l2.1 2.1M4.2 19.8l2.1-2.1m11.4-11.4l2.1-2.1"/></svg>',
    # -- Nodes/correlation: connected dots --
    "nodes": f'<svg {_S}><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="12" cy="18" r="2.5"/><line x1="8" y1="7" x2="16" y2="7"/><line x1="7" y1="8" x2="11" y2="16"/><line x1="17" y1="8" x2="13" y2="16"/></svg>',
    # -- Puzzle/biclustering --
    "puzzle": f'<svg {_S}><path d="M3 12h4a2 2 0 010 4H3V8h4a2 2 0 000-4H3V3h18v18H3z"/><path d="M12 3v4a2 2 0 004 0V3"/></svg>',
    # -- Handshake/co-occurrence: two arrows meeting --
    "handshake": f'<svg {_S}><path d="M3 12h7"/><path d="M14 12h7"/><polyline points="7,9 10,12 7,15"/><polyline points="17,9 14,12 17,15"/></svg>',
    # -- No entry / mutual exclusivity --
    "no_entry": f'<svg {_S}><circle cx="12" cy="12" r="9"/><line x1="5" y1="5" x2="19" y2="19"/></svg>',
    # -- Tag/label --
    "tag": f'<svg {_S}><path d="M11 2H2v9l10 10 9-9z"/><circle cx="6" cy="6" r="1.5" fill="#22c55e"/></svg>',
    # -- Pin/map --
    "pin": f'<svg {_S}><path d="M12 2a7 7 0 00-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 00-7-7z"/><circle cx="12" cy="9" r="2.5"/></svg>',
    # -- Normalize: wavy to straight --
    "normalize": f'<svg {_S}><path d="M2 8c2-3 4 3 6 0s4 3 6 0s4 3 6 0" opacity="0.4"/><line x1="2" y1="16" x2="20" y2="16"/><polyline points="17,13 20,16 17,19"/></svg>',
    # -- Star: H3K79 validation spotlight --
    "star": f'<svg {_S}><polygon points="12,2 15,9 22,9 16.5,14 18.5,21 12,17 5.5,21 7.5,14 2,9 9,9" fill="#22c55e" fill-opacity="0.15"/></svg>',
    # -- Organism badge --
    "organism": f'<svg {_S}><path d="M12 22c-4 0-8-2-8-6s8-14 8-14 8 10 8 14-4 6-8 6z"/><line x1="12" y1="8" x2="12" y2="22"/><path d="M8 14c2-1 6-1 8 0"/></svg>',
    # -- Tissue badge --
    "tissue": f'<svg {_S}><ellipse cx="12" cy="10" rx="8" ry="6"/><line x1="4" y1="10" x2="20" y2="10"/><path d="M8 4c0 3 8 3 8 0"/><path d="M8 16c0-3 8-3 8 0"/></svg>',
    # -- Instrument badge --
    "instrument": f'<svg {_S}><rect x="4" y="8" width="16" height="10" rx="2"/><line x1="8" y1="4" x2="8" y2="8"/><line x1="16" y1="4" x2="16" y2="8"/><line x1="8" y1="4" x2="16" y2="4"/><polyline points="7,13 10,11 13,14 17,11" stroke-width="1.5"/></svg>',
    # -- Export/package box --
    "export": f'<svg {_S}><path d="M12 3v12"/><polyline points="8,11 12,15 16,11"/><path d="M4 15v4a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg>',
}

# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULTS = {
    "PXD046788 ndebug_0": {
        "dir": r"D:\epiprofile_data\PXD046788\MS1_MS2\RawData",
        "ratios": "histone_ratios.xls",
        "ndebug": "ndebug_0", "dataset": "PXD046788",
    },
    "PXD046788 ndebug_2": {
        "dir": r"D:\epiprofile_data\PXD046788\MS1_MS2\RawData",
        "ratios": "histone_ratios_ndebug2.xls",
        "ndebug": "ndebug_2", "dataset": "PXD046788",
    },
    "PXD014739 ndebug_0": {
        "dir": r"D:\epiprofile_data\PXD014739\RawData",
        "ratios": "histone_ratios_ndebug0.xls",
        "ndebug": "ndebug_0", "dataset": "PXD014739",
    },
    "PXD014739 ndebug_2": {
        "dir": r"D:\epiprofile_data\PXD014739\RawData",
        "ratios": "histone_ratios_ndeg2.xls",
        "singleptm_dir": "PXD014738_ndebug2_histone_layouts",
        "ndebug": "ndebug_2", "dataset": "PXD014739",
    },
    "PXD046034": {
        "dir": r"D:\epiprofile_data\PXD046034\MS1_MS2",
        "phenodata": r"E:\EpiProfile_AT_PXD046034_raw\PXD046034\PXD046034\analysis_PXD046034\analysis\post_epiprofile\phenodata_PXD046034.tsv",
        "ndebug": "default", "dataset": "PXD046034",
    },
    "Ontogeny ndebug_0": {
        "dir": r"E:\EpiProfile_Proyecto\EpiProfile_20_AT\RawData",
        "ratios": "histone_ratios_ndebug0_ontogenia.xls",
        "singleptm_dir": "histone_layouts_ndebug_0",
        "ndebug": "ndebug_0", "dataset": "Ontogeny",
    },
    "Ontogeny ndebug_2": {
        "dir": r"E:\EpiProfile_Proyecto\EpiProfile_20_AT\RawData",
        "ratios": "histone_ratios.xls",
        "ndebug": "ndebug_2", "dataset": "Ontogeny",
    },
}

# ---------- Experiment metadata (PXD info, paper summaries) ----------
EXP_INFO = {
    "PXD046788": {
        "title": "Histone PTM landscape under environmental stress in Arabidopsis thaliana",
        "pxd": "PXD046788",
        "organism": "Arabidopsis thaliana (Col-0)",
        "tissue": "Rosette leaves (3-week-old seedlings)",
        "instrument": "Q Exactive HF-X (Thermo Fisher)",
        "method": "Bottom-up MS, propionylation, DDA",
        "conditions": "Control, Heat, Cold, Salt, Drought",
        "summary": "Global histone PTM profiling across five abiotic stress conditions. "
                   "Identifies stress-specific changes in H3K27me3, H3K4me3, and H3K9ac marks. "
                   "Reveals coordinate regulation of methylation and acetylation under heat stress.",
        "icon": "plant",
    },
    "PXD014739": {
        "title": "Comprehensive histone modification atlas of Arabidopsis thaliana",
        "pxd": "PXD014739",
        "organism": "Arabidopsis thaliana (multiple ecotypes)",
        "tissue": "Whole seedlings, flowers, roots, leaves",
        "instrument": "Q Exactive Plus (Thermo Fisher)",
        "method": "Bottom-up MS, propionylation, DDA",
        "conditions": "WT, clf, swn, fie, msi1, emf2, vrn2",
        "summary": "Large-scale atlas of histone H3 and H4 modifications across 7 Polycomb mutant lines. "
                   "Demonstrates PRC2 subunit-specific effects on H3K27me3 and crosstalk with H3K36me marks. "
                   "114 samples provide a comprehensive reference for Arabidopsis histone PTMs.",
        "icon": "dna",
    },
    "PXD046034": {
        "title": "Histone PTMs in FAS1/NAP1 chromatin assembly factor mutants",
        "pxd": "PXD046034",
        "organism": "Arabidopsis thaliana (Col-0)",
        "tissue": "Rosette leaves",
        "instrument": "Q Exactive HF (Thermo Fisher)",
        "method": "Bottom-up MS, propionylation, DDA",
        "conditions": "WT_3905, fas1_3905, fas2_3905, nap1_3905, WT_4105, fas1_4105, fas2_4105, nap1_4105",
        "summary": "Characterization of histone PTM changes in chromatin assembly factor mutants (FAS1, FAS2, NAP1). "
                   "Two experimental designs (3905/4105) reveal replication-dependent vs -independent histone deposition effects. "
                   "FAS mutants show altered H3K56ac and H3.1/H3.3 variant ratios.",
        "icon": "microscope",
    },
    "Ontogeny": {
        "title": "Developmental ontogeny of histone modifications in Arabidopsis",
        "pxd": "Internal",
        "organism": "Arabidopsis thaliana (Col-0)",
        "tissue": "Developmental stages (seedling, rosette, bolting, flowering)",
        "instrument": "Q Exactive HF-X (Thermo Fisher)",
        "method": "Bottom-up MS, propionylation, DDA",
        "conditions": "Stage 1 (seedling), Stage 2 (rosette), Stage 3 (bolting), Stage 4 (flowering)",
        "summary": "Tracks histone PTM dynamics across four developmental stages of Arabidopsis. "
                   "Shows progressive changes in H3K27me3/H3K4me3 bivalent marks during the vegetative-to-reproductive transition. "
                   "Identifies developmentally regulated acetylation patterns.",
        "icon": "seedling",
    },
}

def _get_exp_info(exp_name):
    """Find matching EXP_INFO entry for an experiment name."""
    for key in EXP_INFO:
        if key in exp_name:
            return EXP_INFO[key]
    return None

parser = argparse.ArgumentParser(description="EpiProfile-Plants Dashboard v3.11")
parser.add_argument("dirs", nargs="*", help="EpiProfile output directories")
parser.add_argument("--port", type=int, default=8050)
parser.add_argument("--host", default="0.0.0.0")
args, _ = parser.parse_known_args()

if args.dirs:
    # CLI mode: simple name->path mapping (no ndebug metadata)
    EXPERIMENTS = {}
    for p in args.dirs:
        p = os.path.abspath(p)
        EXPERIMENTS[os.path.basename(p) or p] = {"dir": p, "ndebug": "default", "dataset": os.path.basename(p)}
else:
    EXPERIMENTS = {k: v for k, v in DEFAULTS.items() if os.path.isdir(v.get("dir", ""))}
    if not EXPERIMENTS:
        print("ERROR: No valid experiment directories."); sys.exit(1)

# ======================================================================
# DESIGN SYSTEM
# ======================================================================

FONT = "'Inter','Segoe UI',-apple-system,BlinkMacSystemFont,Arial,sans-serif"
C = {
    "bg":"#f8fdf8","card":"#fff","border":"#e2e8f0","text":"#1e293b","text2":"#475569",
    "accent":"#16a34a","accent_l":"#ecfdf5","accent_d":"#15803d","red":"#dc2626","green":"#059669",
    "muted":"#94a3b8","warn":"#d97706","h3":"#7c3aed","h4":"#0891b2",
    "header_bg":"linear-gradient(135deg, #15803d 0%, #22c55e 50%, #4ade80 100%)",
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

    # Find ALL separator columns for ratio/area/RT blocks
    sep_indices = []
    for i, col in enumerate(raw_df.columns):
        if "Unnamed" in str(col) and i > 0:
            sep_indices.append(i)
    sep_idx = sep_indices[0] if len(sep_indices) >= 1 else None
    sep_idx2 = sep_indices[1] if len(sep_indices) >= 2 else None

    sample_cols_r = raw_df.columns[1:sep_idx] if sep_idx else raw_df.columns[1:]
    sample_names = [c.split(",",1)[1] if "," in c else c for c in sample_cols_r]

    # Build ratio matrix
    ratio_block = raw_df.iloc[:, :sep_idx] if sep_idx else raw_df.copy()
    ratio_block.columns = ["PTM"] + list(sample_names)
    ratio_block = ratio_block.iloc[1:].reset_index(drop=True)  # skip sub-header row

    # Build area matrix (between separator 1 and separator 2)
    area_block = None
    if sep_idx and sep_idx + 1 < len(raw_df.columns):
        area_end = sep_idx2 if sep_idx2 else len(raw_df.columns)
        ab = raw_df.iloc[:, sep_idx+1:area_end].copy()
        ab.insert(0, "PTM", raw_df.iloc[:, 0])
        acols = [re.sub(r"\.\d+$","",c.split(",",1)[1]) if "," in str(c) else str(c) for c in ab.columns[1:]]
        ab.columns = ["PTM"] + acols
        ab = ab.iloc[1:].reset_index(drop=True)
        area_block = ab

    # Build RT matrix (after separator 2)
    rt_block = None
    if sep_idx2 and sep_idx2 + 1 < len(raw_df.columns):
        rb = raw_df.iloc[:, sep_idx2+1:].copy()
        rb.insert(0, "PTM", raw_df.iloc[:, 0])
        rtcols = [re.sub(r"\.\d+$","",c.split(",",1)[1]) if "," in str(c) else str(c) for c in rb.columns[1:]]
        rb.columns = ["PTM"] + rtcols
        rb = rb.iloc[1:].reset_index(drop=True)
        rt_block = rb

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

    # Area matrix (fixed: only columns between sep1 and sep2, not including RT)
    areas_df = None
    if area_block is not None:
        a_filtered = area_block[area_block["PTM"].isin(hpf_df.index)].copy()
        if not a_filtered.empty:
            a_filtered = a_filtered.set_index("PTM")
            a_filtered = a_filtered.apply(pd.to_numeric, errors="coerce")
            # Ensure column names match sample_names exactly
            a_filtered.columns = sample_names[:len(a_filtered.columns)]
            areas_df = a_filtered

    # RT matrix
    rt_df = None
    if rt_block is not None:
        r_filtered = rt_block[rt_block["PTM"].isin(hpf_df.index)].copy()
        if not r_filtered.empty:
            r_filtered = r_filtered.set_index("PTM")
            r_filtered = r_filtered.apply(pd.to_numeric, errors="coerce")
            r_filtered.columns = sample_names[:len(r_filtered.columns)]
            rt_df = r_filtered

    return {
        "sample_names": sample_names,
        "hdp_list": hdp_list,
        "hpf_df": hpf_df,
        "hpf_meta": hpf_meta,
        "var_df": var_df,
        "var_meta": var_meta,
        "areas": areas_df,
        "rt": rt_df,
    }


def load_experiment(base_dir, ratios_file=None, singleptm_dir=None, phenodata_path=None):
    """Load experiment data from a directory.
    Args:
        base_dir: Root directory for this experiment
        ratios_file: Specific histone_ratios filename (e.g. 'histone_ratios_ndebug0.xls')
        singleptm_dir: Subdirectory name for single PTMs (e.g. 'PXD014738_ndebug2_histone_layouts')
        phenodata_path: Explicit absolute path to phenodata TSV file (overrides search)
    """
    data = {"base_dir": base_dir}
    layouts_dir = os.path.join(base_dir, "histone_layouts")
    data["layouts_dir"] = layouts_dir if os.path.isdir(layouts_dir) else base_dir
    ld = data["layouts_dir"]

    # -- single PTMs (hPTM level) --
    # If singleptm_dir is specified, look there first
    sptm_search_dirs = [base_dir, ld]
    if singleptm_dir:
        alt_ld = os.path.join(base_dir, singleptm_dir)
        if os.path.isdir(alt_ld):
            sptm_search_dirs = [alt_ld] + sptm_search_dirs
    for fn in ["histone_ratios_single_PTMs.tsv","histone_ratios_single_PTMs.xls"]:
        found = False
        for search_dir in sptm_search_dirs:
            fp = os.path.join(search_dir, fn)
            if os.path.exists(fp):
                try:
                    df = try_read_tsv(fp)
                    df.columns = ["PTM"] + [c.split(",",1)[1] if "," in c else c for c in df.columns[1:]]
                    df = df.set_index("PTM").apply(pd.to_numeric, errors="coerce")
                    data["hptm"] = df
                    found = True; break
                except Exception: pass
        if found: break

    # -- histone_ratios.xls (hDP/hPF hierarchy) --
    if ratios_file:
        fp = os.path.join(base_dir, ratios_file)
        if not os.path.exists(fp):
            fp = find_file(base_dir, ratios_file)
    else:
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
                log2_a, qn_a = normalize_areas(parsed["areas"])
                if log2_a is not None: data["areas_log2"] = log2_a
                if qn_a is not None: data["areas_norm"] = qn_a
            if parsed.get("rt") is not None:
                data["rt"] = parsed["rt"]
        except Exception as e:
            print(f"    WARN parsing ratios: {e}")

    # -- phenodata --
    # If explicit path provided, use it directly
    if phenodata_path and os.path.exists(phenodata_path):
        try:
            data["phenodata"] = pd.read_csv(phenodata_path, sep="\t")
            print(f"    Phenodata: loaded from explicit path")
        except Exception as pe:
            print(f"    WARN phenodata: {pe}")
    else:
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
    if "areas_norm" in d: parts.append("Areas (normalized)")
    if "rt" in d: parts.append("RT available")
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

def robust_group_test(df, meta, groups, is_log=False):
    """Kruskal-Wallis + pairwise Mann-Whitney with FDR correction.
    is_log: if True, data is log-scale (filter non-finite instead of zeros)."""
    results = []
    for ptm in df.index:
        group_vals = {}
        for g in groups:
            samps = meta[meta["Group"]==g]["Sample"].tolist()
            vals = df.loc[ptm, [s for s in samps if s in df.columns]].dropna()
            if is_log:
                vals = vals[np.isfinite(vals)]
            else:
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


def pairwise_mw(df, meta, g1, g2, is_log=False):
    """Mann-Whitney U between two groups for all PTMs.
    is_log: if True, data is already log-scale so FC = mean_B - mean_A."""
    sa = meta[meta["Group"]==g1]["Sample"].tolist()
    sb = meta[meta["Group"]==g2]["Sample"].tolist()
    ca = [c for c in df.columns if c in sa]
    cb = [c for c in df.columns if c in sb]
    results = []
    for ptm in df.index:
        va = df.loc[ptm, ca].dropna().values.astype(float)
        vb = df.loc[ptm, cb].dropna().values.astype(float)
        if not is_log:
            va = va[va != 0]; vb = vb[vb != 0]
        else:
            va = va[np.isfinite(va)]; vb = vb[np.isfinite(vb)]
        if len(va) >= 2 and len(vb) >= 2:
            try:
                stat, p = mannwhitneyu(va, vb, alternative="two-sided")
                fc = (np.mean(vb) - np.mean(va)) if is_log else np.log2((np.mean(vb)+1e-8)/(np.mean(va)+1e-8))
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


def quantile_normalize(df):
    """Quantile normalization (Bolstad 2003). NaN-aware, pure pandas/numpy.
    Input: features (rows) x samples (columns) DataFrame.
    Returns: quantile-normalized DataFrame, same shape, NaN preserved."""
    if df is None or df.empty: return df
    result = df.copy()
    # Sort each column independently, compute reference distribution
    sorted_df = df.apply(lambda c: c.dropna().sort_values().reset_index(drop=True))
    reference = sorted_df.mean(axis=1)
    if reference.empty: return df
    # For each column, map ranked values to reference distribution
    for col in df.columns:
        valid = df[col].dropna()
        if valid.empty: continue
        ranked = valid.rank(method="average")
        max_rank = ranked.max()
        if max_rank <= 1:
            result.loc[valid.index, col] = reference.iloc[0] if len(reference) > 0 else valid.values
            continue
        # Scale ranks to reference distribution indices
        scaled = (ranked - 1) / (max_rank - 1) * (len(reference) - 1)
        result.loc[valid.index, col] = np.interp(
            scaled.values, np.arange(len(reference)), reference.values)
    return result


def normalize_areas(areas_df):
    """Full normalization pipeline for MS areas.
    1. Replace zeros with NaN (non-detects)
    2. Log2 transform
    3. Quantile normalization
    Returns: (log2_df, qn_df) -- log2-only and log2+QN DataFrames."""
    if areas_df is None or areas_df.empty: return None, None
    clean = areas_df.replace(0, np.nan)
    log2_df = np.log2(clean)
    qn_df = quantile_normalize(log2_df)
    return log2_df, qn_df


def _get_data_source(d, source):
    """Resolve data source key to a DataFrame.
    source: 'ratios'|'hptm'|'hpf'|'areas_norm'|'areas_log2'"""
    if source in ("ratios", None):
        return d.get("hptm", d.get("hpf"))
    elif source == "hptm":
        return d.get("hptm")
    elif source == "hpf":
        return d.get("hpf")
    elif source == "areas_norm":
        return d.get("areas_norm")
    elif source == "areas_log2":
        return d.get("areas_log2")
    return d.get(source, d.get("hptm", d.get("hpf")))


# ======================================================================
# LOAD ALL EXPERIMENTS
# ======================================================================

print("Loading experiments...")
EXP_DATA = {}
for name, cfg in EXPERIMENTS.items():
    base_dir = cfg.get("dir", cfg) if isinstance(cfg, dict) else cfg
    if not os.path.isdir(base_dir):
        continue
    print(f"  {name}...")
    ratios_file = cfg.get("ratios") if isinstance(cfg, dict) else None
    singleptm_dir = cfg.get("singleptm_dir") if isinstance(cfg, dict) else None
    phenodata_path = cfg.get("phenodata") if isinstance(cfg, dict) else None
    EXP_DATA[name] = load_experiment(base_dir, ratios_file=ratios_file,
                                      singleptm_dir=singleptm_dir,
                                      phenodata_path=phenodata_path)
    d = EXP_DATA[name]
    # Store ndebug/dataset metadata for comparison features
    if isinstance(cfg, dict):
        d["ndebug"] = cfg.get("ndebug", "default")
        d["dataset"] = cfg.get("dataset", name)
    else:
        d["ndebug"] = "default"
        d["dataset"] = name
    # Print loading summary
    n_hptm = d["hptm"].shape[0] if "hptm" in d else 0
    n_hpf = d["hpf"].shape[0] if "hpf" in d and not d["hpf"].empty else 0
    n_reg = len(d.get("hdp_list", []))
    meta = d.get("metadata", pd.DataFrame())
    n_samp = len(meta) if not meta.empty else 0
    n_grp = meta["Group"].nunique() if not meta.empty and "Group" in meta.columns else 0
    if "hptm" in d: print(f"    hPTM: {d['hptm'].shape}")
    if n_hpf: print(f"    hPF:  {d['hpf'].shape}")
    if n_reg: print(f"    hDP:  {n_reg} regions")
    if "areas" in d: print(f"    Areas: {d['areas'].shape}")
    if "areas_norm" in d: print(f"    Areas (log2+QN): {d['areas_norm'].shape}")
    if "rt" in d: print(f"    RT: {d['rt'].shape}")
    print(f"    Folders: {len(d['sample_folders'])}")
    psm = d.get("all_psm", pd.DataFrame())
    if not psm.empty: print(f"    PSMs: {len(psm)}")
    ndebug_str = f" [{d['ndebug']}]" if d.get("ndebug","default") != "default" else ""
    print(f"    >> {d.get('description','')}{ndebug_str}")
    log_experiment(name, base_dir, n_samp, n_grp, n_hptm, n_hpf, n_reg)

DEFAULT_EXP = list(EXP_DATA.keys())[0] if EXP_DATA else None

# ======================================================================
# APP
# ======================================================================

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "EpiProfile-Plants Dashboard"

ts = {"color":"#6b7280","backgroundColor":"#fff","border":"none",
      "borderBottom":"3px solid transparent","padding":"15px 22px",
      "fontSize":"14.5px","fontWeight":"500","fontFamily":FONT,
      "transition":"all 0.25s ease","cursor":"pointer","letterSpacing":"0.1px"}
tss = {**ts,"color":"#0f1f13","borderBottom":"3px solid #22c55e","fontWeight":"700",
       "backgroundColor":"#f0fdf4","letterSpacing":"0px"}

# ======================================================================
# LAYOUT
# ======================================================================

app.layout = html.Div(style={"backgroundColor":C["bg"],"minHeight":"100vh","fontFamily":FONT,"color":C["text"]}, children=[
    # ---- HERO HEADER (premium dark gradient with molecular pattern) ----
    html.Div(style={"background":"linear-gradient(155deg, #0a1a0e 0%, #0f2518 20%, #14532d 45%, #166534 65%, #1a7a40 85%, #0f2518 100%)",
                     "padding":"0","color":"white","position":"relative","overflow":"hidden"}, children=[
        # Geometric molecular pattern overlay (CSS-only hexagonal dots)
        html.Div(style={"position":"absolute","inset":"0","opacity":"0.04",
                         "backgroundImage":"radial-gradient(circle, #4ade80 1px, transparent 1px)",
                         "backgroundSize":"28px 28px"}),
        # Decorative accent glows (multiple layered)
        html.Div(style={"position":"absolute","top":"-100px","right":"-80px","width":"400px","height":"400px",
                         "borderRadius":"50%","background":"radial-gradient(circle, rgba(34,197,94,0.18) 0%, transparent 60%)",
                         "filter":"blur(40px)"}),
        html.Div(style={"position":"absolute","bottom":"-80px","left":"3%","width":"300px","height":"300px",
                         "borderRadius":"50%","background":"radial-gradient(circle, rgba(16,185,129,0.12) 0%, transparent 60%)",
                         "filter":"blur(30px)"}),
        html.Div(style={"position":"absolute","top":"30%","right":"25%","width":"150px","height":"150px",
                         "borderRadius":"50%","background":"radial-gradient(circle, rgba(74,222,128,0.06) 0%, transparent 70%)"}),
        html.Div(style={"position":"absolute","top":"10%","left":"40%","width":"200px","height":"200px",
                         "borderRadius":"50%","background":"radial-gradient(circle, rgba(5,150,105,0.08) 0%, transparent 60%)",
                         "filter":"blur(25px)"}),
        # Top accent line (rainbow-ish green spectrum)
        html.Div(style={"height":"3px",
                         "background":"linear-gradient(90deg, transparent 0%, #059669 15%, #10b981 30%, #22c55e 45%, #4ade80 55%, #86efac 65%, #22c55e 80%, transparent 100%)"}),
        # Main content row
        html.Div(style={"display":"flex","alignItems":"center","gap":"36px","flexWrap":"wrap",
                         "position":"relative","zIndex":"1","padding":"38px 52px 18px"}, children=[
            # Title block with enhanced typography
            html.Div([
                html.Div(style={"display":"flex","alignItems":"baseline","gap":"16px"}, children=[
                    html.H1("EpiProfile", style={"margin":"0","fontSize":"46px","fontWeight":"800",
                             "letterSpacing":"-1.5px","color":"#fff","lineHeight":"1",
                             "textShadow":"0 2px 16px rgba(0,0,0,0.4)"}),
                    html.Span("-Plants", style={"fontSize":"46px","fontWeight":"300","color":"#4ade80",
                              "letterSpacing":"-1px","lineHeight":"1",
                              "textShadow":"0 2px 16px rgba(74,222,128,0.3)"}),
                ]),
                # Thin accent underline
                html.Div(style={"height":"2px","width":"180px","marginTop":"8px",
                                 "background":"linear-gradient(90deg, #22c55e 0%, #4ade80 50%, transparent 100%)",
                                 "borderRadius":"2px"}),
                html.Div(style={"display":"flex","gap":"12px","alignItems":"center","marginTop":"12px",
                                 "flexWrap":"wrap"}, children=[
                    html.Span("Histone PTM Quantification Dashboard", style={"color":"#94a3b8",
                              "fontSize":"15px","fontWeight":"400","letterSpacing":"0.5px"}),
                    html.Span("v3.11", style={"background":"linear-gradient(135deg, #22c55e 0%, #10b981 100%)",
                              "padding":"3px 14px","borderRadius":"14px","fontSize":"12px","fontWeight":"700",
                              "color":"#0f1f13","boxShadow":"0 0 12px rgba(34,197,94,0.5), 0 0 4px rgba(34,197,94,0.3)"}),
                ]),
                html.Div(style={"display":"flex","gap":"8px","alignItems":"center","marginTop":"14px",
                                 "flexWrap":"wrap"}, children=[
                    html.Span([_svg_icon(ic, 14, {"filter":"brightness(2)"}), f" {lbl}"],
                              style={"background":"rgba(255,255,255,0.06)","padding":"5px 16px",
                              "borderRadius":"8px","fontSize":"13px","fontWeight":"600","color":"#d1d5db",
                              "border":"1px solid rgba(255,255,255,0.08)","letterSpacing":"0.5px",
                              "display":"inline-flex","alignItems":"center","gap":"6px"})
                    for ic, lbl in [("ptm","hPTM"), ("peptidoform","hPF"), ("dna","hDP"), ("chromatogram","Areas"), ("clock","RT")]
                ]),
            ]),
            html.Div(style={"flex":"1","minWidth":"30px"}),
            # Experiment selector -- enhanced glass card
            html.Div(style={"background":"rgba(255,255,255,0.06)","borderRadius":"16px","padding":"18px 24px",
                             "border":"1px solid rgba(255,255,255,0.10)","backdropFilter":"blur(20px)",
                             "boxShadow":"0 8px 32px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05)"}, children=[
                html.Span([_svg_icon("flask", 13, {"filter":"brightness(2)"}), " EXPERIMENT"],
                          style={"color":"#86efac","fontSize":"10px","fontWeight":"700",
                           "letterSpacing":"2.5px","textTransform":"uppercase","display":"flex","alignItems":"center","gap":"6px","marginBottom":"10px"}),
                dcc.Dropdown(id="exp-sel", options=[{"label":k,"value":k} for k in EXP_DATA],
                             value=DEFAULT_EXP, clearable=False,
                             style={"width":"440px","fontSize":"14px","borderRadius":"10px"}),
            ]),
            # Color palette selector -- enhanced glass card
            html.Div(style={"background":"rgba(255,255,255,0.06)","borderRadius":"16px","padding":"18px 24px",
                             "border":"1px solid rgba(255,255,255,0.10)","backdropFilter":"blur(20px)",
                             "boxShadow":"0 8px 32px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05)"}, children=[
                html.Span([_svg_icon("palette", 13, {"filter":"brightness(2)"}), " PALETTE"],
                          style={"color":"#86efac","fontSize":"10px","fontWeight":"700",
                           "letterSpacing":"2.5px","textTransform":"uppercase","display":"flex","alignItems":"center","gap":"6px","marginBottom":"10px"}),
                dcc.Dropdown(id="palette-sel",
                             options=[{"label":k,"value":k} for k in PALETTES],
                             value="EpiProfile (default)", clearable=False,
                             style={"width":"220px","fontSize":"14px","borderRadius":"10px"}),
            ]),
        ]),
        # Stats ribbon -- enhanced with icons
        html.Div(style={"display":"flex","gap":"20px","alignItems":"center","justifyContent":"center",
                         "padding":"14px 52px 16px","position":"relative","zIndex":"1","flexWrap":"wrap"}, children=[
            html.Span([_svg_icon("folder", 14, {"filter":"brightness(2)"}), f" {len(EXP_DATA)} Experiments"],
                      style={"color":"#86efac","fontSize":"13px","fontWeight":"600","display":"inline-flex","alignItems":"center","gap":"5px"}),
            html.Span(chr(8226), style={"color":"#374151","fontSize":"8px"}),
            html.Span([_svg_icon("dashboard", 14, {"filter":"brightness(1.5)"}), " 14 Analysis Tabs"],
                      style={"color":"#9ca3af","fontSize":"13px","fontWeight":"500","display":"inline-flex","alignItems":"center","gap":"5px"}),
            html.Span(chr(8226), style={"color":"#374151","fontSize":"8px"}),
            html.Span([_svg_icon("bar_chart", 14, {"filter":"brightness(1.5)"}), " KW + MW + FDR"],
                      style={"color":"#9ca3af","fontSize":"13px","fontWeight":"500","display":"inline-flex","alignItems":"center","gap":"5px"}),
            html.Span(chr(8226), style={"color":"#374151","fontSize":"8px"}),
            html.Span([_svg_icon("pca", 14, {"filter":"brightness(1.5)"}), " PCA + Silhouette"],
                      style={"color":"#9ca3af","fontSize":"13px","fontWeight":"500","display":"inline-flex","alignItems":"center","gap":"5px"}),
            html.Span(chr(8226), style={"color":"#374151","fontSize":"8px"}),
            html.Span([_svg_icon("spectra_compare", 14, {"filter":"brightness(1.5)"}), " ndebug Compare"],
                      style={"color":"#9ca3af","fontSize":"13px","fontWeight":"500","display":"inline-flex","alignItems":"center","gap":"5px"}),
        ]),
        # Upload area (collapsible, 3-slot) -- dark glass panel
        html.Details(style={"margin":"6px 52px 0","position":"relative","zIndex":"1",
                             "background":"rgba(255,255,255,0.05)","borderRadius":"12px",
                             "padding":"14px 22px","border":"1px solid rgba(255,255,255,0.08)"}, children=[
            html.Summary("Upload / Replace Data Files",
                         style={"cursor":"pointer","color":"#86efac","fontSize":"14px","fontWeight":"600",
                                "letterSpacing":"0.5px","listStyleType":"none"}),
            # Mode selector
            html.Div(style={"display":"flex","gap":"16px","marginTop":"14px","alignItems":"center","flexWrap":"wrap"}, children=[
                html.Div(style={"display":"flex","gap":"8px","alignItems":"center"}, children=[
                    html.Span("MODE:", style={"color":"#6b7280","fontSize":"11px","fontWeight":"700","letterSpacing":"1px"}),
                    dcc.RadioItems(id="upload-mode",
                        options=[{"label":" Replace files in current experiment","value":"replace"},
                                 {"label":" Upload new experiment","value":"new"}],
                        value="replace", inline=True,
                        style={"color":"#d1d5db","fontSize":"13px"},
                        inputStyle={"marginRight":"4px","marginLeft":"12px"}),
                ]),
                # New experiment name (shown only for "new" mode)
                html.Div(id="new-exp-name-wrap", style={"display":"none"}, children=[
                    html.Span("NAME:", style={"color":"#6b7280","fontSize":"11px","fontWeight":"700","letterSpacing":"1px","marginRight":"8px"}),
                    dcc.Input(id="new-exp-name", type="text", placeholder="My Experiment (species)",
                              style={"borderRadius":"10px","border":"1px solid rgba(255,255,255,0.15)",
                                     "padding":"8px 14px","fontSize":"14px","backgroundColor":"rgba(255,255,255,0.08)",
                                     "color":"white","width":"280px"}),
                ]),
            ]),
            html.Div(style={"display":"flex","gap":"18px","marginTop":"14px","flexWrap":"wrap"}, children=[
                html.Div(style={"flex":"1","minWidth":"250px"}, children=[
                    html.Label("1. Phenodata TSV", style={"color":"#86efac","fontSize":"12px","fontWeight":"600","marginBottom":"4px","display":"block"}),
                    dcc.Upload(id="upload-pheno",
                        children=html.Div(["Drop or ", html.A("select phenodata.tsv",style={"color":"#4ade80","fontWeight":"700"})]),
                        style={"border":"2px dashed rgba(74,222,128,0.25)","borderRadius":"12px","padding":"18px",
                               "textAlign":"center","color":"#9ca3af","fontSize":"13px",
                               "cursor":"pointer","backgroundColor":"rgba(255,255,255,0.03)"},
                        multiple=False),
                ]),
                html.Div(style={"flex":"1","minWidth":"250px"}, children=[
                    html.Label("2. histone_ratios.xls (TSV)", style={"color":"#86efac","fontSize":"12px","fontWeight":"600","marginBottom":"4px","display":"block"}),
                    dcc.Upload(id="upload-ratios",
                        children=html.Div(["Drop or ", html.A("select histone_ratios",style={"color":"#4ade80","fontWeight":"700"})]),
                        style={"border":"2px dashed rgba(74,222,128,0.25)","borderRadius":"12px","padding":"18px",
                               "textAlign":"center","color":"#9ca3af","fontSize":"13px",
                               "cursor":"pointer","backgroundColor":"rgba(255,255,255,0.03)"},
                        multiple=False),
                ]),
                html.Div(style={"flex":"1","minWidth":"250px"}, children=[
                    html.Label("3. Single PTMs TSV", style={"color":"#86efac","fontSize":"12px","fontWeight":"600","marginBottom":"4px","display":"block"}),
                    dcc.Upload(id="upload-singleptm",
                        children=html.Div(["Drop or ", html.A("select single_PTMs",style={"color":"#4ade80","fontWeight":"700"})]),
                        style={"border":"2px dashed rgba(74,222,128,0.25)","borderRadius":"12px","padding":"18px",
                               "textAlign":"center","color":"#9ca3af","fontSize":"13px",
                               "cursor":"pointer","backgroundColor":"rgba(255,255,255,0.03)"},
                        multiple=False),
                ]),
            ]),
            html.Div(id="upload-status", style={"color":"#4ade80","fontSize":"13px","marginTop":"10px","fontWeight":"600"}),
        ]),
        # Bottom padding + accent line
        html.Div(style={"height":"14px"}),
        html.Div(style={"height":"2px","background":"linear-gradient(90deg, transparent 0%, #22c55e 30%, #4ade80 50%, #22c55e 70%, transparent 100%)","opacity":"0.6"}),
    ]),
    # ---- Description bar with experiment stats (clean, modern grey) ----
    html.Div(style={"backgroundColor":"#f8faf9","padding":"14px 52px",
                     "borderBottom":"1px solid #e5e7eb","display":"flex",
                     "justifyContent":"space-between","alignItems":"center","flexWrap":"wrap","gap":"14px"}, children=[
        html.Div(id="desc-bar", style={"fontSize":"14px","color":"#374151","fontWeight":"500"}),
        html.Div(id="exp-stats-bar", style={"display":"flex","gap":"10px","alignItems":"center","flexWrap":"wrap"}),
    ]),
    # ---- Experiment Info Panel (PXD metadata, paper summary) ----
    html.Div(id="exp-info-panel", style={"padding":"0 52px"}),
    # ---- Tabs with icons ----
    dcc.Tabs(id="tabs", value="tab-hpf", style={"backgroundColor":"#fff","borderBottom":"2px solid #d1d5db"},
             colors={"border":"transparent","primary":C["accent"],"background":"#fff"}, children=[
        dcc.Tab(label="Peptidoforms", value="tab-hpf", style=ts, selected_style=tss),
        dcc.Tab(label="Single PTMs", value="tab-hptm", style=ts, selected_style=tss),
        dcc.Tab(label="QC Dashboard", value="tab-qc", style=ts, selected_style=tss),
        dcc.Tab(label="PCA & Clustering", value="tab-pca", style=ts, selected_style=tss),
        dcc.Tab(label="Statistics", value="tab-stats", style=ts, selected_style=tss),
        dcc.Tab(label="UpSet / Co-occur", value="tab-upset", style=ts, selected_style=tss),
        dcc.Tab(label="Region Map", value="tab-region", style=ts, selected_style=tss),
        dcc.Tab(label="H3K79 Validation", value="tab-k79", style=ts, selected_style={**tss, "borderTop":"3px solid #f59e0b","color":"#f59e0b"}),
        dcc.Tab(label="Comparisons", value="tab-cmp", style=ts, selected_style=tss),
        dcc.Tab(label="Phenodata", value="tab-pheno", style=ts, selected_style=tss),
        dcc.Tab(label="Sample Browser", value="tab-browse", style=ts, selected_style=tss),
        dcc.Tab(label="Export to R", value="tab-export", style=ts, selected_style=tss),
        dcc.Tab(label="ndebug Compare", value="tab-ndebug", style=ts, selected_style=tss),
        dcc.Tab(label="Analysis Log", value="tab-log", style=ts, selected_style=tss),
    ]),
    html.Div(id="tab-out", style={"padding":"30px 48px","maxWidth":"1800px","margin":"0 auto"}),
    # ---- Download component (hidden, triggered by export callbacks) ----
    dcc.Download(id="download-data"),
    # ---- Footer (matches header gradient) ----
    html.Div(style={"textAlign":"center","padding":"30px 52px","fontSize":"12px",
                     "background":"linear-gradient(155deg, #0a1a0e 0%, #0f2518 40%, #14532d 70%, #166534 100%)",
                     "borderTop":"2px solid #22c55e","marginTop":"48px","color":"#9ca3af"}, children=[
        html.Div(style={"display":"flex","justifyContent":"center","alignItems":"center","gap":"16px",
                         "flexWrap":"wrap","marginBottom":"10px"}, children=[
            html.Span("EpiProfile-Plants", style={"fontWeight":"800","fontSize":"17px","color":"#fff",
                       "letterSpacing":"-0.5px"}),
            html.Span("v3.11", style={"fontWeight":"700","fontSize":"11px","color":"#0f1f13",
                       "background":"linear-gradient(135deg, #22c55e 0%, #10b981 100%)",
                       "padding":"3px 12px","borderRadius":"10px",
                       "boxShadow":"0 0 8px rgba(34,197,94,0.4)"}),
            html.Span(chr(8226), style={"color":"#374151","fontSize":"8px"}),
            html.Span("Histone PTM Quantification Dashboard", style={"fontWeight":"500","color":"#6b7280",
                       "fontSize":"13px"}),
            html.Span(chr(8226), style={"color":"#374151","fontSize":"8px"}),
            html.A("GitHub", href="https://github.com/biopelayo/epiprofile-dashboard",
                   style={"color":"#86efac","textDecoration":"none","fontWeight":"700",
                          "fontSize":"13px","letterSpacing":"0.5px",
                          "borderBottom":"1px solid rgba(134,239,172,0.3)",
                          "paddingBottom":"1px"}, target="_blank"),
        ]),
        html.P("Publication-quality visualization | KW + MW + FDR | "
               "PCA + Silhouette | ndebug Comparison | Co-occurrence + UpSet | Export to R",
               style={"margin":"8px 0 0","color":"#4b5563","fontSize":"11px","fontWeight":"400",
                       "letterSpacing":"0.5px"}),
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
def _desc_bar(e, pal):
    desc = EXP_DATA[e].get("description","") if e and e in EXP_DATA else ""
    pal_name = pal if pal else "EpiProfile (default)"
    return f"{desc}  |  Palette: {pal_name}"

@callback(Output("exp-info-panel","children"), Input("cur-exp","data"))
def _exp_info(exp):
    """Build the experiment info panel with PXD metadata and paper summary."""
    if not exp or exp not in EXP_DATA:
        return ""
    info = _get_exp_info(exp)
    if info is None:
        return ""
    d = EXP_DATA[exp]; meta = d.get("metadata", pd.DataFrame())
    n_samp = len(meta) if not meta.empty else 0
    n_grp = meta["Group"].nunique() if not meta.empty else 0
    groups = sorted(meta["Group"].unique().tolist()) if not meta.empty else []
    # Info card style
    ics = {"display":"inline-flex","alignItems":"center","gap":"6px","padding":"4px 14px",
           "borderRadius":"8px","fontSize":"12px","fontWeight":"600"}
    return html.Details(open=True, style={"margin":"0","padding":"14px 0",
                                           "borderBottom":"1px solid #e5e7eb"}, children=[
        html.Summary(style={"cursor":"pointer","fontSize":"15px","fontWeight":"700","color":"#0f1f13",
                             "listStyleType":"none","display":"flex","alignItems":"center","gap":"10px"}, children=[
            _svg_icon(info.get("icon",""), size=20),
            html.Span(info.get("title",""), style={"letterSpacing":"-0.2px"}),
            html.Span(info.get("pxd",""), style={"background":"#14532d","color":"#86efac","padding":"2px 10px",
                       "borderRadius":"8px","fontSize":"11px","fontWeight":"700","marginLeft":"4px"}),
        ]),
        html.Div(style={"marginTop":"12px","display":"grid","gridTemplateColumns":"1fr 1fr",
                          "gap":"10px 24px"}, children=[
            # Left column: metadata
            html.Div([
                html.Div(style={"display":"flex","flexWrap":"wrap","gap":"8px","marginBottom":"10px"}, children=[
                    html.Span([_svg_icon("organism", 14), " ", info.get("organism","")], style={**ics,"background":"#f0fdf4","color":"#166534","border":"1px solid #dcfce7"}),
                    html.Span([_svg_icon("tissue", 14), " ", info.get("tissue","")], style={**ics,"background":"#eff6ff","color":"#1e40af","border":"1px solid #dbeafe"}),
                    html.Span([_svg_icon("instrument", 14), " ", info.get("instrument","")], style={**ics,"background":"#faf5ff","color":"#6b21a8","border":"1px solid #f3e8ff"}),
                    html.Span([_svg_icon("gear", 14), " ", info.get("method","")], style={**ics,"background":"#fefce8","color":"#854d0e","border":"1px solid #fef9c3"}),
                ]),
                html.Div(style={"display":"flex","flexWrap":"wrap","gap":"6px"}, children=[
                    html.Span(g, style={"background":"#f1f5f9","color":"#475569","padding":"3px 12px",
                               "borderRadius":"6px","fontSize":"12px","fontWeight":"500",
                               "border":"1px solid #e2e8f0"})
                    for g in groups
                ]) if groups else "",
            ]),
            # Right column: summary
            html.Div([
                html.P(info.get("summary",""), style={"margin":"0","fontSize":"13px","color":"#4b5563",
                        "lineHeight":"1.6","borderLeft":"3px solid #22c55e","paddingLeft":"14px"}),
            ]),
        ]),
    ])

@callback(Output("new-exp-name-wrap","style"),
          Input("upload-mode","value"), prevent_initial_call=True)
def _upload_mode_toggle(mode):
    if mode == "new":
        return {"display":"flex","alignItems":"center"}
    return {"display":"none"}

@callback(Output("exp-stats-bar","children"),
          Input("cur-exp","data"))
def _exp_stats(exp):
    if not exp or exp not in EXP_DATA: return ""
    d = EXP_DATA[exp]; meta = d.get("metadata", pd.DataFrame())
    n_samp = len(meta) if not meta.empty else 0
    n_grp = meta["Group"].nunique() if not meta.empty else 0
    n_hptm = len(d.get("hptm", pd.DataFrame()))
    n_hpf = len(d.get("hpf", pd.DataFrame()))
    has_areas = "areas_norm" in d and d["areas_norm"] is not None
    badge = lambda txt, bg: html.Span(txt, style={"background":bg,"color":"white","padding":"3px 12px",
        "borderRadius":"8px","fontSize":"12px","fontWeight":"600","letterSpacing":"0.2px"})
    chips = [
        badge(f"{n_samp} samples", "#14532d"),
        badge(f"{n_grp} groups", "#166534"),
        badge(f"{n_hptm} hPTMs", "#0891b2"),
    ]
    if n_hpf > 0: chips.append(badge(f"{n_hpf} hPF", "#7c3aed"))
    if has_areas: chips.append(badge("Areas + QN", "#d97706"))
    return chips

@callback(Output("upload-status","children"),
          Output("exp-sel","options"),
          Output("exp-sel","value"),
          Input("upload-pheno","contents"), Input("upload-ratios","contents"),
          Input("upload-singleptm","contents"),
          State("upload-pheno","filename"), State("upload-ratios","filename"),
          State("upload-singleptm","filename"),
          State("upload-mode","value"), State("new-exp-name","value"),
          State("cur-exp","data"), prevent_initial_call=True)
def _upload(pheno_content, ratios_content, sptm_content, pheno_name, ratios_name, sptm_name,
            mode, new_name, exp):
    cur_opts = [{"label":k,"value":k} for k in EXP_DATA]
    cur_val = exp

    # Handle "new experiment" mode
    if mode == "new":
        exp_name = (new_name or "").strip()
        if not exp_name:
            exp_name = f"Uploaded_{datetime.now().strftime('%H%M%S')}"
        if exp_name not in EXP_DATA:
            EXP_DATA[exp_name] = {"description": f"Uploaded: {exp_name}", "layouts_dir": ""}
        exp = exp_name
        cur_opts = [{"label":k,"value":k} for k in EXP_DATA]
        cur_val = exp_name

    if not exp or exp not in EXP_DATA:
        return html.Div("No experiment selected.", style={"color":"#fca5a5"}), cur_opts, cur_val
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
            # Store areas + normalize them immediately
            if parsed["areas"] is not None:
                d["areas"] = parsed["areas"]
                log2_a, qn_a = normalize_areas(parsed["areas"])
                if log2_a is not None: d["areas_log2"] = log2_a
                if qn_a is not None: d["areas_norm"] = qn_a
            # Store RT
            if parsed["rt"] is not None:
                d["rt"] = parsed["rt"]
            # Rebuild metadata if phenodata exists
            pheno = d.get("phenodata")
            if pheno is not None and not pheno.empty:
                d["metadata"] = build_metadata(list(parsed["hpf_df"].columns), pheno)
            d["description"] = build_desc(d)
            extras = []
            if parsed["areas"] is not None: extras.append("Areas+QN")
            if parsed["rt"] is not None: extras.append("RT")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            results.append(html.Span(
                f"Ratios OK: {ratios_name} ({parsed['hpf_df'].shape[0]} hPF, {len(parsed['hdp_list'])} regions){extra_str}. ",
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

    if mode == "new" and results:
        results.insert(0, html.Span(f"NEW EXPERIMENT '{exp}' created! ",
                       style={"color":"#fbbf24","fontWeight":"700","fontSize":"14px"}))

    return (html.Div(results) if results else "", cur_opts, cur_val)

@callback(Output("tab-out","children"), Input("tabs","value"), Input("cur-exp","data"), Input("cur-palette","data"))
def _rt(tab, exp, pal):
    if not exp or exp not in EXP_DATA:
        return html.Div("No experiment loaded", style={"color":C["red"],"textAlign":"center","padding":"80px"})
    d = EXP_DATA[exp]
    try:
        if tab == "tab-log": return tab_log(d, exp)
        if tab == "tab-export": return tab_export(d)
        if tab == "tab-ndebug": return tab_ndebug(exp)
        if tab == "tab-k79": return tab_k79(d, exp)
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

def adaptive_font(n_items, base=14, min_size=7, max_size=18):
    """Compute font size based on number of items to display."""
    if n_items is None or n_items <= 0: return base
    if n_items <= 10: return max_size
    if n_items <= 20: return max(base, 15)
    if n_items <= 35: return base
    if n_items <= 50: return max(min_size + 3, base - 2)
    if n_items <= 80: return max(min_size + 2, base - 3)
    if n_items <= 120: return max(min_size + 1, base - 4)
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

def phm(z, x, y, cs="Viridis", title="", zmin=None, zmax=None, h=None, meta=None):
    """Publication heatmap with adaptive font sizes. If meta is provided, adds a group color bar.
    Height auto-scales to number of features (rows) if h is None."""
    n_y = len(y); n_x = len(x)
    if h is None:
        h = max(350, min(2000, n_y * 16 + 150))
    xsz = adaptive_font(n_x)
    ysz = adaptive_font(n_y)
    ml = adaptive_margin_l(y)
    mb = max(100, min(200, n_x * 4 + 60))  # adaptive bottom margin for rotated x labels
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
        fig.update_layout(template=PUB,height=h,margin=dict(l=ml,b=mb,t=35,r=40))
        fig.update_xaxes(tickangle=45,tickfont=dict(size=xsz), row=2, col=1)
        fig.update_yaxes(tickfont=dict(size=ysz),autorange="reversed", row=2, col=1)
        fig.update_yaxes(tickfont=dict(size=13), row=1, col=1)
        return fig
    else:
        fig = go.Figure(go.Heatmap(z=z,x=x,y=y,colorscale=cs,
            colorbar=dict(thickness=14,len=0.9,title=dict(text=title,side="right",font=dict(size=13)),tickfont=dict(size=12)),
            hoverongaps=False, zmin=zmin, zmax=zmax))
        fig.update_layout(template=PUB,height=h,xaxis=dict(tickangle=45,tickfont=dict(size=xsz)),
                          yaxis=dict(tickfont=dict(size=ysz),autorange="reversed"),margin=dict(l=ml,b=mb,t=35,r=40))
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

def _st(text, sub="", icon=""):
    h3_style = {"color":"#0f1f13","marginTop":"0","marginBottom":"6px","fontSize":"20px","fontWeight":"800",
                "letterSpacing":"-0.3px","display":"flex","alignItems":"center","gap":"8px"}
    h3_children = []
    if icon and icon in ICONS:
        h3_children.append(_svg_icon(icon, size=22))
    h3_children.append(html.Span(text))
    ch = [html.H3(h3_children, style=h3_style)]
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
        return html.Div(style=CS, children=[_st("Peptidoforms (hPF)", icon="peptidoform"),
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
                _st("Top Variable Peptidoforms", icon="fire"), dcc.Graph(figure=vf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"300px"}, children=[
                _st(f"Distribution: {top}"), dcc.Graph(figure=bf)]),
        ]),
        html.Div(style=CS, children=[
            _st("Faceted Violin: Top Variable hPF by Group","Top 6 most variable peptidoforms", icon="violin"),
            dcc.Graph(figure=viol_fig),
        ]),
        html.Div(style=CS, children=[
            _st("Filtered Data Table","Editable | Sortable | Filterable | Export CSV", icon="table"),
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
                _st("Clustered Heatmap","Ward linkage | hPTM ratios | Group color bar", icon="ptm"), dcc.Graph(figure=hm)]),
            html.Div(style={**CS,"flex":"1","minWidth":"450px"}, children=[
                _st("Z-score Heatmap","Row-wise normalization | Group color bar", icon="bar_chart"), dcc.Graph(figure=zhm)]),
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

    # Area normalization: before vs after QN comparison
    areas_log2 = d.get("areas_log2")
    areas_norm = d.get("areas_norm")
    area_children = []
    if areas_log2 is not None:
        la_raw = areas_log2.stack().reset_index(); la_raw.columns = ["PTM","Sample","Log2Area"]
        la_raw = la_raw.merge(meta, on="Sample", how="left")
        ab_raw = px.box(la_raw.dropna(subset=["Log2Area"]), x="Sample", y="Log2Area", color="Group",
                        title="Log2(Area) BEFORE Quantile Normalization", color_discrete_sequence=GC)
        pfig(ab_raw, 380, n_x=len(meta))
        ab_raw.update_layout(xaxis=dict(tickangle=45),showlegend=False)
        area_children.append(html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=ab_raw)]))

    if areas_norm is not None:
        la_qn = areas_norm.stack().reset_index(); la_qn.columns = ["PTM","Sample","Log2Area_QN"]
        la_qn = la_qn.merge(meta, on="Sample", how="left")
        ab_qn = px.box(la_qn.dropna(subset=["Log2Area_QN"]), x="Sample", y="Log2Area_QN", color="Group",
                        title="Log2(Area) AFTER Quantile Normalization", color_discrete_sequence=GC)
        pfig(ab_qn, 380, n_x=len(meta))
        ab_qn.update_layout(xaxis=dict(tickangle=45),showlegend=False)
        area_children.append(html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=ab_qn)]))

    logs = d.get("logs",[]); nc = sum(1 for e in logs if e["n_warnings"]==0)
    nw = sum(1 for e in logs if e["n_warnings"]>0); tw = sum(e["n_warnings"] for e in logs)
    ns = len(meta) if not meta.empty else df.shape[1]; np_ = df.shape[0]
    td = int(binary.sum().sum()); tp = binary.shape[0]*binary.shape[1]
    comp = td/tp*100 if tp>0 else 0

    children = [
        html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}, children=[
            _sc("Samples",str(ns),C["accent"]), _sc("Features",str(np_),C["accent"]),
            _sc("Completeness",f"{comp:.1f}%",C["green"]),
            _sc("Clean Runs",str(nc),C["green"]),
            _sc("Warnings",str(nw),C["red"] if nw>0 else C["green"]),
        ]),
        html.Div(style=CS, children=[_st("Missingness Heatmap","Green=detected | Red=missing", icon="qc"), dcc.Graph(figure=mhm)]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=mb)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=ch)]),
        ]),
    ]
    if area_children:
        children.append(html.Div([
            _st("Area Normalization", "Before vs After Quantile Normalization | log2(MS1 intensity)", icon="normalize"),
            html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=area_children),
        ]))
    return html.Div(children)


# ======================================================================
# TAB: PCA & CLUSTERING (publication quality)
# ======================================================================

def tab_pca(d):
    """PCA & Clustering — layout with Data Source selector."""
    has_areas = "areas_norm" in d and d["areas_norm"] is not None
    source_opts = [{"label":"Ratios (default)","value":"ratios"}]
    if has_areas:
        source_opts.append({"label":"Areas (log2 + QN)","value":"areas_norm"})
        source_opts.append({"label":"Areas (log2 only)","value":"areas_log2"})
    return html.Div([
        html.Div(style={**CS,"display":"flex","gap":"16px","alignItems":"flex-end","flexWrap":"wrap"}, children=[
            html.Div(style={"flex":"1","minWidth":"220px"}, children=[
                _lbl("Data Source"),
                dcc.Dropdown(id="pca-source",options=source_opts,value="ratios",clearable=False,style=DS)]),
        ]),
        html.Div(id="pca-out"),
    ])


@callback(Output("pca-out","children"),
          Input("pca-source","value"), Input("cur-exp","data"), Input("cur-palette","data"),
          prevent_initial_call=True)
def _pca_content(source, exp, pal):
    if not exp or exp not in EXP_DATA: return html.P("N/A")
    d = EXP_DATA[exp]
    df = _get_data_source(d, source)
    meta = d.get("metadata", pd.DataFrame())
    if df is None or df.empty:
        return html.Div(style=CS, children=[html.P("No data for selected source.")])

    is_log = source in ("areas_norm", "areas_log2")
    src_label = {"ratios":"Ratios","areas_norm":"Areas (log2+QN)","areas_log2":"Areas (log2)"}.get(source,"Ratios")

    X = df.T.fillna(0)
    n_comp = min(5, X.shape[1], X.shape[0])
    if n_comp < 2:
        return html.Div(style=CS, children=[html.P("Not enough features for PCA.")])

    pca = skPCA(n_components=n_comp)
    coords = pca.fit_transform(X.values)
    ev = pca.explained_variance_ratio_ * 100

    pc_df = pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"Sample":X.index})
    if n_comp>=3: pc_df["PC3"] = coords[:,2]
    pc_df = pc_df.merge(meta, on="Sample")
    uniq_groups = sorted(pc_df["Group"].unique())
    n_grp = len(uniq_groups)

    # ---- 1. PCA SCATTER (publication quality) ----
    fig1 = px.scatter(pc_df, x="PC1", y="PC2", color="Group", hover_name="Sample",
                      color_discrete_sequence=GC)
    fig1.update_traces(marker=dict(size=12, line=dict(width=1.8, color="white"),
                                    opacity=0.9))
    pfig(fig1, 520, n_groups=n_grp)
    fig1.update_layout(
        xaxis_title=f"PC1 ({ev[0]:.1f}%)", yaxis_title=f"PC2 ({ev[1]:.1f}%)",
        title=dict(text=f"PCA Score Plot ({src_label})", font=dict(size=16)),
        xaxis=dict(zeroline=True, zerolinecolor="#e5e7eb", zerolinewidth=1, showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(zeroline=True, zerolinecolor="#e5e7eb", zerolinewidth=1, showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(font=dict(size=12)))

    # 95% confidence ellipses (proper covariance-based, not axis-aligned)
    for i_grp, grp in enumerate(uniq_groups):
        gd = pc_df[pc_df["Group"] == grp]
        if len(gd) >= 3:
            mx, my = gd["PC1"].mean(), gd["PC2"].mean()
            cov = np.cov(gd["PC1"].values, gd["PC2"].values)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            # 95% chi2 with 2 dof = 5.991
            chi2_val = 5.991
            theta = np.linspace(0, 2 * np.pi, 80)
            a = np.sqrt(eigenvals[1] * chi2_val)
            b = np.sqrt(eigenvals[0] * chi2_val)
            angle = np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])
            ellipse_x = mx + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
            ellipse_y = my + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
            gc_color = GC[i_grp % len(GC)]
            fig1.add_trace(go.Scatter(
                x=ellipse_x, y=ellipse_y, mode="lines", showlegend=False,
                line=dict(width=1.5, dash="dash", color=gc_color), opacity=0.45, hoverinfo="skip"))

    # ---- 2. BIPLOT (publication-quality with gradient arrows) ----
    loadings = pca.components_[:2].T
    load_df = pd.DataFrame({"Feature": df.index, "PC1": loadings[:, 0], "PC2": loadings[:, 1]})
    load_df["mag"] = np.sqrt(load_df["PC1"]**2 + load_df["PC2"]**2)
    top_load = load_df.nlargest(15, "mag")
    scale = np.abs(coords[:, :2]).max() / (np.abs(loadings).max() + 1e-10) * 0.75

    bfig = go.Figure()
    # Sample points (subtle, behind arrows)
    for i_grp, grp in enumerate(uniq_groups):
        gd = pc_df[pc_df["Group"] == grp]
        bfig.add_trace(go.Scatter(x=gd["PC1"], y=gd["PC2"], mode="markers", name=grp,
                                   marker=dict(size=9, opacity=0.35, color=GC[i_grp % len(GC)],
                                               line=dict(width=1, color="white"))))
    # Loading arrows (dark red, elegant)
    for rank, (_, row) in enumerate(top_load.iterrows()):
        ax_val = row["PC1"] * scale
        ay_val = row["PC2"] * scale
        intensity = 0.5 + 0.5 * (1 - rank / len(top_load))
        arrow_color = f"rgba(185, 28, 28, {intensity})"
        bfig.add_annotation(ax=0, ay=0, x=ax_val, y=ay_val, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2, arrowcolor=arrow_color)
        # Feature label with background
        bfig.add_annotation(x=ax_val * 1.12, y=ay_val * 1.12, text=row["Feature"],
                            showarrow=False, font=dict(size=11, color="#991b1b", family=FONT),
                            bgcolor="rgba(255,255,255,0.85)", borderpad=2)
    pfig(bfig, 520, n_groups=n_grp)
    bfig.update_layout(
        xaxis_title=f"PC1 ({ev[0]:.1f}%)", yaxis_title=f"PC2 ({ev[1]:.1f}%)",
        title=dict(text=f"PCA Biplot - Top 15 Loadings ({src_label})", font=dict(size=16)),
        xaxis=dict(zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=1.5),
        yaxis=dict(zeroline=True, zerolinecolor="#d1d5db", zerolinewidth=1.5))

    # ---- 3. SCREE PLOT (extended to all computed components) ----
    all_ev = pca.explained_variance_ratio_
    pc_labels = [f"PC{i+1}" for i in range(len(all_ev))]
    sfig = go.Figure()
    sfig.add_trace(go.Bar(x=pc_labels, y=all_ev * 100, marker_color=C["accent"],
                           name="Individual", text=[f"{v:.1f}%" for v in all_ev * 100],
                           textposition="outside", textfont=dict(size=11)))
    sfig.add_trace(go.Scatter(x=pc_labels, y=np.cumsum(all_ev) * 100, mode="lines+markers",
                              marker=dict(color=C["red"], size=8), line=dict(width=2),
                              name="Cumulative", yaxis="y2"))
    pfig(sfig, 380)
    sfig.update_layout(yaxis_title="Variance Explained (%)",
                       yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
                       title=dict(text="Scree Plot", font=dict(size=16)),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # ---- 4. 3D PCA ----
    fig3d = go.Figure()
    if n_comp >= 3:
        fig3d = px.scatter_3d(pc_df, x="PC1", y="PC2", z="PC3", color="Group",
                               hover_name="Sample", color_discrete_sequence=GC)
        fig3d.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
        pfig(fig3d, 520)
        fig3d.update_layout(
            scene=dict(xaxis_title=f"PC1 ({ev[0]:.1f}%)", yaxis_title=f"PC2 ({ev[1]:.1f}%)",
                       zaxis_title=f"PC3 ({ev[2]:.1f}%)"),
            title=dict(text=f"3D PCA ({src_label})", font=dict(size=16)))

    # ---- 5. HIERARCHICAL CLUSTERING (Ward + colored by group) ----
    try:
        dd = df.T.fillna(0).values
        dist_ = pdist(dd, metric="euclidean"); link_ = linkage(dist_, method="ward")
        dr = scipy_dend(link_, labels=df.columns.tolist(), no_plot=True, color_threshold=0)
        dfig = go.Figure()
        for xc, yc in zip(dr["icoord"], dr["dcoord"]):
            dfig.add_trace(go.Scatter(x=xc, y=yc, mode="lines",
                                       line=dict(color=C["accent"], width=1.8), showlegend=False))
        tp_ = [5 + 10 * i for i in range(len(dr["ivl"]))]
        # Color tick labels by group
        tick_colors = []
        for s in dr["ivl"]:
            grp_match = meta[meta["Sample"] == s]["Group"].values
            if len(grp_match) > 0:
                gi = uniq_groups.index(grp_match[0]) if grp_match[0] in uniq_groups else 0
                tick_colors.append(GC[gi % len(GC)])
            else:
                tick_colors.append(C["text"])
        dfig.update_layout(template=PUB, height=380,
            xaxis=dict(tickmode="array", tickvals=tp_, ticktext=dr["ivl"], tickangle=50,
                       tickfont=dict(size=adaptive_font(len(dr["ivl"])))),
            yaxis_title="Euclidean Distance (Ward)",
            title=dict(text="Hierarchical Clustering (Ward Linkage)", font=dict(size=16)),
            margin=dict(b=130))
    except: dfig = go.Figure(); pfig(dfig, 380)

    # ---- 6. PAM (k-Medoids) CLUSTERING ----
    try:
        from sklearn.metrics import silhouette_score, silhouette_samples
        # Try k from 2 to 6, pick best silhouette
        X_clust = df.T.fillna(0).values
        best_k, best_sil = 2, -1
        sil_scores = {}
        max_k = min(7, X_clust.shape[0] - 1)
        for k_try in range(2, max_k):
            km_try = KMeans(n_clusters=k_try, random_state=42, n_init=10).fit(X_clust)
            s = silhouette_score(X_clust, km_try.labels_)
            sil_scores[k_try] = round(s, 3)
            if s > best_sil:
                best_sil = s; best_k = k_try
        km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_clust)
        sil_vals = silhouette_samples(X_clust, km_best.labels_)

        # Silhouette plot
        sil_fig = go.Figure()
        y_lower = 0
        cluster_colors = [GC[c % len(GC)] for c in range(best_k)]
        for c_idx in range(best_k):
            c_sils = sorted(sil_vals[km_best.labels_ == c_idx])
            y_upper = y_lower + len(c_sils)
            sil_fig.add_trace(go.Bar(
                x=c_sils, y=list(range(y_lower, y_upper)), orientation="h",
                marker_color=cluster_colors[c_idx], name=f"Cluster {c_idx}", showlegend=True,
                width=1.0))
            y_lower = y_upper + 2
        sil_fig.add_vline(x=best_sil, line_dash="dash", line_color=C["red"], annotation_text=f"Mean: {best_sil:.3f}")
        pfig(sil_fig, max(350, X_clust.shape[0] * 8))
        sil_fig.update_layout(
            xaxis_title="Silhouette Coefficient", yaxis=dict(showticklabels=False),
            title=dict(text=f"Silhouette Analysis (K={best_k}, mean={best_sil:.3f})", font=dict(size=16)),
            barmode="stack")

        # Silhouette scores for different k
        sk_fig = go.Figure(go.Bar(
            x=[str(k) for k in sil_scores.keys()],
            y=list(sil_scores.values()),
            marker_color=[C["accent"] if k == best_k else C["muted"] for k in sil_scores.keys()],
            text=[f"{v:.3f}" for v in sil_scores.values()], textposition="outside"))
        pfig(sk_fig, 300)
        sk_fig.update_layout(xaxis_title="Number of Clusters (K)", yaxis_title="Silhouette Score",
                             title=dict(text="Optimal K Selection", font=dict(size=16)))

        # Cluster assignments on PCA space
        pc_df["Cluster"] = [f"C{km_best.labels_[i]}" for i in range(len(km_best.labels_))]
        pam_fig = px.scatter(pc_df, x="PC1", y="PC2", color="Cluster", symbol="Group",
                              hover_name="Sample", color_discrete_sequence=GC)
        pam_fig.update_traces(marker=dict(size=11, line=dict(width=1.5, color="white")))
        pfig(pam_fig, 450, n_groups=best_k)
        pam_fig.update_layout(
            xaxis_title=f"PC1 ({ev[0]:.1f}%)", yaxis_title=f"PC2 ({ev[1]:.1f}%)",
            title=dict(text=f"K-Means Clusters on PCA Space (K={best_k})", font=dict(size=16)))
    except:
        sil_fig = go.Figure(); pfig(sil_fig, 300)
        sk_fig = go.Figure(); pfig(sk_fig, 300)
        pam_fig = go.Figure(); pfig(pam_fig, 350)

    # ---- 7. CORRELATION HEATMAP (Spearman) ----
    corr = df.corr(method="spearman")
    co_ = cluster_order(corr, 0); corr = corr.loc[co_, co_]
    chm = phm(corr.values, corr.columns.tolist(), corr.index.tolist(),
              cs="RdBu_r", title="Spearman rho", zmin=-1, zmax=1, h=max(500, len(corr) * 12))

    # ---- 8. FEATURE DENDROGRAM + K-MEANS ----
    feat_X = df.fillna(0).values; n_feats = feat_X.shape[0]
    try:
        feat_dist = pdist(feat_X, metric="euclidean"); feat_link = linkage(feat_dist, method="ward")
        feat_dr = scipy_dend(feat_link, labels=df.index.tolist(), no_plot=True)
        feat_dfig = go.Figure()
        for xc, yc in zip(feat_dr["icoord"], feat_dr["dcoord"]):
            feat_dfig.add_trace(go.Scatter(x=yc, y=xc, mode="lines",
                                            line=dict(color=C["green"], width=1.3), showlegend=False))
        tp_f = [5 + 10 * i for i in range(len(feat_dr["ivl"]))]
        feat_dfig.update_layout(template=PUB, height=max(400, n_feats * 14),
            yaxis=dict(tickmode="array", tickvals=tp_f, ticktext=feat_dr["ivl"],
                       tickfont=dict(size=adaptive_font(n_feats))),
            xaxis_title="Distance (Ward)", title=dict(text="Feature Dendrogram", font=dict(size=16)),
            margin=dict(l=adaptive_margin_l(feat_dr["ivl"])))
    except: feat_dfig = go.Figure(); pfig(feat_dfig, 350)

    k_vals = min(6, max(2, n_feats // 5))
    try:
        km = KMeans(n_clusters=k_vals, random_state=42, n_init=10).fit(feat_X)
        groups_list = sorted(meta["Group"].unique())
        group_means = pd.DataFrame(index=df.index)
        for g in groups_list:
            samps_g = meta[meta["Group"] == g]["Sample"].tolist()
            cols_g = [c for c in df.columns if c in samps_g]
            if cols_g: group_means[g] = df[cols_g].mean(axis=1)
        group_means["Cluster"] = km.labels_
        cluster_means = group_means.groupby("Cluster")[groups_list].mean()
        km_hm = phm(cluster_means.values, groups_list,
                     [f"Cluster {i}" for i in cluster_means.index],
                     cs="Greens", title=f"Feature Clusters (K={k_vals}) x Group Means",
                     h=max(250, k_vals * 50))
    except: km_hm = go.Figure(); pfig(km_hm, 300)

    # ---- 9. BICLUSTERING ----
    try:
        Z = df.fillna(0).values
        n_r, n_c = Z.shape
        n_bic = min(4, n_r, n_c)
        if n_bic >= 2 and n_r >= 4 and n_c >= 4:
            bic = SpectralBiclustering(n_clusters=n_bic, random_state=42, method="log")
            bic.fit(Z + np.abs(Z.min()) + 1e-6)
            ro = np.argsort(bic.row_labels_); co = np.argsort(bic.column_labels_)
            bic_z = Z[ro][:, co]
            bic_x = [df.columns[i] for i in co]
            bic_y = [df.index[i] for i in ro]
            bic_hm = phm(bic_z, bic_x, bic_y, cs="YlGnBu",
                         title=f"Biclustering (n={n_bic})", h=max(500, len(bic_y) * 14), meta=meta)
            row_labels_sorted = bic.row_labels_[ro]; col_labels_sorted = bic.column_labels_[co]
            for i in range(1, len(row_labels_sorted)):
                if row_labels_sorted[i] != row_labels_sorted[i - 1]:
                    bic_hm.add_hline(y=i - 0.5, line_dash="dot", line_color="white", line_width=2)
            for i in range(1, len(col_labels_sorted)):
                if col_labels_sorted[i] != col_labels_sorted[i - 1]:
                    bic_hm.add_vline(x=i - 0.5, line_dash="dot", line_color="white", line_width=2)
        else:
            bic_hm = go.Figure()
            bic_hm.add_annotation(text=f"Need >= 4 features and 4 samples (have {n_r}x{n_c})",
                                  xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                                  font=dict(size=14, color=C["muted"]))
            pfig(bic_hm, 300)
    except Exception as bic_err:
        bic_hm = go.Figure()
        bic_hm.add_annotation(text=f"Biclustering error: {str(bic_err)[:80]}",
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                              font=dict(size=13, color=C["red"]))
        pfig(bic_hm, 300)

    # ---- ASSEMBLE LAYOUT ----
    children = [
        html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"12px"}, children=[
            _sc("Features", str(df.shape[0]), C["accent"]),
            _sc("Samples", str(df.shape[1]), C["h4"]),
            _sc("Components", str(n_comp), C["h3"]),
            _sc("Var PC1+PC2", f"{ev[0]+ev[1]:.1f}%", C["warn"]),
            _sc("Data", src_label, "#6b7280"),
        ]),
        _st("Principal Component Analysis", f"Score plot with 95% confidence ellipses (chi-squared) | {src_label}", icon="pca"),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"520px"}, children=[dcc.Graph(figure=fig1)]),
            html.Div(style={**CS,"flex":"1","minWidth":"520px"}, children=[dcc.Graph(figure=bfig)]),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"350px"}, children=[dcc.Graph(figure=sfig)]),
            html.Div(style={**CS,"flex":"2","minWidth":"520px"}, children=[dcc.Graph(figure=dfig)]),
        ]),
    ]
    if n_comp >= 3:
        children.append(html.Div(style=CS, children=[dcc.Graph(figure=fig3d)]))

    # Clustering section
    children.append(_st("Sample Clustering & Silhouette Analysis",
                         f"Optimal K selected by silhouette score | Ward linkage dendrogram", icon="bar_chart"))
    children.append(html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
        html.Div(style={**CS,"flex":"1","minWidth":"450px"}, children=[dcc.Graph(figure=pam_fig)]),
        html.Div(style={**CS,"flex":"1","minWidth":"350px"}, children=[dcc.Graph(figure=sil_fig)]),
    ]))
    children.append(html.Div(style={**CS,"maxWidth":"500px"}, children=[dcc.Graph(figure=sk_fig)]))

    children.append(_st("Feature Correlation", icon="nodes"))
    children.append(html.Div(style=CS, children=[dcc.Graph(figure=chm)]))

    children.append(_st("Feature Clustering",
                         f"Ward dendrogram + K-Means (K={k_vals}) group means + Spectral biclustering", icon="puzzle"))
    children.append(html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
        html.Div(style={**CS,"flex":"1","minWidth":"420px"}, children=[dcc.Graph(figure=feat_dfig)]),
        html.Div(style={**CS,"flex":"1","minWidth":"380px"}, children=[dcc.Graph(figure=km_hm)]),
    ]))
    children.append(html.Div(style=CS, children=[dcc.Graph(figure=bic_hm)]))
    return html.Div(children)


# ======================================================================
# TAB: STATISTICS
# ======================================================================

def _enrich_stats(res, groups, is_log=False):
    """Add classification columns to statistics results.
    is_log: if True, data is log-scale so FC = difference of means."""
    histone_col, ptm_type_col, direction_col = [], [], []
    for _, row in res.iterrows():
        h, t = classify_ptm_name(row["PTM"])
        histone_col.append(h); ptm_type_col.append(t)
        means = [row.get(f"mean_{g}", np.nan) for g in groups]
        if is_log:
            means_valid = [(g, m) for g, m in zip(groups, means) if not np.isnan(m) and np.isfinite(m)]
        else:
            means_valid = [(g, m) for g, m in zip(groups, means) if not np.isnan(m) and m > 0]
        if len(means_valid) >= 2:
            max_g = max(means_valid, key=lambda x: x[1])
            min_g = min(means_valid, key=lambda x: x[1])
            if is_log:
                fc = max_g[1] - min_g[1]  # already log-scale
            else:
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

    # Check if Design / Tissue columns exist (for stratified analysis)
    designs = sorted(meta["Design"].unique()) if "Design" in meta.columns and meta["Design"].nunique() > 1 else []
    tissues = sorted(meta["Tissue"].unique()) if "Tissue" in meta.columns and meta["Tissue"].nunique() > 1 else []

    # Data source selector
    has_areas = "areas_norm" in d and d["areas_norm"] is not None
    source_opts = [{"label":"Ratios (default)","value":"ratios"}]
    if has_areas:
        source_opts.append({"label":"Areas (log2+QN)","value":"areas_norm"})
        source_opts.append({"label":"Areas (log2 only)","value":"areas_log2"})

    # Build filter controls
    filter_children = [
        html.Div(style={"flex":"1","minWidth":"200px"}, children=[
            _lbl("Data Source"),
            dcc.Dropdown(id="stats-source",options=source_opts,value="ratios",clearable=False,style=DS)]),
    ]
    if designs:
        filter_children.append(html.Div(style={"flex":"1","minWidth":"220px"}, children=[
            _lbl("Design / Strain"),
            dcc.Dropdown(id="stats-design",
                options=[{"label":"All (pooled - CAUTION)","value":"All"}]+[{"label":f"Design {x}","value":str(x)} for x in designs],
                value=str(designs[0]), clearable=False, style=DS)]))
    else:
        filter_children.append(html.Div(dcc.Store(id="stats-design", data="none")))
    if tissues:
        filter_children.append(html.Div(style={"flex":"1","minWidth":"180px"}, children=[
            _lbl("Tissue"),
            dcc.Dropdown(id="stats-tissue",
                options=[{"label":"All tissues","value":"All"}]+[{"label":t,"value":t} for t in tissues],
                value="All", clearable=False, style=DS)]))
    else:
        filter_children.append(html.Div(dcc.Store(id="stats-tissue", data="All")))

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
          Input("stats-source","value"), Input("stats-design","value"), Input("stats-tissue","value"),
          Input("stats-show","value"),
          Input("stats-classify","value"), Input("stats-fdr","value"),
          Input("cur-exp","data"), prevent_initial_call=True)
def _stats_filtered(source, design, tissue, show, classify, fdr_log, exp):
    t0 = time.time()
    if not exp or exp not in EXP_DATA: return html.P("N/A")
    d = EXP_DATA[exp]
    df = _get_data_source(d, source)
    meta = d.get("metadata", pd.DataFrame())
    is_log = source in ("areas_norm", "areas_log2")
    if df is None or meta.empty: return html.P("No data for selected source.")

    fdr_thresh = 10 ** fdr_log if fdr_log else 0.05

    # Filter to design if applicable
    if design and design not in ("All", "none") and "Design" in meta.columns:
        meta = meta[meta["Design"].astype(str) == str(design)].copy()
        samps = meta["Sample"].tolist()
        cols = [c for c in df.columns if c in samps]; df = df[cols]

    # Filter to tissue if applicable
    if tissue and tissue != "All" and "Tissue" in meta.columns:
        meta = meta[meta["Tissue"] == tissue].copy()
        samps = meta["Sample"].tolist()
        cols = [c for c in df.columns if c in samps]; df = df[cols]

    groups = sorted(meta["Group"].unique())
    if len(groups) < 2:
        return html.P(f"Only {len(groups)} group(s). Need >= 2.", style={"color":C["red"],"padding":"20px"})

    res = robust_group_test(df, meta, groups, is_log=is_log)
    if res.empty: return html.P("Could not compute statistics.", style={"color":C["red"]})

    res = _enrich_stats(res, groups, is_log=is_log)

    # Compute FC for volcano
    fc_list = []
    for _, row in res.iterrows():
        means = [row.get(f"mean_{g}", np.nan) for g in groups]
        means = [m for m in means if not np.isnan(m)]
        if is_log:
            # Data already log-scale: FC = max - min (difference of logs)
            means = [m for m in means if np.isfinite(m)]
            fc_list.append(max(means) - min(means) if len(means) >= 2 else 0)
        else:
            means = [m for m in means if m > 0]
            fc_list.append(np.log2(max(means) / min(means)) if len(means) >= 2 else 0)
    res["maxLog2FC"] = fc_list

    n_total = len(res)
    n_sig = int((res["KW_FDR"] < fdr_thresh).sum())
    n_up = int((res["Direction"].str.startswith("Up") & (res["KW_FDR"] < fdr_thresh)).sum())
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
        _st("Statistical Results", f"Kruskal-Wallis + BH FDR | {show} | Editable | Export CSV", icon="bar_chart"),
        make_table(display_res, "stats-table")]))

    return html.Div(children)


@callback(Output("download-data","data"),
          Input("stats-export","n_clicks"),
          State("stats-source","value"), State("stats-design","value"),
          State("stats-show","value"), State("stats-fdr","value"),
          State("cur-exp","data"), prevent_initial_call=True)
def _stats_export(n, source, design, show, fdr_log, exp):
    if not n or not exp or exp not in EXP_DATA: return no_update
    d = EXP_DATA[exp]; df = _get_data_source(d, source); meta = d.get("metadata", pd.DataFrame())
    is_log = source in ("areas_norm", "areas_log2")
    if df is None: return no_update
    fdr_thresh = 10 ** fdr_log if fdr_log else 0.05
    # Apply design filter
    if design and design not in ("All", "none") and "Design" in meta.columns:
        meta = meta[meta["Design"].astype(str) == str(design)].copy()
        samps = meta["Sample"].tolist()
        cols = [c for c in df.columns if c in samps]; df = df[cols]
    groups = sorted(meta["Group"].unique()) if not meta.empty else []
    if len(groups) < 2: return no_update
    res = robust_group_test(df, meta, groups, is_log=is_log)
    if res.empty: return no_update
    res = _enrich_stats(res, groups, is_log=is_log)
    # Compute FC
    fc_list = []
    for _, row in res.iterrows():
        means = [row.get(f"mean_{g}", np.nan) for g in groups]
        means = [m for m in means if not np.isnan(m)]
        if is_log:
            means = [m for m in means if np.isfinite(m)]
            fc_list.append(max(means) - min(means) if len(means) >= 2 else 0)
        else:
            means = [m for m in means if m > 0]
            fc_list.append(np.log2(max(means) / min(means)) if len(means) >= 2 else 0)
    res["maxLog2FC"] = fc_list
    # Apply show filter
    if show == "sig":
        res = res[res["KW_FDR"] < fdr_thresh]
    elif show == "up":
        res = res[(res["KW_FDR"] < fdr_thresh) & res["Direction"].str.startswith("Up")]
    elif show == "down":
        res = res[(res["KW_FDR"] < fdr_thresh) & ~res["Direction"].str.startswith("Up")]
    src_label = source if source else "ratios"
    fname = f"{exp.split('(')[0].strip()}_stats_{src_label}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    log_analysis(exp, "export_stats", {"format":"csv","source":src_label,"n_rows":len(res)},
                 len(res), 0, f"Exported {len(res)} rows ({src_label})")
    return dcc.send_data_frame(res.to_csv, fname, index=False)


# ======================================================================
# TAB: UPSET / CO-OCCURRENCE
# ======================================================================

def tab_upset(d):
    hpf_meta = d.get("hpf_meta", pd.DataFrame())
    hpf = d.get("hpf", pd.DataFrame())
    meta = d.get("metadata", pd.DataFrame())
    hptm = d.get("hptm")

    if hpf_meta.empty or hpf.empty:
        return html.Div(style=CS, children=[html.P("No peptidoform data for UpSet analysis.")])

    groups = sorted(meta["Group"].unique())
    children = []
    DET_THRESH = 0.005  # detection threshold for ratios

    # ============================================
    # SECTION 1: Sample-level PTM Co-occurrence from combinatorial hPF
    # ============================================
    # For each sample, which PTM pairs ACTUALLY co-occur on the same peptide?
    # This uses the peptidoform-level data (hPF), not aggregated hPTM.
    combos = hpf_meta[hpf_meta["is_combo"]].copy() if "is_combo" in hpf_meta.columns else pd.DataFrame()

    if not combos.empty and not hpf.empty:
        # Count: for each PTM pair, in how many samples is a combinatorial
        # peptidoform carrying BOTH marks detected above threshold?
        pair_sample_counts = {}   # pair -> set of samples where co-detected
        pair_peptidoforms = {}    # pair -> list of peptidoform names carrying both

        for _, row in combos.iterrows():
            ptms = row.get("individual_ptms", [])
            name = row.get("name", "")
            if len(ptms) < 2 or name not in hpf.index:
                continue
            # Which samples have this combinatorial hPF detected?
            vals = hpf.loc[name]
            detected_samples = set(vals[vals > DET_THRESH].index)
            if not detected_samples:
                continue
            for p1, p2 in combinations(sorted(ptms), 2):
                key = f"{p1} + {p2}"
                if key not in pair_sample_counts:
                    pair_sample_counts[key] = set()
                    pair_peptidoforms[key] = []
                pair_sample_counts[key].update(detected_samples)
                pair_peptidoforms[key].append(name)

        if pair_sample_counts:
            # Build table: pair, n_samples, n_peptidoforms, fraction of total samples
            total_samps = hpf.shape[1]
            pair_data = []
            for pair, samps in pair_sample_counts.items():
                pair_data.append({
                    "PTM_Pair": pair,
                    "Samples_Detected": len(samps),
                    "Pct_Samples": round(100 * len(samps) / total_samps, 1),
                    "N_Peptidoforms": len(set(pair_peptidoforms[pair])),
                    "Peptidoforms": ", ".join(sorted(set(pair_peptidoforms[pair])))[:80],
                })
            pair_df = pd.DataFrame(pair_data).sort_values("Samples_Detected", ascending=False).head(30)

            n_pairs = len(pair_df)
            # Bar chart: co-occurrence by sample count
            uf = go.Figure(go.Bar(
                x=pair_df["Samples_Detected"].values, y=pair_df["PTM_Pair"].values,
                orientation="h", text=pair_df["Pct_Samples"].apply(lambda x: f"{x}%"),
                textposition="outside", textfont=dict(size=11),
                marker=dict(color=pair_df["Samples_Detected"].values, colorscale="Viridis",
                            line=dict(width=0), colorbar=dict(title="Samples"))))
            pfig(uf, max(400, n_pairs * 22), n_y=n_pairs)
            uf.update_layout(
                yaxis=dict(autorange="reversed", tickfont=dict(size=adaptive_font(n_pairs))),
                margin=dict(l=adaptive_margin_l(pair_df["PTM_Pair"].tolist())),
                xaxis_title="# Samples with Co-detection",
                title=dict(text="PTM Co-occurrence on Same Peptide (sample-level)", font=dict(size=16)))

            children.append(html.Div(style=CS, children=[
                _st("Peptidoform-level Co-occurrence",
                    f"{len(pair_sample_counts)} unique PTM pairs from {len(combos)} combinatorial peptidoforms | "
                    f"Co-detection = both marks present (ratio > {DET_THRESH}) in the same sample",
                    icon="nodes"),
                dcc.Graph(figure=uf),
                html.Details([
                    html.Summary("Show pair details table", style={"cursor":"pointer","color":C["accent"],
                                  "fontSize":"13px","fontWeight":"600","marginTop":"8px"}),
                    make_table(pair_df.drop(columns=["Peptidoforms"], errors="ignore"), "cooccur-hpf-table"),
                ]),
            ]))

    # ============================================
    # SECTION 2: Pairwise Co-occurrence / Mutual Exclusivity (hPTM level)
    # ============================================
    if hptm is not None and not meta.empty:
        # Binary detection matrix: is PTM present above threshold in each sample?
        binary = (hptm > DET_THRESH).astype(int)
        ptm_names = list(binary.index)
        n_ptms = len(ptm_names)
        n_samps = binary.shape[1]

        # Filter to PTMs detected in at least 2 samples (otherwise pairwise is meaningless)
        ptm_det_count = binary.sum(axis=1)
        keep_mask = ptm_det_count >= 2
        binary_f = binary.loc[keep_mask]
        ptm_names_f = list(binary_f.index)
        n_ptms_f = len(ptm_names_f)

        if n_ptms_f >= 3:
            # Compute 2x2 contingency table for each pair
            # a = both detected, b = A only, c = B only, d = neither
            # Odds ratio = (a*d) / (b*c), with Haldane correction (+0.5)
            # Fisher exact test for significance
            jaccard = np.zeros((n_ptms_f, n_ptms_f))
            log_or = np.zeros((n_ptms_f, n_ptms_f))
            pval_mat = np.ones((n_ptms_f, n_ptms_f))

            pair_results = []
            for i in range(n_ptms_f):
                ai = binary_f.iloc[i].values.astype(bool)
                for j in range(i + 1, n_ptms_f):
                    bj = binary_f.iloc[j].values.astype(bool)
                    a = int(np.sum(ai & bj))       # both
                    b = int(np.sum(ai & ~bj))      # A only
                    c = int(np.sum(~ai & bj))      # B only
                    dd = int(np.sum(~ai & ~bj))    # neither
                    union = a + b + c
                    jac = a / (union + 1e-10)
                    jaccard[i, j] = jaccard[j, i] = jac

                    # Odds ratio with Haldane correction
                    oratio = ((a + 0.5) * (dd + 0.5)) / ((b + 0.5) * (c + 0.5))
                    lor = np.log2(oratio) if oratio > 0 else 0
                    log_or[i, j] = log_or[j, i] = lor

                    # Fisher exact test (two-sided)
                    try:
                        _, pval = fisher_exact([[a, b], [c, dd]])
                    except:
                        pval = 1.0
                    pval_mat[i, j] = pval_mat[j, i] = pval

                    pair_results.append({
                        "PTM_A": ptm_names_f[i], "PTM_B": ptm_names_f[j],
                        "Both": a, "A_only": b, "B_only": c, "Neither": dd,
                        "Jaccard": round(jac, 4), "Log2_OR": round(lor, 3),
                        "Fisher_p": pval,
                    })

            # FDR correction on Fisher p-values
            if pair_results:
                pvals = [r["Fisher_p"] for r in pair_results]
                try:
                    _, fdr_vals, _, _ = multipletests(pvals, method="fdr_bh")
                except:
                    fdr_vals = pvals
                for r, fdr in zip(pair_results, fdr_vals):
                    r["FDR"] = round(fdr, 6)

            # Cluster and plot the log2 OR matrix
            np.fill_diagonal(log_or, 0)
            try:
                dist_mat = 1 - jaccard
                np.fill_diagonal(dist_mat, 0)
                link = linkage(squareform(dist_mat), method="average")
                order = leaves_list(link)
                ptm_ordered = [ptm_names_f[i] for i in order]
                log_or_ordered = log_or[order][:, order]
            except:
                ptm_ordered = ptm_names_f
                log_or_ordered = log_or

            # Clamp extreme values for visual clarity
            log_or_clamped = np.clip(log_or_ordered, -6, 6)
            me_hm = phm(log_or_clamped, ptm_ordered, ptm_ordered,
                         cs="RdBu_r", title="log2(OR)", zmin=-6, zmax=6)
            me_hm.update_layout(title=dict(
                text="Co-occurrence / Mutual Exclusivity (log2 Odds Ratio, Fisher + FDR)",
                font=dict(size=16)))

            # Classify pairs
            res_df = pd.DataFrame(pair_results)
            sig_co = res_df[(res_df["Log2_OR"] > 1) & (res_df["FDR"] < 0.05) & (res_df["Both"] >= 2)].sort_values("Log2_OR", ascending=False)
            sig_ex = res_df[(res_df["Log2_OR"] < -1) & (res_df["FDR"] < 0.05)].sort_values("Log2_OR", ascending=True)
            ns_pairs = len(res_df) - len(sig_co) - len(sig_ex)

            # Summary cards
            summary = html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"16px"}, children=[
                _sc("PTMs Analyzed", str(n_ptms_f), C["accent"]),
                _sc("Total Pairs", str(len(pair_results)), C["h4"]),
                _sc("Sig. Co-occurring", str(len(sig_co)), C["green"]),
                _sc("Sig. Exclusive", str(len(sig_ex)), C["red"]),
                _sc("Not Significant", str(ns_pairs), C["muted"]),
                _sc("Samples", str(n_samps), "#6b7280"),
            ])

            children.append(html.Div(style={"marginTop":"32px"}, children=[
                _st("Co-occurrence & Mutual Exclusivity",
                    "Fisher exact test with BH-FDR correction | "
                    "Positive log2(OR) = co-occurring | Negative = mutually exclusive | "
                    f"Detection threshold: ratio > {DET_THRESH}",
                    icon="dna"),
            ]))
            children.append(summary)
            children.append(html.Div(style=CS, children=[dcc.Graph(figure=me_hm)]))

            # Tables: sig co-occurring and sig exclusive
            co_table = html.Div()
            ex_table = html.Div()
            if not sig_co.empty:
                co_show = sig_co[["PTM_A","PTM_B","Both","Jaccard","Log2_OR","FDR"]].head(25).copy()
                co_show["FDR"] = co_show["FDR"].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
                co_table = html.Div(style=CS, children=[
                    _st("Significantly Co-occurring Pairs",
                        f"{len(sig_co)} pairs | log2(OR) > 1, FDR < 0.05, co-detected in >= 2 samples",
                        icon="handshake"),
                    make_table(co_show, "cooccur-sig-table")])
            if not sig_ex.empty:
                ex_show = sig_ex[["PTM_A","PTM_B","Both","A_only","B_only","Jaccard","Log2_OR","FDR"]].head(25).copy()
                ex_show["FDR"] = ex_show["FDR"].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
                ex_table = html.Div(style=CS, children=[
                    _st("Significantly Mutually Exclusive Pairs",
                        f"{len(sig_ex)} pairs | log2(OR) < -1, FDR < 0.05",
                        icon="no_entry"),
                    make_table(ex_show, "excl-sig-table")])

            children.append(html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
                html.Div(style={"flex":"1","minWidth":"420px"}, children=[co_table]),
                html.Div(style={"flex":"1","minWidth":"420px"}, children=[ex_table]),
            ]))

            # Full results table (collapsible)
            if not res_df.empty:
                full_show = res_df[["PTM_A","PTM_B","Both","A_only","B_only","Neither","Jaccard","Log2_OR","FDR"]].copy()
                full_show["FDR"] = full_show["FDR"].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
                full_show = full_show.sort_values("Log2_OR", ascending=False)
                children.append(html.Details(style={"marginTop":"12px"}, children=[
                    html.Summary("Show all pairwise results", style={"cursor":"pointer","color":C["accent"],
                                  "fontSize":"13px","fontWeight":"600"}),
                    html.Div(style=CS, children=[make_table(full_show, "cooccur-full-table")]),
                ]))

    # ============================================
    # SECTION 3: Per-Group Detection Profile
    # ============================================
    if hptm is not None and not meta.empty:
        # For each group: a PTM is "detected" if ratio > threshold in >= 50% of samples
        det_data = {}
        for g in groups:
            samps = meta[meta["Group"] == g]["Sample"].tolist()
            cols = [c for c in hptm.columns if c in samps]
            if cols:
                frac_detected = (hptm[cols] > DET_THRESH).mean(axis=1)
                det_data[g] = (frac_detected >= 0.5).astype(int)  # robust: majority rule
        det_df = pd.DataFrame(det_data)

        if not det_df.empty:
            # UpSet-style: which groups share which PTMs
            patterns = {}
            ptm_pattern_map = {}
            for ptm in det_df.index:
                pat = tuple(det_df.loc[ptm].values)
                detected_groups = [g for g, v in zip(groups, pat) if v == 1]
                key = " & ".join(detected_groups) if detected_groups else "Not detected"
                patterns[key] = patterns.get(key, 0) + 1
                ptm_pattern_map.setdefault(key, []).append(ptm)

            sp = sorted(patterns.items(), key=lambda x: -x[1])[:25]
            pat_names = [p[0] for p in sp]
            pat_vals = [p[1] for p in sp]
            n_pats = len(pat_names)

            # Color by number of groups involved
            n_groups_in_pat = [len(p[0].split(" & ")) if p[0] != "Not detected" else 0 for p in sp]
            pf = go.Figure(go.Bar(
                x=pat_vals, y=pat_names, orientation="h",
                text=[f"  {v}" for v in pat_vals], textposition="outside",
                marker=dict(color=n_groups_in_pat, colorscale="Greens",
                            line=dict(width=0.5, color="#e5e7eb"),
                            colorbar=dict(title="Groups"))))
            pfig(pf, max(350, n_pats * 24), n_y=n_pats)
            pf.update_layout(
                yaxis=dict(autorange="reversed", tickfont=dict(size=adaptive_font(n_pats))),
                margin=dict(l=adaptive_margin_l(pat_names)), xaxis_title="# PTMs",
                title=dict(text="PTM Detection Patterns (>= 50% samples in group)", font=dict(size=16)))

            # Summary: how many ubiquitous, group-specific, shared
            n_all = patterns.get(" & ".join(groups), 0)
            n_specific = sum(v for k, v in patterns.items()
                             if k != "Not detected" and " & " not in k and k != " & ".join(groups))
            n_shared = sum(v for k, v in patterns.items()
                           if " & " in k and k != " & ".join(groups))
            n_none = patterns.get("Not detected", 0)

            det_summary = html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"12px"}, children=[
                _sc("Ubiquitous (all groups)", str(n_all), C["accent"]),
                _sc("Group-specific", str(n_specific), C["warn"]),
                _sc("Shared (2+ groups)", str(n_shared), C["h4"]),
                _sc("Not detected", str(n_none), C["muted"]),
            ])

            children.append(html.Div(style=CS, children=[
                _st("PTM Detection Patterns",
                    "A PTM is 'detected' in a group if ratio > threshold in >= 50% of group samples | "
                    "Ubiquitous = present in all groups | Group-specific = unique to one group",
                    icon="spectra_compare"),
                det_summary,
                dcc.Graph(figure=pf),
            ]))

            # Group-specific PTMs table (biological insight)
            specific_data = []
            for g in groups:
                if g in patterns:
                    ptms_in_g = ptm_pattern_map.get(g, [])
                    for ptm in ptms_in_g:
                        # Get mean ratio in this group vs others
                        g_samps = meta[meta["Group"] == g]["Sample"].tolist()
                        g_cols = [c for c in hptm.columns if c in g_samps]
                        other_cols = [c for c in hptm.columns if c not in g_samps]
                        mean_g = hptm.loc[ptm, g_cols].mean() if g_cols else 0
                        mean_o = hptm.loc[ptm, other_cols].mean() if other_cols else 0
                        specific_data.append({
                            "PTM": ptm, "Specific_to": g,
                            "Mean_in_group": round(mean_g, 4),
                            "Mean_others": round(mean_o, 5),
                            "Fold_enrichment": round(mean_g / (mean_o + 1e-8), 1),
                        })
            if specific_data:
                spec_df = pd.DataFrame(specific_data).sort_values("Fold_enrichment", ascending=False)
                children.append(html.Div(style=CS, children=[
                    _st("Group-Specific PTMs",
                        "PTMs detected exclusively in one experimental group (>= 50% of samples)",
                        icon="tag"),
                    make_table(spec_df, "group-specific-table"),
                ]))

    # ============================================
    # SECTION 4: Modification complexity per sample
    # ============================================
    n_mods_per_sample = []
    for col in hpf.columns:
        vals = hpf[col].dropna()
        n_detected = int((vals > DET_THRESH).sum())
        combo_detected = 0
        if not hpf_meta.empty and "is_combo" in hpf_meta.columns:
            combo_names = hpf_meta[hpf_meta["is_combo"]]["name"].tolist()
            combo_vals = hpf.loc[[n for n in combo_names if n in hpf.index], col]
            combo_detected = int((combo_vals > DET_THRESH).sum())
        n_mods_per_sample.append({"Sample": col, "Total_hPF": n_detected,
                                   "Combo_hPF": combo_detected,
                                   "Single_hPF": n_detected - combo_detected})

    ndf = pd.DataFrame(n_mods_per_sample).merge(meta, on="Sample", how="left")
    n_samps_plot = len(ndf)
    cf = px.bar(ndf, x="Sample", y=["Single_hPF", "Combo_hPF"],
                color_discrete_sequence=[C["accent"], C["warn"]],
                title="Peptidoform Complexity per Sample",
                barmode="stack", labels={"value":"# Detected hPF","variable":"Type"})
    pfig(cf, 380, n_x=n_samps_plot)
    cf.update_layout(xaxis=dict(tickangle=45, tickfont=dict(size=adaptive_font(n_samps_plot))),
                     yaxis_title="# Detected hPF",
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Complexity by group (box plot)
    if "Group" in ndf.columns:
        bx = px.box(ndf, x="Group", y="Total_hPF", color="Group", points="all",
                     color_discrete_sequence=GC,
                     title="Peptidoform Complexity by Group",
                     labels={"Total_hPF":"# Detected hPF"})
        pfig(bx, 350)
        bx.update_layout(showlegend=False)
        children.append(html.Div(style=CS, children=[
            _st("Modification Complexity",
                "How many peptidoforms are detected per sample? | "
                "Single = one PTM | Combo = multiple PTMs on same peptide",
                icon="dna"),
            html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
                html.Div(style={"flex":"1.2","minWidth":"500px"}, children=[dcc.Graph(figure=cf)]),
                html.Div(style={"flex":"0.8","minWidth":"350px"}, children=[dcc.Graph(figure=bx)]),
            ]),
        ]))
    else:
        children.append(html.Div(style=CS, children=[
            _st("Modification Complexity", icon="dna"),
            dcc.Graph(figure=cf)]))

    return html.Div(children)


# ======================================================================
# TAB: REGION MAP (Modification Landscape with Sequence Context)
# ======================================================================

# Arabidopsis thaliana histone sequences (canonical) for annotation
_HISTONE_SEQS = {
    "H3": ("ARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPGTVALRE"
            "IRRYQKSTELLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEASEAY"
            "LVGLFEDTNLCAIHAKRVTIMPKDIQLARRIRGERA"),
    "H3.3": ("ARTKQTARKSTGGKAPRKQLATKAARKSAPSTGGVKKPHRYRPGTVALRE"
             "IRRYQKSTELLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEASEAY"
             "LVGLFEDTNLCAIHAKRVTIMPKDIQLARRIRGERA"),
    "H4": ("SGRGKGGKGLGKGGAKRHRKVLRDNIQGITKPAIRRLARRGGVKRISGL"
            "IYEETRGVLKVFLENVIRDAVTYTEHAKRKTVTAMDVVYALKRQGRTLY"
            "GFGG"),
}

def _parse_region_positions(region):
    """Extract start/end amino acid positions from region name like H3_3_8."""
    m = re.match(r'H\d+(?:\.\d+)?_(\d+)_(\d+)', region.replace("H33", "H3.3"))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def tab_region(d):
    hpf_meta = d.get("hpf_meta", pd.DataFrame())
    hpf = d.get("hpf", pd.DataFrame())
    hdp_list = d.get("hdp_list", [])
    meta = d.get("metadata", pd.DataFrame())

    if hpf_meta.empty or hpf.empty:
        return html.Div(style=CS, children=[html.P("No peptidoform data for region mapping.")])

    groups = sorted(meta["Group"].unique()) if not meta.empty else []

    # ---- Build region statistics ----
    region_stats = []
    for region in sorted(hpf_meta["region"].unique()):
        rmeta = hpf_meta[hpf_meta["region"] == region]
        n_total = len(rmeta)
        n_combo = int(rmeta["is_combo"].sum())
        n_single = int(((~rmeta["is_combo"]) & (rmeta["modification"] != "unmod")).sum())
        n_unmod = int((rmeta["modification"] == "unmod").sum())
        rnames = rmeta["name"].tolist()
        rdata = hpf.loc[[n for n in rnames if n in hpf.index]]
        mean_ratio = rdata.values.mean() if not rdata.empty else 0
        # Collect individual PTM types in this region
        all_ptms = []
        for _, row in rmeta.iterrows():
            if isinstance(row.get("individual_ptms"), list):
                all_ptms.extend(row["individual_ptms"])
        unique_ptms = sorted(set(all_ptms))
        histone = rmeta["histone"].iloc[0] if len(rmeta) > 0 else ""
        seq = ""
        for hdp in hdp_list:
            if hdp.get("region") == region:
                seq = hdp.get("sequence", ""); break
        start, end = _parse_region_positions(region)

        region_stats.append({
            "Region": region, "Histone": histone, "Sequence": seq,
            "Start": start, "End": end,
            "Total_hPF": n_total, "Single": n_single, "Combo": n_combo,
            "Unmod": n_unmod, "Mean_Ratio": round(mean_ratio, 4),
            "Unique_PTMs": ", ".join(unique_ptms), "n_unique_PTMs": len(unique_ptms),
        })

    rdf = pd.DataFrame(region_stats).sort_values("Region")

    children = []

    # Summary cards
    children.append(html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}, children=[
        _sc("Regions", str(len(rdf)), C["accent"]),
        _sc("Total hPF", str(int(rdf["Total_hPF"].sum())), C["h3"]),
        _sc("Unique PTMs", str(int(rdf["n_unique_PTMs"].sum())), C["warn"]),
        _sc("Histones", str(rdf["Histone"].nunique()), "#6b7280"),
    ]))

    # ============================================================
    # SECTION 1: Sequence Coverage Map (histone track diagram)
    # ============================================================
    hist_colors_map = {"H3": "#2563eb", "H3.3": "#d97706", "H4": "#059669", "Other": "#6b7280"}
    for hist_name in sorted(rdf["Histone"].unique()):
        if not hist_name or hist_name == "Other":
            continue
        hist_key = hist_name if hist_name in _HISTONE_SEQS else ("H3" if hist_name == "H3.3" else None)
        if hist_key is None:
            continue
        full_seq = _HISTONE_SEQS.get(hist_key, "")
        if not full_seq:
            continue

        hist_regions = rdf[rdf["Histone"] == hist_name].copy()
        hist_regions = hist_regions.dropna(subset=["Start", "End"])
        if hist_regions.empty:
            continue

        seq_len = len(full_seq)
        n_tracks = len(hist_regions)

        # Build sequence track figure
        trk = go.Figure()
        # Background: full sequence bar
        trk.add_trace(go.Scatter(
            x=[0.5, seq_len + 0.5], y=[n_tracks + 1.5, n_tracks + 1.5],
            mode="lines", line=dict(color="#e5e7eb", width=10),
            showlegend=False, hoverinfo="skip"))

        # Amino acid letters along top
        aa_x, aa_text = [], []
        for i, aa in enumerate(full_seq):
            if i % 5 == 0 or aa in "KRST":
                aa_x.append(i + 1)
                aa_text.append(f"{aa}{i+1}")
        trk.add_trace(go.Scatter(
            x=aa_x, y=[n_tracks + 2.2] * len(aa_x), mode="text",
            text=aa_text, textfont=dict(size=8, family="Courier New", color="#374151"),
            showlegend=False, hoverinfo="skip"))

        # Region coverage blocks
        base_color = hist_colors_map.get(hist_name, "#6b7280")
        for i, (_, rrow) in enumerate(hist_regions.iterrows()):
            s, e = int(rrow["Start"]), int(rrow["End"])
            y_pos = n_tracks - i
            # Region block
            trk.add_shape(type="rect", x0=s - 0.4, x1=e + 0.4,
                         y0=y_pos - 0.35, y1=y_pos + 0.35,
                         fillcolor=base_color, opacity=0.25,
                         line=dict(color=base_color, width=1.5))
            # Region label
            trk.add_annotation(x=(s + e) / 2, y=y_pos, text=rrow["Sequence"] if rrow["Sequence"] else rrow["Region"],
                              showarrow=False, font=dict(size=10, family="Courier New", color=base_color))
            # Region name on left
            trk.add_annotation(x=-1, y=y_pos, text=rrow["Region"], showarrow=False,
                              font=dict(size=10, color="#374151"), xanchor="right")

            # Mark modification sites with dots
            ptms_str = rrow["Unique_PTMs"]
            if ptms_str:
                for ptm in ptms_str.split(", "):
                    pm = re.match(r'[KRST](\d+)', ptm)
                    if pm:
                        pos = int(pm.group(1))
                        if s <= pos <= e:
                            trk.add_trace(go.Scatter(
                                x=[pos], y=[y_pos + 0.35], mode="markers",
                                marker=dict(size=7, color="#dc2626", symbol="triangle-down",
                                            line=dict(width=0.5, color="white")),
                                text=[ptm], hoverinfo="text", showlegend=False))

        trk.update_layout(
            template=PUB, height=max(180, 50 + n_tracks * 35),
            title=dict(text=f"{hist_name} - Peptide Coverage Map", font=dict(size=16)),
            xaxis=dict(title="Amino Acid Position", range=[0, seq_len + 2],
                      dtick=10, showgrid=True, gridcolor="#f1f5f9",
                      tickfont=dict(size=10)),
            yaxis=dict(showticklabels=False, range=[0, n_tracks + 3],
                      showgrid=False, zeroline=False),
            margin=dict(l=100, r=30, t=50, b=40),
            font=dict(family=FONT))

        children.append(html.Div(style=CS, children=[dcc.Graph(figure=trk)]))

    children.append(_st("Peptide Coverage Map",
        "Histone sequence tracks showing peptide positions and modification sites (red triangles)",
        icon="dna"))

    # ============================================================
    # SECTION 2: PTM Landscape - modification density along sequence
    # ============================================================
    # Build a PTM position map: for each modified residue, count how many distinct
    # modifications target it, and what their mean abundance is
    ptm_positions = {}
    for _, row in hpf_meta.iterrows():
        if isinstance(row.get("individual_ptms"), list):
            region = row["region"]
            histone = row["histone"]
            for ptm in row["individual_ptms"]:
                pm = re.match(r'([KRST])(\d+)(\w+)', ptm)
                if pm:
                    aa, pos, mod_type = pm.group(1), int(pm.group(2)), pm.group(3)
                    key = (histone, pos, aa)
                    if key not in ptm_positions:
                        ptm_positions[key] = {"histone": histone, "pos": pos, "aa": aa,
                                              "mod_types": set(), "regions": set(), "count": 0}
                    ptm_positions[key]["mod_types"].add(mod_type)
                    ptm_positions[key]["regions"].add(region)
                    ptm_positions[key]["count"] += 1

    if ptm_positions:
        ptm_pos_df = pd.DataFrame([
            {"Histone": v["histone"], "Position": v["pos"], "Residue": v["aa"],
             "Site": f"{v['aa']}{v['pos']}", "Modifications": ", ".join(sorted(v["mod_types"])),
             "N_mod_types": len(v["mod_types"]), "N_peptidoforms": v["count"],
             "Regions": ", ".join(sorted(v["regions"]))}
            for v in ptm_positions.values()
        ]).sort_values(["Histone", "Position"])

        # Modification type color scheme
        mod_colors = {"me1": "#3b82f6", "me2": "#2563eb", "me3": "#1e40af",
                      "ac": "#f59e0b", "ub": "#ef4444", "ph": "#8b5cf6",
                      "but": "#10b981", "crot": "#06b6d4", "prop": "#6b7280"}

        landscape = go.Figure()
        for hist_name in sorted(ptm_pos_df["Histone"].unique()):
            hd = ptm_pos_df[ptm_pos_df["Histone"] == hist_name]
            landscape.add_trace(go.Scatter(
                x=hd["Position"], y=hd["N_mod_types"],
                mode="markers+text", name=hist_name,
                marker=dict(size=hd["N_peptidoforms"].clip(upper=25) + 8,
                            color=hist_colors_map.get(hist_name, "#6b7280"),
                            opacity=0.75, line=dict(width=1.5, color="white")),
                text=hd["Site"], textposition="top center",
                textfont=dict(size=9, color="#374151"),
                hovertext=[f"{r['Site']}: {r['Modifications']} ({r['N_peptidoforms']} hPF)"
                           for _, r in hd.iterrows()],
                hoverinfo="text"))

        pfig(landscape, 380)
        landscape.update_layout(
            xaxis_title="Amino Acid Position", yaxis_title="# Modification Types",
            title=dict(text="PTM Landscape: Modification Diversity per Residue", font=dict(size=16)),
            xaxis=dict(dtick=5, showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        children.append(html.Div(style=CS, children=[
            _st("PTM Landscape",
                "Bubble size = number of peptidoforms carrying that site | "
                "Y-axis = number of distinct modification types (me1, me2, ac, ...)",
                icon="ptm"),
            dcc.Graph(figure=landscape),
        ]))

        # Modification site summary table
        children.append(html.Div(style=CS, children=[
            _st("Modification Sites", "All detected modification sites with their types and frequencies"),
            make_table(ptm_pos_df, "ptm-landscape-table"),
        ]))

    # ============================================================
    # SECTION 3: Region Lollipop Charts (improved)
    # ============================================================
    # Cleveland dot plot: Total hPF and unique PTMs side by side
    hist_colors = {"H3": C["h3"], "H3.3": C["warn"], "H4": C["h4"], "Other": C["muted"]}

    lf = go.Figure()
    for _, row in rdf.iterrows():
        region = row["Region"]
        lf.add_trace(go.Scatter(
            x=[0, row["Total_hPF"]], y=[region, region],
            mode="lines", line=dict(color=C["accent"], width=2),
            showlegend=False, hoverinfo="skip"))
    for hist in rdf["Histone"].unique():
        hd = rdf[rdf["Histone"] == hist]
        lf.add_trace(go.Scatter(
            x=hd["Total_hPF"], y=hd["Region"],
            mode="markers+text", name=hist,
            marker=dict(size=14, color=hist_colors.get(hist, C["accent"]),
                        line=dict(width=2, color="white")),
            text=hd["Total_hPF"].astype(str), textposition="middle right",
            textfont=dict(size=13, color=C["text"]),
            hovertext=[f"{r['Region']}: {r['Sequence']} ({r['Start']}-{r['End']})" for _, r in hd.iterrows()],
            hoverinfo="text+x"))
    pfig(lf, max(350, len(rdf) * 32))
    lf.update_layout(
        xaxis_title="Number of Peptidoforms (hPF)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        margin=dict(l=160, r=40), title=dict(text="Peptidoforms per Region", font=dict(size=18)))

    # Unique PTMs lollipop
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
    pfig(uf, max(350, len(rdf) * 32))
    uf.update_layout(
        xaxis_title="# Unique PTM Marks",
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        margin=dict(l=160, r=40), title=dict(text="Unique PTM Marks per Region", font=dict(size=18)))

    children.append(_st("Region Complexity",
        "Lollipop plots: peptidoform count and unique modification marks per derivatized peptide region",
        icon="pin"))
    children.append(html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
        html.Div(style={**CS, "flex": "1", "minWidth": "500px"}, children=[dcc.Graph(figure=lf)]),
        html.Div(style={**CS, "flex": "1", "minWidth": "400px"}, children=[dcc.Graph(figure=uf)]),
    ]))

    # ============================================================
    # SECTION 4: Composition breakdown per region (stacked + heatmap)
    # ============================================================
    sbf = go.Figure()
    sbf.add_trace(go.Bar(x=rdf["Region"], y=rdf["Single"], name="Single-mod",
                          marker_color=C["green"]))
    sbf.add_trace(go.Bar(x=rdf["Region"], y=rdf["Combo"], name="Combinatorial",
                          marker_color=C["warn"]))
    sbf.add_trace(go.Bar(x=rdf["Region"], y=rdf["Unmod"], name="Unmodified",
                          marker_color=C["muted"]))
    pfig(sbf, 400)
    sbf.update_layout(barmode="stack", xaxis=dict(tickangle=45, tickfont=dict(size=12)),
                       yaxis_title="# hPF",
                       title=dict(text="Modification Type per Region", font=dict(size=18)),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Region x Group heatmap
    if groups:
        rg_data = []
        for region in rdf["Region"]:
            rmeta_r = hpf_meta[hpf_meta["region"] == region]
            rnames = rmeta_r["name"].tolist()
            rdata = hpf.loc[[n for n in rnames if n in hpf.index]]
            for g in groups:
                samps = meta[meta["Group"] == g]["Sample"].tolist()
                cols = [c for c in rdata.columns if c in samps]
                gm = rdata[cols].mean().mean() if cols else np.nan
                rg_data.append({"Region": region, "Group": g, "Mean_Ratio": gm})
        rgdf = pd.DataFrame(rg_data)
        rgp = rgdf.pivot(index="Region", columns="Group", values="Mean_Ratio").fillna(0)
        rghm = phm(rgp.values, rgp.columns.tolist(), rgp.index.tolist(),
                    cs="YlOrRd", title="Mean Ratio", h=max(300, len(rgp) * 28))
        rghm.update_layout(title=dict(text="Mean hPF Ratio: Region x Group", font=dict(size=18)))
    else:
        rghm = go.Figure(); pfig(rghm, 300)

    children.append(_st("Composition & Group Patterns",
        "Stacked bar: modification types per region | Heatmap: mean ratios across groups",
        icon="bar_chart"))
    children.append(html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
        html.Div(style={**CS, "flex": "1", "minWidth": "500px"}, children=[dcc.Graph(figure=sbf)]),
        html.Div(style={**CS, "flex": "1", "minWidth": "500px"}, children=[dcc.Graph(figure=rghm)]),
    ]))

    # ============================================================
    # SECTION 5: Faceted box plots for top regions
    # ============================================================
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

        children.append(html.Div(style=CS, children=[
            _st("Top Regions by Group", "Faceted box plots for the 6 most diverse regions", icon="region"),
            dcc.Graph(figure=rbf),
        ]))

    # ============================================================
    # SECTION 6: Sequence context cards
    # ============================================================
    seq_cards = []
    for _, rrow in rdf.iterrows():
        if not rrow["Sequence"] or rrow["Sequence"] == "unmod_header":
            continue
        hist = rrow["Histone"]
        start, end = rrow.get("Start"), rrow.get("End")
        pos_str = f"aa {int(start)}-{int(end)}" if pd.notna(start) and pd.notna(end) else ""

        # Annotated sequence: highlight modifiable residues
        seq_html = []
        seq_str = rrow["Sequence"]
        ptms_in_region = rrow["Unique_PTMs"].split(", ") if rrow["Unique_PTMs"] else []
        # Find which positions in the sequence are modified
        mod_positions = set()
        for ptm in ptms_in_region:
            pm = re.match(r'[KRST](\d+)', ptm)
            if pm and pd.notna(start):
                abs_pos = int(pm.group(1))
                rel_pos = abs_pos - int(start)
                if 0 <= rel_pos < len(seq_str):
                    mod_positions.add(rel_pos)

        for i, aa in enumerate(seq_str):
            if i in mod_positions:
                abs_pos = int(start) + i if pd.notna(start) else i
                seq_html.append(html.Span(aa, style={
                    "color": "#dc2626", "fontWeight": "800", "textDecoration": "underline",
                    "cursor": "pointer"}, title=f"{aa}{abs_pos}"))
            elif aa in "KRST":
                seq_html.append(html.Span(aa, style={"color": "#2563eb", "fontWeight": "700"}))
            else:
                seq_html.append(html.Span(aa, style={"color": "#374151"}))

        badge_style = {"display": "inline-block", "padding": "2px 8px", "borderRadius": "10px",
                       "fontSize": "11px", "fontWeight": "600", "marginRight": "4px"}

        card = html.Div(style={
            "border": "1px solid #e5e7eb", "borderRadius": "10px", "padding": "14px 18px",
            "backgroundColor": "#fafafa", "minWidth": "320px", "flex": "1",
        }, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "8px"}, children=[
                html.Span(rrow["Region"], style={"fontWeight": "800", "fontSize": "15px", "color": "#0f1f13"}),
                html.Span(hist, style={**badge_style,
                    "backgroundColor": hist_colors_map.get(hist, "#6b7280") + "22",
                    "color": hist_colors_map.get(hist, "#6b7280")}),
                html.Span(pos_str, style={**badge_style, "backgroundColor": "#f1f5f9", "color": "#64748b"}) if pos_str else None,
            ]),
            html.Div(style={"fontFamily": "Courier New, monospace", "fontSize": "16px",
                            "letterSpacing": "2px", "marginBottom": "6px", "lineHeight": "1.6"},
                     children=seq_html),
            html.Div(style={"display": "flex", "gap": "12px", "fontSize": "12px", "color": "#6b7280"}, children=[
                html.Span(f"{rrow['Total_hPF']} hPF"),
                html.Span(f"{rrow['n_unique_PTMs']} PTM marks"),
                html.Span(f"{rrow['Single']} single, {rrow['Combo']} combo"),
            ]),
            html.Div(style={"marginTop": "4px", "fontSize": "11px", "color": "#9ca3af"},
                     children=rrow["Unique_PTMs"]) if rrow["Unique_PTMs"] else None,
        ])
        seq_cards.append(card)

    if seq_cards:
        children.append(html.Div(style=CS, children=[
            _st("Sequence Context",
                "Peptide sequences with modified residues highlighted | "
                "Red underlined = modified | Blue = modifiable (K/R/S/T)",
                icon="dna"),
            html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=seq_cards),
        ]))

    # ============================================================
    # SECTION 7: Region summary table
    # ============================================================
    table_cols = ["Region", "Histone", "Sequence", "Start", "End",
                  "Total_hPF", "Single", "Combo", "Unmod", "Mean_Ratio",
                  "n_unique_PTMs", "Unique_PTMs"]
    table_df = rdf[[c for c in table_cols if c in rdf.columns]].copy()
    children.append(html.Div(style=CS, children=[
        _st("Region Summary Table", "Peptide regions with sequence context, modification counts and PTM lists"),
        make_table(table_df, "region-table"),
    ]))

    return html.Div(children)


# ======================================================================
# TAB: H3K79 VALIDATION -- Star Section
# ======================================================================

def tab_k79(d, exp_name):
    """H3K79 validation: dedicated section for H3 73-83 region and K79 mods.
    Aggregates evidence from ratios, areas, RT, PSMs, and per-sample detail files."""
    children = []
    meta = d.get("metadata", pd.DataFrame())
    if meta.empty:
        return html.Div(style=CS, children=[_st("H3K79 Validation", "No metadata available.", icon="star")])
    groups = sorted(meta["Group"].unique())
    hpf = d.get("hpf", pd.DataFrame())
    hptm = d.get("hptm", pd.DataFrame())
    areas = d.get("areas_norm", d.get("areas_log2"))
    rt_df = d.get("rt", pd.DataFrame())
    psm = d.get("all_psm", pd.DataFrame())
    ld = d.get("layouts_dir", "")

    # -- K79 features --
    k79_feats = [f for f in hpf.index if "K79" in str(f)] if not hpf.empty else []
    k79_hptm = [f for f in hptm.index if "K79" in str(f)] if not hptm.empty else []
    region_feats = [f for f in hpf.index if "73_83" in str(f)] if not hpf.empty else []
    unmod_73 = [f for f in region_feats if "unmod" in str(f).lower() or "K79" not in str(f)]
    all_73_83 = sorted(set(region_feats))

    # ---- HEADER with star icon + golden accent ----
    children.append(html.Div(style={**CS, "border":"2px solid #f59e0b","background":"linear-gradient(135deg,#fffbeb 0%,#fef3c7 100%)",
                                     "borderRadius":"12px","padding":"24px"}, children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"14px"}, children=[
            _svg_icon("star", 36),
            html.Div([
                html.H2("H3K79 Validation Panel", style={"margin":"0","fontSize":"26px","fontWeight":"800",
                         "color":"#92400e","letterSpacing":"-0.5px"}),
                html.P("Comprehensive evidence evaluation for H3 73-83 region (EIAQDFKTDLR) and K79 modifications",
                        style={"margin":"4px 0 0","color":"#a16207","fontSize":"14px"}),
            ]),
        ]),
    ]))

    if not k79_feats and not k79_hptm:
        children.append(html.Div(style=CS, children=[
            html.P("No H3K79 features found in this experiment.", style={"color":C["red"],"fontSize":"15px","fontWeight":"600"})]))
        return html.Div(children)

    # ---- 1. SUMMARY CARDS ----
    n_k79_hpf = len(k79_feats)
    n_k79_hptm = len(k79_hptm)
    n_total_73_83 = len(all_73_83)
    # Compute mean detection (non-zero ratio) across samples
    if not hpf.empty and k79_feats:
        k79_df = hpf.loc[k79_feats]
        det_rate = (k79_df > 0.001).mean(axis=1).mean() * 100
    else:
        det_rate = 0
    # Mean ratio
    mean_ratio = hpf.loc[k79_feats].mean(axis=1).mean() if k79_feats and not hpf.empty else 0

    cards = [
        _sc("K79 Peptidoforms", n_k79_hpf, "#f59e0b"),
        _sc("K79 Single PTMs", n_k79_hptm, "#d97706"),
        _sc("Total H3 73-83 Features", n_total_73_83, "#92400e"),
        _sc("Mean Detection %", f"{det_rate:.1f}%", "#22c55e" if det_rate > 50 else "#ef4444"),
        _sc("Mean Ratio", f"{mean_ratio:.4f}", "#0891b2"),
    ]
    children.append(html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"16px"}, children=cards))

    # ---- 2. K79 RATIOS HEATMAP (hPF level) ----
    if k79_feats and not hpf.empty:
        k79_df = hpf.loc[k79_feats].copy()
        # Sort columns by group
        sample_order = meta.sort_values("Group")["Sample"].tolist()
        k79_cols = [c for c in sample_order if c in k79_df.columns]
        k79_ordered = k79_df[k79_cols]
        grp_colors = [meta.set_index("Sample").loc[s,"Group"] if s in meta["Sample"].values else "" for s in k79_cols]

        fig_hm = go.Figure(data=go.Heatmap(
            z=k79_ordered.values, x=k79_cols, y=k79_ordered.index.tolist(),
            colorscale=[[0,"#fefce8"],[0.3,"#fde047"],[0.6,"#f59e0b"],[1.0,"#92400e"]],
            colorbar=dict(title="Ratio")))
        fig_hm.update_layout(**PUB.layout.to_plotly_json(), title="K79 Modification Ratios per Sample",
                             height=200 + 40*len(k79_feats), xaxis_tickangle=-45)
        children.append(html.Div(style=CS, children=[
            _st("K79 Peptidoform Ratios", "Heatmap of K79me1/me2/me3/ac ratios across all samples | golden colorscale", icon="heatmap"),
            dcc.Graph(figure=fig_hm),
        ]))

    # ---- 3. K79 SINGLE PTM RATIOS (hPTM level) ----
    if k79_hptm and not hptm.empty:
        k79_ptm_df = hptm.loc[k79_hptm]
        # Box plot by group
        rows_box = []
        for ptm in k79_hptm:
            for g in groups:
                samps = meta[meta["Group"]==g]["Sample"].tolist()
                vals = k79_ptm_df.loc[ptm, [s for s in samps if s in k79_ptm_df.columns]].dropna()
                for v in vals:
                    if v > 0:
                        rows_box.append({"PTM":ptm, "Group":g, "Ratio":v})
        if rows_box:
            box_df = pd.DataFrame(rows_box)
            fig_box = px.box(box_df, x="PTM", y="Ratio", color="Group", points="all",
                             color_discrete_sequence=GC, template="plotly_white")
            fig_box.update_layout(**PUB.layout.to_plotly_json(), title="H3K79 Single PTM Abundances by Group",
                                  height=450, legend=dict(font=dict(size=11)))
            children.append(html.Div(style=CS, children=[
                _st("K79 Single PTM Abundances", "Box plots showing ratio distribution per K79 modification across groups", icon="bar_chart"),
                dcc.Graph(figure=fig_box),
            ]))

    # ---- 4. AREAS (MS1 intensity) for K79 features ----
    if areas is not None and not areas.empty and k79_feats:
        k79_areas = areas.loc[[f for f in k79_feats if f in areas.index]]
        if not k79_areas.empty:
            rows_area = []
            for feat in k79_areas.index:
                for g in groups:
                    samps = meta[meta["Group"]==g]["Sample"].tolist()
                    vals = k79_areas.loc[feat, [s for s in samps if s in k79_areas.columns]].dropna()
                    for v in vals:
                        if np.isfinite(v):
                            rows_area.append({"Feature":feat, "Group":g, "log2_Area":v})
            if rows_area:
                area_df = pd.DataFrame(rows_area)
                fig_area = px.violin(area_df, x="Feature", y="log2_Area", color="Group", box=True,
                                     color_discrete_sequence=GC, template="plotly_white")
                fig_area.update_layout(**PUB.layout.to_plotly_json(), title="K79 Feature Intensities (log2 normalized)",
                                       height=450)
                children.append(html.Div(style=CS, children=[
                    _st("K79 MS1 Intensities", "Violin plots of log2 quantile-normalized areas | evidence for detection strength", icon="chromatogram"),
                    dcc.Graph(figure=fig_area),
                ]))

    # ---- 5. RETENTION TIME CONSISTENCY ----
    if rt_df is not None and not rt_df.empty and k79_feats:
        k79_rt = rt_df.loc[[f for f in k79_feats if f in rt_df.index]]
        if not k79_rt.empty:
            rows_rt = []
            for feat in k79_rt.index:
                for g in groups:
                    samps = meta[meta["Group"]==g]["Sample"].tolist()
                    vals = k79_rt.loc[feat, [s for s in samps if s in k79_rt.columns]].dropna()
                    for v in vals:
                        if v > 0:
                            rows_rt.append({"Feature":feat, "Group":g, "RT_min":v})
            if rows_rt:
                rt_box = pd.DataFrame(rows_rt)
                fig_rt = px.box(rt_box, x="Feature", y="RT_min", color="Group", points="all",
                                color_discrete_sequence=GC, template="plotly_white")
                fig_rt.update_layout(**PUB.layout.to_plotly_json(), title="K79 Retention Times (minutes)",
                                     height=400, yaxis_title="RT (min)")
                children.append(html.Div(style=CS, children=[
                    _st("K79 Retention Time Consistency", "RT spread across samples validates chromatographic reproducibility", icon="clock"),
                    dcc.Graph(figure=fig_rt),
                ]))

    # ---- 6. PSM EVIDENCE ----
    if not psm.empty:
        # Filter PSMs for H3_73_83 peptide: EIAQDFKTDLR
        seq_col = [c for c in psm.columns if "sequence" in c.lower()]
        if seq_col:
            seq_c = seq_col[0]
            k79_psm = psm[psm[seq_c].str.contains("EIAQDFKTDLR", na=False, case=False)]
            if not k79_psm.empty:
                # Also filter by K79 modifications
                mod_cols = [c for c in psm.columns if "modif" in c.lower()]
                has_k79_mod = k79_psm.copy()
                if mod_cols:
                    mod_c = mod_cols[0]
                    has_k79_mod = k79_psm[k79_psm[mod_c].str.contains("me|ac", na=False, case=False)]

                n_total_psm = len(k79_psm)
                n_k79_mod_psm = len(has_k79_mod)

                children.append(html.Div(style=CS, children=[
                    _st("PSM Evidence for H3 73-83",
                         f"{n_total_psm:,} PSMs for EIAQDFKTDLR | {n_k79_mod_psm:,} with K79 modifications",
                         icon="search"),
                ]))

                # PSM quality: ppm distribution
                ppm_col = [c for c in k79_psm.columns if "ppm" in c.lower()]
                if ppm_col:
                    ppm_c = ppm_col[0]
                    ppm_vals = k79_psm[ppm_c].dropna()
                    if len(ppm_vals) > 0:
                        fig_ppm = go.Figure()
                        fig_ppm.add_trace(go.Histogram(x=ppm_vals, nbinsx=50, marker_color="#f59e0b", name="All EIAQDFKTDLR"))
                        if not has_k79_mod.empty and ppm_c in has_k79_mod.columns:
                            mod_ppm = has_k79_mod[ppm_c].dropna()
                            if len(mod_ppm) > 0:
                                fig_ppm.add_trace(go.Histogram(x=mod_ppm, nbinsx=50, marker_color="#22c55e",
                                                               name="K79 modified", opacity=0.7))
                        fig_ppm.update_layout(**PUB.layout.to_plotly_json(), title="Mass Accuracy (ppm) for H3 73-83 PSMs",
                                              xaxis_title="ppm error", yaxis_title="Count", height=350, barmode="overlay")
                        children.append(html.Div(style=CS, children=[dcc.Graph(figure=fig_ppm)]))

                # RT from PSMs
                rt_col = [c for c in k79_psm.columns if "retention" in c.lower() or "rt" == c.lower()]
                if rt_col:
                    rt_c = rt_col[0]

                # Charge state distribution
                z_col = [c for c in k79_psm.columns if "charge" in c.lower()]
                if z_col:
                    z_c = z_col[0]
                    z_counts = k79_psm[z_c].value_counts().sort_index()
                    fig_z = go.Figure(go.Bar(x=[f"+{int(z)}" for z in z_counts.index],
                                             y=z_counts.values, marker_color="#f59e0b"))
                    fig_z.update_layout(**PUB.layout.to_plotly_json(), title="Charge State Distribution (H3 73-83 PSMs)",
                                        xaxis_title="Charge", yaxis_title="PSM count", height=300)
                    children.append(html.Div(style=CS, children=[dcc.Graph(figure=fig_z)]))

                # Modification breakdown
                if mod_cols:
                    mod_c = mod_cols[0]
                    mod_counts = k79_psm[mod_c].value_counts().head(15)
                    if len(mod_counts) > 0:
                        fig_mc = go.Figure(go.Bar(
                            y=mod_counts.index.tolist(), x=mod_counts.values,
                            orientation="h", marker_color=["#f59e0b" if "79" in str(m) else "#d1d5db" for m in mod_counts.index]))
                        fig_mc.update_layout(**PUB.layout.to_plotly_json(),
                                             title="Top Modification Combinations in H3 73-83 PSMs",
                                             height=max(250, 30*len(mod_counts)),
                                             margin=dict(l=max(200, max(len(str(m)) for m in mod_counts.index)*7)))
                        children.append(html.Div(style=CS, children=[
                            _st("Modification Combinations", "K79-containing modifications highlighted in gold", icon="tag"),
                            dcc.Graph(figure=fig_mc),
                        ]))

                # PSM table (top 50)
                show_cols = [c for c in k79_psm.columns if c != "_sample_folder"][:8]
                psm_show = k79_psm[show_cols].head(50).copy()
                children.append(html.Div(style=CS, children=[
                    _st("PSM Detail Table", f"Top 50 of {n_total_psm:,} PSMs for EIAQDFKTDLR peptide", icon="table"),
                    make_table(psm_show, "k79-psm-table"),
                ]))

    # ---- 7. PER-SAMPLE REGION DETAIL (charge state areas from H3_07_73_83.xls) ----
    folders = d.get("sample_folders", [])
    if ld and folders:
        detail_rows = []
        for fld in folders:
            # Try common naming patterns for H3_73_83 detail file
            for fn in ["H3_07_73_83.xls", "H3_73_83.xls"]:
                fp = os.path.join(ld, fld, "detail", fn)
                if os.path.exists(fp):
                    try:
                        raw = pd.read_csv(fp, sep="\t", header=None)
                        # Parse: first column has modification names, rest are charge states
                        # Find [ratio] and [area] blocks
                        block = None
                        for i, row in raw.iterrows():
                            val0 = str(row.iloc[0]).strip().lower()
                            if val0 == "[ratio]": block = "ratio"; continue
                            if val0 == "[area]": block = "area"; continue
                            if val0 == "[rt]": block = "rt"; continue
                            if val0.startswith("[") or val0 == "": continue
                            if block == "ratio" and "K79" in str(row.iloc[0]):
                                mod_name = str(row.iloc[0]).strip()
                                vals = row.iloc[1:].astype(float, errors="ignore")
                                total = vals.sum() if not vals.isna().all() else 0
                                detail_rows.append({"Sample_Folder": fld, "Modification": mod_name,
                                                    "Block": "ratio", "Total": total})
                            if block == "area" and "K79" in str(row.iloc[0]):
                                mod_name = str(row.iloc[0]).strip()
                                vals = pd.to_numeric(row.iloc[1:], errors="coerce")
                                total = vals.sum() if not vals.isna().all() else 0
                                detail_rows.append({"Sample_Folder": fld, "Modification": mod_name,
                                                    "Block": "area", "Total": total})
                    except:
                        pass
                    break
        if detail_rows:
            det_df = pd.DataFrame(detail_rows)
            # Ratio summary from detail files
            ratio_det = det_df[det_df["Block"] == "ratio"]
            if not ratio_det.empty:
                fig_det = px.box(ratio_det, x="Modification", y="Total", color="Modification",
                                 color_discrete_sequence=["#f59e0b","#d97706","#92400e","#22c55e"],
                                 template="plotly_white", points="all")
                fig_det.update_layout(**PUB.layout.to_plotly_json(),
                                      title="Per-Sample K79 Ratios (from detail files)",
                                      height=400, showlegend=False)
                children.append(html.Div(style=CS, children=[
                    _st("Per-Sample Detail Ratios", f"Extracted from {len(folders)} sample detail files | charge-state aggregated",
                        icon="search"),
                    dcc.Graph(figure=fig_det),
                ]))
            # Area summary from detail files
            area_det = det_df[det_df["Block"] == "area"]
            if not area_det.empty:
                area_det_log = area_det.copy()
                area_det_log["log2_Area"] = np.log2(area_det_log["Total"].clip(lower=1))
                fig_adet = px.box(area_det_log, x="Modification", y="log2_Area", color="Modification",
                                  color_discrete_sequence=["#f59e0b","#d97706","#92400e","#22c55e"],
                                  template="plotly_white", points="all")
                fig_adet.update_layout(**PUB.layout.to_plotly_json(),
                                       title="Per-Sample K79 MS1 Areas (from detail files)",
                                       height=400, showlegend=False)
                children.append(html.Div(style=CS, children=[
                    _st("Per-Sample Detail Areas", "log2(MS1 area) from per-sample region detail files", icon="chromatogram"),
                    dcc.Graph(figure=fig_adet),
                ]))

    # ---- 8. GLOBAL COMPARISON: K79 vs OTHER MODIFICATIONS ----
    if not hpf.empty and k79_feats:
        # Compare K79 detection rate vs all other features
        all_det_rates = (hpf > 0.001).mean(axis=1) * 100
        k79_rates = all_det_rates.loc[k79_feats]
        other_rates = all_det_rates.drop(k79_feats, errors="ignore")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Histogram(x=other_rates, nbinsx=30, name="Other features",
                                         marker_color="#d1d5db", opacity=0.7))
        for feat, rate in k79_rates.items():
            fig_comp.add_vline(x=rate, line_dash="dash", line_color="#f59e0b", line_width=2,
                               annotation_text=str(feat).split()[-1], annotation_position="top")
        fig_comp.update_layout(**PUB.layout.to_plotly_json(),
                               title="K79 Detection Rate vs All Other Features",
                               xaxis_title="Detection Rate (%)", yaxis_title="Count",
                               height=350)
        children.append(html.Div(style=CS, children=[
            _st("K79 in Global Context", "Vertical lines show K79 features against the distribution of all features", icon="spectra_compare"),
            dcc.Graph(figure=fig_comp),
        ]))

    # ---- 9. VALIDATION SUMMARY TABLE ----
    summary_rows = []
    for feat in all_73_83:
        row = {"Feature": feat, "Type": "K79 mod" if "K79" in feat else "Other"}
        if not hpf.empty and feat in hpf.index:
            vals = hpf.loc[feat].dropna()
            vals_nz = vals[vals > 0.001]
            row["Mean_Ratio"] = round(vals.mean(), 6)
            row["Median_Ratio"] = round(vals.median(), 6)
            row["Detection_%"] = round(len(vals_nz)/len(vals)*100, 1) if len(vals) > 0 else 0
            row["CV_%"] = round(vals_nz.std()/vals_nz.mean()*100, 1) if len(vals_nz) > 1 and vals_nz.mean() > 0 else 0
        if areas is not None and not areas.empty and feat in areas.index:
            area_vals = areas.loc[feat].dropna()
            area_vals = area_vals[np.isfinite(area_vals)]
            row["Mean_log2Area"] = round(area_vals.mean(), 2) if len(area_vals) > 0 else 0
        if rt_df is not None and not rt_df.empty and feat in rt_df.index:
            rt_vals = rt_df.loc[feat].dropna()
            rt_vals = rt_vals[rt_vals > 0]
            row["Mean_RT"] = round(rt_vals.mean(), 2) if len(rt_vals) > 0 else 0
            row["RT_SD"] = round(rt_vals.std(), 3) if len(rt_vals) > 1 else 0
        summary_rows.append(row)
    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        children.append(html.Div(style=CS, children=[
            _st("H3 73-83 Validation Summary", "All features in the EIAQDFKTDLR region with detection, ratio, area, and RT metrics", icon="star"),
            make_table(sum_df, "k79-summary-table"),
        ]))

    return html.Div(children)


# ======================================================================
# TAB: ndebug COMPARISON (Detection Efficacy Between ndebug Modes)
# ======================================================================

def _find_ndebug_pairs():
    """Find experiments that share the same dataset but differ in ndebug mode."""
    datasets = {}
    for name, d in EXP_DATA.items():
        ds = d.get("dataset", name)
        nd = d.get("ndebug", "default")
        if nd != "default":
            datasets.setdefault(ds, []).append((name, nd))
    return {ds: exps for ds, exps in datasets.items() if len(exps) >= 2}

def tab_ndebug(current_exp):
    """Build the ndebug comparison tab layout with dropdowns and output area."""
    pairs = _find_ndebug_pairs()
    if not pairs:
        return html.Div(style=CS, children=[
            _st("ndebug Comparison", "No ndebug pairs found. Need same dataset with ndebug_0 and ndebug_2.", icon="spectra_compare"),
            html.P("Upload or load experiments with different ndebug modes for the same dataset to compare detection efficacy.",
                   style={"color": C["muted"], "fontSize": "14px"})])

    # Build dropdown options grouped by dataset
    all_options = []
    for ds, exps in sorted(pairs.items()):
        for exp_name, nd_mode in exps:
            all_options.append({"label": f"{exp_name}", "value": exp_name})

    # Try to auto-select a pair from current experiment's dataset
    cur_ds = EXP_DATA.get(current_exp, {}).get("dataset", "")
    default_a, default_b = all_options[0]["value"], all_options[1]["value"] if len(all_options) > 1 else all_options[0]["value"]
    if cur_ds in pairs:
        p = pairs[cur_ds]
        if len(p) >= 2:
            default_a, default_b = p[0][0], p[1][0]

    return html.Div([
        html.Div(style={**CS, "display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "flex-end"}, children=[
            html.Div(style={"flex": "1", "minWidth": "250px"}, children=[
                _lbl("Experiment A (reference)"),
                dcc.Dropdown(id="ndebug-a", options=all_options, value=default_a, clearable=False, style=DS)]),
            html.Div(style={"flex": "1", "minWidth": "250px"}, children=[
                _lbl("Experiment B (compare)"),
                dcc.Dropdown(id="ndebug-b", options=all_options, value=default_b, clearable=False, style=DS)]),
            html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                _lbl("Data Level"),
                dcc.Dropdown(id="ndebug-level",
                    options=[{"label": "hPTM (single PTMs)", "value": "hptm"},
                             {"label": "hPF (peptidoforms)", "value": "hpf"}],
                    value="hpf", clearable=False, style=DS)]),
        ]),
        html.Div(id="ndebug-out"),
    ])

@callback(Output("ndebug-out", "children"),
          Input("ndebug-a", "value"), Input("ndebug-b", "value"),
          Input("ndebug-level", "value"), Input("cur-palette", "data"),
          prevent_initial_call=True)
def _ndebug_compare(exp_a, exp_b, level, pal):
    if not exp_a or not exp_b or exp_a not in EXP_DATA or exp_b not in EXP_DATA:
        return html.P("Select two experiments to compare.", style={"color": C["muted"]})
    if exp_a == exp_b:
        return html.P("Select two different experiments.", style={"color": C["warn"]})

    da, db = EXP_DATA[exp_a], EXP_DATA[exp_b]
    nd_a = da.get("ndebug", "?")
    nd_b = db.get("ndebug", "?")
    ds_a = da.get("dataset", exp_a)
    ds_b = db.get("dataset", exp_b)

    df_a = da.get(level, da.get("hpf"))
    df_b = db.get(level, db.get("hpf"))
    meta_a = da.get("metadata", pd.DataFrame())
    meta_b = db.get("metadata", pd.DataFrame())

    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return html.P("No data available for selected level.", style={"color": C["red"]})

    DET_THRESH = 0.001
    children = []
    level_label = "hPTM" if level == "hptm" else "hPF"

    # Align features (use intersection)
    common_feats = sorted(set(df_a.index) & set(df_b.index))
    only_a_feats = sorted(set(df_a.index) - set(df_b.index))
    only_b_feats = sorted(set(df_b.index) - set(df_a.index))
    if not common_feats:
        return html.P("No common features between the two experiments.", style={"color": C["red"]})

    dfa = df_a.loc[common_feats]
    dfb = df_b.loc[common_feats]

    # Detection: a feature is "detected" if ratio > threshold in >= 50% of samples
    det_a = (dfa > DET_THRESH).mean(axis=1) >= 0.5
    det_b = (dfb > DET_THRESH).mean(axis=1) >= 0.5
    both_det = det_a & det_b
    only_a = det_a & ~det_b
    only_b = ~det_a & det_b
    neither = ~det_a & ~det_b

    n_both = int(both_det.sum())
    n_only_a = int(only_a.sum())
    n_only_b = int(only_b.sum())
    n_neither = int(neither.sum())
    n_total = len(common_feats)

    # Mean ratios per feature
    mean_a = dfa.mean(axis=1)
    mean_b = dfb.mean(axis=1)

    # ============================================================
    # SECTION 1: Summary Cards
    # ============================================================
    children.append(html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"}, children=[
        _sc(f"Common {level_label}", str(n_total), C["accent"]),
        _sc(f"Both detected", str(n_both), C["h4"]),
        _sc(f"Only {nd_a}", str(n_only_a), C["h3"]),
        _sc(f"Only {nd_b}", str(n_only_b), C["warn"]),
        _sc("Neither", str(n_neither), C["muted"]),
        _sc(f"Unique to A", str(len(only_a_feats)), "#6b7280") if only_a_feats else None,
        _sc(f"Unique to B", str(len(only_b_feats)), "#6b7280") if only_b_feats else None,
    ]))

    # ============================================================
    # SECTION 2: Detection Venn (stacked bar)
    # ============================================================
    venn_fig = go.Figure()
    categories = [f"Both", f"Only {nd_a}", f"Only {nd_b}", "Neither"]
    counts = [n_both, n_only_a, n_only_b, n_neither]
    colors = [C["accent"], C["h3"], C["warn"], C["muted"]]
    venn_fig.add_trace(go.Bar(
        x=counts, y=["Detection"] * 4, orientation="h",
        marker_color=colors, text=[f"{c} ({c/n_total*100:.1f}%)" for c in counts],
        textposition="inside", textfont=dict(size=13, color="white"),
        showlegend=False))
    pfig(venn_fig, 140)
    venn_fig.update_layout(
        barmode="stack", xaxis_title=f"# {level_label} Features",
        yaxis=dict(showticklabels=False),
        title=dict(text=f"Detection Overlap: {nd_a} vs {nd_b} ({ds_a})", font=dict(size=16)))

    # Also make a pie chart
    pie_fig = go.Figure(go.Pie(
        labels=categories, values=counts,
        marker=dict(colors=colors),
        textinfo="label+percent", textfont=dict(size=12),
        hole=0.45))
    pfig(pie_fig, 350)
    pie_fig.update_layout(title=dict(text="Detection Distribution", font=dict(size=16)))

    children.append(html.Div(style=CS, children=[
        _st("Detection Overlap",
            f"Feature detected = ratio > {DET_THRESH} in >= 50% of samples | "
            f"Comparing {exp_a} vs {exp_b}", icon="spectra_compare"),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            html.Div(style={"flex": "2", "minWidth": "450px"}, children=[dcc.Graph(figure=venn_fig)]),
            html.Div(style={"flex": "1", "minWidth": "300px"}, children=[dcc.Graph(figure=pie_fig)]),
        ]),
    ]))

    # ============================================================
    # SECTION 3: Scatter Plot (mean ratio A vs B)
    # ============================================================
    scatter_df = pd.DataFrame({
        "Feature": common_feats,
        f"Mean_{nd_a}": mean_a.values,
        f"Mean_{nd_b}": mean_b.values,
    })
    # Classification for coloring
    status = []
    for f in common_feats:
        if both_det[f]:
            status.append("Both detected")
        elif only_a[f]:
            status.append(f"Only {nd_a}")
        elif only_b[f]:
            status.append(f"Only {nd_b}")
        else:
            status.append("Neither")
    scatter_df["Status"] = status

    status_colors = {"Both detected": C["accent"], f"Only {nd_a}": C["h3"],
                     f"Only {nd_b}": C["warn"], "Neither": C["muted"]}
    sfig = px.scatter(scatter_df, x=f"Mean_{nd_a}", y=f"Mean_{nd_b}",
                       color="Status", hover_name="Feature",
                       color_discrete_map=status_colors)
    sfig.update_traces(marker=dict(size=9, line=dict(width=1, color="white"), opacity=0.8))
    # Identity line
    max_val = max(scatter_df[f"Mean_{nd_a}"].max(), scatter_df[f"Mean_{nd_b}"].max())
    sfig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                               line=dict(dash="dash", color="#9ca3af", width=1.5),
                               showlegend=False, hoverinfo="skip"))
    pfig(sfig, 500)
    sfig.update_layout(
        xaxis_title=f"Mean Ratio ({nd_a})", yaxis_title=f"Mean Ratio ({nd_b})",
        title=dict(text=f"Concordance: {nd_a} vs {nd_b} Mean Ratios", font=dict(size=16)))

    # Spearman correlation
    from scipy.stats import spearmanr as _spearmanr_nd
    valid_mask = (mean_a > 0) & (mean_b > 0)
    if valid_mask.sum() >= 3:
        rho, pval = _spearmanr_nd(mean_a[valid_mask], mean_b[valid_mask])
        corr_text = f"Spearman rho = {rho:.4f} (p = {pval:.2e}, n = {valid_mask.sum()})"
    else:
        corr_text = "Insufficient data for correlation"

    children.append(html.Div(style=CS, children=[
        _st("Ratio Concordance",
            f"Scatter of per-feature mean ratios | {corr_text}", icon="bar_chart"),
        dcc.Graph(figure=sfig),
    ]))

    # ============================================================
    # SECTION 4: Per-Group Concordance Heatmap
    # ============================================================
    if not meta_a.empty and not meta_b.empty:
        groups_a = sorted(meta_a["Group"].unique())
        groups_b = sorted(meta_b["Group"].unique())
        common_groups = sorted(set(groups_a) & set(groups_b))
        if common_groups:
            corr_matrix = []
            for g in common_groups:
                samps_a = meta_a[meta_a["Group"] == g]["Sample"].tolist()
                samps_b = meta_b[meta_b["Group"] == g]["Sample"].tolist()
                cols_a = [c for c in dfa.columns if c in samps_a]
                cols_b = [c for c in dfb.columns if c in samps_b]
                if cols_a and cols_b:
                    ma = dfa[cols_a].mean(axis=1)
                    mb = dfb[cols_b].mean(axis=1)
                    valid = (ma > 0) & (mb > 0)
                    if valid.sum() >= 3:
                        rho, _ = spearmanr(ma[valid], mb[valid])
                    else:
                        rho = np.nan
                else:
                    rho = np.nan
                corr_matrix.append(rho)

            # Bar chart of per-group correlations
            gfig = go.Figure(go.Bar(
                x=common_groups, y=corr_matrix,
                marker_color=[GC[i % len(GC)] for i in range(len(common_groups))],
                text=[f"{r:.3f}" if not np.isnan(r) else "N/A" for r in corr_matrix],
                textposition="outside", textfont=dict(size=12)))
            pfig(gfig, 380)
            gfig.update_layout(
                xaxis_title="Group", yaxis_title="Spearman rho",
                yaxis=dict(range=[0, 1.1]),
                title=dict(text=f"Per-Group Concordance ({nd_a} vs {nd_b})", font=dict(size=16)))

            children.append(html.Div(style=CS, children=[
                _st("Per-Group Concordance",
                    "Spearman correlation of group-level mean ratios between ndebug modes",
                    icon="bar_chart"),
                dcc.Graph(figure=gfig),
            ]))

    # ============================================================
    # SECTION 5: Detection Rate & Sensitivity
    # ============================================================
    # For each feature, calculate detection rate in each mode
    det_rate_a = (dfa > DET_THRESH).mean(axis=1)
    det_rate_b = (dfb > DET_THRESH).mean(axis=1)

    dr_fig = go.Figure()
    dr_fig.add_trace(go.Histogram(x=det_rate_a, name=nd_a, marker_color=C["h3"],
                                   opacity=0.6, nbinsx=20))
    dr_fig.add_trace(go.Histogram(x=det_rate_b, name=nd_b, marker_color=C["warn"],
                                   opacity=0.6, nbinsx=20))
    pfig(dr_fig, 380)
    dr_fig.update_layout(
        barmode="overlay", xaxis_title="Detection Rate (fraction of samples)",
        yaxis_title=f"# {level_label} Features",
        title=dict(text="Detection Rate Distribution", font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # CV comparison
    cv_a = dfa.replace(0, np.nan).std(axis=1) / dfa.replace(0, np.nan).mean(axis=1)
    cv_b = dfb.replace(0, np.nan).std(axis=1) / dfb.replace(0, np.nan).mean(axis=1)

    cv_fig = px.scatter(x=cv_a.values, y=cv_b.values,
                         labels={"x": f"CV ({nd_a})", "y": f"CV ({nd_b})"},
                         hover_name=common_feats)
    cv_fig.update_traces(marker=dict(size=8, color=C["accent"], opacity=0.6,
                                      line=dict(width=1, color="white")))
    max_cv = max(cv_a.dropna().max(), cv_b.dropna().max()) if not cv_a.dropna().empty else 5
    cv_fig.add_trace(go.Scatter(x=[0, max_cv], y=[0, max_cv], mode="lines",
                                 line=dict(dash="dash", color="#9ca3af", width=1.5),
                                 showlegend=False, hoverinfo="skip"))
    pfig(cv_fig, 420)
    cv_fig.update_layout(title=dict(text=f"Coefficient of Variation: {nd_a} vs {nd_b}", font=dict(size=16)))

    children.append(html.Div(style=CS, children=[
        _st("Detection Sensitivity & Reproducibility",
            "Detection rate distribution + CV comparison between ndebug modes",
            icon="ptm"),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            html.Div(style={"flex": "1", "minWidth": "400px"}, children=[dcc.Graph(figure=dr_fig)]),
            html.Div(style={"flex": "1", "minWidth": "400px"}, children=[dcc.Graph(figure=cv_fig)]),
        ]),
    ]))

    # ============================================================
    # SECTION 6: Differential Detection Table
    # ============================================================
    diff_data = []
    for feat in common_feats:
        ma = mean_a[feat]
        mb = mean_b[feat]
        da_ = det_a[feat]
        db_ = det_b[feat]
        if ma > 0 and mb > 0:
            fc = np.log2(mb / ma) if ma > 0 else np.nan
        elif ma > 0:
            fc = -np.inf
        elif mb > 0:
            fc = np.inf
        else:
            fc = 0
        diff_data.append({
            "Feature": feat,
            f"Mean_{nd_a}": round(ma, 5),
            f"Mean_{nd_b}": round(mb, 5),
            "log2FC": round(fc, 3) if np.isfinite(fc) else ("Inf" if fc > 0 else "-Inf"),
            f"Det_{nd_a}": "Yes" if da_ else "No",
            f"Det_{nd_b}": "Yes" if db_ else "No",
            "Status": "Both" if both_det[feat] else (f"Only {nd_a}" if only_a[feat] else
                       (f"Only {nd_b}" if only_b[feat] else "Neither")),
        })
    diff_df = pd.DataFrame(diff_data)
    # Sort by absolute log2FC (biggest differences first)
    diff_df["_abs_fc"] = diff_df["log2FC"].apply(lambda x: abs(float(x)) if isinstance(x, (int, float)) else 999)
    diff_df = diff_df.sort_values("_abs_fc", ascending=False).drop(columns=["_abs_fc"]).head(50)

    children.append(html.Div(style=CS, children=[
        _st("Differential Detection",
            f"Top 50 features with largest ratio differences between {nd_a} and {nd_b} | "
            f"log2FC = log2({nd_b}/{nd_a})", icon="table"),
        make_table(diff_df, "ndebug-diff-table"),
    ]))

    # ============================================================
    # SECTION 7: Area Comparison (if both have areas)
    # ============================================================
    areas_a = da.get("areas_norm")
    areas_b = db.get("areas_norm")
    if areas_a is not None and areas_b is not None and not areas_a.empty and not areas_b.empty:
        common_area_feats = sorted(set(areas_a.index) & set(areas_b.index))
        if common_area_feats:
            area_mean_a = areas_a.loc[common_area_feats].mean(axis=1)
            area_mean_b = areas_b.loc[common_area_feats].mean(axis=1)

            # MA plot (mean vs difference)
            M = (area_mean_a + area_mean_b) / 2  # average intensity
            A_diff = area_mean_b - area_mean_a     # difference (log scale, so this = log2 FC)

            ma_fig = go.Figure()
            ma_fig.add_trace(go.Scatter(
                x=M.values, y=A_diff.values, mode="markers",
                marker=dict(size=7, color=C["accent"], opacity=0.6,
                            line=dict(width=0.5, color="white")),
                text=common_area_feats, hoverinfo="text+x+y",
                name="Features"))
            ma_fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af", line_width=1.5)
            pfig(ma_fig, 450)
            ma_fig.update_layout(
                xaxis_title="Mean log2 Intensity (A)", yaxis_title=f"Difference ({nd_b} - {nd_a})",
                title=dict(text=f"MA Plot: Normalized Areas ({nd_a} vs {nd_b})", font=dict(size=16)))

            # Intensity distributions (box plots)
            box_data = []
            for feat in common_area_feats[:30]:  # top 30 features
                for v in areas_a.loc[feat].dropna():
                    box_data.append({"Feature": feat, "Mode": nd_a, "log2_Area": v})
                for v in areas_b.loc[feat].dropna():
                    box_data.append({"Feature": feat, "Mode": nd_b, "log2_Area": v})
            if box_data:
                bdf = pd.DataFrame(box_data)
                box_fig = px.box(bdf, x="Mode", y="log2_Area", color="Mode",
                                  color_discrete_map={nd_a: C["h3"], nd_b: C["warn"]},
                                  points="all")
                pfig(box_fig, 380)
                box_fig.update_layout(
                    title=dict(text=f"Intensity Distribution Comparison", font=dict(size=16)),
                    showlegend=False)
            else:
                box_fig = go.Figure()

            children.append(html.Div(style=CS, children=[
                _st("Area Intensity Comparison",
                    f"MA plot of log2+QN normalized areas | Positive = higher in {nd_b}",
                    icon="bar_chart"),
                html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                    html.Div(style={"flex": "1.2", "minWidth": "450px"}, children=[dcc.Graph(figure=ma_fig)]),
                    html.Div(style={"flex": "0.8", "minWidth": "350px"}, children=[dcc.Graph(figure=box_fig)]),
                ]),
            ]))

    # ============================================================
    # SECTION 8: Summary Statistics Table
    # ============================================================
    summary_rows = [
        {"Metric": "Total features (common)", "Value_A": n_total, "Value_B": n_total},
        {"Metric": f"Detected (>= 50% samples)", "Value_A": int(det_a.sum()), "Value_B": int(det_b.sum())},
        {"Metric": "Detection rate (mean)", "Value_A": f"{det_rate_a.mean():.3f}", "Value_B": f"{det_rate_b.mean():.3f}"},
        {"Metric": "Mean ratio (detected)", "Value_A": f"{mean_a[det_a].mean():.5f}", "Value_B": f"{mean_b[det_b].mean():.5f}"},
        {"Metric": "Median CV", "Value_A": f"{cv_a.median():.3f}" if not cv_a.dropna().empty else "N/A",
                                "Value_B": f"{cv_b.median():.3f}" if not cv_b.dropna().empty else "N/A"},
        {"Metric": "Samples", "Value_A": dfa.shape[1], "Value_B": dfb.shape[1]},
        {"Metric": "Groups", "Value_A": meta_a["Group"].nunique() if not meta_a.empty else "?",
                              "Value_B": meta_b["Group"].nunique() if not meta_b.empty else "?"},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.columns = ["Metric", nd_a, nd_b]

    children.append(html.Div(style=CS, children=[
        _st("Summary Statistics",
            f"Side-by-side comparison of {exp_a} vs {exp_b}", icon="table"),
        make_table(summary_df, "ndebug-summary-table"),
    ]))

    return html.Div(children)


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
    tissues = sorted(meta["Tissue"].unique()) if "Tissue" in meta.columns and meta["Tissue"].nunique() > 1 else []

    filter_children = [
        html.Div(style={"flex":"1","minWidth":"180px"}, children=[
            _lbl("Group A"),
            dcc.Dropdown(id="cmp-a",options=[{"label":g,"value":g} for g in groups],
                         value=groups[0],clearable=False,style=DS)]),
        html.Div(style={"flex":"1","minWidth":"180px"}, children=[
            _lbl("Group B"),
            dcc.Dropdown(id="cmp-b",options=[{"label":g,"value":g} for g in groups],
                         value=groups[1] if len(groups)>1 else groups[0],clearable=False,style=DS)]),
        html.Div(style={"flex":"1","minWidth":"200px"}, children=[
            _lbl("Data Level"),
            dcc.Dropdown(id="cmp-level",options=[{"label":"hPTM (single, ratios)","value":"hptm"},
                         {"label":"hPF (peptidoforms, ratios)","value":"hpf"}] +
                         ([{"label":"hPF Areas (log2+QN)","value":"areas_norm"},
                           {"label":"hPF Areas (log2)","value":"areas_log2"}]
                          if "areas_norm" in d and d["areas_norm"] is not None else []),
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
    if tissues:
        filter_children.append(html.Div(style={"flex":"1","minWidth":"180px"}, children=[
            _lbl("Tissue"),
            dcc.Dropdown(id="cmp-tissue",
                options=[{"label":"All tissues","value":"All"}]+[{"label":t,"value":t} for t in tissues],
                value="All", clearable=False, style=DS)]))
    else:
        filter_children.append(html.Div(dcc.Store(id="cmp-tissue", data="All")))

    return html.Div([
        html.Div(style={**CS,"display":"flex","gap":"16px","flexWrap":"wrap","alignItems":"flex-end"},
                 children=filter_children),
        html.Div(id="cmp-out"),
    ])

@callback(Output("cmp-out","children"),
          Input("cmp-a","value"),Input("cmp-b","value"),Input("cmp-level","value"),
          Input("cmp-show","value"),Input("cmp-classify","value"),Input("cmp-fdr","value"),
          Input("cmp-tissue","value"),
          Input("cur-exp","data"))
def _cmp(ga, gb, level, show, classify, fdr_log, tissue, exp):
    t0 = time.time()
    if not exp or exp not in EXP_DATA: return html.P("N/A")
    d = EXP_DATA[exp]
    df = _get_data_source(d, level)
    meta = d["metadata"].copy()
    is_log = level in ("areas_norm", "areas_log2")
    if df is None or df.empty: return html.P("No data for level.")

    fdr_thresh = 10 ** fdr_log if fdr_log else 0.05

    # Filter to tissue if applicable
    if tissue and tissue != "All" and "Tissue" in meta.columns:
        meta = meta[meta["Tissue"] == tissue].copy()
        samps = meta["Sample"].tolist()
        cols = [c for c in df.columns if c in samps]; df = df[cols]

    mw = pairwise_mw(df, meta, ga, gb, is_log=is_log)
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
            _st("Mann-Whitney Results",f"FDR-corrected | {show} | Editable | Classified", icon="balance"),
            make_table(display_mw[tbl_cols],"cmp-table")]),
    ])


@callback(Output("download-data","data", allow_duplicate=True),
          Input("cmp-export","n_clicks"),
          State("cmp-a","value"),State("cmp-b","value"),State("cmp-level","value"),
          State("cur-exp","data"), prevent_initial_call=True)
def _cmp_export(n, ga, gb, level, exp):
    if not n or not exp or exp not in EXP_DATA: return no_update
    d = EXP_DATA[exp]; df = _get_data_source(d, level); meta = d["metadata"]
    is_log = level in ("areas_norm", "areas_log2")
    if df is None: return no_update
    mw = pairwise_mw(df, meta, ga, gb, is_log=is_log)
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
            _st("Sample Metadata Table", "Editable | Processing order, groups, batch, data quality metrics", icon="table"),
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
    return html.Div([_st("PTM Profile",f"Sample: {f}", icon="search"), dcc.Graph(figure=fig)])


# ======================================================================
# TAB: ANALYSIS LOG
# ======================================================================

def tab_log(d, exp):
    """Analysis history, session info, upload log."""
    session = get_session_info()
    hist_df = get_analysis_history(experiment=exp, limit=100)
    upl_df = get_upload_history(limit=50)
    has_history = hist_df is not None and not hist_df.empty
    has_uploads = upl_df is not None and not upl_df.empty

    # Session info card
    sess_card = html.Div(style={**CS,"display":"flex","gap":"16px","flexWrap":"wrap","marginBottom":"16px"}, children=[
        _sc("Session ID", str(session.get("id","?")), C["accent"]),
        _sc("Actions", str(session.get("n_actions",0)), C["green"]),
        _sc("Started", session.get("started","?")[:19] if session.get("started") else "?", C["h3"]),
        _sc("Last Activity", session.get("last_activity","?")[:19] if session.get("last_activity") else "?", C["h4"]),
    ])

    # Analysis history table
    if has_history:
        hist_df.columns = ["ID","Experiment","Type","Parameters","Features","Significant","Summary","Duration(ms)","Timestamp"]
        hist_table = html.Div(style=CS, children=[
            _st("Analysis History", f"Last {len(hist_df)} analyses for {exp}", icon="log"),
            make_table(hist_df, "log-hist-table")])
    else:
        hist_table = html.Div(style=CS, children=[html.P("No analyses recorded yet.", style={"color":C["muted"]})])

    # Upload history table
    if has_uploads:
        upl_df.columns = ["ID","Experiment","Filename","Type","Rows","Cols","Status","Message","Timestamp"]
        upl_table = html.Div(style=CS, children=[
            _st("Upload History", f"Last {len(upl_df)} uploads"),
            make_table(upl_df, "log-upl-table")])
    else:
        upl_table = html.Div(style=CS, children=[html.P("No uploads recorded yet.", style={"color":C["muted"]})])

    # Analysis type breakdown chart
    if has_history:
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
    hist_df = get_analysis_history(experiment=exp, limit=1000)
    if hist_df is None or hist_df.empty: return no_update
    hist_df.columns = ["ID","Experiment","Type","Parameters","Features","Significant","Summary","Duration(ms)","Timestamp"]
    fname = f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return dcc.send_data_frame(hist_df.to_csv, fname, index=False)


# ======================================================================
# TAB: EXPORT TO R
# ======================================================================

def tab_export(d):
    """Comprehensive export with user-defined filters and R-ready formats."""
    meta = d.get("metadata", pd.DataFrame())
    has_areas = "areas_norm" in d and d["areas_norm"] is not None
    groups = sorted(meta["Group"].unique()) if not meta.empty else []
    designs = sorted(meta["Design"].unique()) if "Design" in meta.columns and meta["Design"].nunique() > 1 else []

    # Data source options
    source_opts = [
        {"label": "hPTM Ratios (single PTMs)", "value": "hptm"},
        {"label": "hPF Ratios (peptidoforms)", "value": "hpf"},
    ]
    if has_areas:
        source_opts.extend([
            {"label": "hPF Areas (log2 + QN)", "value": "areas_norm"},
            {"label": "hPF Areas (log2 only)", "value": "areas_log2"},
            {"label": "hPF Areas (raw)", "value": "areas"},
            {"label": "Retention Times (RT)", "value": "rt"},
        ])

    # Format options
    format_opts = [
        {"label": "CSV (comma-separated)", "value": "csv"},
        {"label": "TSV (tab-separated)", "value": "tsv"},
        {"label": "R Script + Data (ready-to-load .R + .csv)", "value": "r_bundle"},
    ]

    # Build filters
    filter_children = [
        html.Div(style={"flex":"1","minWidth":"250px"}, children=[
            _lbl("Data Source"),
            dcc.Dropdown(id="exp-source", options=source_opts, value="hptm", clearable=False, style=DS)]),
        html.Div(style={"flex":"1","minWidth":"200px"}, children=[
            _lbl("Export Format"),
            dcc.Dropdown(id="exp-format", options=format_opts, value="csv", clearable=False, style=DS)]),
    ]

    if designs:
        filter_children.append(html.Div(style={"flex":"1","minWidth":"200px"}, children=[
            _lbl("Design / Strain"),
            dcc.Dropdown(id="exp-design",
                options=[{"label":"All designs","value":"All"}]+[{"label":f"Design {x}","value":str(x)} for x in designs],
                value="All", clearable=False, style=DS)]))
    else:
        filter_children.append(html.Div(dcc.Store(id="exp-design", data="All")))

    filter_children.extend([
        html.Div(style={"flex":"1","minWidth":"250px"}, children=[
            _lbl("Groups (select to filter)"),
            dcc.Dropdown(id="exp-groups",
                options=[{"label":g,"value":g} for g in groups],
                value=[], multi=True, placeholder="All groups (default)", style=DS)]),
        html.Div(style={"flex":"1","minWidth":"200px"}, children=[
            _lbl("Feature Filter"),
            dcc.Dropdown(id="exp-feature-filter",
                options=[{"label":"All features","value":"all"},
                         {"label":"Histone H3 only","value":"H3"},
                         {"label":"Histone H3.3 only","value":"H3.3"},
                         {"label":"Histone H4 only","value":"H4"},
                         {"label":"Methylation only","value":"Methylation"},
                         {"label":"Acetylation only","value":"Acetylation"},
                         {"label":"Non-zero only (>1 sample)","value":"nonzero"}],
                value="all", clearable=False, style=DS)]),
    ])

    # Include statistics toggle
    opts_children = [
        html.Div(style={"flex":"1","minWidth":"200px"}, children=[
            _lbl("Include Statistics"),
            dcc.Dropdown(id="exp-include-stats",
                options=[{"label":"Data only","value":"data"},
                         {"label":"Data + KW statistics","value":"stats"},
                         {"label":"Data + KW + pairwise MW","value":"full"}],
                value="data", clearable=False, style=DS)]),
        html.Div(style={"flex":"1","minWidth":"200px"}, children=[
            _lbl("Normalization Info"),
            dcc.Checklist(id="exp-add-meta",
                options=[{"label":" Include sample metadata sheet","value":"meta"},
                         {"label":" Include normalization summary","value":"norm"}],
                value=["meta"], inline=False,
                style={"fontSize":"13px","color":C["text"],"marginTop":"6px"})]),
    ]

    # Export button
    btn_style = {"padding":"12px 28px","borderRadius":"8px","border":"none",
                 "backgroundColor":C["accent"],"color":"white","fontWeight":"700",
                 "cursor":"pointer","fontSize":"15px","letterSpacing":"0.5px"}

    return html.Div([
        _st("Export Data for R / External Analysis",
            "Filter data by source, groups, features | Export as CSV, TSV, or R-ready bundle", icon="export"),
        html.Div(style={**CS,"display":"flex","gap":"16px","alignItems":"flex-end","flexWrap":"wrap"},
                 children=filter_children),
        html.Div(style={**CS,"display":"flex","gap":"16px","alignItems":"flex-end","flexWrap":"wrap","marginTop":"12px"},
                 children=opts_children),
        html.Div(style={"display":"flex","gap":"16px","marginTop":"20px","alignItems":"center"}, children=[
            html.Button("Export Data", id="exp-go", n_clicks=0, style=btn_style),
            html.Button("Preview (first 20 rows)", id="exp-preview", n_clicks=0,
                        style={**btn_style, "backgroundColor":C["h3"]}),
        ]),
        html.Div(id="exp-status", style={"marginTop":"12px"}),
        html.Div(id="exp-preview-out", style={"marginTop":"16px"}),
    ])


def _build_export_data(d, source, design, groups_filter, feature_filter, include_stats, meta_full):
    """Build export DataFrame(s) based on user filters. Returns dict of {name: DataFrame}."""
    meta = meta_full.copy()

    # Resolve data source
    if source == "areas":
        df = d.get("areas")
    elif source == "rt":
        df = d.get("rt")
    else:
        df = _get_data_source(d, source)
    if df is None:
        return None, "Selected data source not available for this experiment."

    is_log = source in ("areas_norm", "areas_log2")

    # Filter by design
    if design and design != "All" and "Design" in meta.columns:
        meta = meta[meta["Design"].astype(str) == str(design)].copy()
        samps = meta["Sample"].tolist()
        cols = [c for c in df.columns if c in samps]
        df = df[cols]

    # Filter by groups
    if groups_filter and len(groups_filter) > 0:
        meta = meta[meta["Group"].isin(groups_filter)].copy()
        samps = meta["Sample"].tolist()
        cols = [c for c in df.columns if c in samps]
        df = df[cols]

    # Filter features
    if feature_filter and feature_filter != "all":
        if feature_filter == "nonzero":
            if is_log:
                nonzero_mask = df.apply(lambda row: row.dropna()[np.isfinite(row.dropna())].shape[0] >= 1, axis=1)
            else:
                nonzero_mask = df.apply(lambda row: (row.dropna() != 0).sum() >= 1, axis=1)
            df = df[nonzero_mask]
        elif feature_filter in ("H3", "H3.3", "H4"):
            keep = [i for i in df.index if classify_ptm_name(i)[0] == feature_filter]
            df = df.loc[keep]
        elif feature_filter in ("Methylation", "Acetylation"):
            keep = [i for i in df.index if classify_ptm_name(i)[1] == feature_filter]
            df = df.loc[keep]

    if df.empty:
        return None, "No data after applying filters."

    groups = sorted(meta["Group"].unique())
    sheets = {"data": df}

    # Add statistics if requested
    if include_stats in ("stats", "full") and len(groups) >= 2 and source != "rt":
        stats_df = robust_group_test(df, meta, groups, is_log=is_log)
        if not stats_df.empty:
            stats_df = _enrich_stats(stats_df, groups, is_log=is_log)
            # Add FC column
            fc_list = []
            for _, row in stats_df.iterrows():
                means = [row.get(f"mean_{g}", np.nan) for g in groups]
                means = [m for m in means if not np.isnan(m)]
                if is_log:
                    means = [m for m in means if np.isfinite(m)]
                    fc_list.append(max(means) - min(means) if len(means) >= 2 else 0)
                else:
                    means = [m for m in means if m > 0]
                    fc_list.append(np.log2(max(means) / min(means)) if len(means) >= 2 else 0)
            stats_df["maxLog2FC"] = fc_list
            sheets["statistics"] = stats_df

        # Full: add all pairwise comparisons
        if include_stats == "full" and len(groups) >= 2:
            pw_frames = []
            for g1, g2 in combinations(groups, 2):
                mw = pairwise_mw(df, meta, g1, g2, is_log=is_log)
                if not mw.empty:
                    mw.insert(0, "Comparison", f"{g1} vs {g2}")
                    pw_frames.append(mw)
            if pw_frames:
                sheets["pairwise"] = pd.concat(pw_frames, ignore_index=True)

    return sheets, None


def _generate_r_script(exp_name, source, sheets, meta_df):
    """Generate R script that loads the exported data and sets up analysis."""
    data_file = f"{exp_name}_data.csv"
    stats_file = f"{exp_name}_statistics.csv" if "statistics" in sheets else None
    pw_file = f"{exp_name}_pairwise.csv" if "pairwise" in sheets else None
    meta_file = f"{exp_name}_metadata.csv"
    is_log = source in ("areas_norm", "areas_log2")

    groups = sorted(meta_df["Group"].unique())
    n_samp = len(meta_df)
    n_feat = len(sheets["data"])

    lines = [
        "# ========================================================",
        f"# EpiProfile-Plants: R Analysis Script",
        f"# Experiment: {exp_name}",
        f"# Data source: {source}",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"# Features: {n_feat} | Samples: {n_samp} | Groups: {len(groups)}",
        "# ========================================================",
        "",
        "# Required packages (install if needed):",
        "# install.packages(c('ggplot2', 'pheatmap', 'dplyr', 'tidyr'))",
        "",
        "library(ggplot2)",
        "library(pheatmap)",
        "library(dplyr)",
        "library(tidyr)",
        "",
        "# ---- Load Data ----",
        f'data <- read.csv("{data_file}", row.names = 1, check.names = FALSE)',
        f'metadata <- read.csv("{meta_file}", stringsAsFactors = TRUE)',
        "",
        "# Verify dimensions",
        f'cat("Data matrix:", nrow(data), "features x", ncol(data), "samples\\n")',
        f'cat("Metadata:", nrow(metadata), "samples\\n")',
        f'cat("Groups:", paste(levels(metadata$Group), collapse=", "), "\\n")',
        "",
    ]

    if stats_file:
        lines.extend([
            "# ---- Load Statistics (Kruskal-Wallis + FDR) ----",
            f'stats <- read.csv("{stats_file}")',
            'cat("Significant features (FDR < 0.05):", sum(stats$KW_FDR < 0.05, na.rm=TRUE), "\\n")',
            "",
        ])

    if pw_file:
        lines.extend([
            "# ---- Load Pairwise Comparisons (Mann-Whitney) ----",
            f'pairwise <- read.csv("{pw_file}")',
            'cat("Pairwise comparisons:", nrow(pairwise), "\\n")',
            "",
        ])

    lines.extend([
        "# ---- Data Properties ----",
        f'is_log_scale <- {"TRUE" if is_log else "FALSE"}',
        f'data_source <- "{source}"',
        "",
        "# ---- Heatmap ----",
        "# Annotation for columns (samples)",
        "ann_col <- data.frame(Group = metadata$Group, row.names = metadata$Sample)",
        "",
        "# Filter to features with variance",
        "data_var <- data[apply(data, 1, function(x) var(x, na.rm=TRUE)) > 0, , drop=FALSE]",
        "",
        "if (nrow(data_var) > 0) {",
        '  pheatmap(as.matrix(data_var),',
        '           annotation_col = ann_col,',
        '           scale = "row",',
        '           clustering_distance_rows = "correlation",',
        '           clustering_distance_cols = "correlation",',
        '           show_colnames = FALSE,',
        f'           main = "EpiProfile: {source}")',
        "}",
        "",
        "# ---- PCA ----",
        "data_complete <- data_var[complete.cases(data_var), , drop=FALSE]",
        "if (nrow(data_complete) >= 3) {",
        "  pca <- prcomp(t(as.matrix(data_complete)), scale. = TRUE)",
        "  pca_df <- data.frame(pca$x[, 1:min(3, ncol(pca$x))], Group = metadata$Group)",
        "",
        '  p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Group)) +',
        '    geom_point(size = 3) +',
        '    stat_ellipse(level = 0.95, linetype = "dashed") +',
        '    theme_minimal() +',
        f'    labs(title = "PCA: {source}",',
        '         x = paste0("PC1 (", round(summary(pca)$importance[2,1]*100, 1), "%)"),',
        '         y = paste0("PC2 (", round(summary(pca)$importance[2,2]*100, 1), "%)"))',
        "  print(p_pca)",
        "}",
        "",
        "# ---- Boxplot of top features ----",
    ])

    if stats_file:
        lines.extend([
            "top_features <- head(stats$PTM[order(stats$KW_pval)], 9)",
        ])
    else:
        lines.extend([
            "# No statistics available; use first 9 features",
            "top_features <- head(rownames(data_var), 9)",
        ])

    lines.extend([
        "data_long <- data[top_features, , drop=FALSE] %>%",
        "  tibble::rownames_to_column('Feature') %>%",
        "  pivot_longer(-Feature, names_to = 'Sample', values_to = 'Value') %>%",
        "  left_join(metadata, by = 'Sample')",
        "",
        "p_box <- ggplot(data_long, aes(x = Group, y = Value, fill = Group)) +",
        "  geom_boxplot(outlier.shape = NA) +",
        "  geom_jitter(width = 0.2, alpha = 0.5, size = 1) +",
        "  facet_wrap(~Feature, scales = 'free_y', ncol = 3) +",
        "  theme_minimal() +",
        "  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +",
        f'  labs(title = "Top Features by Group", y = "{source}")',
        "print(p_box)",
        "",
    ])

    if stats_file:
        lines.extend([
            "# ---- Volcano Plot ----",
            "if ('maxLog2FC' %in% colnames(stats)) {",
            "  stats$negLog10FDR <- -log10(stats$KW_FDR + 1e-300)",
            "  stats$Significant <- stats$KW_FDR < 0.05",
            "  p_vol <- ggplot(stats, aes(x = maxLog2FC, y = negLog10FDR, color = Significant)) +",
            "    geom_point(alpha = 0.7) +",
            '    scale_color_manual(values = c("grey60", "red")) +',
            "    geom_hline(yintercept = -log10(0.05), linetype = 'dashed', color = 'red') +",
            "    theme_minimal() +",
            '    labs(title = "Volcano Plot", x = "max |log2(FC)|", y = "-log10(FDR)")',
            "  print(p_vol)",
            "}",
            "",
        ])

    lines.extend([
        "# ---- Save session ----",
        f'# save.image("{exp_name}_session.RData")',
        f'cat("\\nAnalysis ready. {n_feat} features x {n_samp} samples loaded.\\n")',
    ])

    return "\n".join(lines)


@callback(Output("exp-preview-out", "children"),
          Input("exp-preview", "n_clicks"),
          State("exp-source","value"), State("exp-design","value"),
          State("exp-groups","value"), State("exp-feature-filter","value"),
          State("exp-include-stats","value"), State("exp-add-meta","value"),
          State("cur-exp","data"), prevent_initial_call=True)
def _exp_preview(n, source, design, groups_sel, feat_filter, inc_stats, add_meta, exp):
    if not n or not exp or exp not in EXP_DATA: return html.P("N/A")
    d = EXP_DATA[exp]; meta = d.get("metadata", pd.DataFrame())

    sheets, err = _build_export_data(d, source, design, groups_sel, feat_filter, inc_stats, meta)
    if err: return html.Div(style=CS, children=[html.P(err, style={"color":C["red"]})])

    children = []
    for name, df in sheets.items():
        preview = df.head(20)
        n_rows, n_cols = df.shape
        children.append(html.Div(style=CS, children=[
            _st(f"Preview: {name}", f"{n_rows} rows x {n_cols} columns (showing first 20)"),
            make_table(preview.reset_index() if name == "data" else preview, f"exp-tbl-{name}")
        ]))

    # Show what files would be generated
    exp_name = exp.split("(")[0].strip().replace(" ","_")
    file_list = [f"{exp_name}_data.csv"]
    if "statistics" in sheets: file_list.append(f"{exp_name}_statistics.csv")
    if "pairwise" in sheets: file_list.append(f"{exp_name}_pairwise.csv")
    if add_meta and "meta" in add_meta: file_list.append(f"{exp_name}_metadata.csv")

    src = source if source else "ratios"
    fmt_info = "R bundle (.zip with .R script + .csv files)" if True else "CSV"

    children.insert(0, html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"12px"}, children=[
        _sc("Features", str(sheets["data"].shape[0]), C["accent"]),
        _sc("Samples", str(sheets["data"].shape[1]), C["green"]),
        _sc("Source", src, C["h3"]),
        _sc("Sheets", str(len(sheets)), C["h4"]),
    ]))

    return html.Div(children)


@callback(Output("download-data","data", allow_duplicate=True),
          Output("exp-status","children"),
          Input("exp-go","n_clicks"),
          State("exp-source","value"), State("exp-format","value"),
          State("exp-design","value"), State("exp-groups","value"),
          State("exp-feature-filter","value"), State("exp-include-stats","value"),
          State("exp-add-meta","value"), State("cur-exp","data"),
          prevent_initial_call=True)
def _exp_download(n, source, fmt, design, groups_sel, feat_filter, inc_stats, add_meta, exp):
    if not n or not exp or exp not in EXP_DATA: return no_update, ""
    d = EXP_DATA[exp]; meta = d.get("metadata", pd.DataFrame())
    t0 = time.time()

    sheets, err = _build_export_data(d, source, design, groups_sel, feat_filter, inc_stats, meta)
    if err: return no_update, html.P(err, style={"color":C["red"]})

    exp_name = exp.split("(")[0].strip().replace(" ","_")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    sep_char = "," if fmt == "csv" else "\t"
    ext = "csv" if fmt == "csv" else "tsv"

    if fmt == "r_bundle":
        # Create a ZIP with R script + CSV data files + metadata
        import zipfile
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Data file
            data_csv = sheets["data"].to_csv()
            zf.writestr(f"{exp_name}_data.csv", data_csv)

            # Statistics file
            if "statistics" in sheets:
                zf.writestr(f"{exp_name}_statistics.csv", sheets["statistics"].to_csv(index=False))

            # Pairwise file
            if "pairwise" in sheets:
                zf.writestr(f"{exp_name}_pairwise.csv", sheets["pairwise"].to_csv(index=False))

            # Metadata
            if add_meta and "meta" in add_meta:
                # Filter metadata to match exported samples
                export_samps = list(sheets["data"].columns)
                meta_export = meta[meta["Sample"].isin(export_samps)].copy()
                zf.writestr(f"{exp_name}_metadata.csv", meta_export.to_csv(index=False))

            # Normalization summary
            if add_meta and "norm" in add_meta:
                norm_info = [
                    f"EpiProfile-Plants Export Summary",
                    f"Experiment: {exp}",
                    f"Data source: {source}",
                    f"Date: {datetime.now().isoformat()}",
                    f"Features: {sheets['data'].shape[0]}",
                    f"Samples: {sheets['data'].shape[1]}",
                    f"",
                ]
                if source == "areas_norm":
                    norm_info.extend([
                        "Normalization pipeline:",
                        "  1. Raw MS1 peak areas extracted from EpiProfile output",
                        "  2. Zero values replaced with NaN (missing)",
                        "  3. Log2 transformation applied",
                        "  4. Quantile normalization (Bolstad 2003) across samples",
                        "  5. NaN values preserved (not imputed)",
                        "",
                        "Reference: Bolstad BM et al. (2003) Bioinformatics 19(2):185-93",
                    ])
                elif source == "areas_log2":
                    norm_info.extend([
                        "Normalization pipeline:",
                        "  1. Raw MS1 peak areas extracted from EpiProfile output",
                        "  2. Zero values replaced with NaN",
                        "  3. Log2 transformation applied",
                        "  4. No quantile normalization (log2 only)",
                    ])
                elif source in ("hptm", "hpf", "ratios"):
                    norm_info.extend([
                        "Data type: Relative abundance ratios (0-1)",
                        "  - Calculated by EpiProfile as peak area / total region area",
                        "  - Compositional data (values sum to ~1.0 per peptide region)",
                        "  - Note: spurious negative correlations possible due to closure",
                    ])
                elif source == "areas":
                    norm_info.extend([
                        "Data type: Raw MS1 peak areas (unnormalized)",
                        "  - Direct EpiProfile output without transformation",
                        "  - Recommend log2 + quantile normalization before analysis",
                    ])
                elif source == "rt":
                    norm_info.extend([
                        "Data type: Retention times (minutes)",
                        "  - Chromatographic elution times from LC-MS/MS",
                    ])
                zf.writestr(f"{exp_name}_normalization_info.txt", "\n".join(norm_info))

            # R script
            r_script = _generate_r_script(exp_name, source, sheets, meta)
            zf.writestr(f"{exp_name}_analysis.R", r_script)

        buf.seek(0)
        fname = f"{exp_name}_{source}_R_bundle_{ts}.zip"
        dur = int((time.time() - t0) * 1000)
        n_total = sheets["data"].shape[0]
        log_analysis(exp, "export_r_bundle",
                     {"source": source, "format": "r_bundle", "n_sheets": len(sheets)},
                     n_total, 0, f"R bundle: {n_total} features, {len(sheets)} sheets")

        status = html.Div(style={"display":"flex","gap":"12px","alignItems":"center"}, children=[
            html.Span("Exported!", style={"color":C["green"],"fontWeight":"700","fontSize":"15px"}),
            html.Span(f"{fname} | {n_total} features | {sheets['data'].shape[1]} samples | {len(sheets)} files | {dur}ms",
                      style={"color":C["muted"],"fontSize":"13px"}),
        ])
        return dcc.send_bytes(buf.getvalue(), fname), status

    else:
        # CSV or TSV export
        data_df = sheets["data"]
        # If stats requested, merge stats columns into data
        if "statistics" in sheets:
            stats_cols = sheets["statistics"][["PTM","KW_pval","KW_FDR","maxLog2FC","Direction","Histone","PTM_type"]].copy()
            data_with_stats = data_df.copy()
            data_with_stats.index.name = "PTM"
            merged = data_with_stats.reset_index().merge(stats_cols, on="PTM", how="left").set_index("PTM")
            export_df = merged
        else:
            export_df = data_df

        fname = f"{exp_name}_{source}_{ts}.{ext}"
        dur = int((time.time() - t0) * 1000)
        n_total = len(export_df)
        log_analysis(exp, f"export_{ext}",
                     {"source": source, "format": ext, "n_rows": n_total},
                     n_total, 0, f"Exported {n_total} rows as {ext}")

        status = html.Div(style={"display":"flex","gap":"12px","alignItems":"center"}, children=[
            html.Span("Exported!", style={"color":C["green"],"fontWeight":"700","fontSize":"15px"}),
            html.Span(f"{fname} | {n_total} features | {dur}ms",
                      style={"color":C["muted"],"fontSize":"13px"}),
        ])
        return dcc.send_data_frame(export_df.to_csv, fname, sep=sep_char), status


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EpiProfile-Plants Dashboard v3.11")
    print(f"  Experiments: {len(EXP_DATA)}")
    for n in EXP_DATA: print(f"    * {n}")
    print(f"\n  =>  http://localhost:{args.port}")
    print("="*60 + "\n")
    app.run(debug=False, port=args.port, host=args.host)
