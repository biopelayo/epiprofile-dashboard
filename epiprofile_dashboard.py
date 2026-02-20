"""
EpiProfile-Plants Dashboard v3.1 -- Publication-Quality Visualization
=====================================================================
Interactive Dash/Plotly dashboard for EpiProfile-Plants output.

Properly classifies histone data into three levels:
  hPTM  = individual PTM marks         (from histone_ratios_single_PTMs.xls)
  hPF   = peptidoforms (combinatorial) (from histone_ratios.xls, data rows)
  hDP   = derivatized peptide regions  (from histone_ratios.xls, headers)
  SeqVar = sequence variants per region (from histone_ratios.xls, variant block)

Usage:
  python epiprofile_dashboard.py <dir1> [dir2] ...
  python epiprofile_dashboard.py                    # uses DEFAULTS
Access:  http://localhost:8050
"""

import os, re, sys, base64, math, textwrap, configparser, argparse, warnings
from pathlib import Path
from io import StringIO
from itertools import combinations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram as scipy_dend
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, mannwhitneyu, kruskal
from sklearn.decomposition import PCA as skPCA
from statsmodels.stats.multitest import multipletests
from dash import Dash, html, dcc, callback, Input, Output, State, dash_table, ctx, no_update
import dash

warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULTS = {
    "PXD046788 (Arabidopsis treatments)": r"D:\epiprofile_data\PXD046788\MS1_MS2\RawData",
    "PXD014739 (Arabidopsis histone)": r"D:\epiprofile_data\PXD014739\RawData",
    "PXD046034 (Arabidopsis FAS/NAP)": r"E:\EpiProfile_AT_PXD046034_raw\PXD046034\PXD046034",
    "Ontogeny 1exp (Arabidopsis stages)": r"E:\EpiProfile_Proyecto\EpiProfile_20_AT\histone_layouts_ontogeny_1exp",
}

parser = argparse.ArgumentParser(description="EpiProfile-Plants Dashboard v3.1")
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
    "bg":"#f8f9fc","card":"#fff","border":"#e5e7eb","text":"#1e293b","text2":"#475569",
    "accent":"#4f46e5","accent_l":"#eef2ff","red":"#dc2626","green":"#059669",
    "muted":"#94a3b8","warn":"#d97706","h3":"#7c3aed","h4":"#0891b2",
}
GC = ["#4f46e5","#059669","#d97706","#dc2626","#0891b2","#7c3aed","#db2777",
      "#0d9488","#ea580c","#4338ca","#16a34a","#ca8a04","#9333ea","#2563eb","#c026d3"]

LOGO = "data:image/svg+xml;base64," + base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">'
    b'<defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">'
    b'<stop offset="0%" style="stop-color:#4f46e5"/>'
    b'<stop offset="100%" style="stop-color:#7c3aed"/></linearGradient></defs>'
    b'<rect width="36" height="36" rx="8" fill="url(#g)"/>'
    b'<text x="18" y="24" text-anchor="middle" fill="white" font-family="Arial" '
    b'font-weight="bold" font-size="18">E</text></svg>').decode()

PUB = go.layout.Template()
PUB.layout = go.Layout(
    font=dict(family=FONT, size=12, color=C["text"]),
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    xaxis=dict(gridcolor="#f1f5f9",linecolor="#cbd5e1",linewidth=1,mirror=True,
               title_font=dict(size=13,color="#334155"),tickfont=dict(size=10,color="#64748b")),
    yaxis=dict(gridcolor="#f1f5f9",linecolor="#cbd5e1",linewidth=1,mirror=True,
               title_font=dict(size=13,color="#334155"),tickfont=dict(size=10,color="#64748b")),
    legend=dict(bgcolor="rgba(255,255,255,0.95)",font=dict(size=10,color="#334155"),
                bordercolor="#e5e7eb",borderwidth=1),
    margin=dict(t=40,b=50,l=60,r=20), colorway=GC)

CS = {"backgroundColor":"#fff","borderRadius":"12px","border":"1px solid #e5e7eb",
      "padding":"24px","marginBottom":"16px",
      "boxShadow":"0 1px 3px rgba(0,0,0,0.04),0 1px 2px rgba(0,0,0,0.03)"}
DS = {"fontSize":"13px","borderRadius":"8px"}
TC = {"backgroundColor":"#fff","color":C["text"],"border":"1px solid #f1f5f9",
      "fontSize":"12px","textAlign":"left","padding":"8px 12px","fontFamily":FONT,
      "minWidth":"80px","maxWidth":"280px","overflow":"hidden","textOverflow":"ellipsis"}
TH = {"backgroundColor":"#f8fafc","fontWeight":"600","color":"#334155",
      "borderBottom":f"2px solid {C['accent']}","fontSize":"11px",
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
    for fn in ["phenodata_arabidopsis_project.tsv","phenodata.tsv","phenodata_PXD046034.tsv"]:
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


def build_metadata(sample_names, phenodata=None):
    records = []
    for s in sample_names:
        t, ti, r = parse_sample_name(s.strip())
        records.append({"Sample":s,"Treatment":t,"Tissue":ti,"Replicate":r})
    meta = pd.DataFrame(records)
    if phenodata is not None and "Sample_Name" in phenodata.columns:
        pheno = phenodata.copy()
        pheno["_n"] = pheno["Sample_Name"].str.extract(r"^(\d+)-").astype(float)
        meta["_n"] = meta["Sample"].str.extract(r"^(\d+)").astype(float) if meta["Sample"].str.match(r"^\d").any() else range(len(meta))
        if "Sample_Group" in pheno.columns:
            mm = dict(zip(pheno["Sample_Name"], pheno["Sample_Group"]))
            nm = dict(zip(pheno["_n"].dropna(), pheno.loc[pheno["_n"].notna(),"Sample_Group"]))
            def gg(row):
                if row["Sample"] in mm: return mm[row["Sample"]]
                if row["_n"] in nm: return nm[row["_n"]]
                return row["Treatment"]
            meta["Group"] = meta.apply(gg, axis=1)
        meta = meta.drop(columns=["_n"], errors="ignore")
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
        if "hptm" in d: print(f"    hPTM: {d['hptm'].shape}")
        if "hpf" in d and not d["hpf"].empty: print(f"    hPF:  {d['hpf'].shape}")
        if "hdp_list" in d: print(f"    hDP:  {len(d['hdp_list'])} regions")
        print(f"    Folders: {len(d['sample_folders'])}")
        psm = d.get("all_psm", pd.DataFrame())
        if not psm.empty: print(f"    PSMs: {len(psm)}")
        print(f"    >> {d.get('description','')}")

DEFAULT_EXP = list(EXP_DATA.keys())[0] if EXP_DATA else None

# ======================================================================
# APP
# ======================================================================

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "EpiProfile-Plants Dashboard"

ts = {"color":"#64748b","backgroundColor":"#fff","border":"none",
      "borderBottom":"2px solid transparent","padding":"10px 20px",
      "fontSize":"13px","fontWeight":"500","fontFamily":FONT}
tss = {**ts,"color":C["accent"],"borderBottom":f"2px solid {C['accent']}","fontWeight":"600"}

# ======================================================================
# LAYOUT
# ======================================================================

app.layout = html.Div(style={"backgroundColor":C["bg"],"minHeight":"100vh","fontFamily":FONT,"color":C["text"]}, children=[
    # Header
    html.Div(style={"backgroundColor":"#fff","borderBottom":"1px solid #e5e7eb","padding":"0 32px",
                     "display":"flex","alignItems":"center","gap":"20px","height":"64px","flexWrap":"wrap"}, children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"12px"}, children=[
            html.Img(src=LOGO, style={"height":"32px","width":"32px"}),
            html.Div([
                html.H1("EpiProfile-Plants", style={"margin":"0","fontSize":"18px","fontWeight":"700","color":C["text"]}),
                html.Span("PTM Dashboard v3.1 | hPTM / hPF / hDP", style={"color":C["muted"],"fontSize":"11px"}),
            ]),
        ]),
        html.Div(style={"flex":"1"}),
        html.Div(style={"display":"flex","alignItems":"center","gap":"8px"}, children=[
            html.Span("EXPERIMENT",style={"color":C["muted"],"fontSize":"10px","fontWeight":"600","letterSpacing":"1px"}),
            dcc.Dropdown(id="exp-sel", options=[{"label":k,"value":k} for k in EXP_DATA],
                         value=DEFAULT_EXP, clearable=False, style={"width":"360px",**DS}),
        ]),
    ]),
    # Description bar
    html.Div(id="desc-bar", style={"backgroundColor":C["accent_l"],"padding":"6px 32px",
                                     "fontSize":"12px","color":C["accent"],"fontWeight":"500",
                                     "borderBottom":"1px solid #e5e7eb"}),
    # Tabs
    dcc.Tabs(id="tabs", value="tab-hpf", style={"backgroundColor":"#fff","borderBottom":"1px solid #e5e7eb"},
             colors={"border":"transparent","primary":C["accent"],"background":"#fff"}, children=[
        dcc.Tab(label="Peptidoforms (hPF)", value="tab-hpf", style=ts, selected_style=tss),
        dcc.Tab(label="Single PTMs (hPTM)", value="tab-hptm", style=ts, selected_style=tss),
        dcc.Tab(label="QC Dashboard", value="tab-qc", style=ts, selected_style=tss),
        dcc.Tab(label="PCA & Clustering", value="tab-pca", style=ts, selected_style=tss),
        dcc.Tab(label="Statistics", value="tab-stats", style=ts, selected_style=tss),
        dcc.Tab(label="UpSet / Co-occurrence", value="tab-upset", style=ts, selected_style=tss),
        dcc.Tab(label="Comparisons", value="tab-cmp", style=ts, selected_style=tss),
        dcc.Tab(label="Sample Browser", value="tab-browse", style=ts, selected_style=tss),
    ]),
    html.Div(id="tab-out", style={"padding":"24px 32px","maxWidth":"1600px","margin":"0 auto"}),
    # Footer
    html.Div(style={"textAlign":"center","padding":"20px","color":C["muted"],"fontSize":"11px",
                     "borderTop":"1px solid #e5e7eb","marginTop":"40px"}, children=[
        html.Span("EpiProfile-Plants v3.1 | "),
        html.A("GitHub",href="https://github.com/biopelayo/epiprofile-dashboard",
               style={"color":C["accent"],"textDecoration":"none"},target="_blank"),
    ]),
    dcc.Store(id="cur-exp", data=DEFAULT_EXP),
])


# ======================================================================
# CALLBACKS - ROUTING
# ======================================================================

@callback(Output("cur-exp","data"), Input("exp-sel","value"))
def _se(e): return e

@callback(Output("desc-bar","children"), Input("cur-exp","data"))
def _db(e): return EXP_DATA[e].get("description","") if e and e in EXP_DATA else ""

@callback(Output("tab-out","children"), Input("tabs","value"), Input("cur-exp","data"))
def _rt(tab, exp):
    if not exp or exp not in EXP_DATA:
        return html.Div("No experiment loaded", style={"color":C["red"],"textAlign":"center","padding":"80px"})
    d = EXP_DATA[exp]
    try:
        return {"tab-hpf":tab_hpf,"tab-hptm":tab_hptm,"tab-qc":tab_qc,
                "tab-pca":tab_pca,"tab-stats":tab_stats,"tab-upset":tab_upset,
                "tab-cmp":tab_cmp,"tab-browse":tab_browse}.get(tab, lambda x: html.Div("?"))(d)
    except Exception as e:
        import traceback
        return html.Div([html.H3("Error",style={"color":C["red"]}),
                         html.Pre(traceback.format_exc(),style={"fontSize":"11px","color":C["text2"],"whiteSpace":"pre-wrap"})])

# ======================================================================
# HELPERS
# ======================================================================

def pfig(fig, h=500):
    fig.update_layout(template=PUB, height=h); return fig

def phm(z, x, y, cs="Viridis", title="", zmin=None, zmax=None, h=600):
    fig = go.Figure(go.Heatmap(z=z,x=x,y=y,colorscale=cs,
        colorbar=dict(thickness=12,len=0.9,title=dict(text=title,side="right",font=dict(size=10)),tickfont=dict(size=9)),
        hoverongaps=False, zmin=zmin, zmax=zmax))
    fig.update_layout(template=PUB,height=h,xaxis=dict(tickangle=45,tickfont=dict(size=8)),
                      yaxis=dict(tickfont=dict(size=9),autorange="reversed"),margin=dict(l=180,b=120,t=30,r=30))
    return fig

def cluster_order(df, axis=0):
    try:
        data = df.fillna(0).values if axis==0 else df.fillna(0).values.T
        if data.shape[0] < 3: return list(df.index) if axis==0 else list(df.columns)
        return [list(df.index if axis==0 else df.columns)[i] for i in leaves_list(linkage(pdist(data),method="ward"))]
    except: return list(df.index) if axis==0 else list(df.columns)

def _sc(label, val, color):
    return html.Div(style={**CS,"flex":"1","minWidth":"130px","textAlign":"center","padding":"16px 12px"}, children=[
        html.H2(str(val),style={"color":color,"margin":"0","fontSize":"28px","fontWeight":"700"}),
        html.P(label,style={"color":C["muted"],"margin":"4px 0 0","fontSize":"11px","textTransform":"uppercase",
                             "letterSpacing":"0.5px","fontWeight":"500"})])

def _st(text, sub=""):
    ch = [html.H3(text,style={"color":C["text"],"marginTop":"0","marginBottom":"4px","fontSize":"15px","fontWeight":"600"})]
    if sub: ch.append(html.P(sub,style={"color":C["muted"],"margin":"0 0 12px","fontSize":"12px"}))
    return html.Div(ch)

def _lbl(text):
    return html.Label(text, style={"fontSize":"11px","color":C["muted"],"fontWeight":"500",
                                    "textTransform":"uppercase","letterSpacing":"0.5px"})

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

    # Heatmap
    hm = phm(df.values, df.columns.tolist(), df.index.tolist(),
             cs="Viridis", title="Ratio", h=max(400, len(df)*7))

    # Top variable
    var_s = df.var(axis=1).dropna().sort_values(ascending=False).head(20)
    vf = go.Figure(go.Bar(x=var_s.values, y=var_s.index.tolist(), orientation="h",
                           marker=dict(color=var_s.values, colorscale="Viridis",line=dict(width=0))))
    pfig(vf, 400); vf.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=9)),
                                      margin=dict(l=200),xaxis_title="Variance")

    # Box for top PTM
    top = var_s.index[0] if len(var_s) > 0 else df.index[0]
    melt = df.loc[[top]].T.reset_index(); melt.columns = ["Sample","Ratio"]
    melt = melt.merge(meta, on="Sample")
    bf = px.box(melt, x="Group", y="Ratio", color="Group", points="all",
                title=top, color_discrete_sequence=GC)
    pfig(bf, 350)

    return html.Div([
        html.Div(style=CS, children=[
            _st(f"Peptidoform Heatmap", f"Showing {n_shown} of {n_total} hPF | {df.shape[1]} samples"),
            dcc.Graph(figure=hm),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"2","minWidth":"400px"}, children=[
                _st("Top Variable Peptidoforms"), dcc.Graph(figure=vf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"300px"}, children=[
                _st(f"Distribution: {top}"), dcc.Graph(figure=bf)]),
        ]),
        html.Div(style=CS, children=[
            _st("Filtered Data Table","Editable | Sortable | Filterable | Export CSV"),
            make_table(df, "hpf-table"),
        ]),
    ])


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

    # Clustered heatmap
    ro = cluster_order(df, 0); df_c = df.loc[ro]
    hm = phm(df_c.values, df_c.columns.tolist(), df_c.index.tolist(),
             cs="RdBu_r", title="Ratio", h=max(500, len(df)*11))

    # Z-score
    zs = df_c.apply(lambda r: (r-r.mean())/(r.std()+1e-10), axis=1)
    zhm = phm(zs.values, zs.columns.tolist(), zs.index.tolist(),
              cs="RdBu_r", title="Z-score", zmin=-3, zmax=3, h=max(500, len(df)*11))

    # Violin for key marks
    km = [m for m in ["H3K9me2","H3K14ac","H3K27me3","H3K4me1","H4K16ac","H3K9ac",
                       "H3K36me1","H3K27me1","H4K20me1","H3K4me3"] if m in df.index][:6]
    if km:
        ml = []
        for m in km:
            v = df.loc[m]; t = pd.DataFrame({"Sample":v.index,"Ratio":v.values,"PTM":m})
            t = t.merge(meta, on="Sample"); ml.append(t)
        vf = px.violin(pd.concat(ml), x="PTM", y="Ratio", color="Group", box=True, points="all",
                        color_discrete_sequence=GC)
        pfig(vf, 420); vf.update_layout(xaxis_title="",yaxis_title="Ratio")
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
    pfig(bf, 420); bf.update_layout(xaxis=dict(tickangle=45,tickfont=dict(size=9)),yaxis_title="Mean Ratio")

    return html.Div([
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"450px"}, children=[
                _st("Clustered Heatmap","Ward linkage | hPTM ratios"), dcc.Graph(figure=hm)]),
            html.Div(style={**CS,"flex":"1","minWidth":"450px"}, children=[
                _st("Z-score Heatmap","Row-wise normalization"), dcc.Graph(figure=zhm)]),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[
                _st("Key hPTM Distributions","Violin + box by group"), dcc.Graph(figure=vf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[
                _st("Top 15 hPTMs by Group Mean +/- SD"), dcc.Graph(figure=bf)]),
        ]),
        html.Div(style=CS, children=[
            _st("hPTM Data","Editable | Sortable | Filterable | Export CSV"),
            make_table(df, "hptm-table")]),
    ])


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
    pfig(mb, 350); mb.update_layout(xaxis=dict(tickangle=45,tickfont=dict(size=8)))

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
        pfig(ab, 380); ab.update_layout(xaxis=dict(tickangle=45,tickfont=dict(size=7)),showlegend=False)

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
                       title=dict(text="PCA - Sample Space",font=dict(size=14)))

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
                       title=dict(text="Scree Plot",font=dict(size=14)))

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
    scale = max(abs(coords[:,:2]).max()) / max(abs(loadings).max()+1e-10) * 0.8
    for _, row in top_load.iterrows():
        bfig.add_annotation(ax=0,ay=0,x=row["PC1"]*scale,y=row["PC2"]*scale,
                            xref="x",yref="y",axref="x",ayref="y",
                            showarrow=True,arrowhead=2,arrowsize=1.5,arrowwidth=1.5,
                            arrowcolor=C["red"])
        bfig.add_annotation(x=row["PC1"]*scale*1.1,y=row["PC2"]*scale*1.1,
                            text=row["Feature"],showarrow=False,font=dict(size=8,color=C["red"]))
    pfig(bfig, 500)
    bfig.update_layout(xaxis_title=f"PC1 ({ev[0]:.1f}%)", yaxis_title=f"PC2 ({ev[1]:.1f}%)",
                       title=dict(text="PCA Biplot - Samples + Top Loadings",font=dict(size=14)))

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
            title=dict(text="3D PCA",font=dict(size=14)))

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
            xaxis=dict(tickmode="array",tickvals=tp_,ticktext=dr["ivl"],tickangle=45,tickfont=dict(size=7)),
            yaxis_title="Distance (Ward)",title=dict(text="Hierarchical Clustering",font=dict(size=14)),margin=dict(b=120))
    except:
        dfig = go.Figure(); pfig(dfig, 350)

    # Correlation heatmap
    corr = df.corr(method="spearman")
    co_ = cluster_order(corr, 0); corr = corr.loc[co_,co_]
    chm = phm(corr.values, corr.columns.tolist(), corr.index.tolist(),
              cs="RdBu_r", title="Spearman", zmin=-1, zmax=1, h=max(500,len(corr)*12))

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

    return html.Div(children)


# ======================================================================
# TAB: STATISTICS
# ======================================================================

def tab_stats(d):
    df = d.get("hptm", d.get("hpf"))
    meta = d.get("metadata", pd.DataFrame())
    if df is None or meta.empty:
        return html.Div(style=CS, children=[html.P("No data.")])

    groups = sorted(meta["Group"].unique())
    if len(groups) < 2:
        return html.Div(style=CS, children=[html.P("Need >= 2 groups for statistics.")])

    # Kruskal-Wallis
    res = robust_group_test(df, meta, groups)
    if res.empty:
        return html.Div(style=CS, children=[html.P("Could not compute statistics.")])

    n_sig = int((res["KW_FDR"] < 0.05).sum())
    n_tested = len(res)

    # Volcano-like: -log10(FDR) vs max fold change across groups
    # Compute max absolute FC for each PTM
    fc_data = []
    for _, row in res.iterrows():
        means = [row.get(f"mean_{g}", np.nan) for g in groups]
        means = [m for m in means if not np.isnan(m) and m > 0]
        if len(means) >= 2:
            max_fc = np.log2(max(means)/min(means))
            fc_data.append({"PTM":row["PTM"], "maxLog2FC":max_fc,
                            "negLog10FDR":-np.log10(row["KW_FDR"]+1e-300),
                            "FDR":row["KW_FDR"]})

    if fc_data:
        vdf = pd.DataFrame(fc_data)
        vdf["sig"] = vdf["FDR"] < 0.05
        vfig = px.scatter(vdf, x="maxLog2FC", y="negLog10FDR", hover_name="PTM",
                          color="sig", color_discrete_map={True:C["red"],False:C["muted"]},
                          labels={"maxLog2FC":"Max |log2(FC)|","negLog10FDR":"-log10(FDR)"},
                          title="Volcano Plot (Kruskal-Wallis)")
        pfig(vfig, 450)
        vfig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color=C["red"],
                       annotation_text="FDR=0.05")
        vfig.update_traces(marker=dict(size=8,line=dict(width=0.5,color="white")))
    else:
        vfig = go.Figure(); pfig(vfig, 450)

    # Significant PTMs bar
    sig_res = res[res["KW_FDR"] < 0.05].head(30)
    if not sig_res.empty:
        sbf = go.Figure(go.Bar(
            x=-np.log10(sig_res["KW_FDR"].values+1e-300),
            y=sig_res["PTM"].tolist(), orientation="h",
            marker=dict(color=-np.log10(sig_res["KW_FDR"].values+1e-300),colorscale="Reds",line=dict(width=0))))
        pfig(sbf, max(300,len(sig_res)*18))
        sbf.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=9)),
                          margin=dict(l=180),xaxis_title="-log10(FDR)")
    else:
        sbf = go.Figure(); pfig(sbf, 300)

    return html.Div([
        html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px","flexWrap":"wrap"}, children=[
            _sc("Tested",str(n_tested),C["accent"]),
            _sc("Significant (FDR<0.05)",str(n_sig),C["red"] if n_sig>0 else C["green"]),
            _sc("Groups",str(len(groups)),C["h4"]),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[
                _st("Volcano Plot","Kruskal-Wallis FDR vs max fold change"), dcc.Graph(figure=vfig)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[
                _st(f"Top Significant Features (FDR<0.05)","n={n_sig}"), dcc.Graph(figure=sbf)]),
        ]),
        html.Div(style=CS, children=[
            _st("Full Statistical Results","Kruskal-Wallis + BH FDR correction | Editable"),
            make_table(res, "stats-table")]),
    ])


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
    uf.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=10)),
                     margin=dict(l=220),xaxis_title="Co-occurrence Count",
                     title=dict(text="PTM Co-occurrence (from combinatorial hPF)",font=dict(size=14)))

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
        pf.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=9)),
                         margin=dict(l=250),xaxis_title="# PTMs",
                         title=dict(text="PTM Detection Patterns Across Groups",font=dict(size=14)))
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
    pfig(cf, 380); cf.update_layout(xaxis=dict(tickangle=45,tickfont=dict(size=7)),yaxis_title="# Detected hPF")

    return html.Div([
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=uf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=pf)]),
        ]),
        html.Div(style=CS, children=[dcc.Graph(figure=cf)]),
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

    return html.Div([
        html.Div(style={**CS,"display":"flex","gap":"16px","flexWrap":"wrap","alignItems":"flex-end"}, children=[
            html.Div(style={"flex":"1","minWidth":"200px"}, children=[
                _lbl("Group A"),
                dcc.Dropdown(id="cmp-a",options=[{"label":g,"value":g} for g in groups],
                             value=groups[0],clearable=False,style=DS)]),
            html.Div(style={"flex":"1","minWidth":"200px"}, children=[
                _lbl("Group B"),
                dcc.Dropdown(id="cmp-b",options=[{"label":g,"value":g} for g in groups],
                             value=groups[1],clearable=False,style=DS)]),
            html.Div(style={"flex":"1","minWidth":"200px"}, children=[
                _lbl("Data Level"),
                dcc.Dropdown(id="cmp-level",options=[{"label":"hPTM (single)","value":"hptm"},
                             {"label":"hPF (peptidoforms)","value":"hpf"}],
                             value="hptm",clearable=False,style=DS)]),
        ]),
        html.Div(id="cmp-out"),
    ])

@callback(Output("cmp-out","children"),
          Input("cmp-a","value"),Input("cmp-b","value"),Input("cmp-level","value"),Input("cur-exp","data"))
def _cmp(ga, gb, level, exp):
    if not exp or exp not in EXP_DATA: return html.P("N/A")
    d = EXP_DATA[exp]
    df = d.get(level, d.get("hptm", d.get("hpf")))
    meta = d["metadata"]
    if df is None or df.empty: return html.P("No data for level.")

    mw = pairwise_mw(df, meta, ga, gb)
    if mw.empty: return html.P("Could not compute comparison.")

    n_sig = int((mw["FDR"]<0.05).sum())

    # Volcano
    mw["negLog10FDR"] = -np.log10(mw["FDR"]+1e-300)
    mw["sig"] = mw["FDR"] < 0.05
    vf = px.scatter(mw, x="log2FC", y="negLog10FDR", hover_name="PTM",
                    color="sig", color_discrete_map={True:C["red"],False:C["muted"]},
                    labels={"log2FC":f"log2(FC) {gb}/{ga}","negLog10FDR":"-log10(FDR)"},
                    title=f"Volcano: {gb} vs {ga} | Mann-Whitney U + FDR")
    pfig(vf, 480)
    vf.add_hline(y=-np.log10(0.05),line_dash="dash",line_color=C["red"],annotation_text="FDR=0.05")
    vf.add_vline(x=0,line_dash="dash",line_color=C["muted"])
    vf.update_traces(marker=dict(size=8,line=dict(width=0.5,color="white")))

    # FC bar
    fc_sorted = mw.sort_values("log2FC")
    colors = [C["green"] if v>0.5 else C["red"] if v<-0.5 else C["muted"] for v in fc_sorted["log2FC"]]
    ff = go.Figure(go.Bar(x=fc_sorted["log2FC"].values, y=fc_sorted["PTM"].tolist(),
                           orientation="h", marker_color=colors))
    pfig(ff, max(400, len(fc_sorted)*14))
    ff.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=9)),
                     margin=dict(l=160),xaxis_title=f"log2(FC) {gb}/{ga}")
    ff.add_vline(x=0,line_color=C["muted"],line_dash="dash")

    # MA plot
    mw["A"] = 0.5*(np.log2(mw["mean_A"]+1e-8)+np.log2(mw["mean_B"]+1e-8))
    maf = px.scatter(mw, x="A", y="log2FC", hover_name="PTM", color="sig",
                     color_discrete_map={True:C["red"],False:C["muted"]},
                     title="MA Plot",labels={"A":"Avg Intensity","log2FC":"log2(FC)"})
    pfig(maf, 400); maf.add_hline(y=0,line_color=C["muted"],line_dash="dash")

    return html.Div([
        html.Div(style={"display":"flex","gap":"12px","marginBottom":"16px"}, children=[
            _sc(f"Sig (FDR<0.05)",str(n_sig),C["red"]),
            _sc("Tested",str(len(mw)),C["accent"]),
        ]),
        html.Div(style={"display":"flex","gap":"16px","flexWrap":"wrap"}, children=[
            html.Div(style={**CS,"flex":"1","minWidth":"500px"}, children=[dcc.Graph(figure=vf)]),
            html.Div(style={**CS,"flex":"1","minWidth":"400px"}, children=[dcc.Graph(figure=ff)]),
        ]),
        html.Div(style=CS, children=[dcc.Graph(figure=maf)]),
        html.Div(style=CS, children=[
            _st("Mann-Whitney Results","FDR-corrected | Editable"),
            make_table(mw[["PTM","log2FC","pval","FDR","mean_A","mean_B","median_A","median_B"]],"cmp-table")]),
    ])


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
    fig.update_layout(yaxis=dict(autorange="reversed",tickfont=dict(size=9)),margin=dict(l=120,t=10),xaxis_title="Ratio")
    return html.Div([_st("PTM Profile",f"Sample: {f}"), dcc.Graph(figure=fig)])


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EpiProfile-Plants Dashboard v3.1")
    print(f"  Experiments: {len(EXP_DATA)}")
    for n in EXP_DATA: print(f"    * {n}")
    print(f"\n  =>  http://localhost:{args.port}")
    print("="*60 + "\n")
    app.run(debug=False, port=args.port, host=args.host)
