"""Extra readers and exporters for the EpiProfile-Plants dashboard.

* per-sample region detail files  (histone_layouts/<sample>/detail/<REGION>.xls)
* PSM identification lists        (histone_layouts/<sample>/detail/psm/identification_list.xls)
* Dinosaur MS1 feature tables     (*.features.tsv)
* mzML spectra via pyteomics (optional dependency)
* PTMVision plain-CSV export of site-level modifications
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Region detail files: [ratio] / [area] / [rt] blocks, rows = peptidoforms,
# columns = charge states (or whatever the sub-header says)
# ----------------------------------------------------------------------
_BLOCK_RE = re.compile(r"^\[(\w+)\]$")


def parse_region_detail(path: str) -> pd.DataFrame:
    """Parse one region detail file into a long DataFrame with columns
    block, modification, column, value. Unknown layouts yield an empty frame."""
    try:
        with open(path, "r", errors="replace") as fh:
            lines = [ln.rstrip("\r\n") for ln in fh]
    except Exception:
        return pd.DataFrame(columns=["block", "modification", "column", "value"])
    rows = []
    block, header = None, None
    for ln in lines:
        cells = ln.split("\t")
        first = cells[0].strip()
        m = _BLOCK_RE.match(first.lower())
        if m:
            block, header = m.group(1), None
            continue
        if block is None or first == "":
            continue
        vals = [v.strip() for v in cells[1:]]
        if header is None:
            nums = pd.to_numeric(pd.Series(vals), errors="coerce")
            # a header row is one whose values are not all numeric and whose first
            # cell is not a modification name (heuristic: contains 'z' or 'charge')
            if nums.notna().sum() == 0 or first.lower() in ("mod", "modification", "ptm", "name"):
                header = [v or f"c{i+1}" for i, v in enumerate(vals)]
                continue
            header = [f"z{i+1}" for i in range(len(vals))]
        for col, v in zip(header, vals):
            num = pd.to_numeric(v, errors="coerce")
            if pd.notna(num):
                rows.append({"block": block, "modification": first, "column": col, "value": float(num)})
    return pd.DataFrame(rows, columns=["block", "modification", "column", "value"])


def load_region_details(layouts_dir: str, folders, max_files: int = 5000) -> pd.DataFrame:
    """Scan every sample folder's detail/ directory for region files
    (anything that is not psm/ and ends with .xls/.tsv/.txt). Returns a long
    DataFrame: sample_folder, region_file, region, block, modification, column, value."""
    rows = []
    n = 0
    for f in folders:
        ddir = os.path.join(layouts_dir, f, "detail")
        if not os.path.isdir(ddir):
            continue
        for fn in sorted(os.listdir(ddir)):
            fp = os.path.join(ddir, fn)
            if not os.path.isfile(fp) or not fn.lower().endswith((".xls", ".tsv", ".txt")):
                continue
            n += 1
            if n > max_files:
                break
            df = parse_region_detail(fp)
            if df.empty:
                continue
            df.insert(0, "region", _region_from_filename(fn))
            df.insert(0, "region_file", fn)
            df.insert(0, "sample_folder", f)
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["sample_folder", "region_file", "region", "block", "modification", "column", "value"])
    return pd.concat(rows, ignore_index=True)


def _region_from_filename(fn: str) -> str:
    """H3_07_73_83.xls -> H3_73_83 ; H4_4_17.xls -> H4_4_17 ; H33_27_40.xls -> H33_27_40"""
    stem = os.path.splitext(fn)[0]
    m = re.match(r"^(H\d+[A-Za-z]?)_(?:\d+_)?(\d+)_(\d+)$", stem)
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}" if m else stem


def summarize_region_details(det: pd.DataFrame) -> pd.DataFrame:
    """Per sample/region/modification: total area (sum over charge states),
    mean RT, ratio total. Wide-ish table used by the Region Map tab."""
    if det.empty:
        return det
    g = det.groupby(["sample_folder", "region", "modification", "block"])["value"]
    agg = g.sum().unstack("block")
    if "rt" in agg.columns:
        agg["rt"] = det[det["block"] == "rt"].groupby(["sample_folder", "region", "modification"])["value"].mean()
    return agg.reset_index()


# ----------------------------------------------------------------------
# PSM identification lists
# ----------------------------------------------------------------------
_PSM_COLS = {
    "sequence": ["sequence", "peptide", "pep_seq", "modified_sequence", "seq"],
    "charge": ["charge", "z", "precursor_charge"],
    "score": ["score", "xcorr", "hyperscore", "ionscore", "evalue", "e-value", "pep_score"],
    "scan": ["scan", "scan_number", "scannum", "spectrum", "first_scan"],
    "rt": ["rt", "retention_time", "rt_min", "retention time"],
    "mz": ["mz", "m/z", "precursor_mz", "observed_mz"],
    "mods": ["modification", "modifications", "mods", "ptm", "variable_modifications"],
}


def _find_col(columns, candidates):
    low = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for cand in candidates:
        for k, c in low.items():
            if cand in k:
                return c
    return None


def summarize_psms(psm: pd.DataFrame) -> dict:
    """Column-agnostic PSM summary. Returns dict with per_sample (DataFrame),
    charge (Series), score (Series), per_peptide (DataFrame) and the detected
    column mapping. Works with whatever columns identification_list.xls has."""
    out = {"n": 0, "columns": {}, "per_sample": pd.DataFrame(), "charge": pd.Series(dtype=int),
           "score": pd.Series(dtype=float), "per_peptide": pd.DataFrame()}
    if psm is None or psm.empty:
        return out
    cols = {k: _find_col(psm.columns, v) for k, v in _PSM_COLS.items()}
    out["columns"] = {k: v for k, v in cols.items() if v is not None}
    out["n"] = len(psm)
    sf = "_sample_folder" if "_sample_folder" in psm.columns else None
    if sf:
        per = psm.groupby(sf).size().rename("n_psm").to_frame()
        if cols["sequence"]:
            per["n_peptides"] = psm.groupby(sf)[cols["sequence"]].nunique()
        if cols["charge"]:
            per["mean_charge"] = pd.to_numeric(psm[cols["charge"]], errors="coerce").groupby(psm[sf]).mean()
        out["per_sample"] = per.reset_index().rename(columns={sf: "sample_folder"})
    if cols["charge"]:
        out["charge"] = pd.to_numeric(psm[cols["charge"]], errors="coerce").dropna().astype(int).value_counts().sort_index()
    if cols["score"]:
        out["score"] = pd.to_numeric(psm[cols["score"]], errors="coerce").dropna()
    if cols["sequence"]:
        pp = psm.groupby(cols["sequence"]).size().rename("n_psm").to_frame()
        if sf:
            pp["n_samples"] = psm.groupby(cols["sequence"])[sf].nunique()
        if cols["charge"]:
            pp["charges"] = psm.groupby(cols["sequence"])[cols["charge"]].apply(
                lambda s: ",".join(sorted(set(str(int(v)) for v in pd.to_numeric(s, errors="coerce").dropna()))))
        out["per_peptide"] = pp.sort_values("n_psm", ascending=False).reset_index().rename(columns={cols["sequence"]: "sequence"})
    return out


# ----------------------------------------------------------------------
# Dinosaur MS1 features
# ----------------------------------------------------------------------
DINOSAUR_COLS = ["mz", "mostAbundantMz", "charge", "rtStart", "rtApex", "rtEnd", "fwhm",
                 "nIsotopes", "nScans", "averagineCorr", "mass", "massCalib", "intensityApex", "intensitySum"]


def find_dinosaur_features(base_dir: str) -> list[str]:
    pats = [os.path.join(base_dir, "**", "*.features.tsv"), os.path.join(base_dir, "**", "*features.tsv")]
    found = set()
    for p in pats:
        found.update(glob.glob(p, recursive=True))
    return sorted(found)


def load_dinosaur_features(base_dir: str, max_rows_per_file: int | None = None) -> pd.DataFrame:
    """Concatenate all Dinosaur feature tables under base_dir, adding a
    ``sample`` column derived from the filename (<sample>.features.tsv)."""
    frames = []
    for fp in find_dinosaur_features(base_dir):
        try:
            df = pd.read_csv(fp, sep="\t", nrows=max_rows_per_file)
        except Exception:
            continue
        if "mz" not in df.columns or "charge" not in df.columns:
            continue
        name = os.path.basename(fp)
        name = re.sub(r"\.features\.tsv$|features\.tsv$", "", name).rstrip("._-")
        df.insert(0, "sample", name)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["sample"] + DINOSAUR_COLS)
    return pd.concat(frames, ignore_index=True)


def summarize_dinosaur(feat: pd.DataFrame) -> pd.DataFrame:
    if feat.empty:
        return feat
    g = feat.groupby("sample")
    s = pd.DataFrame({
        "n_features": g.size(),
        "median_intensityApex": g["intensityApex"].median() if "intensityApex" in feat else np.nan,
        "median_fwhm_min": g["fwhm"].median() if "fwhm" in feat else np.nan,
        "frac_z2plus": g["charge"].apply(lambda s: float((s >= 2).mean())),
        "rt_span_min": (g["rtApex"].max() - g["rtApex"].min()) if "rtApex" in feat else np.nan,
    })
    return s.reset_index()


# ----------------------------------------------------------------------
# mzML (optional: pyteomics)
# ----------------------------------------------------------------------
def find_mzml(base_dir: str) -> list[str]:
    return sorted(set(glob.glob(os.path.join(base_dir, "**", "*.mzML"), recursive=True)
                      + glob.glob(os.path.join(base_dir, "**", "*.mzml"), recursive=True)))


def have_pyteomics() -> bool:
    try:
        import pyteomics.mzml  # noqa: F401
        import psims  # noqa: F401  (pyteomics >= 5 needs it for PSI formats)
        return True
    except Exception:
        return False


def read_spectrum(mzml_path: str, scan: int | None = None, index: int | None = None) -> dict | None:
    """Return {'mz': ndarray, 'intensity': ndarray, 'ms_level': int, 'rt': float,
    'precursor_mz': float|None, 'charge': int|None, 'id': str} for one spectrum.
    Select by native scan number or by 0-based index."""
    if not have_pyteomics():
        return None
    from pyteomics import mzml as _mzml
    with _mzml.MzML(mzml_path) as reader:
        spec = None
        if index is not None:
            spec = reader[int(index)]
        elif scan is not None:
            # try the two most common id layouts, then a linear scan
            for sid in (f"controllerType=0 controllerNumber=1 scan={scan}", f"scan={scan}", f"index={scan}"):
                try:
                    spec = reader.get_by_id(sid)
                    break
                except Exception:
                    continue
            if spec is None:
                reader.reset()
                for s in reader:
                    if re.search(rf"scan={scan}\b", s.get("id", "")) or s.get("index") == scan:
                        spec = s; break
        if spec is None:
            return None
        out = {"id": spec.get("id", ""), "ms_level": int(spec.get("ms level", 1)),
               "mz": np.asarray(spec.get("m/z array", [])), "intensity": np.asarray(spec.get("intensity array", [])),
               "rt": None, "precursor_mz": None, "charge": None}
        try:
            out["rt"] = float(spec["scanList"]["scan"][0]["scan start time"])
        except Exception:
            pass
        try:
            ion = spec["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]
            out["precursor_mz"] = float(ion.get("selected ion m/z"))
            out["charge"] = int(ion.get("charge state")) if ion.get("charge state") else None
        except Exception:
            pass
        return out


# ----------------------------------------------------------------------
# .ms1 / .ms2 (RawConverter / EpiProfile input format)
#   H<tab>...           header lines
#   S<tab>first<tab>last<tab>precursor m/z      scan start
#   I<tab>RTime<tab>12.34                        info lines
#   Z<tab>charge<tab>M+H                         charge lines (ms2)
#   <m/z> <intensity>                            peaks
# ----------------------------------------------------------------------
def find_ms_files(base_dir: str) -> list[str]:
    """mzML, .ms2 and .ms1 files under base_dir (recursive)."""
    out = set(find_mzml(base_dir))
    for ext in ("ms2", "MS2", "ms1", "MS1"):
        out.update(glob.glob(os.path.join(base_dir, "**", f"*.{ext}"), recursive=True))
    return sorted(out)


def iter_ms2_scans(path: str):
    """Yield dicts (same keys as read_spectrum) for every scan in an .ms1/.ms2 file."""
    cur = None
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            if not ln or ln[0] == "H":
                continue
            tag = ln[0]
            if tag == "S":
                if cur is not None:
                    cur["mz"] = np.asarray(cur["mz"]); cur["intensity"] = np.asarray(cur["intensity"]); yield cur
                parts = ln.rstrip("\n").split("\t")
                scan = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                cur = {"id": f"scan={scan}", "scan": scan, "ms_level": 1 if path.lower().endswith(".ms1") else 2,
                       "mz": [], "intensity": [], "rt": None, "charge": None,
                       "precursor_mz": float(parts[3]) if len(parts) > 3 and parts[3].strip() else None}
            elif cur is None:
                continue
            elif tag == "I":
                parts = ln.rstrip("\n").split("\t")
                if len(parts) > 2 and parts[1].strip().lower() in ("rtime", "rt", "retentiontime"):
                    try: cur["rt"] = float(parts[2])
                    except ValueError: pass
            elif tag == "Z":
                parts = ln.rstrip("\n").split("\t")
                try: cur["charge"] = int(float(parts[1]))
                except (ValueError, IndexError): pass
            elif tag == "D":
                continue
            else:
                parts = ln.split()
                if len(parts) >= 2:
                    try:
                        cur["mz"].append(float(parts[0])); cur["intensity"].append(float(parts[1]))
                    except ValueError:
                        pass
    if cur is not None:
        cur["mz"] = np.asarray(cur["mz"]); cur["intensity"] = np.asarray(cur["intensity"]); yield cur


def read_ms2_scan(path: str, scan: int | None = None, index: int | None = None) -> dict | None:
    for i, sp in enumerate(iter_ms2_scans(path)):
        if (scan is not None and sp.get("scan") == scan) or (index is not None and i == index) or (scan is None and index is None):
            return sp
    return None


def read_spectrum_any(path: str, scan: int | None = None) -> dict | None:
    """Dispatch on extension: mzML (pyteomics) or ms1/ms2 (native)."""
    if path.lower().endswith((".ms1", ".ms2")):
        return read_ms2_scan(path, scan=scan)
    return read_spectrum(path, scan=scan)


def list_scans(path: str, limit: int = 2000) -> list[dict]:
    """Light index of scans: scan, ms_level, rt, precursor_mz, charge (first `limit`)."""
    out = []
    if path.lower().endswith((".ms1", ".ms2")):
        for i, sp in enumerate(iter_ms2_scans(path)):
            out.append({"scan": sp.get("scan"), "ms_level": sp["ms_level"], "rt": sp.get("rt"),
                        "precursor_mz": sp.get("precursor_mz"), "charge": sp.get("charge"), "n_peaks": len(sp["mz"])})
            if i + 1 >= limit: break
        return out
    if not have_pyteomics():
        return out
    from pyteomics import mzml as _mzml
    with _mzml.MzML(path) as reader:
        for i, spec in enumerate(reader):
            m = re.search(r"scan=(\d+)", spec.get("id", ""))
            rt = None
            try: rt = float(spec["scanList"]["scan"][0]["scan start time"])
            except Exception: pass
            pmz, z = None, None
            try:
                ion = spec["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]
                pmz = float(ion.get("selected ion m/z")); z = int(ion["charge state"]) if ion.get("charge state") else None
            except Exception: pass
            out.append({"scan": int(m.group(1)) if m else spec.get("index"), "ms_level": int(spec.get("ms level", 1)),
                        "rt": rt, "precursor_mz": pmz, "charge": z, "n_peaks": len(spec.get("m/z array", []))})
            if i + 1 >= limit: break
    return out


# ----------------------------------------------------------------------
# Fragment annotation ("annotator"): ProForma ladders via pyteomics,
# optional cross-check with rustyms (the engine behind Snijder lab's Annotator)
# ----------------------------------------------------------------------
ION_TYPES_DEFAULT = ("b", "y")
_PROTON = 1.007276466812


def have_rustyms() -> bool:
    try:
        import rustyms  # noqa: F401
        return True
    except Exception:
        return False


def theoretical_fragments(proforma: str, max_charge: int = 2, ion_types=ION_TYPES_DEFAULT) -> pd.DataFrame:
    """Theoretical fragment m/z for a ProForma peptidoform (e.g. KSTGGK[Acetyl]APR,
    K[Trimethyl]STGGKAPR, EIAQDFK[Dimethyl]TDLR). Columns: ion, series, position, charge, mz."""
    from pyteomics import proforma as _pf
    pep = _pf.ProForma.parse(proforma)
    rows = []
    n = len(pep.sequence)
    for series in ion_types:
        for z in range(1, max_charge + 1):
            try:
                mzs = list(pep.fragments(series, z))
            except Exception:
                continue
            for i, mz in enumerate(mzs, start=1):
                pos = i
                rows.append({"ion": f"{series}{pos}{'+' * z}", "series": series, "position": pos, "charge": z, "mz": float(mz)})
    df = pd.DataFrame(rows)
    df.attrs["length"] = n
    df.attrs["sequence"] = "".join(str(a[0]) for a in pep.sequence)
    return df


def annotate_peaks(spec: dict, proforma: str, tol: float = 20.0, unit: str = "ppm",
                   max_charge: int | None = None, ion_types=ION_TYPES_DEFAULT, min_rel_int: float = 0.0) -> dict:
    """Match experimental peaks to theoretical fragments.
    Returns dict(matches=DataFrame[mz,intensity,ion,theo_mz,error_ppm,error_da],
                 coverage={'b': frac, 'y': frac}, explained_intensity=frac, sequence=str, n_theoretical=int)."""
    mz = np.asarray(spec["mz"], float); it = np.asarray(spec["intensity"], float)
    if max_charge is None:
        max_charge = max(1, min(3, (spec.get("charge") or 2)))
    theo = theoretical_fragments(proforma, max_charge=max_charge, ion_types=ion_types)
    if theo.empty or len(mz) == 0:
        return {"matches": pd.DataFrame(), "coverage": {}, "explained_intensity": 0.0,
                "sequence": theo.attrs.get("sequence", ""), "n_theoretical": len(theo)}
    keep = it >= (min_rel_int * it.max() if len(it) else 0)
    mzk, itk = mz[keep], it[keep]
    order = np.argsort(mzk); mzk, itk = mzk[order], itk[order]
    rows = []
    for _, f in theo.iterrows():
        t = f["mz"]
        tol_da = t * tol * 1e-6 if unit == "ppm" else tol
        lo, hi = np.searchsorted(mzk, t - tol_da), np.searchsorted(mzk, t + tol_da, side="right")
        if hi > lo:
            j = lo + int(np.argmax(itk[lo:hi]))  # most intense peak in window
            rows.append({"mz": mzk[j], "intensity": itk[j], "ion": f["ion"], "series": f["series"], "position": int(f["position"]),
                         "charge": int(f["charge"]), "theo_mz": t, "error_da": mzk[j] - t, "error_ppm": (mzk[j] - t) / t * 1e6})
    m = pd.DataFrame(rows)
    n = theo.attrs.get("length", 0) or 1
    cov = {}
    for ser in ion_types:
        cov[ser] = float(m[m["series"] == ser]["position"].nunique() / max(1, n - 1)) if not m.empty else 0.0
    expl = float(m["intensity"].sum() / it.sum()) if not m.empty and it.sum() > 0 else 0.0
    return {"matches": m.sort_values("mz") if not m.empty else m, "coverage": cov, "explained_intensity": expl,
            "sequence": theo.attrs.get("sequence", ""), "n_theoretical": len(theo)}


def rustyms_annotate(spec: dict, proforma: str, model: str = "CidHcd") -> dict:
    """Cross-check with rustyms (0.9 API). Returns dict(n_matched, n_peaks, formulas=[...]) or {} if unavailable."""
    if not have_rustyms():
        return {}
    try:
        import rustyms
        z = int(spec.get("charge") or 2)
        pm = (float(spec["precursor_mz"]) * z - z * _PROTON) if spec.get("precursor_mz") else None
        sp = rustyms.RawSpectrum("spec", 1, [float(x) for x in spec["mz"]], [float(x) for x in spec["intensity"]],
                                 rt=spec.get("rt") or 0.0, precursor_charge=z, precursor_mass=pm)
        pep_cls = getattr(rustyms, "CompoundPeptidoform", None) or getattr(rustyms, "CompoundPeptidoformIon")
        fm = getattr(rustyms.FragmentationModel, model, rustyms.FragmentationModel.CidHcd)
        try:
            ann = sp.annotate(pep_cls(proforma), fm, rustyms.MassMode.Monoisotopic)
        except TypeError:
            ann = sp.annotate(pep_cls(proforma), fm)
        matched = [(pk.experimental_mz, [str(f.formula) for f in pk.annotation]) for pk in ann.spectrum if pk.annotation]
        return {"n_matched": len(matched), "n_peaks": len(ann.spectrum),
                "formulas": [f"{m:.4f}: {', '.join(fs)}" for m, fs in matched[:50]]}
    except Exception as e:
        return {"error": str(e)}


def annotate_spectrum(spec: dict, proforma: str, tol_da: float = 0.02):
    """Backward-compatible helper: list of (mz, intensity, label)."""
    res = annotate_peaks(spec, proforma, tol=tol_da, unit="Da")
    m = res["matches"]
    return [] if m.empty else list(zip(m["mz"], m["intensity"], m["ion"]))


def spectrum_to_mgf(spec: dict, title: str = "spectrum", proforma: str | None = None) -> str:
    """MGF text for one spectrum (what Annotator / IPSA / USE open directly)."""
    lines = ["BEGIN IONS", f"TITLE={title}"]
    if spec.get("precursor_mz"): lines.append(f"PEPMASS={spec['precursor_mz']:.6f}")
    if spec.get("charge"): lines.append(f"CHARGE={int(spec['charge'])}+")
    if spec.get("rt") is not None: lines.append(f"RTINSECONDS={float(spec['rt']) * 60:.2f}")
    if spec.get("scan") is not None or spec.get("id"): lines.append(f"SCANS={spec.get('scan') or spec.get('id')}")
    if proforma: lines.append(f"SEQ={proforma}")
    for a, b in zip(spec["mz"], spec["intensity"]):
        lines.append(f"{a:.5f} {b:.2f}")
    lines.append("END IONS")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# PTMVision export (plain CSV)
# ----------------------------------------------------------------------
# UniProt accessions for Arabidopsis thaliana core histones used by EpiProfile-Plants.
HISTONE_UNIPROT = {"H3": "P59226", "H3.1": "P59226", "H3.3": "P59169", "H33": "P59169",
                   "H4": "P59259", "H2A": "O23628", "H2B": "Q9LQQ4"}
UNIMOD = {"me1": ("Methyl", 34), "me2": ("Dimethyl", 36), "me3": ("Trimethyl", 37),
          "ac": ("Acetyl", 1), "ph": ("Phospho", 21), "pr": ("Propionyl", 58),
          "bu": ("Butyryl", 1289), "cr": ("Crotonyl", 1363), "ub": ("GG", 121),
          "fo": ("Formyl", 122), "su": ("Succinyl", 64), "hib": ("2-hydroxyisobutyryl", 1849),
          "la": ("Lactylation", 2114)}
_SITE_RE = re.compile(r"^(H\d+(?:\.\d)?|H33)K(\d+)([a-z0-9]+)$")


def ptm_site_table(hptm_names, hptm_df: pd.DataFrame | None = None, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn hPTM names like H3K27me3 into PTMVision-style rows:
    uniprot_id, position, modification_unimod_name, modification_unimod_id,
    histone, residue, ptm, plus mean ratio per group when data are given."""
    rows = []
    for name in hptm_names:
        m = _SITE_RE.match(str(name).replace("H3.3", "H33"))
        if not m:
            continue
        hist, pos, mod = m.group(1), int(m.group(2)), m.group(3)
        hist_key = "H3.3" if hist == "H33" else hist
        uni_name, uni_id = UNIMOD.get(mod, (mod, ""))
        # Arabidopsis H3 proteins carry the initiator Met in UniProt: histone
        # nomenclature numbers from the Ala after Met, so UniProt position = K + 1.
        row = {"uniprot_id": HISTONE_UNIPROT.get(hist_key, ""), "position": pos + 1,
               "modification_unimod_name": uni_name, "modification_unimod_id": uni_id,
               "histone": hist_key, "residue": f"K{pos}", "ptm": mod, "feature": name}
        if hptm_df is not None and name in hptm_df.index and meta is not None and not meta.empty:
            for g, sub in meta.groupby("Group"):
                cols = [c for c in sub["Sample"] if c in hptm_df.columns]
                row[f"mean_{g}"] = float(hptm_df.loc[name, cols].mean()) if cols else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
