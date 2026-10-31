"""Unit tests for the helper modules (ep_stats, ep_io, ep_tools, ep_report)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ep_io  # noqa: E402
import ep_report  # noqa: E402
import ep_stats  # noqa: E402
import ep_tools  # noqa: E402

DEMO = os.path.join(ROOT, "examples", "demo_experiment")


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    samples = [f"S{i}" for i in range(12)]
    meta = pd.DataFrame({"Sample": samples, "Group": ["A"] * 6 + ["B"] * 6, "Batch": ["1", "2"] * 6})
    x = pd.DataFrame(rng.lognormal(-2, 0.5, (40, 12)), index=[f"f{i}" for i in range(40)], columns=samples)
    x.iloc[:5, 6:] *= 3                                   # true effect in f0..f4
    x.loc[:, (meta.Batch == "2").values] *= 1.8           # batch effect
    x = x.mask(rng.random((40, 12)) < 0.08, 0.0)          # non-detects as zeros
    return x, meta


# ---------------------------------------------------------------- ep_stats
@pytest.mark.parametrize("method", ["none", "min_half", "row_median", "knn"])
def test_impute(toy, method):
    x, _ = toy
    y = ep_stats.impute_missing(x, method)
    assert y.shape == x.shape
    if method != "none":
        assert y.isna().sum().sum() == 0
    else:
        assert y.isna().sum().sum() == int((x == 0).sum().sum())


@pytest.mark.parametrize("method", ["median_center", "combat"])
def test_batch_correct_removes_batch_shift(toy, method):
    x, meta = toy
    y = ep_stats.batch_correct(x, meta, method)
    lb = np.log2(ep_stats._as_missing(y, False))
    b2 = lb.loc[:, (meta.Batch == "2").values].mean().mean()
    b1 = lb.loc[:, (meta.Batch == "1").values].mean().mean()
    assert abs(b2 - b1) < 0.15   # uncorrected difference is ~0.85
    # no batch column -> unchanged
    assert ep_stats.batch_correct(x, meta.drop(columns="Batch"), method).equals(x)


@pytest.mark.parametrize("method", ["mannwhitney", "welch", "moderated_t"])
def test_pairwise_test_finds_true_effects(toy, method):
    x, meta = toy
    r = ep_stats.pairwise_test(x, meta, "A", "B", method, batch="combat", impute="min_half")
    assert {"PTM", "pval", "FDR", "log2FC", "cliffs_delta", "hedges_g", "mean_A", "mean_B"} <= set(r.columns)
    top = set(r.head(5)["PTM"])
    assert len(top & {"f0", "f1", "f2", "f3", "f4"}) >= 4
    assert (r.loc[r["PTM"].isin(["f0", "f1", "f2", "f3", "f4"]), "log2FC"] > 0.8).all()  # true effect is log2(3)=1.58


def test_effect_sizes():
    assert ep_stats.cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    assert ep_stats.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert ep_stats.hedges_g([1, 2, 3, 4], [3, 4, 5, 6]) > 1.0
    assert 0 < ep_stats.epsilon_squared(12.0, 30, 3) < 1
    assert np.isnan(ep_stats.epsilon_squared(np.nan, 30, 3))


def test_moderated_t_shrinks_variance():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, 1, (200, 8)), columns=[f"s{i}" for i in range(8)])
    res = ep_stats.moderated_t(df, df.columns[:4], df.columns[4:])
    assert np.isfinite(res.attrs["df0"]) and res.attrs["s0"] > 0
    assert len(res) == 200 and (res["df"] > 6).all()   # prior df added


# ---------------------------------------------------------------- ep_io
def test_region_detail_parsing(tmp_path):
    fp = tmp_path / "H3_07_73_83.xls"
    fp.write_text("[ratio]\nmod\tz1\tz2\nK79me1\t0.3\t0.1\nK79me2\t0.2\t0\n[area]\nmod\tz1\tz2\nK79me1\t1e6\t2e5\n[rt]\nmod\tz1\tz2\nK79me1\t35.2\t35.4\n")
    df = ep_io.parse_region_detail(str(fp))
    assert set(df["block"]) == {"ratio", "area", "rt"}
    assert df[(df.block == "area") & (df.modification == "K79me1")]["value"].sum() == 1.2e6
    assert ep_io._region_from_filename("H3_07_73_83.xls") == "H3_73_83"
    assert ep_io._region_from_filename("H33_27_40.xls") == "H33_27_40"


def test_region_details_from_demo():
    folders = sorted(os.listdir(os.path.join(DEMO, "histone_layouts")))
    det = ep_io.load_region_details(os.path.join(DEMO, "histone_layouts"), folders)
    assert set(det["region"]) == {"H3_9_17", "H3_73_83"}
    assert det["sample_folder"].nunique() == 12
    summ = ep_io.summarize_region_details(det)
    assert {"area", "ratio", "rt"} <= set(summ.columns)


def test_psm_summary_is_column_agnostic():
    psm = pd.DataFrame({"Peptide": ["AAA", "BBB", "AAA"], "Charge": [2, 3, 2], "XCorr": [2.1, 3.0, 1.5],
                        "_sample_folder": ["1_a", "1_a", "2_b"]})
    s = ep_io.summarize_psms(psm)
    assert s["columns"] == {"sequence": "Peptide", "charge": "Charge", "score": "XCorr"}
    assert s["per_sample"]["n_psm"].tolist() == [2, 1]
    assert s["per_peptide"].iloc[0]["sequence"] == "AAA"
    assert ep_io.summarize_psms(pd.DataFrame())["n"] == 0


def test_dinosaur_loader():
    feat = ep_io.load_dinosaur_features(DEMO)
    assert feat["sample"].nunique() == 2 and "intensityApex" in feat.columns
    s = ep_io.summarize_dinosaur(feat)
    assert set(s["sample"]) == {"Control_Sh1", "Control_Sh2"} and (s["n_features"] > 0).all()
    assert ep_io.load_dinosaur_features("/nonexistent").empty


def test_ms_file_discovery_and_readers():
    files = ep_io.find_ms_files(DEMO)
    assert any(f.endswith(".ms2") for f in files) and any(f.endswith(".mzML") for f in files)
    ms2 = next(f for f in files if f.endswith(".ms2"))
    sp = ep_io.read_spectrum_any(ms2, scan=2001)
    assert sp["charge"] == 2 and abs(sp["precursor_mz"] - 478.7765) < 1e-4 and len(sp["mz"]) > 10
    scans = ep_io.list_scans(ms2)
    assert [s["scan"] for s in scans] == [2001, 2002, 2003]
    if ep_io.have_pyteomics():
        mz = next(f for f in files if f.endswith(".mzML"))
        sp2 = ep_io.read_spectrum_any(mz, scan=2)
        assert sp2["ms_level"] == 2 and sp2["charge"] == 2
        assert [s["ms_level"] for s in ep_io.list_scans(mz)] == [1, 2]


def test_annotator_matches_ladder():
    ms2 = os.path.join(DEMO, "ms2", "Control_Sh2.ms2")
    sp = ep_io.read_spectrum_any(ms2, scan=2001)
    res = ep_io.annotate_peaks(sp, "KSTGGK[Acetyl]APR", tol=20, unit="ppm")
    ions = set(res["matches"]["ion"])
    assert {"b1+", "b2+", "b3+", "b4+", "b5+", "y1+", "y2+", "y3+", "y4+"} <= ions
    assert (res["matches"]["error_ppm"].abs() < 5).all()
    assert res["coverage"]["b"] > 0.5 and res["sequence"] == "KSTGGKAPR"
    # scan without the ladder -> essentially nothing
    sp2 = ep_io.read_spectrum_any(ms2, scan=2002)
    assert len(ep_io.annotate_peaks(sp2, "KSTGGK[Acetyl]APR", tol=5, unit="ppm")["matches"]) <= 2
    mgf = ep_io.spectrum_to_mgf(sp, "t", "KSTGGK[Acetyl]APR")
    assert mgf.startswith("BEGIN IONS") and "PEPMASS=478.776500" in mgf and "CHARGE=2+" in mgf and "SEQ=" in mgf


@pytest.mark.skipif(not ep_io.have_rustyms(), reason="rustyms not installed")
def test_rustyms_cross_check():
    sp = ep_io.read_spectrum_any(os.path.join(DEMO, "ms2", "Control_Sh2.ms2"), scan=2001)
    rx = ep_io.rustyms_annotate(sp, "KSTGGK[Acetyl]APR")
    assert rx.get("n_matched", 0) >= 8


def test_ptm_site_table():
    t = ep_io.ptm_site_table(["H3K27me3", "H3.3K36me2", "H4K16ac", "H3K9me2K14ac", "H3K79me1"])
    assert len(t) == 4   # combinatorial name skipped
    row = t[t.feature == "H3K27me3"].iloc[0]
    assert row.uniprot_id == "P59226" and row.position == 28 and row.modification_unimod_id == 37
    assert t[t.feature == "H3.3K36me2"].iloc[0].uniprot_id == "P59169"


# ---------------------------------------------------------------- ep_tools / ep_report
def test_dinosaur_command_and_java():
    if not ep_tools.dinosaur_jar():
        pytest.skip("Dinosaur jar not installed")
    cmd = ep_tools.dinosaur_command("/x/run.mzML", "/out", concurrency=3)
    assert cmd[1:3] == ["-jar", ep_tools.dinosaur_jar()] and "--concurrency=3" in cmd and "--outDir=/out" in cmd and cmd[-1] == "/x/run.mzML"


def test_annotator_binary_lookup(monkeypatch, tmp_path):
    exe = tmp_path / "annotator"
    exe.write_text("#!/bin/sh\nexit 0\n"); exe.chmod(0o755)
    monkeypatch.setenv("ANNOTATOR_BIN", str(exe))
    assert ep_tools.annotator_binary() == str(exe)
    assert ep_tools.launch_annotator()["ok"] is True
    monkeypatch.setenv("ANNOTATOR_BIN", "/nonexistent")
    monkeypatch.setattr(ep_tools, "TOOLS_DIR", str(tmp_path / "none"))
    monkeypatch.setattr(ep_tools.shutil, "which", lambda n: None)
    assert ep_tools.annotator_binary() is None and ep_tools.launch_annotator()["ok"] is False


def test_report_builder():
    import plotly.graph_objects as go
    html = ep_report.build_report([("Table", "desc", pd.DataFrame({"a": [1, 2]})), ("Fig", "", go.Figure()), ("Text", "", "<b>x</b>")],
                                  title="T", subtitle="S")
    assert html.startswith("<!doctype html>") and "plotly" in html and "<table" in html and "<b>x</b>" in html
