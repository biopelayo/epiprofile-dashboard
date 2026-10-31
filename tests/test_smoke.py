"""Smoke tests: load the bundled demo dataset and render every tab.

The dashboard module parses argv and loads experiments at import time, so the
module is imported once here with argv pointing at the demo directory.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "examples", "demo_experiment")
SCRIPT = os.path.join(ROOT, "epiprofile_dashboard.py")

TABS = ["tab-hpf", "tab-hptm", "tab-qc", "tab-pca", "tab-stats", "tab-upset",
        "tab-region", "tab-k79", "tab-cmp", "tab-ndebug", "tab-pheno",
        "tab-browse", "tab-paper", "tab-export", "tab-log"]


@pytest.fixture(scope="session")
def dash_mod(tmp_path_factory):
    # Keep the SQLite log out of the repository during tests.
    db_dir = tmp_path_factory.mktemp("db")
    old_argv = sys.argv
    sys.argv = ["epiprofile_dashboard.py", f"Demo ndebug_0={DEMO}", f"Demo ndebug_2={DEMO}"]
    try:
        spec = importlib.util.spec_from_file_location("epiprofile_dashboard", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        mod_src = open(SCRIPT, encoding="utf-8").read()
        # Redirect DB_PATH before init_db runs by executing with a patched source.
        mod_src = mod_src.replace(
            'DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epiprofile.db")',
            f'DB_PATH = {str(db_dir / "epiprofile.db")!r}')
        mod.__file__ = SCRIPT
        exec(compile(mod_src, SCRIPT, "exec"), mod.__dict__)
    finally:
        sys.argv = old_argv
    return mod


def test_demo_loaded(dash_mod):
    assert set(dash_mod.EXP_DATA) == {"Demo ndebug_0", "Demo ndebug_2"}
    d = dash_mod.EXP_DATA["Demo ndebug_0"]
    assert d["hpf"].shape == (39, 12)
    assert d["hptm"].shape == (26, 12)
    assert d["areas_norm"].shape == (39, 12)
    assert d["rt"].shape == (39, 12)
    assert len(d["hdp_list"]) == 7
    meta = d["metadata"]
    assert sorted(meta["Group"].unique()) == ["Control", "NaB", "TSA"]
    assert set(meta["Tissue"]) == {"Shoot", "Root"}


def test_ndebug_inferred_from_name(dash_mod):
    assert dash_mod.EXP_DATA["Demo ndebug_0"]["ndebug"] == "ndebug_0"
    assert dash_mod.EXP_DATA["Demo ndebug_2"]["dataset"] == "Demo"
    pairs = dash_mod._find_ndebug_pairs()
    assert "Demo" in pairs and len(pairs["Demo"]) == 2


@pytest.mark.parametrize("name,path,ratios,expected", [
    ("PXD014739 ndebug_2", "D:/x/PXD014739/RawData", "histone_ratios_ndeg2.xls", ("ndebug_2", "PXD014739")),
    ("Ontogeny ndebug_0", "/e/RawData", None, ("ndebug_0", "Ontogeny")),
    ("RawData", "/data/exp_ndebug2/RawData", None, ("ndebug_2", "RawData")),
    ("mydata", "/x/mydata", None, ("default", "mydata")),
])
def test_infer_exp_meta(dash_mod, name, path, ratios, expected):
    assert dash_mod.infer_exp_meta(name, path, ratios) == expected


def test_config_file_relative_paths(dash_mod, tmp_path):
    cfg = tmp_path / "experiments.json"
    cfg.write_text('{"Rel": {"dir": "sub/exp"}, "Str": "other", "Bad": {"ratios": "x"}}')
    out = dash_mod.load_config_file(str(cfg))
    assert out["Rel"]["dir"] == os.path.normpath(str(tmp_path / "sub" / "exp"))
    assert out["Str"]["dir"] == os.path.normpath(str(tmp_path / "other"))
    assert "Bad" not in out
    assert dash_mod.load_config_file(None) == {}


@pytest.mark.parametrize("tab", TABS)
def test_every_tab_renders(dash_mod, tab):
    out = dash_mod._rt(tab, "Demo ndebug_0", "npg")
    rendered = str(out)
    assert "Traceback" not in rendered, rendered[:2000]


def test_ndebug_compare_runs(dash_mod):
    out = dash_mod._ndebug_compare("Demo ndebug_0", "Demo ndebug_2", "hpf", "npg")
    assert "Traceback" not in str(out)


def test_statistics_tab_callback(dash_mod):
    out = dash_mod._stats_filtered("ratios", "All", "All", "all", "all", False, "Demo ndebug_0")
    assert "Traceback" not in str(out)


def test_read_table_auto_formats(dash_mod, tmp_path):
    import pandas as pd
    df = pd.DataFrame({"PTM": ["H3K4me1", "H3K9ac"], "1,S1": [0.1, 0.2], "2,S2": [0.3, 0.4]})
    (tmp_path / "a.tsv").write_text(df.to_csv(sep="\t", index=False))
    (tmp_path / "b.csv").write_text(df.to_csv(index=False))
    (tmp_path / "c.xls").write_text(df.to_csv(sep="\t", index=False))  # EpiProfile-style TSV named .xls
    df.to_excel(tmp_path / "d.xlsx", index=False)
    for fn in ["a.tsv", "b.csv", "c.xls", "d.xlsx"]:
        out = dash_mod.read_table_auto(str(tmp_path / fn))
        assert list(out.columns) == ["PTM", "1,S1", "2,S2"], fn
        assert out.shape == (2, 3), fn
    # bytes payload as produced by dcc.Upload
    raw = (tmp_path / "b.csv").read_bytes()
    assert dash_mod.read_table_auto(raw, "b.csv").shape == (2, 3)
    assert dash_mod.read_table_auto((tmp_path / "d.xlsx").read_bytes(), "d.xlsx").shape == (2, 3)


def test_normalize_phenodata_aliases(dash_mod):
    import pandas as pd
    df = pd.DataFrame({"Sample": ["A"], "Condition": ["WT"], "Tissue": ["Leaf"]})
    out = dash_mod.normalize_phenodata(df)
    assert list(out.columns) == ["Sample_Name", "Sample_Group", "Tissue"]
    # already-correct names are left untouched
    df2 = pd.DataFrame({"Sample_Name": ["A"], "Sample_Group": ["WT"], "Group": ["x"]})
    assert list(dash_mod.normalize_phenodata(df2).columns) == ["Sample_Name", "Sample_Group", "Group"]


def test_xlsx_export(dash_mod):
    import base64, io
    import pandas as pd
    data, status = dash_mod._exp_download(1, "hptm", "xlsx", "All", [], "all", "none", ["meta"], "Demo ndebug_0")
    assert data and data.get("filename", "").endswith(".xlsx")
    content = base64.b64decode(data["content"])
    book = pd.read_excel(io.BytesIO(content), sheet_name=None)
    assert "data" in book and "metadata" in book
    assert book["data"].shape[0] == 26
