"""Extra statistics for the EpiProfile-Plants dashboard.

Pure numpy/pandas/scipy, no Dash. Everything takes a features x samples
DataFrame and a metadata DataFrame with at least ``Sample`` and ``Group``
columns (``Batch`` for batch correction).

Contents
--------
impute_missing      : none | min_half | row_median | knn
batch_correct       : none | median_center | combat (parametric EB, Johnson 2007)
pairwise_test       : mannwhitney | welch | moderated_t (limma-style squeezeVar)
cliffs_delta, hedges_g, epsilon_squared : effect sizes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import digamma, polygamma
from statsmodels.stats.multitest import multipletests

IMPUTE_METHODS = {
    "none": "No imputation (drop missing per test)",
    "min_half": "Half of the feature minimum (left-censored, MNAR)",
    "row_median": "Feature median across samples (MAR)",
    "knn": "k-nearest-neighbour samples (k=5)",
}
BATCH_METHODS = {
    "none": "No batch correction",
    "median_center": "Per-batch median centering",
    "combat": "ComBat (parametric empirical Bayes)",
}
TEST_METHODS = {
    "mannwhitney": "Mann-Whitney U (non-parametric)",
    "welch": "Welch t-test",
    "moderated_t": "Moderated t (limma-style empirical Bayes)",
}


# ----------------------------------------------------------------------
# Missing values
# ----------------------------------------------------------------------
def _as_missing(df: pd.DataFrame, is_log: bool) -> pd.DataFrame:
    """Return a float copy where non-detects are NaN (zeros for ratio data)."""
    x = df.astype(float).copy()
    if not is_log:
        x = x.mask(x == 0)
    x = x.mask(~np.isfinite(x))
    return x


def impute_missing(df: pd.DataFrame, method: str = "none", is_log: bool = False, k: int = 5) -> pd.DataFrame:
    """Impute non-detects. Rows that are entirely missing are left as NaN."""
    x = _as_missing(df, is_log)
    if method in (None, "none") or x.isna().sum().sum() == 0:
        return x
    if method == "min_half":
        row_min = x.min(axis=1)
        fill = row_min / 2.0 if not is_log else row_min - 1.0  # log2 scale: -1 == half
        return x.apply(lambda r: r.fillna(fill[r.name]), axis=1)
    if method == "row_median":
        return x.apply(lambda r: r.fillna(r.median()), axis=1)
    if method == "knn":
        try:
            from sklearn.impute import KNNImputer
        except Exception:  # pragma: no cover
            return x.apply(lambda r: r.fillna(r.median()), axis=1)
        full_na = x.isna().all(axis=1)
        sub = x[~full_na]
        if sub.empty:
            return x
        # impute samples (columns) using feature profiles -> transpose
        imp = KNNImputer(n_neighbors=min(k, max(1, sub.shape[1] - 1)))
        vals = imp.fit_transform(sub.T.values).T
        out = x.copy()
        out.loc[sub.index, :] = vals
        return out
    raise ValueError(f"unknown imputation method: {method}")


# ----------------------------------------------------------------------
# Batch correction
# ----------------------------------------------------------------------
def _batches(meta: pd.DataFrame, columns, batch_col="Batch"):
    if batch_col not in meta.columns:
        return None
    m = meta.set_index("Sample")[batch_col]
    b = pd.Series([m.get(c, np.nan) for c in columns], index=columns).astype(str)
    b = b.replace({"nan": np.nan, "None": np.nan}).dropna()
    if b.nunique() < 2:
        return None
    return b


def batch_correct(df: pd.DataFrame, meta: pd.DataFrame, method: str = "none",
                  is_log: bool = False, batch_col: str = "Batch", group_col: str = "Group") -> pd.DataFrame:
    """Remove batch effects. Data are transformed to log2 for ratio inputs,
    corrected, and transformed back so callers see the same scale."""
    if method in (None, "none"):
        return df
    b = _batches(meta, df.columns, batch_col)
    if b is None:
        return df
    x = _as_missing(df, is_log)
    work = x if is_log else np.log2(x)
    cols = list(b.index)
    w = work[cols]
    if method == "median_center":
        out = w.copy()
        grand = w.median(axis=1)
        for bt in b.unique():
            bc = b[b == bt].index
            shift = w[bc].median(axis=1) - grand
            out[bc] = w[bc].sub(shift, axis=0)
    elif method == "combat":
        groups = None
        if group_col in meta.columns:
            g = meta.set_index("Sample")[group_col]
            groups = pd.Series([g.get(c, "NA") for c in cols], index=cols).astype(str)
        out = _combat(w, b, groups)
    else:
        raise ValueError(f"unknown batch method: {method}")
    res = work.copy()
    res[cols] = out
    if not is_log:
        res = np.power(2.0, res)
    return res


def _combat(x: pd.DataFrame, batch: pd.Series, groups: pd.Series | None) -> pd.DataFrame:
    """Parametric empirical-Bayes ComBat (Johnson, Li & Rabinovic 2007).
    x: features x samples (log scale, NaN allowed; NaN cells are skipped in the fits).
    Rows with fewer than 2 non-missing values in any batch are returned uncorrected."""
    samples = list(x.columns)
    batches = list(pd.unique(batch))
    n_b = len(batches)
    B = np.zeros((len(samples), n_b))
    for j, bt in enumerate(batches):
        B[[samples.index(s) for s in batch[batch == bt].index], j] = 1.0
    design = [B]
    if groups is not None and groups.nunique() > 1:
        G = pd.get_dummies(groups).astype(float)
        G = G.iloc[:, 1:]  # drop one level to avoid collinearity with batch intercepts
        design.append(G.values)
    X = np.hstack(design)
    Y = x.values.astype(float)
    out = Y.copy()
    n_feat = Y.shape[0]

    # 1. standardise each feature with a least-squares fit (per row, NaN-aware)
    gamma_hat = np.full((n_feat, n_b), np.nan)
    delta_hat = np.full((n_feat, n_b), np.nan)
    stand = np.full_like(Y, np.nan)
    grand_mean = np.full(n_feat, np.nan)
    var_pooled = np.full(n_feat, np.nan)
    ok_rows = np.zeros(n_feat, dtype=bool)
    for i in range(n_feat):
        y = Y[i]
        m = np.isfinite(y)
        if m.sum() < X.shape[1] + 1:
            continue
        # each batch must have >= 2 observations
        if any((m & (B[:, j] == 1)).sum() < 2 for j in range(n_b)):
            continue
        beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        n_per_batch = B[m].sum(axis=0)
        gm = (n_per_batch / n_per_batch.sum()) @ beta[:n_b]
        fitted_nb = X[:, n_b:] @ beta[n_b:] if X.shape[1] > n_b else 0.0
        resid = y - (X @ beta)
        vp = np.nanvar(resid[m], ddof=0)
        if not np.isfinite(vp) or vp <= 0:
            continue
        s = (y - gm - fitted_nb) / np.sqrt(vp)
        stand[i] = s
        grand_mean[i] = gm
        var_pooled[i] = vp
        for j in range(n_b):
            mb = m & (B[:, j] == 1)
            gamma_hat[i, j] = np.nanmean(s[mb])
            delta_hat[i, j] = np.nanvar(s[mb], ddof=1) if mb.sum() > 1 else 1.0
        ok_rows[i] = True
    if ok_rows.sum() == 0:
        return x

    # 2. empirical Bayes priors per batch
    def _aprior(d):
        m, s2 = np.nanmean(d), np.nanvar(d)
        return (2 * s2 + m ** 2) / s2 if s2 > 0 else 1.0

    def _bprior(d):
        m, s2 = np.nanmean(d), np.nanvar(d)
        return (m * s2 + m ** 3) / s2 if s2 > 0 else 1.0

    gamma_star = np.full_like(gamma_hat, np.nan)
    delta_star = np.full_like(delta_hat, np.nan)
    for j in range(n_b):
        g = gamma_hat[ok_rows, j]
        d = delta_hat[ok_rows, j]
        g_bar, t2 = np.nanmean(g), np.nanvar(g)
        a_p, b_p = _aprior(d), _bprior(d)
        mb = B[:, j] == 1
        n_j = np.array([(np.isfinite(stand[i]) & mb).sum() for i in np.where(ok_rows)[0]], dtype=float)
        s_b = stand[ok_rows][:, mb]
        g_new, d_new = g.copy(), d.copy()
        for _ in range(100):  # iterative solution as in sva::ComBat
            g_prev, d_prev = g_new.copy(), d_new.copy()
            g_new = (t2 * n_j * g + d_new * g_bar) / (t2 * n_j + d_new) if t2 > 0 else g
            sum2 = np.nansum((s_b - g_new[:, None]) ** 2, axis=1)
            d_new = (0.5 * sum2 + b_p) / (n_j / 2.0 + a_p - 1.0)
            d_new = np.where(np.isfinite(d_new) & (d_new > 0), d_new, d)
            if np.nanmax(np.abs(g_new - g_prev)) < 1e-4 and np.nanmax(np.abs(d_new - d_prev)) < 1e-4:
                break
        gamma_star[ok_rows, j] = g_new
        delta_star[ok_rows, j] = d_new

    # 3. adjust
    for i in np.where(ok_rows)[0]:
        s = stand[i].copy()
        for j in range(n_b):
            mb = B[:, j] == 1
            s[mb] = (s[mb] - gamma_star[i, j]) / np.sqrt(delta_star[i, j])
        fitted_nb = X[:, n_b:] @ np.zeros(X.shape[1] - n_b) if X.shape[1] > n_b else 0.0
        # add back grand mean and (optionally) the biological covariate fit
        yfit = Y[i]
        m = np.isfinite(yfit)
        beta, *_ = np.linalg.lstsq(X[m], yfit[m], rcond=None)
        cov = X[:, n_b:] @ beta[n_b:] if X.shape[1] > n_b else 0.0
        out[i] = s * np.sqrt(var_pooled[i]) + grand_mean[i] + cov
    return pd.DataFrame(out, index=x.index, columns=x.columns)


# ----------------------------------------------------------------------
# Effect sizes
# ----------------------------------------------------------------------
def cliffs_delta(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(a) * len(b)))


def hedges_g(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    d = (b.mean() - a.mean()) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)
    return float(d * J)


def epsilon_squared(h_stat: float, n: int, k: int) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis H."""
    if not np.isfinite(h_stat) or n <= k:
        return np.nan
    return float((h_stat - k + 1) / (n - k))


# ----------------------------------------------------------------------
# Moderated t (limma squeezeVar)
# ----------------------------------------------------------------------
def _fit_f_dist(var: np.ndarray, df1: float):
    """Estimate prior (df0, s0^2) for residual variances ~ s0^2 * F(df1, df0).
    Port of limma::fitFDist (Smyth 2004) without covariates."""
    var = var[np.isfinite(var) & (var > 0)]
    if len(var) < 2:
        return np.inf, float(np.nanmedian(var)) if len(var) else np.nan
    z = np.log(var)
    e = z - digamma(df1 / 2.0) + np.log(df1 / 2.0)
    emean = e.mean()
    evar = np.sum((e - emean) ** 2) / (len(e) - 1) - polygamma(1, df1 / 2.0)
    if evar > 0:
        # solve trigamma(df0/2) = evar for df0 (Newton on log scale)
        y = evar
        x = 0.5 + 1.0 / y
        for _ in range(50):
            tri = polygamma(1, x)
            dif = tri * (1 - tri / y) / polygamma(2, x)
            x = x + dif
            if x <= 0:
                x = 1e-8
            if abs(dif) < 1e-8:
                break
        df0 = 2.0 * x
        s0 = np.exp(emean + digamma(df0 / 2.0) - np.log(df0 / 2.0))
    else:
        df0 = np.inf
        s0 = np.exp(emean)
    return float(df0), float(s0)


def moderated_t(df: pd.DataFrame, cols_a, cols_b):
    """Two-group moderated t-test with limma-style variance shrinkage.
    Returns DataFrame with t, pval, log2FC(mean_B - mean_A), df_total, s2_post."""
    A = df[cols_a].astype(float).values
    Bv = df[cols_b].astype(float).values
    na = np.isfinite(A).sum(axis=1)
    nb = np.isfinite(Bv).sum(axis=1)
    ma = np.nanmean(np.where(np.isfinite(A), A, np.nan), axis=1)
    mb = np.nanmean(np.where(np.isfinite(Bv), Bv, np.nan), axis=1)
    ssa = np.nansum((A - ma[:, None]) ** 2, axis=1)
    ssb = np.nansum((Bv - mb[:, None]) ** 2, axis=1)
    dfr = na + nb - 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = (ssa + ssb) / dfr
    ok = (na >= 2) & (nb >= 2) & np.isfinite(s2)
    df0, s0 = _fit_f_dist(s2[ok], float(np.median(dfr[ok])) if ok.any() else 1.0)
    if np.isinf(df0):
        s2_post = np.where(ok, s2, np.nan)
        df_tot = dfr
    else:
        s2_post = (df0 * s0 + dfr * s2) / (df0 + dfr)
        df_tot = df0 + dfr
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(s2_post * (1.0 / na + 1.0 / nb))
        t = (mb - ma) / se
    p = 2 * sps.t.sf(np.abs(t), df_tot)
    res = pd.DataFrame({"t": t, "pval": p, "log2FC": mb - ma, "mean_A": ma, "mean_B": mb,
                        "n_A": na, "n_B": nb, "df": df_tot, "s2_post": s2_post}, index=df.index)
    res.attrs["df0"] = df0; res.attrs["s0"] = s0
    return res[ok]


# ----------------------------------------------------------------------
# Unified pairwise test
# ----------------------------------------------------------------------
def pairwise_test(df: pd.DataFrame, meta: pd.DataFrame, g1: str, g2: str, method: str = "mannwhitney",
                  is_log: bool = False, impute: str = "none", batch: str = "none") -> pd.DataFrame:
    """Two-group comparison for every feature. Output columns are a superset
    of the dashboard's legacy ``pairwise_mw`` (PTM, pval, FDR, log2FC, mean_A,
    mean_B, median_A, median_B) plus ``stat``, ``cliffs_delta``, ``hedges_g``
    and ``method``."""
    sa = meta[meta["Group"] == g1]["Sample"].tolist()
    sb = meta[meta["Group"] == g2]["Sample"].tolist()
    ca = [c for c in df.columns if c in sa]
    cb = [c for c in df.columns if c in sb]
    if not ca or not cb:
        return pd.DataFrame()
    x = batch_correct(df, meta, batch, is_log=is_log)
    x = impute_missing(x, impute, is_log=is_log)
    # for log-scale data the test runs on the values; for ratios we keep the linear
    # scale for MW (rank-based) and use log2 for parametric tests
    if method == "mannwhitney":
        rows = []
        for ptm in x.index:
            va = x.loc[ptm, ca].dropna().values.astype(float)
            vb = x.loc[ptm, cb].dropna().values.astype(float)
            if len(va) < 2 or len(vb) < 2:
                continue
            try:
                stat, p = sps.mannwhitneyu(va, vb, alternative="two-sided")
            except Exception:
                continue
            fc = (vb.mean() - va.mean()) if is_log else np.log2((vb.mean() + 1e-8) / (va.mean() + 1e-8))
            rows.append({"PTM": ptm, "stat": stat, "U_stat": stat, "pval": p, "log2FC": fc,
                         "mean_A": va.mean(), "mean_B": vb.mean(),
                         "median_A": np.median(va), "median_B": np.median(vb),
                         "n_A": len(va), "n_B": len(vb),
                         "cliffs_delta": cliffs_delta(va, vb), "hedges_g": hedges_g(va, vb)})
        res = pd.DataFrame(rows)
    else:
        xl = x if is_log else np.log2(x)
        if method == "welch":
            rows = []
            for ptm in xl.index:
                va = xl.loc[ptm, ca].dropna().values.astype(float)
                vb = xl.loc[ptm, cb].dropna().values.astype(float)
                if len(va) < 2 or len(vb) < 2:
                    continue
                t, p = sps.ttest_ind(va, vb, equal_var=False)
                rows.append({"PTM": ptm, "stat": t, "pval": p, "log2FC": vb.mean() - va.mean(),
                             "n_A": len(va), "n_B": len(vb),
                             "cliffs_delta": cliffs_delta(va, vb), "hedges_g": hedges_g(va, vb)})
            res = pd.DataFrame(rows)
        elif method == "moderated_t":
            mt = moderated_t(xl, ca, cb)
            if mt.empty:
                return pd.DataFrame()
            res = mt.rename(columns={"t": "stat"}).reset_index().rename(columns={"index": "PTM"})
            res["cliffs_delta"] = [cliffs_delta(xl.loc[p, ca].dropna(), xl.loc[p, cb].dropna()) for p in res["PTM"]]
            res["hedges_g"] = [hedges_g(xl.loc[p, ca].dropna(), xl.loc[p, cb].dropna()) for p in res["PTM"]]
            res.attrs.update(mt.attrs)
        else:
            raise ValueError(f"unknown test method: {method}")
        if res.empty:
            return res
        # report means/medians on the original scale for readability
        lin = x
        res["mean_A"] = [lin.loc[p, ca].mean() for p in res["PTM"]]
        res["mean_B"] = [lin.loc[p, cb].mean() for p in res["PTM"]]
        res["median_A"] = [lin.loc[p, ca].median() for p in res["PTM"]]
        res["median_B"] = [lin.loc[p, cb].median() for p in res["PTM"]]
    if res.empty:
        return res
    if len(res) > 1:
        res["FDR"] = multipletests(res["pval"].values, method="fdr_bh")[1]
    else:
        res["FDR"] = res["pval"]
    res["method"] = method
    return res.sort_values("pval").reset_index(drop=True)
