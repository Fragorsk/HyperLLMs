"""Shared inferential statistics for validation.py and run_error_analysis.py.

Notes of a single patient are correlated (treatment history copied between visits),
so the primary unit of analysis is the PATIENT, and the document is a sensitivity analysis.

  * Unit metrics are computed from count sums (ratio of sums): patient = sum of TP/FP/FN
    (or of error counts and denominators) over all of their documents.
  * The cluster bootstrap samples patients with replacement (with all their documents);
    the document bootstrap is the same algorithm with each document as a separate cluster.
    The statistic in each sample = the same formula as the point estimate (micro from sums).
  * Rank tests (Friedman, Wilcoxon, Holm) operate on the units × models matrix;
    the unit (patient / document) only determines the rows and N.

Edge rules:
  Unit F1: TP+FP+FN = 0 → 1; predictions without GT → 0 (from the formula).
  Unit error rate: denominator 0 → NaN; the unit drops out of the given test for
  all models (pairing preserved), the number of exclusions is reported.
  Corpus / bootstrap-sample micro-F1: 2TP+FP+FN = 0 → NaN (as in the P/R/F1 table).

The seed (42) and the number of resamples (10000) are shared by all CIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm, rankdata, spearmanr, wilcoxon

UNITS = ("patient", "document")
N_BOOT = 10000
SEED = 42
ALPHA = 0.05


# ─── Aggregation ───────────────────────────────────────────────────────────────

def unit_f1(tp, fp, fn, empty_doc_policy: str = "one"):
    """F1 = 2TP / (2TP + FP + FN) per unit (numpy arrays or scalars).
    TP+FP+FN = 0: 1.0 for "one", NaN for "exclude". Predictions without GT → 0 (from the formula)."""
    if empty_doc_policy not in ("one", "exclude"):
        raise ValueError(f"empty_doc_policy musi być 'one' lub 'exclude', jest {empty_doc_policy!r}")
    tp, fp, fn = (np.asarray(a, dtype=float) for a in (tp, fp, fn))
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / den, 1.0 if empty_doc_policy == "one" else np.nan)


def unit_rate(num, den):
    """Unit error rate = num / den; den = 0 → NaN."""
    num, den = (np.asarray(a, dtype=float) for a in (num, den))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def micro_f1(tp, fp, fn):
    """Micro-F1 from TP/FP/FN sums over the last axis; arrays (R, U) → mean over runs R.
    2TP+FP+FN = 0 → NaN."""
    tp, fp, fn = (np.asarray(a, dtype=float).sum(axis=-1) for a in (tp, fp, fn))
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(den > 0, 2 * tp / den, np.nan)
    return np.mean(f1, axis=-1) if f1.ndim else f1


def aggregate_to_patient(
    per_doc: pd.DataFrame,
    group_cols: Sequence[str] = ("model", "category"),
    patient_col: str = "patient",
    doc_col: str = "doc",
    count_cols: Optional[Sequence[str]] = None,
    rates: Optional[dict[str, tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Table (group…, patient) with SUMS of counts over the patient's documents.

    `per_doc` has one row per (group…, patient, document); `count_cols` (default: all
    numeric columns other than the keys) are summed. Added columns:
      n_docs — number of the patient's documents,
      f1     — 2ΣTP/(2ΣTP+ΣFP+ΣFN), when tp/fp/fn columns exist (TP+FP+FN = 0 → 1),
      {name} for each pair rates[name] = (numerator, denominator): Σnumerator/Σdenominator, 0 → NaN.
    Patient metrics are a ratio of sums, not a mean of document metrics."""
    keys = [*group_cols, patient_col]
    if count_cols is None:
        count_cols = [c for c in per_doc.columns
                      if c not in (*keys, doc_col) and pd.api.types.is_numeric_dtype(per_doc[c])
                      and not pd.api.types.is_bool_dtype(per_doc[c])]
    count_cols = list(count_cols)
    out = per_doc.groupby(keys, sort=True, dropna=False)[count_cols].sum()
    out["n_docs"] = per_doc.groupby(keys, sort=True, dropna=False)[doc_col].nunique()
    out = out.reset_index()

    # Σ over patients == Σ over documents
    for c in count_cols:
        if not np.isclose(out[c].sum(), per_doc[c].sum()):
            raise AssertionError(f"aggregate_to_patient: Σ{c} po pacjentach ≠ Σ po dokumentach")

    if {"tp", "fp", "fn"} <= set(out.columns):
        out["f1"] = unit_f1(out["tp"], out["fp"], out["fn"])
    for name, (num, den) in (rates or {}).items():
        out[name] = unit_rate(out[num], out[den])
    return out


def unit_table(
    per_doc: pd.DataFrame,
    unit: str,
    group_cols: Sequence[str] = ("model", "category"),
    patient_col: str = "patient",
    doc_col: str = "doc",
    count_cols: Optional[Sequence[str]] = None,
    rates: Optional[dict[str, tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Unit-level table with a `unit_id` column: patient (aggregate_to_patient)
    or document (the same formula computed from a single document's counts)."""
    _check_unit(unit)
    if unit == "patient":
        out = aggregate_to_patient(per_doc, group_cols, patient_col, doc_col, count_cols, rates)
        return out.assign(unit_id=out[patient_col])
    # document = a patient with a single document: same function, key (patient, document)
    tmp = per_doc.assign(_unit=per_doc[patient_col].astype(str) + "\x1f" + per_doc[doc_col].astype(str))
    out = aggregate_to_patient(tmp, group_cols, "_unit", doc_col, count_cols, rates)
    doc_of = tmp.drop_duplicates("_unit").set_index("_unit")
    out[patient_col] = out["_unit"].map(doc_of[patient_col])
    out[doc_col] = out["_unit"].map(doc_of[doc_col])
    return out.drop(columns="_unit").assign(unit_id=lambda d: d[doc_col])


def _check_unit(unit: str) -> None:
    if unit not in UNITS:
        raise ValueError(f"unit musi być jednym z {UNITS}, jest {unit!r}")


# ─── Bootstrap ────────────────────────────────────────────────────────────────

def resample_multiplicities(n_units: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """(n_boot, n_units): how many times each unit landed in the bootstrap sample
    (drawing n_units units with replacement; rng = default_rng(seed))."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_units, size=(n_boot, n_units))
    mult = np.zeros((n_boot, n_units))
    np.add.at(mult, (np.arange(n_boot)[:, None], idx), 1.0)
    return mult


def cluster_bootstrap(
    counts: np.ndarray,
    clusters: Sequence,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> np.ndarray:
    """Cluster bootstrap of count sums: samples CLUSTERS (patients) with replacement, each
    with all of its documents. `counts` has shape (..., n_docs), `clusters` — the
    cluster label of each document. Returns the sums in each sample: (n_boot, ...).
    All rows of `counts` (models, categories, TP/FP/FN) get the same samples,
    so the result is paired. The statistic (e.g. micro-F1) is then computed from these sums."""
    counts = np.asarray(counts, dtype=float)
    labels, inv = np.unique(np.asarray(clusters), return_inverse=True)
    agg = np.zeros(counts.shape[:-1] + (labels.size,))
    np.add.at(agg, (..., inv), counts)
    return cluster_bootstrap_sums(agg, n_boot, seed)


def cluster_bootstrap_sums(unit_counts: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Like cluster_bootstrap, but `unit_counts` (..., n_units) are already summed per cluster."""
    unit_counts = np.asarray(unit_counts, dtype=float)
    mult = resample_multiplicities(unit_counts.shape[-1], n_boot, seed)
    return np.moveaxis(np.tensordot(unit_counts, mult, axes=([-1], [1])), -1, 0)


def document_bootstrap(counts: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Bootstrap over documents (sensitivity analysis): each document is a separate cluster."""
    counts = np.asarray(counts, dtype=float)
    return cluster_bootstrap_sums(counts, n_boot, seed)


def bootstrap_sums(counts: np.ndarray, unit: str, clusters: Sequence,
                   n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Dispatcher: unit='patient' → cluster_bootstrap(counts, clusters), 'document' → document_bootstrap."""
    _check_unit(unit)
    if unit == "patient":
        return cluster_bootstrap(counts, clusters, n_boot, seed)
    return document_bootstrap(counts, n_boot, seed)


def f1_from_sums(tp, fp, fn):
    """F1 from sums (no summing over an axis); 2TP+FP+FN = 0 → NaN."""
    tp, fp, fn = (np.asarray(a, dtype=float) for a in (tp, fp, fn))
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / den, np.nan)


def percentile_ci(boot: np.ndarray, min_valid: float = 0.5) -> Optional[tuple[float, float]]:
    """Percentile 95% CI ignoring NaN; None when defined samples < min_valid·n_boot."""
    boot = np.asarray(boot, dtype=float)
    valid = boot[~np.isnan(boot)]
    if valid.size == 0 or valid.size < boot.size * min_valid:
        return None
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_p_two_sided(boot_delta: np.ndarray, delta: float) -> float:
    """Two-sided p of a difference: bootstrap distribution shifted to H0 (Δ = 0)."""
    d = np.asarray(boot_delta, dtype=float)
    d = d[~np.isnan(d)]
    return float((1 + np.sum(np.abs(d - delta) >= abs(delta))) / (d.size + 1))


# ─── Rank tests ────────────────────────────────────────────────────────────────

def holm(pvals) -> np.ndarray:
    """Holm–Bonferroni correction (statsmodels multipletests, if installed;
    otherwise an equivalent implementation: step-down with enforced monotonicity)."""
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return p
    try:
        from statsmodels.stats.multitest import multipletests
        return multipletests(p, method="holm")[1]
    except ImportError:
        pass
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def rank_biserial(x, y) -> float:
    """Rank-biserial r = (W+ − W−) / (n(n+1)/2) on non-zero differences d = x − y
    (positive: x > y). No non-zero differences → NaN."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    nz = d[d != 0]
    if nz.size == 0:
        return float("nan")
    ranks = rankdata(np.abs(nz))
    n = nz.size
    return float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / (n * (n + 1) / 2))


def wilcoxon_pair(x, y, zero_method: str = "wilcox") -> dict:
    """Two-sided Wilcoxon signed-rank test for paired x, y.
    Returns n_ties (difference 0), n_eff (non-zero differences), W, p_raw, Z, r = |Z|/sqrt(n_eff)
    and the rank-biserial r_rb. Z = norm.isf(p/2) with the sign of the median of the differences x − y (when the median = 0:
    the sign of the excess of positive over negative rank sums). All differences zero: p = 1, Z = 0."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    nz = d[d != 0]
    n_ties = int(d.size - nz.size)
    n_eff = int(nz.size)
    if n_eff == 0:
        return {"n_ties": n_ties, "n_eff": 0, "W": float("nan"), "p_raw": 1.0,
                "Z": 0.0, "r": float("nan"), "r_rb": float("nan"), "median_diff": 0.0}

    res = wilcoxon(x, y, zero_method=zero_method, alternative="two-sided")
    p = float(res.pvalue)
    med = float(np.median(d))
    if med != 0:
        sign = np.sign(med)
    else:
        sign = np.sign(np.sum(rankdata(np.abs(nz)) * np.sign(nz)))
    z = float(sign * norm.isf(p / 2))
    return {"n_ties": n_ties, "n_eff": n_eff, "W": float(res.statistic), "p_raw": p,
            "Z": z, "r": abs(z) / np.sqrt(n_eff), "r_rb": rank_biserial(x, y), "median_diff": med}


def friedman(X: np.ndarray, models: Sequence[str]) -> dict:
    """Friedman (scipy, with tie correction) on a matrix (N units, k models) without NaN;
    Kendall W = χ² / (N(k−1)); mean ranks (ascending: 1 = lowest value, ties averaged)
    are always returned."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    mean_ranks = dict(zip(models, rankdata(X, axis=1).mean(axis=0))) if n else {m: float("nan") for m in models}
    out = {"chi2": float("nan"), "df": k - 1, "p": float("nan"), "kendall_w": float("nan"),
           "N": n, "k": k, "mean_ranks": {m: float(v) for m, v in mean_ranks.items()}}
    if k >= 3 and n > 0:
        chi2, p = friedmanchisquare(*X.T)
        out.update(chi2=float(chi2), p=float(p), kendall_w=kendall_w(chi2, n, k))
    return out


def kendall_w(chi2: float, n: int, k: int) -> float:
    """Kendall's W = χ²_Friedman / (N·(k−1))."""
    return float(chi2 / (n * (k - 1)))


def complete_rows(X: np.ndarray) -> np.ndarray:
    """Mask of rows without NaN (a unit is excluded for all models)."""
    return ~np.isnan(np.asarray(X, dtype=float)).any(axis=1)


def rank_tests(
    X: np.ndarray,
    models: Sequence[str],
    unit: str,
    zero_method: str = "wilcox",
    sensitivity_methods: Iterable[str] = ("pratt", "zsplit"),
    min_nonzero: int = 0,
) -> dict:
    """Friedman + pairwise Wilcoxon with Holm on a (units × models) matrix.

    Rows with NaN are excluded (reported as n_excluded). A pair with < `min_nonzero`
    non-zero differences is not tested (p = None); Holm covers the pairs with a result.
    Pair columns: model_a, model_b, n_eff, n_ties, W, Z, r (|Z|/√n), r_rb, median_diff,
    p_raw, p_holm, p_{method} for each sensitivity method."""
    _check_unit(unit)
    if zero_method not in ("wilcox", "pratt", "zsplit"):
        raise ValueError(f"Nieznany zero_method: {zero_method!r}")
    X = np.asarray(X, dtype=float)
    keep = complete_rows(X)
    Xk = X[keep]
    fr = friedman(Xk, models)

    rows: list[dict] = []
    k = len(models)
    for i in range(k):
        for j in range(i + 1, k):
            x, y = Xk[:, i], Xk[:, j]
            n_nz = int(np.sum((x - y) != 0))
            row = {"model_a": models[i], "model_b": models[j]}
            if n_nz < min_nonzero:
                row.update(n_eff=n_nz, n_ties=int(x.size - n_nz), W=None, Z=None, r=None, r_rb=None,
                           median_diff=float(np.median(x - y)) if x.size else float("nan"),
                           p_raw=None, **{f"p_{m}": None for m in sensitivity_methods})
            else:
                w = wilcoxon_pair(x, y, zero_method)
                row.update({kk: w[kk] for kk in ("n_eff", "n_ties", "W", "Z", "r", "r_rb", "median_diff", "p_raw")})
                for m in sensitivity_methods:
                    row[f"p_{m}"] = wilcoxon_pair(x, y, m)["p_raw"]
            rows.append(row)
    valid = [r for r in rows if r["p_raw"] is not None]
    for r, ph in zip(valid, holm([r["p_raw"] for r in valid])):
        r["p_holm"] = float(ph)
    for r in rows:
        r.setdefault("p_holm", None)
    return {
        "unit": unit, "N": int(keep.sum()), "n_total": int(X.shape[0]),
        "n_excluded": int((~keep).sum()), "keep": keep, "zero_method": zero_method,
        "friedman": fr, "pairs": pd.DataFrame(rows), "n_comparisons": len(valid),
    }


# ─── ICC ──────────────────────────────────────────────────────────────────────

def _icc_from_sums(g, N, sum_n2, S, A, Q):
    """ICC(1) from cluster sums (all arguments may be arrays of samples).
    g — number of clusters, N — Σn_i, sum_n2 — Σn_i², S — Σ y, A — Σ (Σ_i y)²/n_i, Q — Σ y²."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ssb = A - S ** 2 / N
        ssw = Q - A
        msb = ssb / (g - 1)
        msw = ssw / (N - g)
        n0 = (N - sum_n2 / N) / (g - 1)
        return (msb - msw) / (msb + (n0 - 1) * msw), n0


def icc_by_patient(values, clusters, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """ICC(1) from a one-way ANOVA with a random patient effect:
        ICC = (MSB − MSW) / (MSB + (n₀ − 1)·MSW),  n₀ = (N − Σn_i²/N) / (g − 1)
    (n₀ — adjusted mean cluster size for unbalanced clusters). NaN values
    (e.g. a rate with denominator 0) are skipped. 95% CI: percentile cluster bootstrap
    (patients with replacement; a patient drawn multiple times = separate clusters)."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    ok = ~np.isnan(v)
    v, c = v[ok], c[ok]
    labels, inv = np.unique(c, return_inverse=True)
    g = labels.size
    n_i = np.bincount(inv, minlength=g).astype(float)
    s_i = np.bincount(inv, weights=v, minlength=g)
    q_i = np.bincount(inv, weights=v ** 2, minlength=g)
    per = np.vstack([np.ones(g), n_i, n_i ** 2, s_i, s_i ** 2 / n_i, q_i])  # (6, g)

    icc, n0 = _icc_from_sums(*per.sum(axis=1))
    sums = cluster_bootstrap_sums(per, n_boot, seed)  # (n_boot, 6)
    boot, _ = _icc_from_sums(*sums.T)
    ci = percentile_ci(boot)
    return {"icc": float(icc), "ci_low": ci[0] if ci else float("nan"), "ci_high": ci[1] if ci else float("nan"),
            "n_docs": int(v.size), "n_patients": int(g), "n0": float(n0),
            "n_docs_excluded_nan": int((~ok).sum())}


# ─── Spearman with cluster CI ──────────────────────────────────────────────────

def _spearman_cols(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman's ρ of x with each column of Y (Pearson on ranks; ties averaged)."""
    rx = rankdata(x)
    rY = rankdata(Y, axis=0)
    rx = rx - rx.mean()
    rY = rY - rY.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx @ rY) / np.sqrt((rx @ rx) * (rY * rY).sum(axis=0))


def spearman_bootstrap(
    x, Y, clusters, unit: str = "patient", n_boot: int = N_BOOT, seed: int = SEED,
) -> pd.DataFrame:
    """Spearman's ρ (on documents) of the variable x with each column of Y (DataFrame, columns = models),
    with a percentile 95% CI from the cluster bootstrap (unit='patient': patients drawn
    with all their documents) or the document bootstrap (unit='document'), and p from spearmanr on
    documents (sensitivity analysis). Rows with NaN in x or in the given Y column are skipped
    only for that column. Returns: model_name, spearman_rho, ci_low, ci_high,
    ci_excludes_zero, p_value, n_docs, n_patients."""
    _check_unit(unit)
    x = np.asarray(x, dtype=float)
    Ydf = pd.DataFrame(Y)
    c = np.asarray(clusters)
    rows = []
    for col in Ydf.columns:
        y = Ydf[col].to_numpy(dtype=float)
        ok = ~(np.isnan(x) | np.isnan(y))
        xs, ys, cs = x[ok], y[ok], c[ok]
        rho, p = spearmanr(xs, ys)
        labels, inv = np.unique(cs if unit == "patient" else np.arange(xs.size), return_inverse=True)
        members = [np.flatnonzero(inv == u) for u in range(labels.size)]
        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, labels.size, size=labels.size)
            idx = pick if unit == "document" else np.concatenate([members[u] for u in pick])
            boot[b] = _spearman_cols(xs[idx], ys[idx][:, None])[0]
        ci = percentile_ci(boot)
        lo, hi = ci if ci else (float("nan"), float("nan"))
        rows.append({"model_name": col, "spearman_rho": float(rho), "ci_low": lo, "ci_high": hi,
                     "ci_excludes_zero": bool(ci is not None and (lo > 0 or hi < 0)),
                     "p_value": float(p), "n_docs": int(ok.sum()), "n_patients": int(np.unique(cs).size)})
    return pd.DataFrame(rows)


# ─── Unit comparison report ────────────────────────────────────────────────────

COMPARISON_COLUMNS = [
    "source", "test", "item", "statistic_name",
    "stat_document", "stat_patient", "p_holm_document", "p_holm_patient",
    "ci_document", "ci_patient", "effect_document", "effect_patient", "r_rb_document", "r_rb_patient",
    "kendall_w_document", "kendall_w_patient", "N_document", "N_patient",
    "excluded_document", "excluded_patient",
    "significant_document", "significant_patient", "conclusion_changed",
]


def fmt_ci(ci) -> str:
    if ci is None or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in ci):
        return ""
    return f"[{ci[0]:.4g}, {ci[1]:.4g}]"


def write_unit_comparison_part(rows: list[dict], path: str | Path) -> pd.DataFrame:
    """Writes a part of the unit comparison report (one script); `merge_unit_comparison` joins the parts.
    Empty significant_* (no test, e.g. only the F1 CI) → conclusion_changed = False."""
    df = pd.DataFrame(rows).reindex(columns=COMPARISON_COLUMNS)
    tested = df["significant_document"].notna() & df["significant_patient"].notna()
    df["conclusion_changed"] = tested & (df["significant_document"] != df["significant_patient"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def merge_unit_comparison(out_dir: str | Path, parts: Sequence[str | Path], alpha: float = ALPHA) -> Optional[pd.DataFrame]:
    """Joins the existing parts into {out_dir}/unit_comparison.csv and .md (list of changed conclusions at the top)."""
    frames = [pd.read_csv(p) for p in parts if Path(p).exists()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    out_dir = Path(out_dir)
    df.to_csv(out_dir / "unit_comparison.csv", index=False)

    changed = df[df["conclusion_changed"].astype(bool)]
    lines = [
        "# Unit of analysis: document (sensitivity) vs patient (primary)",
        "",
        f"Significance: p_Holm < {alpha:g} for rank tests (post-hoc only when the Friedman test of that "
        "error type is significant at the same unit), 95% CI excluding 0 for bootstrap differences and "
        "Spearman correlations, CI given for F1 (no test).",
        "",
        f"## Changed conclusions ({len(changed)})",
        "",
    ]
    if changed.empty:
        lines.append("None.")
    else:
        for _, r in changed.iterrows():
            detail = []
            if pd.notna(r["p_holm_document"]) or pd.notna(r["p_holm_patient"]):
                detail.append(f"p {_fmt_num(r['p_holm_document'])} → {_fmt_num(r['p_holm_patient'])}")
            if isinstance(r["ci_document"], str) or isinstance(r["ci_patient"], str):
                detail.append(f"CI {r['ci_document'] if isinstance(r['ci_document'], str) else '—'} → "
                              f"{r['ci_patient'] if isinstance(r['ci_patient'], str) else '—'}")
            lines.append(f"- **{r['source']} / {r['test']} / {r['item']}**: document → "
                         f"{'significant' if r['significant_document'] else 'n.s.'}, patient → "
                         f"{'significant' if r['significant_patient'] else 'n.s.'} ({'; '.join(detail)})")
    lines += ["", "## All comparisons", ""]
    show = ["source", "test", "item", "stat_document", "stat_patient", "p_holm_document", "p_holm_patient",
            "ci_document", "ci_patient", "effect_document", "effect_patient", "kendall_w_document",
            "kendall_w_patient", "N_document", "N_patient", "excluded_document", "excluded_patient",
            "conclusion_changed"]
    lines.append("| " + " | ".join(show) + " |")
    lines.append("|" + "---|" * len(show))
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt_num(r[c]) for c in show) + " |")
    (out_dir / "unit_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Zapisano porównanie jednostek ({len(df)} wierszy, zmienione wnioski: {len(changed)}) → "
          f"{out_dir / 'unit_comparison.csv'}, .md")
    return df


def merge_icc_report(out_dir: str | Path, parts: Sequence[str | Path]) -> Optional[pd.DataFrame]:
    frames = [pd.read_csv(p) for p in parts if Path(p).exists()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(Path(out_dir) / "icc_report.csv", index=False)
    print(f"Zapisano ICC ({len(df)} wierszy) → {Path(out_dir) / 'icc_report.csv'}")
    return df


def _fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        return f"{v:.3g}" if abs(v) < 1e-3 and v != 0 else f"{v:.4g}"
    return str(v)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
