import csv
import json
from collections import defaultdict
from pathlib import Path

import matching
import stats_clustered as sc
from matching import (  # noqa: F401 — re-export for older scripts
    BINARY_LABEL_MAP,
    CATEGORIES,
    PL_DIAG_TO_EN,
    THERAPY_MAP,
    _norm,
)

# Matching, label mapping and the error taxonomy live in matching.py — this module
# only computes metrics from them; CIs and statistical tests (patient / document) — stats_clustered.py.


def get_category(label: str) -> str:
    """Category (EN) of a GT label according to matching.classify_label; rejected labels → 'Other'."""
    cat, _key, _value, _reason = matching.classify_label(label)
    return cat or "Other"


DEFAULT_MODELS = [
    # "Divide_Bielik-v3-11_extraction_17_test",
    # "Divide_Qwen3-14_extraction_17_test",
    # "Divide_PLLuM-v2-12_extraction_17_test",
    "PLLuM-v2-12",
    "Qwen3_8-27",
    "Bielik-v3-11",
    "Qwen3-14",
    "Ministral-3-14",
    "Gemma3-12",
]

DEFAULT_CSV = "annotations/validated_annotations.csv"


def _model_path(output_dir: str, model: str) -> Path:
    return Path(output_dir) / f"{model}_200_patients.json"


def _evaluate(gt_rows: list[dict], models: list[str], output_dir: str) -> tuple[matching.GoldStandard, dict[str, matching.ModelResult]]:
    """GT + matching.evaluate_model results for the models whose file exists."""
    gold = matching.load_gold(gt_rows)
    results: dict[str, matching.ModelResult] = {}
    for model in models:
        path = _model_path(output_dir, model)
        if not path.exists():
            print(f"[POMINIĘTO] Brak pliku: {path.name}")
            continue
        results[model] = matching.evaluate_model_file(gold, path, model)
    return gold, results


def _collect_results(
    gt_rows: list[dict],
    models: list[str],
    output_dir: str,
) -> tuple[list[str], dict[str, dict[str, list[bool]]]]:
    """
    Returns:
      available_models  — models for which a JSON file was found
      entity_results    — {GT label: {model: [hit?, ...]}} (one entry per document)
    """
    _gold, results = _evaluate(gt_rows, models, output_dir)
    entity_results: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for model, res in results.items():
        for r in res.iter_results():
            for g, _p in r.tp:
                entity_results[g.label][model].append(True)
            for g, _l in r.fn:
                entity_results[g.label][model].append(False)
    return list(results), entity_results


def evaluate_models(
    csv_path: str = DEFAULT_CSV,
    output_dir: str = "outputs_v5",
    models: list[str] | None = None,
) -> None:
    """Prints a Recall report per category for each model (Recall = TP/(TP+FN))."""
    if models is None:
        models = DEFAULT_MODELS

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))

    gold, results = _evaluate(gt_rows, models, output_dir)
    print(f"GT: {gold.n_rows} wierszy, {gold.n_gt()} ocenianych encji w {len(gold.docs)} dokumentach "
          f"(odrzucone wiersze: {dict(gold.excluded_rows)})\n")

    col_w = max(len(c) for c in CATEGORIES) + 2
    for model, res in results.items():
        tot = {"tp": 0, "fn": 0}
        print("=" * 70)
        print(f"Model: {model}")
        print("=" * 70)
        print(f"  {'Kategoria':<{col_w}} {'Correct':>8} {'Total':>7} {'Recall':>10}")
        print(f"  {'-' * col_w} {'-'*8} {'-'*7} {'-'*10}")
        for cat in CATEGORIES:
            tp = sum(len(res.docs[o][cat].tp) for o in res.docs)
            n = sum(res.docs[o][cat].n_gt for o in res.docs)
            if not n:
                continue
            print(f"  {cat:<{col_w}} {tp:>8} {n:>7} {tp / n:>9.1%}")
            tot["tp"] += tp
            tot["fn"] += n - tp
        n_all = tot["tp"] + tot["fn"]
        print(f"  {'─' * col_w} {'─'*8} {'─'*7} {'─'*10}")
        print(f"  {'RAZEM':<{col_w}} {tot['tp']:>8} {n_all:>7} {tot['tp'] / n_all if n_all else 0:>9.1%}")
        print()


def evaluate_entities_to_csv(
    csv_path: str = DEFAULT_CSV,
    output_dir: str = "outputs_v5",
    output_csv: str = "results/entity_accuracy.csv",
    models: list[str] | None = None,
) -> None:
    """
    Writes the entity × model matrix to a CSV file.

    Columns: Encja, Kategoria, Count_GT, {model_1}, {model_2}, …
    Count_GT = number of documents with the given entity (after merging duplicates).
    Values = entity recall (0.0–1.0), rounded to 4 decimal places.
    """
    if models is None:
        models = DEFAULT_MODELS

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))

    gold = matching.load_gold(gt_rows)
    available, entity_results = _collect_results(gt_rows, models, output_dir)

    gt_counts: dict[str, int] = defaultdict(int)
    label_cat: dict[str, str] = {}
    for cats in gold.items.values():
        for cat, items in cats.items():
            for g in items:
                gt_counts[g.label] += 1
                label_cat[g.label] = cat

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    sorted_labels = sorted(gt_counts, key=lambda lbl: (label_cat[lbl], -gt_counts[lbl], lbl))

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Encja", "Kategoria", "Count_GT"] + available)
        for label in sorted_labels:
            row_out = [label, label_cat[label], gt_counts[label]]
            for model in available:
                hits = entity_results[label].get(model, [])
                row_out.append(round(sum(hits) / len(hits), 4) if hits else "")
            writer.writerow(row_out)

    print(f"Zapisano {len(sorted_labels)} encji × {len(available)} modeli → {output_csv}")


def _compute_full_metrics(
    gt_rows: list[dict],
    available: list[str],
    output_dir: str,
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, dict[str, int]]]]]:
    """
    Returns (metrics, metrics_by_opis):
      metrics         — {category: {model: {tp, fp, fn}}} aggregated over all OPIS
      metrics_by_opis — {category: {model: {opis: {tp, fp, fn}}}}, the same split
                         not summed over documents (bootstrap CI, per-document tests).

    TP/FP/FN come exclusively from matching.evaluate_model (1:1 matching within
    a document and category; definitions in the matching.py docstring). Every GT document has an entry
    in every category, including zero ones, so the bootstrap samples from the full OPIS population.
    """
    _gold, results = _evaluate(gt_rows, available, output_dir)
    metrics: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    )
    metrics_by_opis: dict[str, dict[str, dict[str, dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    )
    for model, res in results.items():
        for r in res.iter_results():
            c = {"tp": len(r.tp), "fp": len(r.fp), "fn": len(r.fn)}
            for k, v in c.items():
                metrics[r.category][model][k] += v
                metrics_by_opis[r.category][model][r.opis][k] += v
    return metrics, metrics_by_opis


# Model names in LaTeX tables and plots (number = billions of parameters, hence "B"; default: file name)
MODEL_DISPLAY: dict[str, str] = {
    "PLLuM-v2-12": "PLLuM-v2-12B",
    "Qwen3_8-27": "Qwen3.8-27B",
    "Bielik-v3-11": "Bielik-v3-11B",
    "Qwen3-14": "Qwen3-14B",
    "Ministral-3-14": "Ministral-3-14B",
    "Gemma3-12": "Gemma3-12B",
    "Llama3_3-70": "Llama3.3-70B",
}


PRF_CAPTION = (
    "F1 score (with 95\\% confidence interval), Precision (P) and Recall (R), in \\%, "
    "of LLM annotation models per annotation category. "
    "$\\mathrm{F1} = 2\\mathrm{TP}/(2\\mathrm{TP}+\\mathrm{FP}+\\mathrm{FN})$. "
    "Micro avg.: TP, FP, FN summed over all categories; Macro avg.: unweighted mean of the "
    "per-category values. Best F1, P and R in each category in bold."
)


def _prf_unit_caption(unit: str, n_patients: int, n_docs: int, n_boot: int) -> str:
    """Addendum to the P/R/F1 table caption describing the unit of analysis and the bootstrap."""
    if unit == "patient":
        return (f" Unit of analysis: patient, $N = {n_patients}$ (P, R and F1 pooled over all "
                f"{n_docs} visits); 95\\% CI from a cluster bootstrap resampling patients with all their "
                f"visits ({n_boot} resamples, percentile intervals).")
    return (f" 95\\% CI from a bootstrap resampling documents, $N = {n_docs}$ ({n_boot} resamples, "
            "percentile intervals); sensitivity analysis (unit of analysis: document, visits of the same "
            "patient treated as independent).")


def evaluate_models_to_latex(
    csv_path: str = DEFAULT_CSV,
    output_dir: str = "outputs_v5",
    output_tex: str = "results/evaluation_table.tex",
    models: list[str] | None = None,
    caption: str | None = None,
    label: str = "tab:annotation_metrics",
    n_boot: int = sc.N_BOOT,
    seed: int = sc.SEED,
    unit: str = "patient",
) -> "pd.DataFrame":
    """
    Writes the evaluation results (F1 / P / R) as a LaTeX table and a CSV next to it (same name, .csv).

    Columns: Category | metric | model …
    Rows: per category, Micro avg. (TP/FP/FN summed over categories) and Macro avg.
    (unweighted mean of per-category F1 / P / R); a block of three rows F1 (with CI as
    \\ci{lo}{hi}), P, R; the best value of each metric in a row is bold. Requires
    in the preamble: booktabs, multirow and the \\ci macro, e.g. \\newcommand{\\ci}[2]{{\\scriptsize(#1--#2)}}.

    Metrics (TP/FP/FN from matching.py), independent of `unit`:
    P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN) from sums over all documents.
    95% CI of F1: `unit="patient"` — cluster bootstrap (patients with replacement, with all
    their documents), `unit="document"` — document bootstrap (sensitivity); `n_boot` resamples,
    seed `seed`, a single sample matrix for all table cells (paired across
    models and categories). In each sample F1 = 2ΣTP/(2ΣTP+ΣFP+ΣFN) as in the point
    estimate; Macro = mean F1 of the categories defined in the given sample.
    """
    import warnings

    import numpy as np
    import pandas as pd

    if models is None:
        models = DEFAULT_MODELS

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))

    available_set: list[str] = []
    for m in models:
        if (Path(output_dir) / f"{m}_200_patients.json").exists():
            available_set.append(m)
        else:
            print(f"[POMINIĘTO] Brak pliku: {m}_200_patients.json")

    raw, raw_by_opis = _compute_full_metrics(gt_rows, available_set, output_dir)
    gold = matching.load_gold(gt_rows)
    docs = gold.docs
    clusters = np.array([gold.patient[d] for d in docs])
    n_patients = len(set(clusters))

    ordered_cats = [c for c in CATEGORIES if c in raw]

    # ── Counts (category, model, TP/FP/FN, document) and bootstrap samples ──
    pos = {d: i for i, d in enumerate(docs)}
    arr = np.zeros((len(ordered_cats), len(available_set), 3, len(docs)))
    for c_i, cat in enumerate(ordered_cats):
        for m_i, model in enumerate(available_set):
            for opis, om in raw_by_opis.get(cat, {}).get(model, {}).items():
                arr[c_i, m_i, :, pos[opis]] = (om["tp"], om["fp"], om["fn"])
            if tuple(arr[c_i, m_i].sum(axis=-1)) != tuple(raw[cat][model][k] for k in ("tp", "fp", "fn")):
                raise AssertionError(f"{model}/{cat}: Σ dokumentów ≠ zliczenia zagregowane")
    boot = sc.bootstrap_sums(arr, unit, clusters, n_boot, seed)  # (n_boot, C, M, 3)
    f1_cat = sc.f1_from_sums(boot[..., 0], boot[..., 1], boot[..., 2])  # (n_boot, C, M)
    tot = boot.sum(axis=1)
    f1_avg = sc.f1_from_sums(tot[..., 0], tot[..., 1], tot[..., 2])  # (n_boot, M)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # sample without any defined category → NaN
        f1_macro = np.nanmean(f1_cat, axis=1)

    def _safe(num: int, den: int) -> float | None:
        return num / den if den else None

    # cell[(row, model)] = {"F1": .., "P": .., "R": .., "ci": ..}; row = EN category, "Average" or "Macro"
    cell: dict[tuple[str, str], dict] = {}
    total: dict[str, dict[str, int]] = {m: {"tp": 0, "fp": 0, "fn": 0} for m in available_set}

    def _metrics(tp: int, fp: int, fn: int, ci) -> dict:
        p = _safe(tp, tp + fp)
        r = _safe(tp, tp + fn)
        # F1 = 2TP/(2TP+FP+FN) — same formula as in the comparison table; = 0 also when P is undefined
        return {"F1": _safe(2 * tp, 2 * tp + fp + fn), "P": p, "R": r, "ci": ci,
                "tp": tp, "fp": fp, "fn": fn}

    for c_i, cat in enumerate(ordered_cats):
        for m_i, model in enumerate(available_set):
            cm = raw[cat][model]
            cell[(cat, model)] = _metrics(cm["tp"], cm["fp"], cm["fn"], sc.percentile_ci(f1_cat[:, c_i, m_i]))
            for k in ("tp", "fp", "fn"):
                total[model][k] += cm[k]

    for m_i, model in enumerate(available_set):
        t = total[model]
        cell[("Average", model)] = _metrics(t["tp"], t["fp"], t["fn"], sc.percentile_ci(f1_avg[:, m_i]))

    # Macro: mean of per-category F1 / P / R
    for m_i, model in enumerate(available_set):
        cat_cells = [cell[(cat, model)] for cat in ordered_cats]
        macro = {}
        for metric in ("F1", "P", "R"):
            vals = [c[metric] for c in cat_cells if c[metric] is not None]
            macro[metric] = sum(vals) / len(vals) if vals else None
        macro["ci"] = sc.percentile_ci(f1_macro[:, m_i])
        cell[("Macro", model)] = macro

    # Assertion: F1 in the table = 2TP/(2TP+FP+FN) from the same counts
    for model in available_set:
        for cat in ordered_cats + ["Average"]:
            c = raw[cat][model] if cat != "Average" else total[model]
            den = 2 * c["tp"] + c["fp"] + c["fn"]
            expected = 2 * c["tp"] / den if den else None
            got = cell[(cat, model)]["F1"]
            if (expected is None) != (got is None) or (got is not None and not np.isclose(got, expected)):
                raise AssertionError(f"{model}/{cat}: F1 {got} ≠ 2TP/(2TP+FP+FN) {expected}")

    # ── Formatting ───────────────────────────────────────────────────────────
    def _pct(v: float | None) -> str:
        return f"{v * 100:.1f}" if v is not None else "---"

    def _metric_cells(row: str, metric: str) -> list[str]:
        """Cells of one metric row; the best value (after rounding
        to the displayed precision, so ties are all bold) in \\textbf."""
        vals = [cell[(row, m)][metric] for m in available_set]
        shown = [round(v * 100, 1) if v is not None else None for v in vals]
        best = max((s for s in shown if s is not None), default=None)
        out: list[str] = []
        for m, v, s in zip(available_set, vals, shown):
            txt = _pct(v)
            if s is not None and s == best:
                txt = f"\\textbf{{{txt}}}"
            ci = cell[(row, m)]["ci"]
            if metric == "F1" and v is not None and ci is not None:
                txt += f" \\ci{{{ci[0] * 100:.1f}}}{{{ci[1] * 100:.1f}}}"
            out.append(txt)
        return out

    def _block(row: str, row_title: str) -> list[str]:
        return [f"\\multirow{{3}}{{*}}{{{row_title}}}"] + [
            f" & {metric:<2} & {' & '.join(_metric_cells(row, metric))} \\\\"
            for metric in ("F1", "P", "R")
        ]

    if caption is None:
        caption = PRF_CAPTION + _prf_unit_caption(unit, n_patients, len(docs), n_boot)

    n_models = len(available_set)
    model_headers = " & ".join(
        f"\\textbf{{{MODEL_DISPLAY.get(m, m)}}}" for m in available_set
    )

    lines: list[str] = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{ll{'c' * n_models}}}",
        "\\toprule",
        f"\\textbf{{Category}} & & {model_headers} \\\\",
        "\\midrule",
    ]
    for i, cat in enumerate(ordered_cats):
        if i:
            lines.append("\\addlinespace")
        lines += _block(cat, cat)
    lines.append("\\midrule")
    lines += _block("Average", "\\textbf{Micro avg.}")
    lines.append("\\addlinespace")
    lines += _block("Macro", "\\textbf{Macro avg.}")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ]

    tex = "\n".join(lines) + "\n"

    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    rows = []
    for row in ordered_cats + ["Average", "Macro"]:
        for model in available_set:
            c = cell[(row, model)]
            rows.append({"row": row, "model": model, "unit": unit,
                         "tp": c.get("tp"), "fp": c.get("fp"), "fn": c.get("fn"),
                         "F1": c["F1"], "P": c["P"], "R": c["R"],
                         "ci_low": c["ci"][0] if c["ci"] else None, "ci_high": c["ci"][1] if c["ci"] else None,
                         "n_patients": n_patients, "n_documents": len(docs), "n_boot": n_boot, "seed": seed})
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_tex).with_suffix(".csv"), index=False)

    print(f"Zapisano tabelę LaTeX ({len(ordered_cats)} kategorii × {n_models} modeli × 3 metryki, "
          f"CI: {unit}) → {output_tex}")
    return df


def icc_f1_by_model(
    csv_path: str = DEFAULT_CSV,
    output_dir: str = "outputs_v5",
    models: list[str] | None = None,
    n_boot: int = sc.N_BOOT,
    seed: int = sc.SEED,
) -> "pd.DataFrame":
    """ICC(1) of document F1 (TP/FP/FN summed over categories, TP+FP+FN = 0 → 1) within
    a patient, per model, with a 95% CI from the cluster bootstrap."""
    import numpy as np
    import pandas as pd

    if models is None:
        models = DEFAULT_MODELS
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))
    gold, results = _evaluate(gt_rows, models, output_dir)
    rows = []
    for model, res in results.items():
        c = {k: np.zeros(len(gold.docs)) for k in ("tp", "fp", "fn")}
        for d_i, opis in enumerate(gold.docs):
            for r in res.docs[opis].values():
                c["tp"][d_i] += len(r.tp)
                c["fp"][d_i] += len(r.fp)
                c["fn"][d_i] += len(r.fn)
        f1 = sc.unit_f1(c["tp"], c["fp"], c["fn"])
        icc = sc.icc_by_patient(f1, [gold.patient[d] for d in gold.docs], n_boot, seed)
        rows.append({"metric": "document_f1", "error_type": "", "model": model, **icc})
    return pd.DataFrame(rows)


def _fmt_p_tex(p: float, bold: bool) -> str:
    if p < 0.001:
        return "$\\mathbf{<.001}$" if bold else "$<.001$"
    txt = f"{p:.3f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def _fmt_p_text(p: float) -> str:
    return "$p < .001$" if p < 0.001 else f"$p = {p:.3f}$"


def _legacy_entity_wilcoxon(
    gt_rows: list[dict],
    available: list[str],
    output_dir: str,
    output_tex: str,
    caption: str,
    label: str,
) -> dict:
    """Former procedure (accuracy per entity, n = number of unique GT labels).
    Kept only for comparisons — it violates the independence assumption and does not concern F1."""
    import numpy as np
    from scipy.stats import wilcoxon as _wilcoxon

    _, entity_results = _collect_results(gt_rows, available, output_dir)

    # Accuracy per entity per model: {label: {model: float}}
    all_labels = sorted(entity_results.keys())
    entity_acc: dict[str, dict[str, float]] = {}
    for lbl in all_labels:
        entity_acc[lbl] = {}
        for model in available:
            hits = entity_results[lbl].get(model, [])
            if hits:
                entity_acc[lbl][model] = sum(hits) / len(hits)

    # Paired Wilcoxon test for each pair (i, j), i < j
    PairKey = tuple[str, str]
    pval: dict[PairKey, float | None] = {}
    neff: dict[PairKey, int] = {}

    for i, m1 in enumerate(available):
        for j, m2 in enumerate(available):
            if j <= i:
                continue
            common = [
                lbl for lbl in all_labels
                if m1 in entity_acc.get(lbl, {}) and m2 in entity_acc.get(lbl, {})
            ]
            x = np.array([entity_acc[lbl][m1] for lbl in common])
            y = np.array([entity_acc[lbl][m2] for lbl in common])
            diffs = x - y
            n_nonzero = int(np.sum(diffs != 0))
            neff[(m1, m2)] = n_nonzero
            if n_nonzero < 10:
                pval[(m1, m2)] = None
                continue
            try:
                _, p = _wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                pval[(m1, m2)] = float(p)
            except Exception:
                pval[(m1, m2)] = None

    def _fmt_cell(m_row: str, m_col: str) -> str:
        if m_row == m_col:
            return "---"
        key: PairKey = (m_row, m_col) if (m_row, m_col) in pval else (m_col, m_row)
        p = pval.get(key)
        if p is None:
            return "---"
        if p < 1e-4:
            exp = int(f"{p:.2e}".split("e")[1])
            mant = p / (10 ** exp)
            return f"${mant:.2f} \\times 10^{{{exp}}}$"
        return f"{p:.4f}"

    n = len(available)
    col_spec = "l" + "c" * n

    headers = " & ".join(f"\\textbf{{{m}}}" for m in available)

    lines: list[str] = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\textbf{{Model}} & {headers} \\\\",
        "\\midrule",
    ]
    for m_row in available:
        cells = [_fmt_cell(m_row, m_col) for m_col in available]
        lines.append(f"{m_row} & {' & '.join(cells)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Zapisano tabelę Wilcoxona ({n}×{n}) → {output_tex}")
    for (m1, m2), p in sorted(pval.items()):
        p_str = f"{p:.4f}" if p is not None else "n/a"
        print(f"  {m1} vs {m2}: p={p_str}, n_nonzero={neff.get((m1,m2),0)}")

    return {"unit": "entity", "p_values": pval, "n_eff": neff, "tex": tex}


LEGACY_WILCOXON_CAPTION = (
    "Wilcoxon signed-rank test p-values for pairwise model comparisons "
    "(entity-level accuracy, two-sided). "
    "Values below $p < 0.05$ are considered statistically significant."
)


def evaluate_wilcoxon_to_latex(
    csv_path: str = DEFAULT_CSV,
    output_dir: str = "outputs_v5",
    output_tex: str = "results/wilcoxon_table.tex",
    models: list[str] | None = None,
    caption: str | None = None,
    label: str = "tab:wilcoxon",
    unit: str = "patient",
    empty_doc_policy: str = "one",
    zero_method: str = "wilcox",
    n_boot: int = sc.N_BOOT,
    seed: int = sc.SEED,
    alpha: float = sc.ALPHA,
    runs: dict[str, list[str]] | None = None,
) -> dict | None:
    """
    Pairwise model comparisons on F1 computed per unit of analysis (default: patient).

    TP/FP/FN per document come from `_compute_full_metrics` — the same matching logic
    as in `evaluate_models_to_latex` (Average row), summed over all categories.
    Documents = OPIS present in GT (as in the main evaluation).

      unit             — "patient" (TP/FP/FN summed over all of the patient's documents,
                         patient F1 = 2ΣTP/(2ΣTP+ΣFP+ΣFN)), "document" (sensitivity analysis),
                         or the name of another GT column grouping documents. "entity" = the former
                         procedure (accuracy per entity), legacy only, with a warning.
      empty_doc_policy — TP+FP+FN = 0: "one" (F1 = 1) or "exclude" (unit excluded
                         from all models to keep the pairing).
      zero_method      — passed to scipy.stats.wilcoxon; "pratt" and "zsplit" are always computed in addition.
      runs             — optionally {model: [stem, …]} for multiple runs/seeds, where the file
                         is {output_dir}/{stem}_200_patients.json; F1 per unit is computed
                         per run and averaged, and so is micro-F1.

    Procedure (stats_clustered.rank_tests): Friedman (Kendall's W, mean ranks) on the
    units × models matrix; two-sided Wilcoxon for k(k−1)/2 pairs with Holm correction; Δ corpus
    micro-F1 (row − column, p.p.) with 95% CI and two-sided p from the paired bootstrap
    of units (`n_boot`, `seed`) — for patients this is the cluster bootstrap.

    Writes `output_tex` (upper triangle: Holm p, lower: Δ [CI]) and next to it
    {base}_details.csv, {base}_friedman.csv and {base}_per_document.csv. Returns a dict with
    all statistics.
    """
    import numpy as np
    import pandas as pd

    if models is None:
        models = DEFAULT_MODELS

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))

    model_runs = {m: list(runs.get(m, [m])) if runs else [m] for m in models}
    available = []
    for m in models:
        missing = [s for s in model_runs[m] if not (Path(output_dir) / f"{s}_200_patients.json").exists()]
        if missing:
            print(f"[POMINIĘTO] {m}: brak plików " + ", ".join(f"{s}_200_patients.json" for s in missing))
        else:
            available.append(m)
    if len(available) < 2:
        print("Za mało modeli do porównania parowego.")
        return None

    if unit == "entity":
        import warnings
        msg = ("unit='entity' to procedura legacy: accuracy per encja narusza założenie "
               "niezależności (encje z jednego dokumentu są skorelowane) i ignoruje FP; "
               "F1 per encja nie istnieje. Użyj unit='patient'.")
        warnings.warn(msg, stacklevel=2)
        print(f"[OSTRZEŻENIE] {msg}")
        return _legacy_entity_wilcoxon(
            gt_rows, available, output_dir, output_tex,
            caption if caption is not None else LEGACY_WILCOXON_CAPTION, label,
        )

    if zero_method not in ("wilcox", "pratt", "zsplit"):
        raise ValueError(f"Nieznany zero_method: {zero_method!r}")

    # ── Units of analysis ────────────────────────────────────────────────────
    docs = sorted({r["OPIS"] for r in gt_rows})  # same order as in _compute_full_metrics
    patient_col = "ID_PACJENTA" if gt_rows and "ID_PACJENTA" in gt_rows[0] else None
    doc_patient = {r["OPIS"]: r[patient_col] for r in gt_rows} if patient_col else {}
    group_col = patient_col if unit == "patient" else (None if unit == "document" else unit)
    if unit == "patient" and patient_col is None:
        raise ValueError(f"unit='patient': brak kolumny ID_PACJENTA w {csv_path}")
    if group_col is None:
        doc_unit = {d: d for d in docs}
    else:
        if not gt_rows or group_col not in gt_rows[0]:
            raise ValueError(f"unit={unit!r}: brak takiej kolumny w {csv_path}")
        groups: dict[str, set[str]] = defaultdict(set)
        for r in gt_rows:
            groups[r["OPIS"]].add(r[group_col])
        ambiguous = [d for d, g in groups.items() if len(g) > 1]
        if ambiguous:
            raise ValueError(f"unit={unit!r}: {len(ambiguous)} dokumentów ma więcej niż jedną wartość grupy")
        doc_unit = {d: next(iter(groups[d])) for d in docs}
    units = docs if group_col is None else sorted(set(doc_unit.values()))
    unit_pos = {u: i for i, u in enumerate(units)}
    unit_name = unit
    test_unit = "document" if group_col is None else "patient"  # group of documents = cluster

    # ── TP/FP/FN per document (main evaluation logic) ────────────────────
    stems = [s for m in available for s in model_runs[m]]
    raw, raw_by_opis = _compute_full_metrics(gt_rows, stems, output_dir)

    # documents missing from the results of some run → unpaired unit
    missing_units: set[str] = set()
    for s in stems:
        with (Path(output_dir) / f"{s}_200_patients.json").open(encoding="utf-8") as f:
            present = {rec["OPIS"] for rec in json.load(f)}
        lacking = [d for d in docs if d not in present]
        if lacking:
            print(f"[BRAKI] {s}: {len(lacking)} dokumentów GT bez predykcji")
            missing_units.update(doc_unit[d] for d in lacking)

    # summing documents into units (for a patient: Σ over visits, ratio of sums)
    n_units = len(units)
    counts: dict[str, dict[str, np.ndarray]] = {}
    for m in available:
        c = {k: np.zeros((len(model_runs[m]), n_units)) for k in ("tp", "fp", "fn")}
        for r_i, s in enumerate(model_runs[m]):
            for cat_metrics in raw_by_opis.values():
                for opis, om in cat_metrics.get(s, {}).items():
                    for k in ("tp", "fp", "fn"):
                        c[k][r_i, unit_pos[doc_unit[opis]]] += om[k]
        counts[m] = c

    # Σ over units == Σ over documents == Average row of the main evaluation
    for m in available:
        for r_i, s in enumerate(model_runs[m]):
            tot = {k: sum(raw[cat].get(s, {}).get(k, 0) for cat in raw) for k in ("tp", "fp", "fn")}
            for k in ("tp", "fp", "fn"):
                if counts[m][k][r_i].sum() != tot[k]:
                    raise AssertionError(f"{s}: Σ{k} po jednostkach {counts[m][k][r_i].sum()} ≠ Σ po dokumentach {tot[k]}")
            main_f1 = float(sc.micro_f1(tot["tp"], tot["fp"], tot["fn"]))
            ours = float(sc.micro_f1(counts[m]["tp"][r_i], counts[m]["fp"][r_i], counts[m]["fn"][r_i]))
            if not np.isclose(main_f1, ours):
                raise AssertionError(f"{s}: mikro-F1 {ours:.6f} ≠ ewaluacja główna {main_f1:.6f}")

    # ── Empty units ──────────────────────────────────────────────────────────
    gold = counts[available[0]]["tp"][0] + counts[available[0]]["fn"][0]  # TP+FN = number of gold entities
    n_gold_empty = int(np.sum(gold == 0))
    empty_stats: dict[str, dict[str, int]] = {}
    for m in available:
        c = counts[m]
        all_zero = (c["tp"] + c["fp"] + c["fn"]) == 0
        pred_no_gold = (gold == 0) & (c["fp"] > 0)
        empty_stats[m] = {
            "tp_fp_fn_zero": int(np.sum(all_zero.any(axis=0))),
            "no_gold_with_predictions": int(np.sum(pred_no_gold.any(axis=0))),
        }
    print(f"Jednostka analizy: {unit_name}; jednostek: {n_units}; bez złotych encji: {n_gold_empty}")
    for m in available:
        e = empty_stats[m]
        print(f"  {m}: TP+FP+FN=0 → {e['tp_fp_fn_zero']} "
              f"(F1={'1' if empty_doc_policy == 'one' else 'wykluczone'}); "
              f"brak złotych, są predykcje → {e['no_gold_with_predictions']} (F1=0)")

    # ── F1 per unit (mean over runs) and pairing ──────────────
    f1_runs = {m: sc.unit_f1(counts[m]["tp"], counts[m]["fp"], counts[m]["fn"], empty_doc_policy)
               for m in available}
    f1_unit = {m: v.mean(axis=0) for m, v in f1_runs.items()}
    excluded_empty = np.zeros(n_units, dtype=bool)
    for m in available:
        excluded_empty |= np.isnan(f1_unit[m])
    excluded_missing = np.array([u in missing_units for u in units])
    keep = ~(excluded_empty | excluded_missing)
    if excluded_empty.any():
        print(f"[WYKLUCZONO] {int(excluded_empty.sum())} jednostek z TP+FP+FN=0 (empty_doc_policy='exclude')")
    if excluded_missing.any():
        print(f"[WYKLUCZONO] {int(excluded_missing.sum())} jednostek bez kompletu predykcji wszystkich modeli")
    n_ok = int(keep.sum())
    X = np.column_stack([f1_unit[m][keep] for m in available])  # (N, k)
    k = len(available)

    # ── Friedman + pairwise Wilcoxon (Holm) ─────────────────────────────────
    tests = sc.rank_tests(X, available, test_unit, zero_method, sensitivity_methods=("pratt", "zsplit"))
    friedman = tests["friedman"]
    if k < 3:
        print("[INFO] Test Friedmana wymaga co najmniej 3 modeli — pominięto.")

    # ── Micro-F1 and paired bootstrap of units ────────────────────────────
    counts_ok = {m: {kk: v[:, keep] for kk, v in c.items()} for m, c in counts.items()}
    micro = {m: float(sc.micro_f1(c["tp"], c["fp"], c["fn"])) for m, c in counts_ok.items()}
    stacked = np.stack([np.stack([counts_ok[m][kk] for kk in ("tp", "fp", "fn")]) for m in available])  # (M, 3, R, U)
    sums = sc.cluster_bootstrap_sums(stacked, n_boot, seed)  # (n_boot, M, 3, R): the same samples for all models
    boot_f1 = sc.f1_from_sums(sums[:, :, 0], sums[:, :, 1], sums[:, :, 2]).mean(axis=-1)  # (n_boot, M)
    boot = {m: boot_f1[:, i] for i, m in enumerate(available)}

    pairs: list[dict] = []
    for _, w in tests["pairs"].iterrows():
        a, b = w["model_a"], w["model_b"]
        delta = micro[a] - micro[b]
        d_boot = boot[a] - boot[b]
        lo, hi = sc.percentile_ci(d_boot, min_valid=0.0)
        pairs.append({
            "pair": f"{a} vs {b}", "model_a": a, "model_b": b,
            "n_eff": w["n_eff"], "n_ties": w["n_ties"], "W": w["W"], "Z": w["Z"], "r": w["r"], "r_rb": w["r_rb"],
            "median_diff_f1": w["median_diff"],
            "p_raw": w["p_raw"], "p_holm": w["p_holm"], "p_pratt": w["p_pratt"], "p_zsplit": w["p_zsplit"],
            "micro_f1_a": micro[a] * 100, "micro_f1_b": micro[b] * 100,
            "delta_pp": delta * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
            "p_bootstrap": sc.bootstrap_p_two_sided(d_boot, delta),
        })

    # Δ in the comparison table = micro-F1 difference from the main table (when no unit is excluded)
    if keep.all():
        main_micro = {
            m: float(np.mean([sc.micro_f1(*(sum(raw[cat].get(s, {}).get(k, 0) for cat in raw)
                                            for k in ("tp", "fp", "fn")))
                              for s in model_runs[m]]))
            for m in available
        }
        for p in pairs:
            expected = (main_micro[p["model_a"]] - main_micro[p["model_b"]]) * 100
            if not np.isclose(p["delta_pp"], expected):
                raise AssertionError(f"{p['pair']}: Δ {p['delta_pp']:.6f} ≠ różnica tabeli głównej {expected:.6f}")
    details = pd.DataFrame(pairs)
    pair_idx = {(p["model_a"], p["model_b"]): p for p in pairs}

    # ── LaTeX table ─────────────────────────────────────────────────────────
    def _cell(m_row: str, m_col: str) -> str:
        r_i, c_i = available.index(m_row), available.index(m_col)
        if r_i == c_i:
            return "---"
        if r_i < c_i:  # upper triangle: Holm p
            p = pair_idx[(m_row, m_col)]["p_holm"]
            return _fmt_p_tex(p, p < alpha)
        # lower triangle: Δ = row − column (pair stored as column − row → sign flip)
        rec = pair_idx[(m_col, m_row)]
        d, lo, hi = -rec["delta_pp"], -rec["ci_high_pp"], -rec["ci_low_pp"]
        return f"{d:+.1f} [{lo:.1f}, {hi:.1f}]"

    if caption is None:
        caption = _wilcoxon_caption(unit, group_col, n_ok, k, zero_method, alpha, empty_doc_policy,
                                    empty_stats, friedman, n_boot, len(docs))

    headers = " & ".join(f"\\textbf{{{MODEL_DISPLAY.get(m, m)}}}" for m in available)
    lines: list[str] = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{l{'c' * k}}}",
        "\\toprule",
        f"\\textbf{{Model}} & {headers} \\\\",
        "\\midrule",
    ]
    for m_row in available:
        cells = [_cell(m_row, m_col) for m_col in available]
        lines.append(f"{MODEL_DISPLAY.get(m_row, m_row)} & {' & '.join(cells)} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}",
    ]
    tex = "\n".join(lines) + "\n"

    # ── Saving ────────────────────────────────────────────────────────────────
    tex_path = Path(output_tex)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    base = tex_path.with_suffix("") if tex_path.suffix == ".tex" else tex_path
    details_csv = base.parent / f"{base.name}_details.csv"
    friedman_csv = base.parent / f"{base.name}_friedman.csv"
    per_doc_csv = base.parent / f"{base.name}_per_document.csv"
    details.to_csv(details_csv, index=False, encoding="utf-8")
    pd.DataFrame([{
        "unit": unit_name, "N": friedman["N"], "n_units_total": n_units, "n_excluded": n_units - n_ok,
        "k": friedman["k"], "chi2": friedman["chi2"], "df": friedman["df"], "p": friedman["p"],
        "kendall_w": friedman["kendall_w"],
        **{f"mean_rank_{m}": v for m, v in friedman["mean_ranks"].items()},
    }]).to_csv(friedman_csv, index=False, encoding="utf-8")

    per_rows = []
    for m in available:
        for r_i, s in enumerate(model_runs[m]):
            for u_i, u in enumerate(units):
                row = {"unit_index": u_i, "unit": unit_name}
                if group_col is None and patient_col:
                    row[patient_col] = doc_patient.get(u, "")
                row.update({
                    "unit_id": u, "model": m, "run": s,
                    "tp": int(counts[m]["tp"][r_i, u_i]), "fp": int(counts[m]["fp"][r_i, u_i]),
                    "fn": int(counts[m]["fn"][r_i, u_i]),
                    "f1_run": float(f1_runs[m][r_i, u_i]), "f1_unit": float(f1_unit[m][u_i]),
                    "included": bool(keep[u_i]),
                })
                per_rows.append(row)
    per_unit = pd.DataFrame(per_rows)
    per_unit.to_csv(per_doc_csv, index=False, encoding="utf-8")

    print(f"Zapisano tabelę Wilcoxona ({k}×{k}, {unit_name}-level F1, N={n_ok}) → {tex_path}")
    print(f"  szczegóły → {details_csv}")
    print(f"  F1 per jednostka → {per_doc_csv}")
    if not np.isnan(friedman["chi2"]):
        print(f"  Friedman: chi2({friedman['df']})={friedman['chi2']:.2f}, p={friedman['p']:.3g}, "
              f"W Kendalla={friedman['kendall_w']:.3f}")
    for p in pairs:
        print(f"  {p['pair']}: n_eff={p['n_eff']}, ties={p['n_ties']}, p_raw={p['p_raw']:.3g}, "
              f"p_holm={p['p_holm']:.3g}, Δ={p['delta_pp']:+.1f} p.p. "
              f"[{p['ci_low_pp']:.1f}, {p['ci_high_pp']:.1f}], p_boot={p['p_bootstrap']:.3g}")

    return {
        "unit": unit_name,
        "models": available,
        "runs": {m: model_runs[m] for m in available},
        "n_units_total": n_units,
        "n_units": n_ok,
        "n_gold_empty": n_gold_empty,
        "empty": empty_stats,
        "excluded_empty": int(excluded_empty.sum()),
        "excluded_missing": int(excluded_missing.sum()),
        "friedman": friedman,
        "micro_f1": micro,
        "pairs": details,
        "per_unit": per_unit,
        "tex": tex,
        "paths": {"tex": str(tex_path), "details_csv": str(details_csv),
                  "friedman_csv": str(friedman_csv), "per_document_csv": str(per_doc_csv)},
    }


DIFF_NOTE = (
    "Differences are computed from unrounded micro-F1 values; CIs of differences come from the paired "
    "bootstrap and cannot be derived from the per-model CIs in the F1 table."
)


def _wilcoxon_caption(unit, group_col, n_ok, k, zero_method, alpha, empty_doc_policy,
                      empty_stats, friedman, n_boot, n_docs) -> str:
    """Comparison table caption: patient variant (main) or document variant (sensitivity)."""
    import numpy as np

    if group_col is None:
        unit_tex, units_tex = "document", "documents"
    elif unit == "patient":
        unit_tex, units_tex = "patient", "patients"
    else:
        unit_tex, units_tex = f"group (\\texttt{{{unit.replace('_', chr(92) + '_')}}})", "groups"
    n_empty_any = int(max((e["tp_fp_fn_zero"] for e in empty_stats.values()), default=0))
    empty_txt = (
        f"{units_tex.capitalize()} with no gold and no predicted entities were scored F1 $= 1$"
        if empty_doc_policy == "one" else
        f"{units_tex.capitalize()} with no gold and no predicted entities were excluded"
    ) + f" ({n_empty_any} such {units_tex}); those with predictions but no gold entities score F1 $= 0$."
    ties_txt = {
        "wilcox": "zero differences were discarded (Wilcoxon method; Pratt and zero-split methods gave the sensitivity analysis)",
        "pratt": "zero differences were ranked and then dropped (Pratt method)",
        "zsplit": "zero-difference ranks were split between signs",
    }[zero_method]
    if np.isnan(friedman["chi2"]):
        fr_txt = "Friedman test not applicable (fewer than three models)."
    else:
        fr_txt = (f"Friedman test: $\\chi^2({friedman['df']}) = {friedman['chi2']:.2f}$, "
                  f"{_fmt_p_text(friedman['p'])}, Kendall's $W = {friedman['kendall_w']:.3f}$.")
    f1_def = "$\\mathrm{F1} = 2\\mathrm{TP}/(2\\mathrm{TP}+\\mathrm{FP}+\\mathrm{FN})$"
    if unit == "patient":
        return (
            "Pairwise comparison of LLM extraction models. "
            "Upper triangle: Holm-adjusted $p$-values of two-sided Wilcoxon signed-rank tests on paired "
            f"patient-level F1 ({f1_def} with TP, FP, FN summed over all visits of a patient); "
            f"unit of analysis: patient, $N = {n_ok}$ (patient-level micro-F1 aggregated over all visits); "
            f"{k * (k - 1) // 2} comparisons; {ties_txt}; "
            f"bold: $p_{{\\mathrm{{Holm}}}} < {alpha:g}$. {empty_txt} "
            "Lower triangle: difference in corpus micro-F1 (TP, FP, FN summed over all "
            f"{n_docs} documents; row $-$ column, percentage points) with 95\\% percentile CI from a paired "
            f"cluster bootstrap resampling patients with all their visits ({n_boot} resamples). "
            f"{DIFF_NOTE} "
            f"{fr_txt}"
        )
    return (
        "Pairwise comparison of LLM extraction models. "
        "Upper triangle: Holm-adjusted $p$-values of two-sided Wilcoxon signed-rank tests on paired "
        f"{unit_tex}-level F1 ({f1_def} per {unit_tex}; unit of analysis: {unit_tex}, $N = {n_ok}$; "
        f"{k * (k - 1) // 2} comparisons); {ties_txt}; "
        f"bold: $p_{{\\mathrm{{Holm}}}} < {alpha:g}$. {empty_txt} "
        "Lower triangle: difference in corpus micro-F1 (TP, FP, FN summed over all "
        f"{units_tex}; row $-$ column, percentage points) with 95\\% percentile CI from a paired "
        f"bootstrap resampling whole {units_tex} with replacement ({n_boot} resamples). "
        f"{DIFF_NOTE} "
        f"{fr_txt}"
        + (" Sensitivity analysis: visits of the same patient treated as independent." if group_col is None else "")
    )


def evaluate_json_validity_to_latex(
    csv_path: str = DEFAULT_CSV,
    output_dir: str = "outputs_v5",
    output_tex: str = "results/json_validity.tex",
    models: list[str] | None = None,
    label: str = "tab:json_validity",
) -> "pd.DataFrame":
    """JSON validity per model on GT documents (D8: skeleton after JSONDecodeError,
    matching.is_invalid_output) + number of duplicated output records (D6).
    Writes `output_tex` and a CSV next to it."""
    import pandas as pd

    if models is None:
        models = DEFAULT_MODELS
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))
    gold, results = _evaluate(gt_rows, models, output_dir)

    rows = []
    for model, res in results.items():
        n_docs = len(gold.docs)
        n_inv = len(res.invalid_docs)
        n_gt_inv = sum(len(r.fn) for r in res.iter_results() if r.opis in res.invalid_docs)
        rows.append({
            "model": model, "n_documents": n_docs, "n_invalid_json": n_inv,
            "valid_json_pct": 100 * (n_docs - n_inv) / n_docs if n_docs else float("nan"),
            "n_gt_entities_in_invalid_docs": n_gt_inv,
            "pct_gt_entities_in_invalid_docs": 100 * n_gt_inv / gold.n_gt() if gold.n_gt() else float("nan"),
            "n_missing_documents": len(res.missing_docs),
            "n_duplicate_records": len(res.duplicate_opis),
            "n_conflicting_duplicates": len(res.conflicting_opis),
        })
    df = pd.DataFrame(rows)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Output validity per model over the "
        f"{len(gold.docs)} evaluated documents. Invalid: the response could not be parsed as JSON; "
        "the document stays in the evaluation and all its gold entities count as false negatives "
        "(\\textit{invalid\\_output}).}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Invalid JSON (docs) & Valid JSON (\\%) & GT entities in invalid docs (\\%) \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"{MODEL_DISPLAY.get(r['model'], r['model'])} & {r['n_invalid_json']} & "
                     f"{r['valid_json_pct']:.1f} & {r['pct_gt_entities_in_invalid_docs']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(output_tex).write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_csv(Path(output_tex).with_suffix(".csv"), index=False)
    print(f"Zapisano tabelę poprawności JSON → {output_tex}")
    return df


def write_counts_csv(
    csv_path: str = DEFAULT_CSV,
    output_dir: str = "outputs_v5",
    output_csv: str = "results/counts_per_document.csv",
    models: list[str] | None = None,
) -> None:
    """Counts per (model, document, category) from matching.py — the source of all tables."""
    if models is None:
        models = DEFAULT_MODELS
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))
    gold, results = _evaluate(gt_rows, models, output_dir)
    rows = [row for res in results.values() for row in matching.counts_rows(res, gold.patient)]
    matching.write_counts_csv(rows, output_csv)
    print(f"Zapisano zliczenia ({len(rows)} wierszy) → {output_csv}")


def unit_comparison_rows(prf: dict, wil: dict, alpha: float = sc.ALPHA) -> list[dict]:
    """unit_comparison rows for validation.py: F1 CI from the P/R/F1 table, Friedman, pairwise Wilcoxon
    and Δ micro-F1 (bootstrap) — document and patient variants side by side."""
    import numpy as np

    rows: list[dict] = []
    d, p = wil["document"], wil["patient"]
    rows.append({
        "source": "validation", "test": "Friedman (document/patient F1; p unadjusted, single test)",
        "item": "all models", "statistic_name": "chi2",
        "stat_document": d["friedman"]["chi2"], "stat_patient": p["friedman"]["chi2"],
        "p_holm_document": d["friedman"]["p"], "p_holm_patient": p["friedman"]["p"],
        "kendall_w_document": d["friedman"]["kendall_w"], "kendall_w_patient": p["friedman"]["kendall_w"],
        "N_document": d["n_units"], "N_patient": p["n_units"],
        "excluded_document": d["n_units_total"] - d["n_units"], "excluded_patient": p["n_units_total"] - p["n_units"],
        "significant_document": d["friedman"]["p"] < alpha, "significant_patient": p["friedman"]["p"] < alpha,
    })
    pd_ = d["pairs"].set_index("pair")
    pp_ = p["pairs"].set_index("pair")
    for pair in pd_.index:
        a, b = pd_.loc[pair], pp_.loc[pair]
        common = {"source": "validation", "item": pair, "N_document": d["n_units"], "N_patient": p["n_units"],
                  "excluded_document": d["n_units_total"] - d["n_units"],
                  "excluded_patient": p["n_units_total"] - p["n_units"]}
        rows.append({
            **common, "test": "Wilcoxon signed-rank on unit F1 (Holm, 15 comparisons)", "statistic_name": "W",
            "stat_document": a["W"], "stat_patient": b["W"],
            "p_holm_document": a["p_holm"], "p_holm_patient": b["p_holm"],
            "effect_document": a["r"], "effect_patient": b["r"],
            "r_rb_document": a["r_rb"], "r_rb_patient": b["r_rb"],
            "significant_document": a["p_holm"] < alpha, "significant_patient": b["p_holm"] < alpha,
        })
        rows.append({
            **common, "test": "Delta micro-F1, paired bootstrap 95% CI (pp)", "statistic_name": "delta_pp",
            "stat_document": a["delta_pp"], "stat_patient": b["delta_pp"],
            "ci_document": sc.fmt_ci((a["ci_low_pp"], a["ci_high_pp"])),
            "ci_patient": sc.fmt_ci((b["ci_low_pp"], b["ci_high_pp"])),
            "significant_document": a["ci_low_pp"] > 0 or a["ci_high_pp"] < 0,
            "significant_patient": b["ci_low_pp"] > 0 or b["ci_high_pp"] < 0,
        })
    fd = prf["document"].set_index(["row", "model"])
    fp = prf["patient"].set_index(["row", "model"])
    for key in fd.index:
        a, b = fd.loc[key], fp.loc[key]
        rows.append({
            "source": "validation", "test": "F1 95% CI (estimate, no test)", "item": f"{key[0]} / {key[1]}",
            "statistic_name": "F1", "stat_document": a["F1"], "stat_patient": b["F1"],
            "ci_document": sc.fmt_ci((a["ci_low"], a["ci_high"])), "ci_patient": sc.fmt_ci((b["ci_low"], b["ci_high"])),
            "N_document": int(a["n_documents"]), "N_patient": int(b["n_patients"]),
        })
    return rows


def _tabular_body_without_ci(tex: str) -> list[str]:
    import re

    lines = tex.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("\\begin{tabular}"))
    end = next(i for i, l in enumerate(lines) if l.startswith("\\end{tabular}"))
    return [re.sub(r" \\ci\{[^}]*\}\{[^}]*\}", "", l) for l in lines[start:end + 1]]


def check_against_baseline(results_dir: Path, baseline_dir: Path | None, prf: dict, wil: dict, n_patients: int) -> list[str]:
    """Phase 3 assertions: point estimates identical across units and to the results before the change;
    number of patients in the tests = n_patients − exclusions. Returns the list of checks performed."""
    import numpy as np
    import pandas as pd

    done: list[str] = []
    point = ["tp", "fp", "fn", "F1", "P", "R"]
    if "patient" in prf and "document" in prf:
        a = prf["patient"][["row", "model"] + point].set_index(["row", "model"])
        b = prf["document"][["row", "model"] + point].set_index(["row", "model"])
        pd.testing.assert_frame_equal(a, b)
        done.append("P/R/F1 (per kategoria, Micro, Macro) identyczne: pacjent == dokument")
    for u, res in wil.items():
        if u == "patient":
            if res["n_units_total"] != n_patients:
                raise AssertionError(f"jednostek pacjentowych {res['n_units_total']} ≠ {n_patients} pacjentów w GT")
            if res["n_units"] != n_patients - res["excluded_empty"] - res["excluded_missing"]:
                raise AssertionError("N pacjentów w teście ≠ 100 − raportowane wykluczenia")
            done.append(f"Wilcoxon/Friedman F1 (pacjent): N = {res['n_units']} = {n_patients} − "
                        f"{res['excluded_empty'] + res['excluded_missing']} wykluczeń")
    if "patient" in wil and "document" in wil:
        a, b = wil["patient"]["pairs"].set_index("pair"), wil["document"]["pairs"].set_index("pair")
        for c in ("micro_f1_a", "micro_f1_b", "delta_pp"):
            np.testing.assert_allclose(a[c], b[c], rtol=0, atol=1e-12)
        done.append("micro-F1 i Δ micro-F1 identyczne: pacjent == dokument")

    if baseline_dir is None or not baseline_dir.exists():
        print(f"[UWAGA] Brak katalogu wyników bazowych {baseline_dir} — pominięto porównanie z wynikami sprzed zmiany")
        return done
    base_tex = baseline_dir / "evaluation_table.tex"
    if base_tex.exists():
        base_body = _tabular_body_without_ci(base_tex.read_text(encoding="utf-8"))
        for u in prf:
            new_body = _tabular_body_without_ci((results_dir / f"evaluation_table_{u}.tex").read_text(encoding="utf-8"))
            if new_body != base_body:
                raise AssertionError(f"evaluation_table_{u}.tex: wartości F1/P/R ≠ {base_tex}")
        done.append(f"tabela P/R/F1 (bez CI) identyczna z {base_tex}")
    base_counts = baseline_dir / "counts_per_document.csv"
    if base_counts.exists():
        a = pd.read_csv(base_counts).sort_values(["model", "opis", "category"]).reset_index(drop=True)
        b = pd.read_csv(results_dir / "counts_per_document.csv").sort_values(["model", "opis", "category"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        done.append(f"zliczenia per (model, dokument, kategoria) identyczne z {base_counts}")
    base_det = baseline_dir / "wilcoxon_table_details.csv"
    if base_det.exists() and "document" in wil:
        a = pd.read_csv(base_det).set_index("pair")
        b = wil["document"]["pairs"].set_index("pair").loc[a.index]
        for c in ("micro_f1_a", "micro_f1_b", "delta_pp"):
            np.testing.assert_allclose(b[c], a[c], rtol=0, atol=1e-9)
        done.append(f"micro-F1 i Δ identyczne z {base_det}")
        # same method and same samples → the document-level test reproduces the baseline results
        for c in ("n_eff", "n_ties", "W", "p_raw", "p_holm", "p_pratt", "ci_low_pp", "ci_high_pp", "p_bootstrap"):
            np.testing.assert_allclose(b[c], a[c], rtol=1e-9, atol=1e-12, err_msg=c)
        done.append("wariant dokumentowy odtwarza bazowe W, p_raw, p_Holm, p_Pratt i CI Δ")
    return done


def main(argv: list[str] | None = None) -> None:
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description="Ewaluacja modeli: P/R/F1, porównania par, poprawność JSON.")
    ap.add_argument("--gold-csv", default=DEFAULT_CSV)
    ap.add_argument("--outputs-dir", default="outputs_v5", help="katalog z {model}_200_patients.json")
    ap.add_argument("--results-dir", default="results_v5_clustered_d9", help="katalog wyników (nowy, nie nadpisuje starych)")
    ap.add_argument("--baseline-dir", default="",
                    help="wyniki sprzed zmiany do asercji estymat punktowych ('' = bez porównania)")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--unit", choices=sc.UNITS, default="patient", help="główna jednostka analizy")
    ap.add_argument("--sensitivity", action=argparse.BooleanOptionalAction, default=True,
                    help="dodatkowo druga jednostka (dokument przy --unit patient) jako analiza wrażliwości")
    ap.add_argument("--n-boot", type=int, default=sc.N_BOOT)
    ap.add_argument("--seed", type=int, default=sc.SEED)
    args = ap.parse_args(argv)

    models = args.models or DEFAULT_MODELS
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    units = [args.unit] + ([u for u in sc.UNITS if u != args.unit] if args.sensitivity else [])
    common = {"csv_path": args.gold_csv, "output_dir": args.outputs_dir, "models": models}

    evaluate_models(**common)
    evaluate_entities_to_csv(output_csv=str(out / "entity_accuracy.csv"), **common)
    prf, wil = {}, {}
    for u in units:
        primary = u == args.unit
        prf[u] = evaluate_models_to_latex(
            output_tex=str(out / f"evaluation_table_{u}.tex"), unit=u, n_boot=args.n_boot, seed=args.seed,
            label="tab:annotation_metrics" if primary else f"tab:annotation_metrics_{u}", **common)
        wil[u] = evaluate_wilcoxon_to_latex(
            output_tex=str(out / f"wilcoxon_table_{u}.tex"), unit=u, n_boot=args.n_boot, seed=args.seed,
            label="tab:wilcoxon" if primary else f"tab:wilcoxon_{u}", **common)
    evaluate_json_validity_to_latex(output_tex=str(out / "json_validity.tex"), **common)
    write_counts_csv(output_csv=str(out / "counts_per_document.csv"), **common)

    icc = icc_f1_by_model(n_boot=args.n_boot, seed=args.seed, **common)
    icc.to_csv(out / "icc_validation.csv", index=False)
    print("ICC(1) F1 dokumentów w obrębie pacjenta:")
    for _, r in icc.iterrows():
        print(f"  {r['model']:<16} ICC={r['icc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  n0={r['n0']:.2f}")

    gold = matching.load_gold(args.gold_csv)
    checks = check_against_baseline(out, Path(args.baseline_dir) if args.baseline_dir else None, prf, wil, len(set(gold.patient.values())))
    print("Asercje:\n  " + "\n  ".join(checks))

    if set(units) == set(sc.UNITS):
        sc.write_unit_comparison_part(unit_comparison_rows(prf, wil), out / "unit_comparison_validation.csv")
    sc.merge_unit_comparison(out, [out / "unit_comparison_validation.csv", out / "unit_comparison_error_analysis.csv"])
    sc.merge_icc_report(out, [out / "icc_validation.csv", out / "icc_error_analysis.csv"])


if __name__ == "__main__":
    main()
