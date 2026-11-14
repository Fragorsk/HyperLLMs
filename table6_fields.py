"""Table 6: sensitivity (Se), PPV and F1 per field for comorbidities and symptoms.

Replaces the per-field accuracy (``error_patterns.field_accuracy_table``), which is dominated by
true negatives. TP/FP/FN counts come directly from ``matching.evaluate_model_file``
(the same matches as Table 2), grouped by ``GoldItem.key`` / ``PredItem.key`` —
without a separate binarization: ``present: true`` → prediction, ``false``/``null`` → no prediction,
negation_error_fp → FP, negation_error_fn → FN (D1, D2 in ``matching.py``).

Outputs (in ``--results-dir``):
    table6_fields_main.{tex,csv}   — main text: n GT (prevalence), F1 per model, Mean
    tableS_fields_full.{tex,csv}   — supplementary materials: Se/PPV/F1 with 95% CI, TP/FP/FN
    table6_accuracy_legacy.csv     — the previous accuracy (for comparison only)

Assertions (abort the run): Σ fields == Table 2 category row (TP/FP/FN, F1/P/R,
F1 CI), Σ GT entities == Table 1 counts, accuracy from these counts == field_accuracy.csv.

Usage:
    python table6_fields.py [--gold-csv ...] [--outputs-dir outputs_v5]
                            [--results-dir results_v5_clustered_d9]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matching
import stats_clustered as sc
from error_analysis import DEFAULT_CSV, DEFAULT_MODELS
from error_patterns import FIELD_DISPLAY_NAMES
from validation import MODEL_DISPLAY

SMALL_N = 10  # fields with n GT < SMALL_N: raw TP/(TP+FN) next to F1, plus a footnote
UNIT = "patient"

# Extraction-schema fields excluded from evaluation (D2) → positive GT label (for the footnote)
NOT_EVALUATED_LABELS = {
    "asthma": "Astma", "smoking": "Palenie", "syncope": "Omdlenia", "palpitations": "Kołatanie serca",
}
NOT_EVALUATED_DISPLAY = {
    "asthma": "asthma", "smoking": "smoking", "syncope": "syncope", "palpitations": "palpitations",
}


def evaluated_fields() -> list[tuple[str, str]]:
    return [(cat, f) for cat in matching.BINARY_CATEGORIES for f in matching.EVALUATED_FIELDS[cat]]


# ─── Counts ────────────────────────────────────────────────────────────────────

def field_count_array(results: dict[str, matching.ModelResult], docs: Sequence[str],
                      fields: Sequence[tuple[str, str]]) -> np.ndarray:
    """(fields, models, TP/FP/FN, documents) from the shared matching results."""
    f_idx = {f: i for i, f in enumerate(fields)}
    arr = np.zeros((len(fields), len(results), 3, len(docs)))
    for m_i, res in enumerate(results.values()):
        for d_i, opis in enumerate(docs):
            for cat in matching.BINARY_CATEGORIES:
                r = res.docs[opis][cat]
                for g, _ in r.tp:
                    arr[f_idx[(cat, g.key)], m_i, 0, d_i] += 1
                for p, _ in r.fp:
                    arr[f_idx[(cat, p.key)], m_i, 1, d_i] += 1
                for g, _ in r.fn:
                    arr[f_idx[(cat, g.key)], m_i, 2, d_i] += 1
    return arr


def prevalence(gold: matching.GoldStandard, fields: Sequence[tuple[str, str]]) -> dict[tuple[str, str], tuple[int, int]]:
    """(category, field) → (number of GT entities, number of documents with ≥1 GT entity)."""
    n_ent: dict[tuple[str, str], int] = defaultdict(int)
    n_doc: dict[tuple[str, str], int] = defaultdict(int)
    for cats in gold.items.values():
        for cat in matching.BINARY_CATEGORIES:
            keys = [g.key for g in cats.get(cat, [])]
            for k in keys:
                n_ent[(cat, k)] += 1
            for k in set(keys):
                n_doc[(cat, k)] += 1
    return {f: (n_ent[f], n_doc[f]) for f in fields}


def field_metrics(tp, fp, fn):
    """Se, PPV, F1 (tablice; NaN = n/a). TP+FN = 0 → Se i F1 n/a; TP+FP = 0 → PPV n/a."""
    tp, fp, fn = (np.asarray(a, dtype=float) for a in (tp, fp, fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        ppv = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        f1 = np.where(tp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return se, ppv, f1


def _nan_to_none(x: float) -> Optional[float]:
    return None if np.isnan(x) else float(x)


def field_table(gold: matching.GoldStandard, results: dict[str, matching.ModelResult],
                n_boot: int = sc.N_BOOT, seed: int = sc.SEED, unit: str = UNIT) -> pd.DataFrame:
    """One row per (field, model): TP/FP/FN, Se/PPV/F1 with 95% CI, prevalence."""
    fields = evaluated_fields()
    docs = gold.docs
    arr = field_count_array(results, docs, fields)
    clusters = np.array([gold.patient[d] for d in docs])
    boot = sc.bootstrap_sums(arr, unit, clusters, n_boot, seed)  # (n_boot, F, M, 3)
    b_se, b_ppv, b_f1 = field_metrics(boot[..., 0], boot[..., 1], boot[..., 2])
    sums = arr.sum(axis=-1)
    se, ppv, f1 = field_metrics(sums[..., 0], sums[..., 1], sums[..., 2])
    prev = prevalence(gold, fields)

    rows = []
    for f_i, (cat, key) in enumerate(fields):
        n_gt, n_docs = prev[(cat, key)]
        for m_i, model in enumerate(results):
            row = {
                "category": cat, "field": key, "field_display": FIELD_DISPLAY_NAMES.get(key, key),
                "model": model, "n_gt": n_gt, "n_docs_gt": n_docs,
                "prevalence_pct": 100 * n_docs / len(docs), "n_documents": len(docs),
                "tp": int(sums[f_i, m_i, 0]), "fp": int(sums[f_i, m_i, 1]), "fn": int(sums[f_i, m_i, 2]),
            }
            for name, point, b in (("se", se, b_se), ("ppv", ppv, b_ppv), ("f1", f1, b_f1)):
                ci = sc.percentile_ci(b[:, f_i, m_i]) if not np.isnan(point[f_i, m_i]) else None
                row[name] = _nan_to_none(point[f_i, m_i])
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci if ci else (None, None)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs.update(n_boot=n_boot, seed=seed, unit=unit, n_patients=len(set(clusters)), n_documents=len(docs))
    df.attrs["boot_category_sums"] = {
        cat: boot[:, [i for i, (c, _) in enumerate(fields) if c == cat]].sum(axis=1)  # (n_boot, M, 3)
        for cat in matching.BINARY_CATEGORIES
    }
    return df


def legacy_accuracy(table: pd.DataFrame) -> pd.DataFrame:
    """Previous accuracy = 1 − (FP + FN) / number of documents (field_accuracy.csv)."""
    t = table.assign(accuracy=1 - (table["fp"] + table["fn"]) / table["n_documents"])
    return t.pivot_table(index=["category", "field", "field_display"], columns="model",
                         values="accuracy", sort=False)[list(table["model"].unique())]


# ─── Assertions ────────────────────────────────────────────────────────────────

def category_counts(results: dict[str, matching.ModelResult]) -> dict[tuple[str, str], tuple[int, int, int]]:
    out: dict[tuple[str, str], tuple[int, int, int]] = {}
    for model, res in results.items():
        for cat in matching.BINARY_CATEGORIES:
            rs = [res.docs[o][cat] for o in res.docs]
            out[(cat, model)] = (sum(len(r.tp) for r in rs), sum(len(r.fp) for r in rs), sum(len(r.fn) for r in rs))
    return out


def check_category_sums(table: pd.DataFrame, results: dict[str, matching.ModelResult],
                        gold: matching.GoldStandard) -> None:
    """Σ fields == category counts from the shared matching; Σ GT entities == category n_gt."""
    cat_counts = category_counts(results)
    for (cat, model), sub in table.groupby(["category", "model"]):
        got = tuple(int(sub[k].sum()) for k in ("tp", "fp", "fn"))
        if got != cat_counts[(cat, model)]:
            raise AssertionError(f"{model}/{cat}: Σ pól TP/FP/FN {got} ≠ kategoria {cat_counts[(cat, model)]}")
        p, r, f1 = matching.prf(*got)
        _, ppv_s, f1_s = field_metrics(*got)
        if not (np.isclose(ppv_s, p if p is not None else np.nan, equal_nan=True)
                and np.isclose(f1_s, f1 if f1 is not None else np.nan, equal_nan=True)):
            raise AssertionError(f"{model}/{cat}: metryki z sum pól ≠ metryki kategorii")
    for cat in matching.BINARY_CATEGORIES:
        n = int(table[(table["category"] == cat)].drop_duplicates("field")["n_gt"].sum())
        if n != gold.n_gt(cat):
            raise AssertionError(f"{cat}: Σ encji GT po polach {n} ≠ {gold.n_gt(cat)}")


def check_against_tables(table: pd.DataFrame, results_dir: Path) -> list[str]:
    """Comparison with the saved Tables 1, 2 and the previous Table 6. Returns the list of checks performed."""
    done: list[str] = []
    unit = table.attrs["unit"]
    t2_path = results_dir / f"evaluation_table_{unit}.csv"
    if t2_path.exists():
        t2 = pd.read_csv(t2_path).set_index(["row", "model"])
        boot_sums = table.attrs["boot_category_sums"]
        models = list(table["model"].unique())
        for (cat, model), sub in table.groupby(["category", "model"]):
            ref = t2.loc[(cat, model)]
            tp, fp, fn = (int(sub[k].sum()) for k in ("tp", "fp", "fn"))
            if (tp, fp, fn) != (int(ref["tp"]), int(ref["fp"]), int(ref["fn"])):
                raise AssertionError(f"{model}/{cat}: Σ pól ({tp}, {fp}, {fn}) ≠ Tabela 2 {t2_path}")
            se, ppv, f1 = (float(x) for x in field_metrics(tp, fp, fn))
            if not np.allclose([se, ppv, f1], [ref["R"], ref["P"], ref["F1"]], rtol=0, atol=1e-12):
                raise AssertionError(f"{model}/{cat}: Se/PPV/F1 z sum pól ≠ R/P/F1 Tabeli 2")
            b = boot_sums[cat][:, models.index(model)]
            ci = sc.percentile_ci(sc.f1_from_sums(b[:, 0], b[:, 1], b[:, 2]))
            if not np.allclose(ci, [ref["ci_low"], ref["ci_high"]], rtol=0, atol=1e-12):
                raise AssertionError(f"{model}/{cat}: CI F1 z sum pól {ci} ≠ Tabela 2 — inne próbki bootstrapu?")
        done.append(f"Σ pól == Tabela 2 (TP/FP/FN, F1/P/R, 95% CI F1): {t2_path}")

    t1_path = results_dir / "entity_accuracy.csv"
    if t1_path.exists():
        t1 = pd.read_csv(t1_path)
        for cat in matching.BINARY_CATEGORIES:
            ref = int(t1.loc[t1["Kategoria"] == cat, "Count_GT"].sum())
            n = int(table[table["category"] == cat].drop_duplicates("field")["n_gt"].sum())
            if n != ref:
                raise AssertionError(f"{cat}: Σ encji GT po polach {n} ≠ Tabela 1 {ref} ({t1_path})")
        done.append(f"Σ encji GT == Tabela 1: {t1_path}")

    legacy_path = results_dir / "error_analysis" / "tables" / "field_accuracy.csv"
    if legacy_path.exists():
        ref = pd.read_csv(legacy_path).set_index(["Category", "Field"])
        acc = legacy_accuracy(table).reset_index()
        for _, row in acc.iterrows():
            old = ref.loc[(row["category"], row["field_display"])]
            for m in table["model"].unique():
                if not np.isclose(row[m], old[m], rtol=0, atol=1e-12):
                    raise AssertionError(f"{m}/{row['field']}: accuracy z nowych zliczeń ≠ {legacy_path}")
        done.append(f"accuracy z nowych zliczeń == dotychczasowa Tabela 6: {legacy_path}")
    return done


# ─── Tables ────────────────────────────────────────────────────────────────────

def main_table(table: pd.DataFrame) -> pd.DataFrame:
    """Row per field: n GT, prevalence, F1 and TP per model, Mean F1; sorted within the block
    ascending by Mean F1, zero-prevalence fields at the end of the block."""
    models = list(table["model"].unique())
    idx = ["category", "field", "field_display", "n_gt", "n_docs_gt", "prevalence_pct"]
    rows = []
    for key, sub in table.groupby(idx, sort=False):
        row = dict(zip(idx, key))
        by_model = sub.set_index("model")
        for m in models:
            row[f"f1_{m}"] = by_model.loc[m, "f1"]
            for k in ("tp", "fp", "fn"):
                row[f"{k}_{m}"] = int(by_model.loc[m, k])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["mean_f1"] = out[[f"f1_{m}" for m in models]].mean(axis=1, skipna=True)
    out["small_n"] = (out["n_gt"] > 0) & (out["n_gt"] < SMALL_N)
    out["zero_prevalence"] = out["n_gt"] == 0
    cat_order = {c: i for i, c in enumerate(matching.BINARY_CATEGORIES)}
    out = out.sort_values(
        ["category", "zero_prevalence", "mean_f1"],
        key=lambda s: s.map(cat_order) if s.name == "category" else s,
        na_position="last",
    ).reset_index(drop=True)
    return out


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}"


def raw_not_evaluated_counts(gold_csv: str | Path) -> dict[str, int]:
    """Number of documents with a positive GT label for excluded fields (D2) — for the footnote only."""
    docs: dict[str, set] = defaultdict(set)
    with open(gold_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            docs[row["Label"]].add(row["OPIS"])
    return {k: len(docs[label]) for k, label in NOT_EVALUATED_LABELS.items()}


def main_table_tex(main: pd.DataFrame, models: list[str], attrs: dict, not_evaluated: dict[str, int]) -> str:
    n_docs, n_pat, n_boot = attrs["n_documents"], attrs["n_patients"], attrs["n_boot"]
    excluded = ", ".join(f"{NOT_EVALUATED_DISPLAY[k]} ({n})" for k, n in not_evaluated.items())
    caption = (
        "Per-field extraction performance for comorbidities and symptoms: F1 (\\%) per model. "
        "For each field and visit, a model prediction is positive only if the field is reported as "
        "present (\\texttt{present} = true); \\texttt{false} or \\texttt{null} is negative. "
        "TP, FP and FN come from the same matching as Table~2, so a field reported present although "
        "the note negates it counts as FP and a gold-positive field reported as absent counts as FN; "
        "per-field TP, FP and FN sum to the Comorbidities and Symptoms rows of Table~2. "
        "Sensitivity $\\mathrm{Se} = \\mathrm{TP}/(\\mathrm{TP}+\\mathrm{FN})$, "
        "positive predictive value $\\mathrm{PPV} = \\mathrm{TP}/(\\mathrm{TP}+\\mathrm{FP})$, "
        "$\\mathrm{F1} = 2\\mathrm{TP}/(2\\mathrm{TP}+\\mathrm{FP}+\\mathrm{FN})$; true negatives do not "
        "enter any of these measures. Se, PPV and F1 with 95\\% CIs and TP/FP/FN counts are given in "
        "Supplementary Table~\\ref{tab:field_metrics_full}; CIs come from a paired cluster bootstrap "
        f"resampling patients with all their visits ($N = {n_pat}$ patients, {n_docs} visits; "
        f"{n_boot} resamples, percentile intervals). "
        f"$n_\\mathrm{{GT}}$: gold-standard entities; in parentheses, the percentage of the {n_docs} "
        "visits with the field present in the gold standard. Mean: unweighted mean of the per-model F1 "
        "(n/a values omitted). Fields sorted from hardest to easiest (ascending mean F1) within each block. "
        f"$^{{a}}$Fewer than {SMALL_N} gold-standard entities: raw TP/(TP+FN) shown next to F1; "
        "estimates are unstable. "
        "n/a: undefined because TP+FN = 0 (no gold-standard entity); for such fields the number of "
        "false-positive predictions is shown instead. "
        "Fields outside the evaluation schema are not scored in Table~2 and are omitted "
        f"(number of visits with a positive gold-standard label: {excluded})."
    )
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:field_metrics}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lr" + "c" * len(models) + "c}",
        "\\toprule",
        "Field & $n_\\mathrm{GT}$ (\\%) & " + " & ".join(MODEL_DISPLAY.get(m, m) for m in models) + " & Mean \\\\",
        "\\midrule",
    ]
    for cat in matching.BINARY_CATEGORIES:
        rows = main[main["category"] == cat]
        if rows.empty:
            continue
        lines.append(f"\\multicolumn{{{len(models) + 3}}}{{l}}{{\\textit{{{cat}}}}} \\\\")
        for _, r in rows.iterrows():
            name = r["field_display"] + ("$^{a}$" if r["small_n"] else "")
            cells = []
            for m in models:
                if r["zero_prevalence"]:
                    cells.append(f"n/a (FP = {r[f'fp_{m}']})")
                elif r["small_n"]:
                    cells.append(f"{_pct(r[f'f1_{m}'])} ({r[f'tp_{m}']}/{r[f'tp_{m}'] + r[f'fn_{m}']})")
                else:
                    cells.append(_pct(r[f"f1_{m}"]))
            lines.append(f"{name} & {r['n_gt']} ({r['prevalence_pct']:.1f}) & " + " & ".join(cells)
                         + f" & {_pct(r['mean_f1'])} \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def full_table_tex(table: pd.DataFrame, main: pd.DataFrame, attrs: dict) -> str:
    """Supplementary materials (longtable): Se, PPV, F1 with 95% CI and TP/FP/FN per field and model."""
    def _ci(r: pd.Series, k: str) -> str:
        if r[k] is None or pd.isna(r[k]):
            return "n/a"
        lo, hi = r[f"{k}_ci_low"], r[f"{k}_ci_high"]
        return f"{r[k] * 100:.1f}" + ("" if lo is None or pd.isna(lo) else f" \\ci{{{lo * 100:.1f}}}{{{hi * 100:.1f}}}")

    caption = (
        "Per-field sensitivity (Se), positive predictive value (PPV) and F1 (\\%, with 95\\% CI), "
        "and TP/FP/FN counts, for comorbidities and symptoms, per model. Definitions, the "
        "positive/negative rule and the handling of negations are as in Table~\\ref{tab:field_metrics}. "
        f"95\\% CI: paired cluster bootstrap resampling patients with all their visits "
        f"($N = {attrs['n_patients']}$ patients, {attrs['n_documents']} visits; {attrs['n_boot']} resamples, "
        "percentile intervals; the same resamples as Table~2). n/a: undefined (Se and F1 when TP+FN = 0, "
        "PPV when TP+FP = 0); no CI is shown when fewer than half of the resamples give a defined value. "
        "$n_\\mathrm{GT}$: gold-standard entities (percentage of visits with the field present)."
    )
    models = list(table["model"].unique())
    head = ("Field & Model & Se & PPV & F1 & TP & FP & FN \\\\")
    lines = [
        "\\begin{small}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{longtable}{llcccrrr}",
        f"\\caption{{{caption}}}\\label{{tab:field_metrics_full}} \\\\",
        "\\toprule", head, "\\midrule", "\\endfirsthead",
        "\\toprule", head, "\\midrule", "\\endhead",
        "\\bottomrule", "\\endfoot",
    ]
    t = table.set_index(["category", "field", "model"])
    for cat in matching.BINARY_CATEGORIES:
        rows = main[main["category"] == cat]
        if rows.empty:
            continue
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{cat}}}}} \\\\*")
        for _, f in rows.iterrows():
            for i, m in enumerate(models):
                r = t.loc[(cat, f["field"], m)]
                first = (f"{f['field_display']} ($n_\\mathrm{{GT}} = {f['n_gt']}$, {f['prevalence_pct']:.1f}\\%)"
                         if i == 0 else "")
                end = " \\\\*" if i < len(models) - 1 else " \\\\"
                lines.append(f"{first} & {MODEL_DISPLAY.get(m, m)} & {_ci(r, 'se')} & {_ci(r, 'ppv')} & "
                             f"{_ci(r, 'f1')} & {r['tp']} & {r['fp']} & {r['fn']}{end}")
            lines.append("\\addlinespace")
    if lines[-1] == "\\addlinespace":
        lines.pop()
    lines += ["\\end{longtable}", "\\end{small}"]
    return "\n".join(lines) + "\n"


# ─── Runner ───────────────────────────────────────────────────────────────────

def run(gold_csv: str, outputs_dir: str, results_dir: str, models: Optional[list[str]] = None,
        n_boot: int = sc.N_BOOT, seed: int = sc.SEED) -> pd.DataFrame:
    models = models or DEFAULT_MODELS
    gold = matching.load_gold(gold_csv)
    results = {}
    for m in models:
        path = Path(outputs_dir) / f"{m}_200_patients.json"
        if not path.exists():
            print(f"[POMINIĘTO] {m}: {path} not found.")
            continue
        results[m] = matching.evaluate_model_file(gold, path, m)

    table = field_table(gold, results, n_boot, seed)
    check_category_sums(table, results, gold)
    out = Path(results_dir)
    for msg in check_against_tables(table, out):
        print(f"[OK] {msg}")

    main = main_table(table)
    available = list(results)
    not_evaluated = raw_not_evaluated_counts(gold_csv)

    full_cols = ["category", "field", "field_display", "n_gt", "n_docs_gt", "prevalence_pct", "model",
                 "tp", "fp", "fn"] + [f"{k}{s}" for k in ("se", "ppv", "f1") for s in ("", "_ci_low", "_ci_high")]
    order = {f: i for i, f in enumerate(main["field"])}
    full = table.sort_values(["category", "field"], key=lambda s: s.map(order) if s.name == "field" else s)
    full = full.assign(n_boot=n_boot, seed=seed, unit=UNIT)[full_cols + ["n_boot", "seed", "unit"]]

    out.mkdir(parents=True, exist_ok=True)
    main.to_csv(out / "table6_fields_main.csv", index=False)
    (out / "table6_fields_main.tex").write_text(main_table_tex(main, available, table.attrs, not_evaluated),
                                               encoding="utf-8")
    full.to_csv(out / "tableS_fields_full.csv", index=False)
    (out / "tableS_fields_full.tex").write_text(full_table_tex(table, main, table.attrs), encoding="utf-8")
    legacy_accuracy(table).reset_index().to_csv(out / "table6_accuracy_legacy.csv", index=False)
    print(f"Wrote table6_fields_main.{{csv,tex}}, tableS_fields_full.{{csv,tex}}, "
          f"table6_accuracy_legacy.csv -> {out.resolve()}")
    return table


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-field Se/PPV/F1 for comorbidities and symptoms (Table 6)")
    p.add_argument("--gold-csv", default=DEFAULT_CSV)
    p.add_argument("--outputs-dir", default="outputs_v5", help="Directory with {model}_200_patients.json")
    p.add_argument("--results-dir", default="results_v5_clustered_d9",
                   help="Results directory of the current version (Tables 1/2 read from here, outputs written here)")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n-boot", type=int, default=sc.N_BOOT)
    p.add_argument("--seed", type=int, default=sc.SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    run(a.gold_csv, a.outputs_dir, a.results_dir, a.models, a.n_boot, a.seed)
