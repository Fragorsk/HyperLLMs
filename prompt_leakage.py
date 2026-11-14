"""Prompt-example leakage (Table 7): predicted medications whose normalised name is one
of the rule-E brand-to-generic example substances of the v5 extraction prompt.

None of these substances occurs in the notes, in the text given to the model or in the
gold standard (checked here and in leakage_preflight.md), so every such prediction is a FP.

Counts come from ``matching.evaluate_model_file`` (the matching behind Tables 2–4).
The sensitivity analysis removes the leaked medications from the model output (copies in a
temporary directory; ``outputs_v5`` is not touched) and re-runs the unchanged Table 2 and
Table 3 code (``validation.evaluate_models_to_latex`` / ``evaluate_wilcoxon_to_latex``,
patient cluster bootstrap, same seed and number of resamples).

Name normalisation = ``error_patterns._drug_names_to_inn`` (mapping file + overrides, as
in the medication error tables), extended by the brand → INN pairs of rule E itself (the
mapping file has no row for these brands because they do not occur in the data).

Usage:
    python prompt_leakage.py [--gold-csv ...] [--outputs-dir outputs_v5]
                             [--results-dir results_v5_clustered_d9]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matching
import stats_clustered as sc
import validation
from error_patterns import _drug_key, _drug_names_to_inn
from main_neurosymbolic import V5_PROMPT_EXAMPLE_DRUGS
from validation import DEFAULT_MODELS, MODEL_DISPLAY

# Diagnostic only: inflection-tolerant stems of every rule-E name (brand, Polish, INN).
# Used to check the corpus and to cross-check that the normalisation misses no variant.
LEAK_STEMS = ("edarbi", "azylsart", "azilsart", "rasilez", "alisk", "ranexa", "ranola",
              "lescol", "fluwast", "fluvast", "minipress", "prazos")
LEAK_STEM_RX = re.compile("|".join(LEAK_STEMS), re.IGNORECASE)


# ─── PROMPT_EXAMPLE_SET and name normalisation ───────────────────────────────

def prompt_example_pairs() -> list[tuple[str, str, str]]:
    """(brand, Polish generic, INN) for each rule-E example."""
    pl = pd.Series([g for _, g in V5_PROMPT_EXAMPLE_DRUGS], dtype=str)
    inn = _drug_names_to_inn(pl)
    return [(b, g, i) for (b, g), i in zip(V5_PROMPT_EXAMPLE_DRUGS, inn)]


PROMPT_EXAMPLE_SET: frozenset[str] = frozenset(i for _, _, i in prompt_example_pairs())
BRAND_TO_INN: dict[str, str] = {_drug_key(b): i for b, _, i in prompt_example_pairs()}


def to_inn(names: Iterable[str]) -> dict[str, str]:
    """name → normalised INN label (evaluation normalisation + rule-E brands)."""
    names = list(dict.fromkeys(str(n) for n in names))
    if not names:
        return {}
    inn = _drug_names_to_inn(pd.Series(names, dtype=str))
    return {n: BRAND_TO_INN.get(_drug_key(n), i) for n, i in zip(names, inn)}


def leak_map(names: Iterable[str]) -> dict[str, Optional[str]]:
    """name → leaked INN, or None if the name is not a prompt example."""
    return {n: (i if i in PROMPT_EXAMPLE_SET else None) for n, i in to_inn(names).items()}


# ─── Leaked predictions in an evaluated model ────────────────────────────────

def leak_records(result: matching.ModelResult, patient: dict[str, str]) -> pd.DataFrame:
    """One row per leaked Medication prediction: label = 'TP' or its FP label."""
    rows = []
    for opis, cats in result.docs.items():
        r = cats["Medication"]
        lm = leak_map(p.name for p in r.preds)
        labelled = [(p, "TP", g.label) for g, p in r.tp] + [(p, label, "") for p, label in r.fp]
        rows += [{"model": result.model, "patient": patient[opis], "opis": opis, "name": p.name,
                  "inn": lm[p.name], "label": label, "matched_gt": gt, "span": p.span, "index": p.index}
                 for p, label, gt in labelled if lm[p.name]]
    return pd.DataFrame(rows, columns=["model", "patient", "opis", "name", "inn", "label",
                                       "matched_gt", "span", "index"])


def exclusion_plan(leaks: pd.DataFrame) -> dict[str, set[int]]:
    """opis → indices (``PredItem.index`` in Medication) of leaked predictions to exclude.
    Only FP-labelled ones (user decision 2026-11-07): the few leaked names that the
    span-based ``matching.med_edge`` matches to a GT drug stay, so TP and FN are unchanged."""
    fp = leaks[leaks["label"] != "TP"]
    return {o: set(g["index"]) for o, g in fp.groupby("opis")}


def filter_facts(facts: dict, drop: Optional[set[int]] = None) -> tuple[dict, int]:
    """Copy of ``facts`` without leaked medications. ``drop`` = Medication prediction indices
    (positions among non-null-named medications, as in ``matching.prediction_items``);
    None = every leaked medication. Returns (facts, number removed)."""
    meds = facts.get("medications")
    if not meds:
        return facts, 0
    nonnull = [i for i, m in enumerate(meds) if not matching.is_null_like(m.get("name"))]
    lm = leak_map(str(meds[i].get("name") or "") for i in nonnull)
    if drop is None:
        drop = {j for j, i in enumerate(nonnull) if lm[str(meds[i].get("name") or "")]}
    if not drop:
        return facts, 0
    remove = {nonnull[j] for j in drop}
    for i in remove:
        if not lm[str(meds[i].get("name") or "")]:
            raise AssertionError(f"exclusion would remove a non-leaked medication: {meds[i].get('name')!r}")
    out = copy.deepcopy(facts)
    out["medications"] = [m for i, m in enumerate(meds) if i not in remove]
    return out, len(remove)


def write_filtered_outputs(src: Path, dst: Path, plan: dict[str, set[int]]) -> int:
    """Writes ``dst`` = ``src`` with the planned medications removed from the first record of
    each OPIS (the record ``matching.load_predictions`` evaluates, D6). Asserts that no
    document changes its invalid_output status (that would change FN labels)."""
    with src.open(encoding="utf-8") as f:
        records = json.load(f)
    removed, seen = 0, set()
    for rec in records:
        opis = rec["OPIS"]
        if opis in seen:
            continue
        seen.add(opis)
        if opis not in plan:
            continue
        new, n = filter_facts(rec["facts"], plan[opis])
        if matching.is_invalid_output(new) != matching.is_invalid_output(rec["facts"]):
            raise AssertionError(f"{src.name}: removing leakage changes invalid_output of a document")
        rec["facts"] = new
        removed += n
    with dst.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    return removed


# ─── Summary per model ───────────────────────────────────────────────────────

def summarise(leaks: pd.DataFrame, results: dict[str, matching.ModelResult], n_docs: int,
              n_patients: int) -> pd.DataFrame:
    rows = []
    for m, res in results.items():
        lk = leaks[leaks["model"] == m]
        lab = Counter(lk["label"])
        fp_all = sum(len(r.fp) for r in res.iter_results())
        med_halluc = sum(1 for r in res.iter_results() if r.category == "Medication"
                         for _p, l in r.fp if l == "hallucination_entity")
        n_leak = len(lk)
        n_fp = n_leak - lab["TP"]
        rows.append({
            "model": m, "n_leak": n_leak, "n_leak_tp": lab["TP"], "n_leak_fp": n_fp,
            "n_leak_hallucination_entity": lab["hallucination_entity"], "n_leak_duplicate": lab["duplicate"],
            "med_hallucination_entity": med_halluc, "fp_all": fp_all,
            "pct_of_med_hallucinations": 100 * lab["hallucination_entity"] / med_halluc if med_halluc else None,
            "pct_of_all_fp": 100 * n_fp / fp_all if fp_all else None,
            "visits_affected": lk["opis"].nunique(), "visits_total": n_docs,
            "pct_visits": 100 * lk["opis"].nunique() / n_docs,
            "patients_affected": lk["patient"].nunique(), "patients_total": n_patients,
        })
    return pd.DataFrame(rows)


def ranking(values: dict[str, float]) -> list[str]:
    return sorted(values, key=lambda m: -values[m])


# ─── Output formatting ───────────────────────────────────────────────────────

def _f(v: Optional[float]) -> str:
    return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


TABLE7_ROWS = ["Leaked predictions (n)", "\\% of medication hallucinations", "\\% of all false positives",
               "Visits affected, n (\\%)", "Micro-P: all $\\rightarrow$ excl. leakage",
               "Micro-F1: all $\\rightarrow$ excl. leakage"]


def table7_cells(s: pd.Series, before: dict, after: dict) -> list[str]:
    """Cells of one model column (LaTeX). ``before``/``after``: P, F1, ci_low, ci_high (fractions)."""
    if s["n_leak"] == 0:
        return ["0", "0.0", "0.0", f"0 (0.0)", f"{_f(before['P'] * 100)} $\\rightarrow$ ---",
                f"{_f(before['F1'] * 100)} $\\rightarrow$ ---"]
    marks = ("a" if s["n_leak_duplicate"] else "") + ("b" if s["n_leak_tp"] else "")
    marks = f"$^{{{','.join(marks)}}}$" if marks else ""
    return [
        f"{int(s['n_leak'])}{marks}",
        _f(s["pct_of_med_hallucinations"]),
        _f(s["pct_of_all_fp"]),
        f"{int(s['visits_affected'])} ({_f(s['pct_visits'])})",
        f"{_f(before['P'] * 100)} $\\rightarrow$ {_f(after['P'] * 100)}",
        f"{_f(before['F1'] * 100)} $\\rightarrow$ {_f(after['F1'] * 100)} "
        f"\\ci{{{_f(after['ci_low'] * 100)}}}{{{_f(after['ci_high'] * 100)}}}",
    ]


def caption7(pairs, n_boot: int, n_patients: int, footnote: str) -> str:
    lst = ", ".join(f"{b} $\\rightarrow$ {i}" for b, _, i in pairs)
    return ("Prompt-example leakage per model. Leaked predictions: predicted medications whose normalised "
            f"name matches one of the five brand-to-generic example substances given in the prompt ({lst}). "
            "None of these substances occurs in the corpus or in the gold standard, so each such prediction "
            "is wrong. \\% of medication hallucinations: leaked predictions labelled "
            "hallucination\\_entity among all Medication hallucination\\_entity errors (Table~4); "
            "\\% of all false positives: leaked false positives among all false positives of the model. "
            "Visits affected: visits with at least one leaked prediction (of 504). "
            "Excluding the leaked false positives changes only the number of false positives; TP and FN are "
            "unaffected. Micro values are pooled over all categories (Table~2). "
            f"95\\% CI (excluding leakage) from a cluster bootstrap resampling patients, $N = {n_patients}$ "
            f"({n_boot} resamples, percentile intervals)." + footnote)


# ─── Main ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold-csv", default=validation.DEFAULT_CSV)
    ap.add_argument("--outputs-dir", default="outputs_v5")
    ap.add_argument("--results-dir", default="results_v5_clustered_d9")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--n-boot", type=int, default=sc.N_BOOT)
    ap.add_argument("--seed", type=int, default=sc.SEED)
    args = ap.parse_args(argv)
    out = Path(args.results_dir)
    models = args.models
    checks: list[str] = []

    gold = matching.load_gold(args.gold_csv)
    n_docs, n_patients = len(gold.docs), len(set(gold.patient.values()))
    pairs = prompt_example_pairs()
    print("PROMPT_EXAMPLE_SET:", pairs)

    # ── Original evaluation (same call as Tables 2–4) ──
    results = {m: matching.evaluate_model_file(gold, Path(args.outputs_dir) / f"{m}_200_patients.json", m)
               for m in models}
    leaks = pd.concat([leak_records(results[m], gold.patient) for m in models], ignore_index=True)
    bad = set(leaks["label"]) - {"TP", "hallucination_entity", "duplicate"}
    if bad:
        raise AssertionError(f"unexpected labels of leaked predictions: {bad}")
    checks.append("leaked predictions carry only the labels hallucination_entity, duplicate or TP "
                  f"(TP: {int((leaks.label == 'TP').sum())}, kept in the sensitivity analysis)")

    # cross-check: every predicted medication name containing a rule-E stem is counted as leakage
    for m, res in results.items():
        names = [p.name for r in res.iter_results() if r.category == "Medication" for p in r.preds]
        stem_hits = {n for n in names if LEAK_STEM_RX.search(n)}
        lm = leak_map(stem_hits)
        missed = sorted(n for n in stem_hits if not lm[n])
        if missed:
            raise AssertionError(f"{m}: names with a rule-E stem not normalised to the set: {missed}")
    checks.append("stem cross-check: every predicted name containing a rule-E stem maps to PROMPT_EXAMPLE_SET")

    summary = summarise(leaks, results, n_docs, n_patients)

    # ── Table 2 / Table 3 (existing results) ──
    t2 = pd.read_csv(out / "evaluation_table_patient.csv")
    t3 = pd.read_csv(out / "wilcoxon_table_patient_details.csv")
    if set(t2["n_boot"]) != {args.n_boot} or set(t2["seed"]) != {args.seed}:
        raise AssertionError("Table 2 was computed with a different n_boot/seed")
    for m, res in results.items():
        tot = {k: sum(len(getattr(r, k)) for r in res.iter_results()) for k in ("tp", "fp", "fn")}
        row = t2[(t2["row"] == "Average") & (t2["model"] == m)].iloc[0]
        if (row.tp, row.fp, row.fn) != (tot["tp"], tot["fp"], tot["fn"]):
            raise AssertionError(f"{m}: matching counts ≠ Table 2 micro row")
        if summary.loc[summary.model == m, "fp_all"].item() != row.fp:
            raise AssertionError(f"{m}: FP ≠ Table 2")
    checks.append("TP/FP/FN of the current matching == Table 2 micro row (results dir)")

    # ── Sensitivity: unchanged Table 2/3 code on filtered outputs ──
    with tempfile.TemporaryDirectory(prefix="noleak_") as tmp:
        tmp = Path(tmp)
        for m in models:
            lk = leaks[leaks.model == m]
            n = write_filtered_outputs(Path(args.outputs_dir) / f"{m}_200_patients.json",
                                       tmp / f"{m}_200_patients.json", exclusion_plan(lk))
            if n != int((lk.label != "TP").sum()):
                raise AssertionError(f"{m}: removed {n} ≠ FP-labelled leaked predictions")
        common = {"csv_path": args.gold_csv, "output_dir": str(tmp), "models": models,
                  "n_boot": args.n_boot, "seed": args.seed, "unit": "patient"}
        t2x = validation.evaluate_models_to_latex(output_tex=str(tmp / "eval.tex"), **common)
        wx = validation.evaluate_wilcoxon_to_latex(output_tex=str(tmp / "wil.tex"), **common)
    t3x = wx["pairs"]

    # ── Assertions: TP, FN unchanged; FP down by exactly n_leak (per category and per patient) ──
    fp_leaks = leaks[leaks.label != "TP"]
    leak_by_cat = {m: int((fp_leaks.model == m).sum()) for m in models}
    for m in models:
        for cat in t2["row"].unique():
            if cat == "Macro":
                continue
            a = t2[(t2.row == cat) & (t2.model == m)].iloc[0]
            b = t2x[(t2x.row == cat) & (t2x.model == m)].iloc[0]
            if (a.tp, a.fn) != (b.tp, b.fn):
                raise AssertionError(f"{m}/{cat}: TP/FN changed after exclusion")
            expected = leak_by_cat[m] if cat in ("Medication", "Average") else 0
            if a.fp - b.fp != expected:
                raise AssertionError(f"{m}/{cat}: FP drop {a.fp - b.fp} ≠ n_leak {expected}")
    per_before = pd.read_csv(out / "wilcoxon_table_patient_per_document.csv")
    per_after = wx["per_unit"].copy()
    per_before["unit_id"] = per_before["unit_id"].astype(str)
    per_after["unit_id"] = per_after["unit_id"].astype(str)
    mrg = per_before.merge(per_after, on=["model", "unit_id"], suffixes=("_b", "_a"))
    if len(mrg) != len(per_before):
        raise AssertionError("per-patient tables do not align")
    leak_pat = fp_leaks.groupby(["model", "patient"]).size()
    exp_drop = mrg.apply(lambda r: leak_pat.get((r.model, str(r.unit_id)), 0), axis=1)
    if not ((mrg.tp_b == mrg.tp_a) & (mrg.fn_b == mrg.fn_a)).all():
        raise AssertionError("per-patient TP/FN changed after exclusion")
    if not ((mrg.fp_b - mrg.fp_a) == exp_drop).all():
        raise AssertionError("per-patient FP drop ≠ n_leak of that patient")
    checks.append("after exclusion TP and FN are unchanged for every model, category and patient")
    checks.append("FP decreases by exactly n_leak_fp = n_leak − n_leak_tp (Medication and micro rows; per patient)")
    checks.append(f"bootstrap: patient cluster, n_boot={args.n_boot}, seed={args.seed} (same as Table 2)")

    # ── Metrics before / after ──
    def _get(t, row, m):
        return t[(t.row == row) & (t.model == m)].iloc[0]

    metr = []
    for m in models:
        b, a = _get(t2, "Average", m), _get(t2x, "Average", m)
        bm, am = _get(t2, "Medication", m), _get(t2x, "Medication", m)
        bM, aM = _get(t2, "Macro", m), _get(t2x, "Macro", m)
        metr.append({"model": m,
                     "micro_P_all": b.P, "micro_P_excl": a.P, "micro_R_all": b.R, "micro_R_excl": a.R,
                     "micro_F1_all": b.F1, "micro_F1_excl": a.F1,
                     "micro_F1_all_ci_low": b.ci_low, "micro_F1_all_ci_high": b.ci_high,
                     "micro_F1_excl_ci_low": a.ci_low, "micro_F1_excl_ci_high": a.ci_high,
                     "med_P_all": bm.P, "med_P_excl": am.P, "med_F1_all": bm.F1, "med_F1_excl": am.F1,
                     "macro_F1_all": bM.F1, "macro_F1_excl": aM.F1,
                     "fp_excl": a.fp, "tp": b.tp, "fn": b.fn})
    metr = pd.DataFrame(metr)
    for r in metr.itertuples():
        if leak_by_cat[r.model] == 0:
            for k in ("micro_P", "micro_F1", "med_P", "med_F1", "macro_F1"):
                if not np.isclose(getattr(r, f"{k}_all"), getattr(r, f"{k}_excl")):
                    raise AssertionError(f"{r.model}: n_leak = 0 but {k} changed")
    full = summary.merge(metr, on="model")

    rank = {k: (ranking(dict(zip(full.model, full[f"{k}_all"]))), ranking(dict(zip(full.model, full[f"{k}_excl"]))))
            for k in ("micro_F1", "macro_F1")}

    # ── Table 3 significance changes ──
    sig = t3[["model_a", "model_b", "p_holm"]].merge(
        t3x[["model_a", "model_b", "p_holm", "delta_pp", "ci_low_pp", "ci_high_pp"]],
        on=["model_a", "model_b"], suffixes=("_all", "_excl"))
    sig["sig_all"] = sig.p_holm_all < sc.ALPHA
    sig["sig_excl"] = sig.p_holm_excl < sc.ALPHA
    if len(sig) != 15:
        raise AssertionError(f"expected 15 pairwise comparisons, got {len(sig)}")

    # ── Write outputs ──
    full.to_csv(out / "table7_prompt_leakage.csv", index=False)
    by_sub = (leaks.groupby(["model", "inn"]).agg(n=("label", "size"),
                                                  n_hallucination_entity=("label", lambda s: (s == "hallucination_entity").sum()),
                                                  n_duplicate=("label", lambda s: (s == "duplicate").sum()),
                                                  visits=("opis", "nunique"), patients=("patient", "nunique"))
              .reset_index().rename(columns={"inn": "substance"}))
    grid = pd.MultiIndex.from_product([models, sorted(PROMPT_EXAMPLE_SET)], names=["model", "substance"])
    by_sub = by_sub.set_index(["model", "substance"]).reindex(grid, fill_value=0).reset_index()
    by_sub["model_display"] = by_sub.model.map(lambda m: MODEL_DISPLAY.get(m, m))
    by_sub.to_csv(out / "leakage_by_substance.csv", index=False)
    leaks.drop(columns="index").to_csv(out / "leakage_predictions.csv", index=False)
    sig.to_csv(out / "leakage_wilcoxon_changes.csv", index=False)

    foot = ""
    if full.n_leak_duplicate.sum():
        foot += (" $^{a}$Includes leaked predictions labelled duplicate (repeated within a visit): "
                 + ", ".join(f"{MODEL_DISPLAY.get(r.model, r.model)} {r.n_leak_duplicate}"
                             for r in full.itertuples() if r.n_leak_duplicate)
                 + "; the \\% of medication hallucinations counts only those labelled hallucination\\_entity.")
    if full.n_leak_tp.sum():
        foot += (" $^{b}$Includes leaked predictions counted as true positives: "
                 + ", ".join(f"{MODEL_DISPLAY.get(r.model, r.model)} {r.n_leak_tp}"
                             for r in full.itertuples() if r.n_leak_tp)
                 + ". The model output a prompt substance as the drug name but quoted a gold-standard drug in "
                 "the evidence span, which the span-based matching accepts; these are not excluded.")
    lines = ["\\begin{table*}[htbp]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
             f"\\begin{{tabular}}{{l{'c' * len(models)}}}", "\\toprule",
             " & " + " & ".join(f"\\textbf{{{MODEL_DISPLAY.get(m, m)}}}" for m in models) + " \\\\", "\\midrule"]
    cols = []
    for m in models:
        s = full[full.model == m].iloc[0]
        cols.append(table7_cells(s, {"P": s.micro_P_all, "F1": s.micro_F1_all},
                                 {"P": s.micro_P_excl, "F1": s.micro_F1_excl,
                                  "ci_low": s.micro_F1_excl_ci_low, "ci_high": s.micro_F1_excl_ci_high}))
    for i, title in enumerate(TABLE7_ROWS):
        if i == 4:
            lines.append("\\midrule")
        lines.append(f"{title} & " + " & ".join(c[i] for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{{caption7(pairs, args.n_boot, n_patients, foot)}}}",
              "\\label{tab:prompt_leakage}", "\\end{table*}"]
    (out / "table7_prompt_leakage.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table 7 'all predictions' == Table 2 micro (rounded as displayed)
    for m in models:
        b = _get(t2, "Average", m)
        s = full[full.model == m].iloc[0]
        if _f(b.P * 100) != _f(s.micro_P_all * 100) or _f(b.F1 * 100) != _f(s.micro_F1_all * 100):
            raise AssertionError(f"{m}: Table 7 'all' ≠ Table 2 micro")
    checks.append("Table 7 'all predictions' P and F1 == Table 2 micro values")

    write_report(out / "leakage_report.md", pairs, full, by_sub, rank, sig, leaks, checks, n_docs, n_patients,
                 args, models)
    print("Asercje:\n  " + "\n  ".join(checks))
    print(f"Zapisano → {out}/table7_prompt_leakage.{{tex,csv}}, leakage_by_substance.csv, leakage_report.md")


def rq2_comment(full: pd.DataFrame, rank: dict, changed: pd.DataFrame, D) -> str:
    """One–two sentences generated from the computed values (no hand-written numbers)."""
    top = full.loc[full.n_leak.idxmax()]
    gain = (top.micro_F1_excl - top.micro_F1_all) * 100
    moved = [k for k, (b, a) in rank.items() if b != a]
    s = (f"Przeciek dotyczy głównie {D(top.model)} ({_f(top.pct_of_all_fp)} % jego FP; micro-F1 "
         f"+{gain:.1f} p.p. po wykluczeniu) i w mniejszym stopniu pozostałych modeli, więc wnioski do RQ2 "
         f"wymagają korekty, jeśli opierają się na kolejności modeli lub istotności par: ")
    parts = []
    if moved:
        parts.append("zmienia się ranking (" + ", ".join(moved) + ")")
    if not changed.empty:
        parts.append(f"{len(changed)} z 15 porównań w Tabeli 3 zmienia istotność ("
                     + "; ".join(f"{D(r.model_a)} vs {D(r.model_b)}" for r in changed.itertuples()) + ")")
    if not parts:
        return ("Ranking i istotność porównań w Tabeli 3 nie zmieniają się; wnioski do RQ2 nie wymagają "
                "korekty, wystarczy wspomnieć przeciek jako źródło FP.")
    return s + "; ".join(parts) + ". Proponujemy raportować wyniki główne bez zmian i przeciek jako analizę wrażliwości (Tabela 7)."


def write_report(path, pairs, full, by_sub, rank, sig, leaks, checks, n_docs, n_patients, args, models):
    D = lambda m: MODEL_DISPLAY.get(m, m)  # noqa: E731
    pp = lambda v: f"{v * 100:.1f}"  # noqa: E731
    L = ["# Prompt-example leakage — wartości do artykułu", "",
         f"Źródło: `prompt_leakage.py`, wyniki w `{args.results_dir}/`, wyjścia `{args.outputs_dir}/`, "
         f"GT `{args.gold_csv}` ({n_docs} wizyt, {n_patients} pacjentów). Bootstrap: klastrowy po pacjentach, "
         f"{args.n_boot} resampli, seed {args.seed}.", "",
         "## Lista substancji (sekcja 3.1)", "",
         "PROMPT_EXAMPLE_SET (reguła E promptu v5, nazwa handlowa → nazwa generyczna w prompcie → INN po normalizacji):", ""]
    L += [f"- {b} → {g} → **{i}**" for b, g, i in pairs]
    L += ["", "Do tekstu: " + ", ".join(f"{i} ({b})" for b, _, i in pairs) + ".", "",
          "## Wartości per model", "",
          "| Model | n_leak | halluc. / dup. / TP | [X] % halucynacji leków | [Y] % wszystkich FP | [n] wizyt (%) | pacjentów |",
          "|---|---|---|---|---|---|---|"]
    for r in full.itertuples():
        L.append(f"| {D(r.model)} | {r.n_leak} | {r.n_leak_hallucination_entity} / {r.n_leak_duplicate} / {r.n_leak_tp} | "
                 f"{_f(r.pct_of_med_hallucinations)} ({r.n_leak_hallucination_entity}/{r.med_hallucination_entity}) | "
                 f"{_f(r.pct_of_all_fp)} ({r.n_leak_fp}/{r.fp_all}) | {r.visits_affected} ({_f(r.pct_visits)}) | "
                 f"{r.patients_affected}/{r.patients_total} |")
    L += ["", "[X] liczony tylko z predykcji `hallucination_entity` (część n_leak ma etykietę `duplicate` — "
          "powtórzenie w tej samej wizycie). [Y] = FP z przecieku (hallucination_entity + duplicate) / wszystkie FP "
          "modelu (Tabela 2/4). Wizyty i pacjenci liczeni ze wszystkich n_leak (także TP).", "",
          "Predykcje z przecieku liczone jako TP (decyzja użytkownika 2026-11-07: zostają w analizie wrażliwości, "
          "bo dopasowanie po cytacie je akceptuje):", ""]
    for r in leaks[leaks.label == "TP"].itertuples():
        L.append(f"- {D(r.model)}, pacjent {r.patient}: name = `{r.name}`, cytat „{r.span}” → {r.matched_gt}")
    L += ["", "## Metryki przed → po wykluczeniu", "",
          "| Model | micro-P | micro-R | micro-F1 [95% CI po wykl.] | Medication P | Medication F1 | macro-F1 |",
          "|---|---|---|---|---|---|---|"]
    for r in full.itertuples():
        L.append(f"| {D(r.model)} | {pp(r.micro_P_all)} → {pp(r.micro_P_excl)} | {pp(r.micro_R_all)} → {pp(r.micro_R_excl)} | "
                 f"{pp(r.micro_F1_all)} → {pp(r.micro_F1_excl)} [{pp(r.micro_F1_excl_ci_low)}, {pp(r.micro_F1_excl_ci_high)}] | "
                 f"{pp(r.med_P_all)} → {pp(r.med_P_excl)} | {pp(r.med_F1_all)} → {pp(r.med_F1_excl)} | "
                 f"{pp(r.macro_F1_all)} → {pp(r.macro_F1_excl)} |")
    pl = full[full.model == "PLLuM-v2-12"].iloc[0] if (full.model == "PLLuM-v2-12").any() else None
    others = full[full.model != "PLLuM-v2-12"]
    dF1 = (others.micro_F1_excl - others.micro_F1_all) * 100
    zmax = others.iloc[int(dF1.values.argmax())]
    L += [""]
    if pl is not None:
        L += [f"- **[P′] PLLuM-v2-12B micro-P po wykluczeniu: {pp(pl.micro_P_excl)} %** (było {pp(pl.micro_P_all)}); "
              f"**[F1′] micro-F1: {pp(pl.micro_F1_excl)} %** [95% CI {pp(pl.micro_F1_excl_ci_low)}–{pp(pl.micro_F1_excl_ci_high)}] "
              f"(było {pp(pl.micro_F1_all)}; zmiana +{(pl.micro_F1_excl - pl.micro_F1_all) * 100:.2f} p.p.)."]
    L += [f"- **[Z] maks. zmiana micro-F1 wśród pozostałych modeli: +{dF1.max():.2f} p.p.** ({D(zmax.model)}: "
          f"{pp(zmax.micro_F1_all)} → {pp(zmax.micro_F1_excl)})."]
    L += ["", "## Ranking modeli", ""]
    for k, (b, a) in rank.items():
        ch = "bez zmian" if b == a else "ZMIANA"
        L.append(f"- {k}: {ch}. Przed: {' > '.join(map(D, b))}. Po: {' > '.join(map(D, a))}.")
    tot = full.n_leak.sum()
    tot_h = full.n_leak_hallucination_entity.sum()
    tot_mh = full.med_hallucination_entity.sum()
    L += ["", "## Sekcja 4.3", "",
          f"Łącznie (6 modeli) predykcji z przecieku w kategorii Medication: **{tot}**, w tym "
          f"**{tot_h} z etykietą hallucination_entity** spośród {tot_mh} halucynacji leków wszystkich modeli "
          f"({100 * tot_h / tot_mh:.1f} %), {full.n_leak_duplicate.sum()} duplicate, {full.n_leak_tp.sum()} TP. "
          f"Jeśli zdanie w 4.3 dotyczy halucynacji leków („…of which [n] were prompt-example leakage”), "
          f"właściwa liczba to **[n] = {tot_h}**.", "",
          "Rozkład per substancja (wszystkie modele): " +
          ", ".join(f"{s} {n}" for s, n in by_sub.groupby("substance").n.sum().sort_values(ascending=False).items()) + ".",
          "", "## Tabela 3 po wykluczeniu (Wilcoxon na F1 pacjenta, Holm, 15 par, α = 0.05)", ""]
    changed = sig[sig.sig_all != sig.sig_excl]
    if changed.empty:
        L.append("Żadna para nie zmienia istotności.")
    else:
        for r in changed.itertuples():
            L.append(f"- {D(r.model_a)} vs {D(r.model_b)}: p_Holm {r.p_holm_all:.3g} → {r.p_holm_excl:.3g} "
                     f"({'istotna → nieistotna' if r.sig_all else 'nieistotna → istotna'})")
    L += ["", "Pełne zestawienie: `leakage_wilcoxon_changes.csv`.", "",
          "## Komentarz do RQ2 (propozycja)", "",
          "(Założenie: RQ2 dotyczy porównania modeli — definicji RQ2 nie ma w repozytorium.) "
          + rq2_comment(full, rank, changed, D), "",
          "## Asercje (Faza 3)", ""] + [f"- OK: {c}" for c in checks]
    L += ["", "## Decyzje nieopisane w zadaniu (propozycje)", "",
          "1. Wyniki i nowe pliki w `results_v5_clustered_d9/` — to aktualny katalog (zgodny z obecnym `matching.py` "
          "po poprawce D9, domyślny w `validation.py`); `results_v5_clustered/` to wersja sprzed D9 (inne FP).",
          "2. Normalizacja: `error_patterns._drug_names_to_inn` (ta sama co w tabelach błędów leków) + pary marka → INN "
          "z reguły E (plik mapowania nie zawiera tych marek). Dopasowanie TP/FP w `matching.med_edge` nie używa INN "
          "(porównuje surowe nazwy i cytat), więc INN służy tu tylko do identyfikacji przecieku.",
          "2a. (Decyzja użytkownika) Predykcje z przecieku, które dopasowanie liczy jako TP, wchodzą do n_leak, ale "
          "nie są wykluczane — wykluczenie dotyczy tylko predykcji z etykietą FP; dzięki temu TP/FN bez zmian.",
          "3. Kontrola kompletności: każda przewidziana nazwa leku zawierająca rdzeń z reguły E (np. `azylsart`, `prazos`) "
          "musi trafić do zbioru — asercja w skrypcie.",
          "4. Wykluczenie realizowane przez usunięcie leków z kopii wyjść (katalog tymczasowy) i ponowne uruchomienie "
          "niezmienionych funkcji Tabel 2 i 3 (`validation.evaluate_models_to_latex`, `evaluate_wilcoxon_to_latex`). "
          "Asercja: żaden dokument nie zmienia statusu invalid_output.",
          "5. „% of all false positives”: licznik = FP z przecieku (hallucination_entity + duplicate), "
          "mianownik = FP z Tabeli 2/4. „Visits affected” liczone ze wszystkich n_leak.",
          "6. Skutki pośrednie nieuwzględnione: lek z przecieku może wpływać na `diagnoses_from_medications` (Stage 1.5); "
          "liczymy wyłącznie predykcje kategorii Medication, zgodnie z zadaniem.",
          "7. Dodatkowe pliki: `leakage_predictions.csv` (lista predykcji) i `leakage_wilcoxon_changes.csv`."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
