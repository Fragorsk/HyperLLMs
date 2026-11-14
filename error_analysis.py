"""Level 1 error analysis: one error record per unmatched prediction (FP), per
unmatched gold entity (FN) and per correct extraction with an ungrounded
evidence span (TP flag), all taken from the shared matching in ``matching.py``.

Gold standard used
-------------------
The flat CSV produced by the annotation tool (``annotations/validated_annotations.csv``,
columns ``OPIS, Text, Label, Model, ID_PACJENTA``), the same file and the same
loader (``matching.load_gold``) used by ``validation.py``. It records *entity
presence* (a drug/diagnosis/comorbidity/symptom was mentioned, or a categorical
field took a given value) but no attribute-level gold values (dose_mg, frequency,
is_current, AHI, confidence, ...). ``temporal_error``, ``name_mapping_error``,
``attribute_error`` and ``confidence_error`` therefore cannot be computed and are
not emitted.

Taxonomy (definitions and priority order: docstring of ``matching.py``)
------------------------------------------------------------------------
Every FP gets exactly one of: duplicate, negation_error, classification_error,
hallucination_entity. Every FN gets exactly one of: invalid_output,
classification_error, negation_error, omission. ``hallucination_evidence`` flags a
TP whose evidence span is not found in the text the model saw (after abbreviation
expansion); it is not an FP. Hence, per model and category:
    #FP records == FP and #FN records == FN of validation.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matching
from matching import (  # noqa: F401 — re-export for older scripts
    NEGATION_MARKERS,
    WELLBEING_SCALE,
    evidence_span_grounded,
    find_span_position,
    has_negation_nearby,
)

# ─── Constants ────────────────────────────────────────────────────────────

CATEGORIES = matching.CATEGORIES

SEVERITY_MAP: dict[str, str] = {
    "hallucination_entity": "critical",
    "omission": "critical",
    "negation_error": "critical",
    "invalid_output": "critical",
    "temporal_error": "moderate",
    "name_mapping_error": "moderate",
    "attribute_error": "moderate",
    "classification_error": "moderate",
    "duplicate": "minor",
    "hallucination_evidence": "minor",
    "confidence_error": "minor",
}


# ─── Data model ───────────────────────────────────────────────────────────

@dataclass
class ErrorRecord:
    note_id: str
    model_name: str
    category: str
    field: str
    error_type: str
    error_subtype: Optional[str]
    side: str                       # "FP", "FN" or "TP" (quality flag on a correct extraction)
    gold_value: Optional[str]
    predicted_value: Optional[str]
    evidence_span_gold: Optional[str]
    evidence_span_predicted: Optional[str]
    severity: str

    def to_dict(self) -> dict:
        return asdict(self)


def note_id_for(patient_id: int | str, opis: str) -> str:
    digest = hashlib.sha1(opis.encode("utf-8")).hexdigest()[:8]
    return f"P{patient_id}_{digest}"


def count_model_entities(facts: dict) -> int:
    """Number of positive model predictions (n_pred) — matching.count_predictions."""
    return sum(matching.count_predictions(facts).values())


# ─── Records from matching results ────────────────────────────────────────

def _gold_field(g: matching.GoldItem) -> str:
    if g.category == "Medication":
        return g.label[5:].strip()
    if g.category == "Diagnosis":
        return g.label[13:].strip()
    return g.key


def _gold_value(g: matching.GoldItem) -> str:
    return g.value if g.value is not None else _gold_field(g)


def _pred_field(p: matching.PredItem) -> str:
    return p.name if p.category in matching.LIST_CATEGORIES else p.key


def records_for(note_id: str, model: str, r: matching.DocCategoryResult) -> list[ErrorRecord]:
    """ErrorRecords for one (document, category) result of matching.evaluate_document."""
    recs: list[ErrorRecord] = []
    wrong_pred = r.preds[0] if r.category in matching.CATEGORICAL_CATEGORIES and r.preds else None

    for g, label in r.fn:
        subtype = None
        pred_value = None
        if label == "classification_error" and wrong_pred is not None:
            pred_value = wrong_pred.value
            if r.category == "Well-being":
                subtype = matching.wellbeing_subtype(g.value, wrong_pred.value)
        recs.append(ErrorRecord(
            note_id, model, r.category, _gold_field(g), label, subtype, "FN",
            _gold_value(g), pred_value, g.texts[0] if g.texts else None,
            wrong_pred.span if wrong_pred is not None else None, SEVERITY_MAP[label],
        ))

    for p, label in r.fp:
        subtype = None
        gold_value = None
        if label == "classification_error" and r.gold:
            gold_value = r.gold[0].value
            if r.category == "Well-being":
                subtype = matching.wellbeing_subtype(r.gold[0].value, p.value)
        recs.append(ErrorRecord(
            note_id, model, r.category, _pred_field(p), label, subtype, "FP",
            gold_value, p.value or p.name, None, p.span, SEVERITY_MAP[label],
        ))

    for g, p in r.tp_ungrounded:
        recs.append(ErrorRecord(
            note_id, model, r.category, _gold_field(g), "hallucination_evidence", None, "TP",
            _gold_value(g), p.value or p.name, g.texts[0] if g.texts else None, p.span,
            SEVERITY_MAP["hallucination_evidence"],
        ))
    return recs


# ─── Runner ───────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    "PLLuM-v2-12",
    "Qwen3_8-27",
    "Bielik-v3-11",
    "Qwen3-14",
    "Ministral-3-14",
    "Gemma3-12"
]

DEFAULT_CSV = "annotations/validated_annotations.csv"


def run_level1(
    csv_path: str = DEFAULT_CSV,
    outputs_dir: str = "outputs",
    models: Optional[list[str]] = None,
    output_csv: str = "output/tables/error_analysis_results.csv",
    output_summary_json: str = "output/tables/error_analysis_summary.json",
    output_notes_meta_csv: str = "output/tables/notes_meta.csv",
    output_notes_model_meta_csv: str = "output/tables/notes_model_meta.csv",
) -> list[ErrorRecord]:
    """Load gold + model outputs, run the shared matching for every note x model pair,
    and write the full error table + summary JSON + per-note metadata CSVs
    (note length / gold entity count / model entity count / TP count, used by error_patterns.py)."""
    if models is None:
        models = DEFAULT_MODELS

    gold = matching.load_gold(csv_path)
    if not gold.docs:
        raise RuntimeError(f"No gold notes could be loaded from {csv_path}")
    note_ids = {opis: note_id_for(gold.patient[opis], opis) for opis in gold.docs}

    all_records: list[ErrorRecord] = []
    available_models: list[str] = []
    meta: dict[tuple[str, str], dict[str, int]] = {}
    totals: dict[str, dict[str, int]] = {}

    for model in models:
        path = Path(outputs_dir) / f"{model}_200_patients.json"
        if not path.exists():
            print(f"[POMINIĘTO] {model}: {path} not found.")
            continue
        res = matching.evaluate_model_file(gold, path, model)
        available_models.append(model)
        t = defaultdict(int)
        for r in res.iter_results():
            nid = note_ids[r.opis]
            recs = records_for(nid, model, r)
            all_records.extend(recs)
            m = meta.setdefault((nid, model), {"n_gt": 0, "n_pred": 0, "n_tp": 0})
            m["n_gt"] += r.n_gt
            m["n_pred"] += r.n_pred
            m["n_tp"] += len(r.tp)
            # #FP/#FN records == FP/FN of the shared matching
            if sum(x.side == "FP" for x in recs) != len(r.fp) or sum(x.side == "FN" for x in recs) != len(r.fn):
                raise AssertionError(f"{model}/{r.category}: liczba rekordów błędów ≠ FP/FN")
            for k, v in (("tp", len(r.tp)), ("fp", len(r.fp)), ("fn", len(r.fn))):
                t[k] += v
        totals[model] = dict(t)

    _write_error_csv(all_records, output_csv)
    _write_summary_json(all_records, available_models, gold, totals, output_summary_json)
    _write_notes_meta(gold, note_ids, output_notes_meta_csv)
    _write_notes_model_meta(gold, note_ids, meta, available_models, output_notes_model_meta_csv)

    return all_records


def _write_error_csv(records: list[ErrorRecord], output_csv: str) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["note_id", "model_name", "category", "field", "error_type", "error_subtype", "side",
              "gold_value", "predicted_value", "evidence_span_gold", "evidence_span_predicted",
              "severity"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())
    print(f"Wrote {len(records)} error records -> {output_csv}")


def _write_summary_json(records: list[ErrorRecord], models: list[str],
                        gold: matching.GoldStandard, totals: dict[str, dict[str, int]],
                        output_path: str) -> None:
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_severity: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        key = matching.error_key(r.error_type, r.side)
        by_model[r.model_name][key] += 1
        by_category[r.category][key] += 1
        by_severity[r.model_name][r.severity] += 1

    summary = {
        "n_notes": len(gold.docs),
        "n_gt_entities": gold.n_gt(),
        "n_gt_rows": gold.n_rows,
        "excluded_gt_rows": dict(gold.excluded_rows),
        "n_models": len(models),
        "models": models,
        "counts_by_model": totals,
        "n_error_records_total": len(records),
        "errors_by_model": {m: dict(v) for m, v in by_model.items()},
        "errors_by_category": {c: dict(v) for c, v in by_category.items()},
        "severity_by_model": {m: dict(v) for m, v in by_severity.items()},
        "evaluated_fields": matching.EVALUATED_FIELDS,
        "not_evaluated_fields": matching.NOT_EVALUATED_FIELDS,
        "not_evaluable_error_types": [
            "temporal_error", "name_mapping_error", "attribute_error", "confidence_error",
        ],
        "not_evaluable_reason": (
            "Gold standard CSV records entity presence only, no per-attribute "
            "values (dose_mg, frequency, is_current, confidence)."
        ),
        "known_gold_coverage_gaps": (
            "Medication/diagnosis annotation is not exhaustive per note: some notes "
            "with clear drug mentions (e.g. containing 'mg') have zero 'Lek:' gold rows "
            "(~7% of notes). Under the closed-world assumption (no gold row = gold-empty), "
            "a correct extraction from such a note is an FP (hallucination_entity), so the "
            "absolute hallucination rate for these categories is an upper bound."
        ),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote summary JSON -> {output_path}")


def _write_notes_model_meta(
    gold: matching.GoldStandard,
    note_ids: dict[str, str],
    meta: dict[tuple[str, str], dict[str, int]],
    models: list[str],
    output_path: str,
) -> None:
    """Per-note × per-model counts: GT entities (n_gt), model predictions (n_pred), TP.
    Used by error_patterns.py as rate denominators (FN types / n_gt, FP types / n_pred,
    TP flags / n_tp)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["note_id", "model_name", "n_gt_entities", "n_model_entities", "n_tp"])
        for opis in gold.docs:
            nid = note_ids[opis]
            for model in models:
                m = meta[(nid, model)]
                writer.writerow([nid, model, m["n_gt"], m["n_pred"], m["n_tp"]])
    print(f"Wrote notes model metadata -> {output_path}")


def _write_notes_meta(gold: matching.GoldStandard, note_ids: dict[str, str], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["note_id", "patient_id", "note_length_chars", "n_gold_entities"])
        for opis in gold.docs:
            n_gt = sum(len(v) for v in gold.items.get(opis, {}).values())
            writer.writerow([note_ids[opis], gold.patient[opis], len(opis), n_gt])
    print(f"Wrote notes metadata -> {output_path}")


if __name__ == "__main__":
    run_level1()
