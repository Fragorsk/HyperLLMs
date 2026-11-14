"""Shared definition of TP / FP / FN for validation.py and error_analysis.py.

The only place where:
  * the gold standard (GT) is loaded and labels are mapped to categories/fields,
  * model outputs are loaded (duplicate records, invalid JSON),
  * predictions are matched to GT (1:1 matching),
  * every FP / FN gets exactly one error label.

Methodological decisions (details and rationale: changes_summary.md):

  Evaluation unit — a document-level fact: (document, category, key).
      Duplicated GT rows (same OPIS and same entity) are merged.
  D1  Only positive assertions are evaluated. GT labels „X – wykluczone” ("ruled out") are dropped
      from GT; ``present: false`` in the model output is not a prediction.
  D2  Labels outside the evaluation schema („Inne”, e.g. „Palenie”, „Astma”) are dropped from GT.
      As a consequence of D1+D2, fields without any positive GT label (asthma, smoking,
      syncope, palpitations) are not evaluated — neither on the GT nor on the prediction side.
  D3  Well-being / Therapy adherence: closed-world — a prediction with no GT = FP.
      The value „średnieskargi na dolegliwości” (concatenated due to a missing comma in the
      extraction schema enum) is treated as „średnie”, both in GT and in predictions.
  D4  No diagnosis fallback via ``choroby_towarzyszace``: a diagnosis is TP only
      if the model put it in the diagnosis list.
  D5  ``hallucination_evidence`` is a TP quality flag (not FP). The quote is grounded
      in the text the model saw: ``OPIS_MODEL_INPUT`` from the record (v5), and for older
      files without that field — ``expand_abbreviations(OPIS)``.
  D6  Duplicated output records (same OPIS): the first one wins, with a warning.
  D7  Medications: substring matching (no fuzzy n-grams), names normalized with ``_norm``.
  D8  Invalid JSON detected by the skeleton heuristic (``is_invalid_output``); the document
      stays in the denominator, all its GT entities are FN labelled ``invalid_output``.
  D9  A medication / diagnosis whose name (after ``_norm``) is an empty value (``NULL_LIKE``:
      „brak danych”, „brak”, „null”, „none”, „unknown”, „nie podano”, …) is not a prediction —
      the model encoded a missing value as text. Only exact matches are filtered.

Matching: within (document, category), maximum 1:1 matching (Kuhn's
algorithm, augmenting paths). Deterministic order: GT entities in the order of
sorted keys, predictions in model-output order. Every prediction and every
GT entity is used at most once.

Labels (first match wins):
  FN: invalid_output → classification_error → negation_error → omission
  FP: duplicate → negation_error → classification_error → hallucination_entity
  TP: hallucination_evidence (flag; TP remains TP)
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from schema import MEDICAL_ABBREVIATIONS

# ─── GT label → JSON field mappings ───────────────────────────────────────────

# Comorbidities and symptoms: label → (group, json_key, expected_present).
# Variants with expected_present=False („– wykluczone”, i.e. "ruled out") are dropped from GT (D1).
BINARY_LABEL_MAP: dict[str, tuple[str, str, bool]] = {
    "Cukrzyca":                        ("choroby_towarzyszace", "diabetes",              True),
    "Choroba wieńcowa":                ("choroby_towarzyszace", "coronary_artery_disease",True),
    "Choroba wieńcowa – wykluczona":   ("choroby_towarzyszace", "coronary_artery_disease",False),
    "Choroby tarczycy":                ("choroby_towarzyszace", "thyroid_disease",        True),
    "Bezdech senny":                   ("choroby_towarzyszace", "sleep_apnea",            True),
    "Otyłość":                         ("choroby_towarzyszace", "obesity",                True),
    "Migotanie przedsionków":          ("choroby_towarzyszace", "atrial_fibrillation",    True),
    "Udar / TIA":                      ("choroby_towarzyszace", "stroke_or_tia",          True),
    "POChP":                           ("choroby_towarzyszace", "copd",                   True),
    "Przewlekła choroba nerek":        ("choroby_towarzyszace", "chronic_kidney_disease", True),
    "Zmiana masy ciała":               ("choroby_towarzyszace", "weight_change",          True),
    "Palenie – wykluczone":            ("choroby_towarzyszace", "smoking",                False),
    "Duszność":                        ("objawy",               "dyspnea",                True),
    "Duszność – wykluczona":           ("objawy",               "dyspnea",                False),
    "Ból w klatce piersiowej":         ("objawy",               "chest_pain",             True),
    "Ból w klatce piersiowej – wykluczony": ("objawy",          "chest_pain",             False),
    "Kołatanie serca – wykluczone":    ("objawy",               "palpitations",           False),
    "Zawroty głowy":                   ("objawy",               "dizziness",              True),
    "Obrzęki":                         ("objawy",               "edema",                  True),
    "Obrzęki – wykluczone":            ("objawy",               "edema",                  False),
}

# Mapping of the Polish diagnosis name → English key in JSON
PL_DIAG_TO_EN: dict[str, list[str]] = {
    "nadciśnienie tętnicze":       ["hypertension", "nadciśnienie tętnicze", "nadciśnienie_tętnicze"],
    "nadciśnienie_tętnicze":       ["hypertension", "nadciśnienie tętnicze", "nadciśnienie_tętnicze"],
    "nadciśnienie_tętnicze_oporne":["nadciśnienie_tętnicze_oporne", "oporne_nadciśnienie_tętnicze"],
    "oporne_nadciśnienie_tętnicze":["oporne_nadciśnienie_tętnicze", "nadciśnienie_tętnicze_oporne"],
    "cukrzyca":                    ["diabetes"],
    "bezdech senny":               ["sleep_apnea"],
    "choroba wieńcowa":            ["coronary_artery_disease"],
    "choroba wieńcowa stabilna":   ["coronary_artery_disease", "choroba wieńcowa stabilna"],
    "choroby tarczycy":            ["thyroid_disease"],
    "otyłość":                     ["obesity"],
    "migotanie przedsionków":      ["atrial_fibrillation"],
    "udar / tia":                  ["stroke_or_tia"],
    "udar tia":                    ["stroke_or_tia"],
    "pochp":                       ["copd"],
    "przewlekła choroba nerek":    ["chronic_kidney_disease"],
    "dyslipidemia":                ["dyslipidemia", "hyperlipidemia"],
    "dysplipidemia":               ["dyslipidemia", "hyperlipidemia"],
    "hiperlipidemia":              ["hyperlipidemia", "dyslipidemia"],
    "hyperlipidemia":              ["hyperlipidemia", "dyslipidemia"],
    "hiperlipidemia mixed":        ["hyperlipidemia", "hyperlipidemia_mixed"],
    "hyperlipidemia mixed":        ["hyperlipidemia_mixed", "hyperlipidemia"],
    "astma":                       ["asthma"],
    "palenie":                     ["smoking"],
    "pcos":                        ["pcos"],
    "zawroty głowy":               ["dizziness"],
    "anemia":                      ["anemia"],
    "anemia microcytic":           ["anemia_microcytic", "anemia"],
    "białkomocz":                  ["białkomocz", "proteinuria"],
    "hiperurykemia":               ["hiperurykemia", "hyperuricemia"],
    "hyperkalemia":                ["hyperkalemia"],
    "dna moczanowa":               ["dna moczanowa", "gout"],
    "osteoporoza":                 ["osteoporoza", "osteoporosis"],
    "retinopatia cukrzycowa":      ["retinopatia_cukrzycowa"],
}

THERAPY_MAP = {"dobre": "good", "częściowe": "partial", "słabe": "poor"}

# Well-being: 7 allowed values + an alias for the concatenated value from the schema enum (D3).
WELLBEING_VALUES = ["dobre", "dość dobre", "lepsze", "średnie", "gorsze", "skargi na dolegliwości", "złe"]
WELLBEING_ALIASES = {"średnieskargi na dolegliwości": "średnie"}
# Ordinal scale (best → worst) for the classification_error distance subtype; „lepsze” ("better")
# is a relative assessment and has no position on the scale.
WELLBEING_SCALE = ["dobre", "dość dobre", "średnie", "gorsze", "skargi na dolegliwości", "złe"]

# ─── Categories and fields ─────────────────────────────────────────────────────

CATEGORIES = [
    "ABPM", "Comorbidities", "Diagnosis", "Medication",
    "Symptoms", "Well-being", "Therapy adherence",
]
BINARY_CATEGORIES = ("Comorbidities", "Symptoms")
LIST_CATEGORIES = ("Medication", "Diagnosis")
CATEGORICAL_CATEGORIES = ("Well-being", "Therapy adherence")
_GROUP_TO_CATEGORY = {"choroby_towarzyszace": "Comorbidities", "objawy": "Symptoms"}
_CATEGORY_TO_GROUP = {v: k for k, v in _GROUP_TO_CATEGORY.items()}

# Evaluated binary fields = fields for which a positive GT label exists (D1, D2).
EVALUATED_FIELDS: dict[str, list[str]] = {cat: [] for cat in BINARY_CATEGORIES}
for _group, _key, _expected in BINARY_LABEL_MAP.values():
    _fields = EVALUATED_FIELDS[_GROUP_TO_CATEGORY[_group]]
    if _expected and _key not in _fields:
        _fields.append(_key)
# Extraction-schema fields (main_neurosymbolic_v4.FACT_SCHEMA) that are not evaluated, with the reason.
NOT_EVALUATED_FIELDS: dict[str, str] = {
    "asthma": "pozytywne etykiety GT („Astma”) poza schematem ewaluacji (D2)",
    "smoking": "pozytywne etykiety GT („Palenie”) poza schematem (D2), negacje odrzucone (D1)",
    "syncope": "pozytywne etykiety GT („Omdlenia”) poza schematem ewaluacji (D2)",
    "palpitations": "pozytywne etykiety GT („Kołatanie serca”) poza schematem (D2), negacje odrzucone (D1)",
}

# ─── Error labels ──────────────────────────────────────────────────────────────

FN_LABELS = ["invalid_output", "classification_error", "negation_error", "omission"]
FP_LABELS = ["duplicate", "negation_error", "classification_error", "hallucination_entity"]
TP_FLAGS = ["hallucination_evidence"]

# Keys used in Level 2 tables/plots: type + side where the type occurs on both sides.
ERROR_KEYS_FN = ["omission", "invalid_output", "negation_error_fn", "classification_error_fn"]
ERROR_KEYS_FP = ["hallucination_entity", "duplicate", "negation_error_fp", "classification_error_fp"]
ERROR_KEYS_TP = ["hallucination_evidence"]
ERROR_KEYS = ERROR_KEYS_FN + ERROR_KEYS_FP + ERROR_KEYS_TP
_TWO_SIDED = {"negation_error", "classification_error"}


def error_key(error_type: str, side: str) -> str:
    return f"{error_type}_{side.lower()}" if error_type in _TWO_SIDED else error_type


def error_key_side(key: str) -> str:
    if key in ERROR_KEYS_FN:
        return "FN"
    if key in ERROR_KEYS_FP:
        return "FP"
    return "TP"


# ─── Normalization and text ────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ").strip()


def expand_abbreviations(text: str, abbreviations: dict | None = None) -> str:
    """Copy of ``main_neurosymbolic_v4.expand_abbreviations`` (that module imports torch).
    Consistency with the source is checked by tests/test_matching.py::test_expand_abbreviations_matches_pipeline."""
    if abbreviations is None:
        abbreviations = MEDICAL_ABBREVIATIONS

    def replace_abbr(match):
        abbr = match.group(0)
        expansion = abbreviations.get(abbr)
        if expansion:
            return f"{expansion} ({abbr})"
        return abbr

    sorted_abbrs = sorted(abbreviations.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(a) for a in sorted_abbrs) + r")\b"
    return re.sub(pattern, replace_abbr, text)


NEGATION_MARKERS: list[str] = [
    "nie", "bez", "neguje", "wyklucza", "brak", "nie stwierdza", "nie stwierdzono",
    "wykluczono", "zaprzecza", "nie występuje", "nie zgłasza", "nie podaje",
    "odmawia", "nie potwierdza",
]
NEGATION_WINDOW = 50
FUZZY_EVIDENCE_THRESHOLD = 0.90

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
# Sentence/line boundary: a newline or sentence punctuation before whitespace
# (so decimals like "0.77" are not boundaries).
_SENT_BOUNDARY_RE = re.compile(r"\n|[.!?;](?=\s)")


def _normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def find_span_position(span: str, source: str,
                       threshold: float = FUZZY_EVIDENCE_THRESHOLD) -> Optional[tuple[int, int]]:
    """Locate ``span`` inside ``source`` allowing for whitespace/punctuation
    drift. Returns (start, end) char offsets in the *original* source, or
    None if no sufficiently similar window is found."""
    if not span or not source:
        return None

    idx = source.find(span)
    if idx != -1:
        return idx, idx + len(span)

    norm_source = _normalize(source)
    norm_span = _normalize(span)
    if not norm_span:
        return None
    idx = norm_source.find(norm_span)
    if idx != -1:
        approx_len = len(span)
        start = max(0, idx)
        return start, min(len(source), start + approx_len)

    # Fuzzy fallback: slide a window of len(span) (+/-20%) across the source
    # and keep the best SequenceMatcher ratio.
    best_ratio, best_pos = 0.0, None
    win = len(norm_span)
    if win == 0:
        return None
    step = max(1, win // 4)
    for lo in range(0, max(1, len(norm_source) - win + 1), step):
        for wlen in (win, int(win * 0.8), int(win * 1.2)):
            wlen = max(1, wlen)
            window = norm_source[lo:lo + wlen]
            if not window:
                continue
            ratio = difflib.SequenceMatcher(None, norm_span, window).ratio()
            if ratio > best_ratio:
                best_ratio, best_pos = ratio, lo
    if best_pos is not None and best_ratio >= threshold:
        return best_pos, min(len(source), best_pos + win)
    return None


@lru_cache(maxsize=None)
def evidence_span_grounded(span: Optional[str], source: str,
                           threshold: float = FUZZY_EVIDENCE_THRESHOLD) -> bool:
    """True if ``span`` exists in ``source`` verbatim or with >= threshold
    normalized similarity (accounts for minor whitespace/punctuation diffs)."""
    if not span:
        return True  # nothing to ground
    if not source:
        return False
    norm_source = _normalize(source)
    norm_span = _normalize(span)
    if not norm_span:
        return True
    if norm_span in norm_source:
        return True
    return find_span_position(span, source, threshold=threshold) is not None


def _sentence_window(source: str, start: int, end: int, window: int) -> str:
    """Text around ``source[start:end]``: at most ``window`` characters on each side,
    clipped to the sentence/line containing the anchor."""
    lo = max(0, start - window)
    hi = min(len(source), end + window)
    left_bounds = [m.end() for m in _SENT_BOUNDARY_RE.finditer(source, lo, start)]
    if left_bounds:
        lo = left_bounds[-1]
    right = _SENT_BOUNDARY_RE.search(source, end, hi)
    if right:
        hi = right.start()
    return source[lo:hi]


def has_negation_nearby(anchors: Sequence[Optional[str]], source: str,
                        window: int = NEGATION_WINDOW) -> bool:
    """Negation marker in the sentence around the first anchor found (±``window`` characters).
    No anchor in the text → False."""
    return _has_negation_nearby(tuple(anchors), source, window)


@lru_cache(maxsize=None)
def _has_negation_nearby(anchors: tuple, source: str, window: int) -> bool:
    for anchor in anchors:
        if not anchor:
            continue
        pos = find_span_position(anchor, source)
        if pos is None:
            continue
        window_text = _normalize(_sentence_window(source, pos[0], pos[1], window))
        return any(re.search(rf"\b{re.escape(m)}\b", window_text) for m in NEGATION_MARKERS)
    return False


# ─── Gold standard ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoldItem:
    category: str
    key: str                       # JSON field / normalized drug or diagnosis name
    label: str                     # original GT label (first row)
    texts: tuple[str, ...] = ()    # Text column from all merged rows
    value: Optional[str] = None    # value of categorical fields


@dataclass
class GoldStandard:
    docs: list[str]                                   # sorted OPIS (all from the GT file)
    items: dict[str, dict[str, list[GoldItem]]]       # opis → category → entities
    patient: dict[str, str]                           # opis → ID_PACJENTA
    n_rows: int                                       # rows in the GT file
    excluded_rows: Counter                            # reason → number of rows
    excluded_labels: Counter                          # (reason, label) → number of rows

    def n_gt(self, category: Optional[str] = None) -> int:
        return sum(len(cats.get(category, [])) if category else sum(len(v) for v in cats.values())
                   for cats in self.items.values())


def classify_label(label: str) -> tuple[Optional[str], Optional[str], Optional[str], str]:
    """(category, key, value, reason). Category None = rejected label; reason:
    'negated' (D1), 'other' (D2) or 'ok'."""
    if label == "ABPM / Holter":
        return "ABPM", "abpm_holter", None, "ok"
    if label in BINARY_LABEL_MAP:
        group, key, expected = BINARY_LABEL_MAP[label]
        if not expected:
            return None, None, None, "negated"
        return _GROUP_TO_CATEGORY[group], key, None, "ok"
    if label.startswith("Lek: ") and label[5:].strip():
        return "Medication", _norm(label[5:]), None, "ok"
    if label.startswith("Rozpoznanie: ") and label[13:].strip():
        return "Diagnosis", _norm(label[13:]), None, "ok"
    if label.startswith("Samopoczucie: "):
        value = normalize_wellbeing(label[14:])
        if value in WELLBEING_VALUES:
            return "Well-being", "samopoczucie", value, "ok"
    if label.startswith("Stosowanie terapii: "):
        value = THERAPY_MAP.get(label[20:].strip().lower())
        if value:
            return "Therapy adherence", "therapy_adherence", value, "ok"
    return None, None, None, "other"


def normalize_wellbeing(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    return WELLBEING_ALIASES.get(v, v) or None


def load_gold(source: str | Path | Iterable[dict]) -> GoldStandard:
    """Loads GT from a CSV file (OPIS, Text, Label, …) or from a list of rows."""
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = list(source)

    acc: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    patient: dict[str, str] = {}
    excluded_rows: Counter = Counter()
    excluded_labels: Counter = Counter()
    for row in rows:
        opis, label = row["OPIS"], row["Label"]
        patient.setdefault(opis, row.get("ID_PACJENTA", ""))
        acc.setdefault(opis, {})
        cat, key, value, reason = classify_label(label)
        if cat is None:
            excluded_rows[reason] += 1
            excluded_labels[(reason, label)] += 1
            continue
        slot = acc[opis].get((cat, key))
        if slot is None:
            acc[opis][(cat, key)] = {"label": label, "texts": [row.get("Text", "") or ""], "value": value}
            continue
        excluded_rows["duplicate"] += 1
        excluded_labels[("duplicate", label)] += 1
        if cat in CATEGORICAL_CATEGORIES and slot["value"] != value:
            raise ValueError(f"Sprzeczne wartości {cat} w jednym dokumencie: {slot['value']!r} vs {value!r}")
        slot["texts"].append(row.get("Text", "") or "")

    items: dict[str, dict[str, list[GoldItem]]] = {}
    for opis, slots in acc.items():
        by_cat: dict[str, list[GoldItem]] = defaultdict(list)
        for (cat, key), s in sorted(slots.items()):
            by_cat[cat].append(GoldItem(cat, key, s["label"], tuple(s["texts"]), s["value"]))
        items[opis] = dict(by_cat)

    return GoldStandard(
        docs=sorted(items), items=items, patient=patient, n_rows=len(rows),
        excluded_rows=excluded_rows, excluded_labels=excluded_labels,
    )


# ─── Model outputs ─────────────────────────────────────────────────────────────

_EMPTY_ENTITY = {"present": None, "evidence_span": None, "details": None, "confidence": "low"}


def is_invalid_output(facts: dict) -> bool:
    """Skeleton substituted by main_neurosymbolic_v4.py after a JSONDecodeError (D8).
    ``diagnoses_from_medications`` is skipped — Stage 1.5 fills it in after parsing."""
    return (
        facts.get("medications") == []
        and facts.get("diagnoses_from_notes") == []
        and facts.get("field_confidence") == {}
        and facts.get("therapy_adherence") == {"status": None, "confidence": "low"}
        and facts.get("abpm_holter") == {"mentioned": False, "confidence": "low"}
        and facts.get("samopoczucie") == {"category": None, "confidence": "low"}
        and all(v == _EMPTY_ENTITY for v in facts.get("choroby_towarzyszace", {}).values())
        and all(v == _EMPTY_ENTITY for v in facts.get("objawy", {}).values())
    )


@dataclass
class Predictions:
    facts: dict[str, dict]                      # opis → facts (first record, D6)
    model_inputs: dict[str, str] = field(default_factory=dict)  # v5: opis → OPIS_MODEL_INPUT (D5)
    duplicate_opis: list[str] = field(default_factory=list)
    conflicting_opis: list[str] = field(default_factory=list)


def load_predictions(path: str | Path, warn: bool = True) -> Predictions:
    with Path(path).open(encoding="utf-8") as f:
        records = json.load(f)
    facts: dict[str, dict] = {}
    model_inputs: dict[str, str] = {}
    dup: list[str] = []
    conflict: list[str] = []
    for rec in records:
        opis = rec["OPIS"]
        if opis in facts:
            dup.append(opis)
            if json.dumps(rec["facts"], sort_keys=True) != json.dumps(facts[opis], sort_keys=True):
                conflict.append(opis)
            continue
        facts[opis] = rec["facts"]
        if rec.get("OPIS_MODEL_INPUT"):  # v5: text given to the model (D5)
            model_inputs[opis] = rec["OPIS_MODEL_INPUT"]
    if dup and warn:
        msg = (f"{Path(path).name}: {len(dup)} zduplikowanych rekordów OPIS "
               f"({len(conflict)} z różnymi wynikami) — użyto pierwszego rekordu (D6)")
        warnings.warn(msg, stacklevel=2)
        print(f"[OSTRZEŻENIE] {msg}")
    return Predictions(facts=facts, model_inputs=model_inputs,
                       duplicate_opis=dup, conflicting_opis=conflict)


@dataclass(frozen=True)
class PredItem:
    category: str
    key: str                       # JSON field / normalized name / value
    name: str                      # name for reports
    span: Optional[str]
    index: int                     # position in the model output (within the category)
    value: Optional[str] = None


# D9: values meaning "no name" (compared after _norm and stripping dots/spaces)
NULL_LIKE = frozenset({
    "", "null", "none", "unknown", "brak", "brak danych", "nie podano", "nie dotyczy",
    "nieznany", "nieznana", "n/a", "b/d",
})


def is_null_like(name: Optional[str]) -> bool:
    return _norm(name or "").strip(" .") in NULL_LIKE


def prediction_items(facts: dict) -> dict[str, list[PredItem]]:
    """Positive model assertions per category (D1: ``present: false`` is not a prediction)."""
    out: dict[str, list[PredItem]] = {c: [] for c in CATEGORIES}

    abpm = facts.get("abpm_holter") or {}
    if abpm.get("mentioned") is True:
        out["ABPM"].append(PredItem("ABPM", "abpm_holter", "abpm_holter",
                                           abpm.get("evidence_span"), 0))

    for cat in BINARY_CATEGORIES:
        group = facts.get(_CATEGORY_TO_GROUP[cat]) or {}
        for i, key in enumerate(EVALUATED_FIELDS[cat]):
            entry = group.get(key) or {}
            if isinstance(entry, dict) and entry.get("present") is True:
                out[cat].append(PredItem(cat, key, key, entry.get("evidence_span"), i))

    meds = [m for m in (facts.get("medications") or []) if not is_null_like(m.get("name"))]
    for i, m in enumerate(meds):
        out["Medication"].append(PredItem("Medication", _norm(m.get("name") or ""), m.get("name") or "",
                                          m.get("evidence_span"), i))

    diags = list(facts.get("diagnoses_from_notes") or []) + list(facts.get("diagnoses_from_medications") or [])
    diags = [d for d in diags if not is_null_like(d.get("diagnosis"))]
    for i, d in enumerate(diags):
        out["Diagnosis"].append(PredItem("Diagnosis", _norm(d.get("diagnosis") or ""), d.get("diagnosis") or "",
                                         d.get("phrase"), i))

    samo = facts.get("samopoczucie") or {}
    wb = normalize_wellbeing(samo.get("category"))
    if wb:
        out["Well-being"].append(PredItem("Well-being", "samopoczucie", wb, samo.get("evidence_span"), 0, wb))

    ther = facts.get("therapy_adherence") or {}
    status = ther.get("status")
    if status in THERAPY_MAP.values():  # "unknown"/None → no prediction
        out["Therapy adherence"].append(PredItem("Therapy adherence", "therapy_adherence", status,
                                                 ther.get("evidence_span"), 0, status))
    return out


def count_predictions(facts: dict) -> dict[str, int]:
    return {c: len(v) for c, v in prediction_items(facts).items()}


# ─── Matching ──────────────────────────────────────────────────────────────────

def med_edge(pred: PredItem, gold: GoldItem) -> bool:
    """Model medication ↔ GT label 'Lek: X': X is contained in the quote or name (or the name in X).
    Empty names/spans do not match (the empty string is contained in every string)."""
    gt_name = gold.key
    span = _norm(pred.span or "")
    name = pred.key
    if not gt_name:
        return False
    return gt_name in span or gt_name in name or (bool(name) and name in gt_name)


def diag_edge(pred: PredItem, gold: GoldItem) -> bool:
    """Model diagnosis ↔ GT label 'Rozpoznanie: X': Text ↔ phrase, label ↔ diagnosis
    name, PL → EN mapping. Empty phrase/diagnosis do not match."""
    diag_label = gold.key
    phrase = _norm(pred.span or "")
    diag_name = pred.key
    for text in gold.texts:
        t = text.lower()
        if t and phrase and (t in phrase or phrase in t):
            return True
    if not diag_name:
        return False
    if diag_label and (diag_label in diag_name or diag_name in diag_label):
        return True
    en_names = PL_DIAG_TO_EN.get(diag_label, [])
    return any(_norm(e) == diag_name or _norm(e) in diag_name for e in en_names)


def _key_edge(pred: PredItem, gold: GoldItem) -> bool:
    return pred.key == gold.key


def _value_edge(pred: PredItem, gold: GoldItem) -> bool:
    return pred.value == gold.value


EDGES: dict[str, Callable[[PredItem, GoldItem], bool]] = {
    "ABPM": _key_edge, "Comorbidities": _key_edge, "Symptoms": _key_edge,
    "Medication": med_edge, "Diagnosis": diag_edge,
    "Well-being": _value_edge, "Therapy adherence": _value_edge,
}


def match_one_to_one(golds: Sequence[GoldItem], preds: Sequence[PredItem],
                     edge: Callable[[PredItem, GoldItem], bool]) -> list[tuple[int, int]]:
    """Maximum 1:1 matching (Kuhn's algorithm). Returns pairs (GT index, prediction index)
    sorted by GT index."""
    adj = [[j for j, p in enumerate(preds) if edge(p, g)] for g in golds]
    owner: dict[int, int] = {}

    def _augment(gi: int, seen: set[int]) -> bool:
        for j in adj[gi]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or _augment(owner[j], seen):
                owner[j] = gi
                return True
        return False

    for gi in range(len(golds)):
        _augment(gi, set())
    return sorted((gi, j) for j, gi in owner.items())


# ─── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class DocCategoryResult:
    opis: str
    category: str
    gold: list[GoldItem]
    preds: list[PredItem]
    tp: list[tuple[GoldItem, PredItem]]
    fn: list[tuple[GoldItem, str]]
    fp: list[tuple[PredItem, str]]
    tp_ungrounded: list[tuple[GoldItem, PredItem]]

    @property
    def n_gt(self) -> int:
        return len(self.gold)

    @property
    def n_pred(self) -> int:
        return len(self.preds)


@dataclass
class ModelResult:
    model: str
    docs: dict[str, dict[str, DocCategoryResult]]     # opis → category → result
    invalid_docs: set[str]
    missing_docs: set[str]
    duplicate_opis: list[str]
    conflicting_opis: list[str]

    def iter_results(self) -> Iterable[DocCategoryResult]:
        for opis in sorted(self.docs):
            for cat in CATEGORIES:
                yield self.docs[opis][cat]


def wellbeing_subtype(gold_value: Optional[str], pred_value: Optional[str]) -> Optional[str]:
    try:
        return f"distance={abs(WELLBEING_SCALE.index(gold_value) - WELLBEING_SCALE.index(pred_value))}"
    except ValueError:
        return "value_outside_defined_scale"


def evaluate_document(opis: str, gold_by_cat: dict[str, list[GoldItem]], facts: Optional[dict],
                      model_input: Optional[str] = None) -> dict[str, DocCategoryResult]:
    invalid = facts is None or is_invalid_output(facts)
    preds_by_cat = prediction_items(facts) if facts is not None else {c: [] for c in CATEGORIES}
    # v5: the text the model saw (D5): OPIS_MODEL_INPUT or reconstructed from OPIS
    source = model_input if model_input else expand_abbreviations(opis)

    out: dict[str, DocCategoryResult] = {}
    for cat in CATEGORIES:
        golds = gold_by_cat.get(cat, [])
        preds = preds_by_cat[cat]
        pairs = match_one_to_one(golds, preds, EDGES[cat])
        matched_g = {gi for gi, _ in pairs}
        matched_p = {pj for _, pj in pairs}

        tp = [(golds[gi], preds[pj]) for gi, pj in pairs]
        tp_ungrounded = [(g, p) for g, p in tp if p.span and not evidence_span_grounded(p.span, source)]

        fn: list[tuple[GoldItem, str]] = []
        for gi, g in enumerate(golds):
            if gi in matched_g:
                continue
            if invalid:
                label = "invalid_output"
            elif cat in CATEGORICAL_CATEGORIES and preds:
                label = "classification_error"
            elif cat in BINARY_CATEGORIES and (
                    ((facts.get(_CATEGORY_TO_GROUP[cat]) or {}).get(g.key) or {}).get("present") is False):
                label = "negation_error"  # the model explicitly negated an entity present in GT
            else:
                label = "omission"
            fn.append((g, label))

        fp: list[tuple[PredItem, str]] = []
        seen_keys: set[str] = set()
        for pj, p in enumerate(preds):
            earlier = p.key in seen_keys
            seen_keys.add(p.key)
            if pj in matched_p:
                continue
            if cat in LIST_CATEGORIES and (earlier or any(EDGES[cat](p, g) for g in golds)):
                label = "duplicate"  # repeated prediction, or the GT entity is already taken by another prediction
            elif cat in BINARY_CATEGORIES + ("ABPM",) and has_negation_nearby([p.span], source):
                label = "negation_error"  # the model reported an entity that the text negates
            elif cat in CATEGORICAL_CATEGORIES and golds:
                label = "classification_error"
            else:
                label = "hallucination_entity"
            fp.append((p, label))

        out[cat] = DocCategoryResult(opis, cat, list(golds), list(preds), tp, fn, fp, tp_ungrounded)
    return out


def evaluate_model(gold: GoldStandard, preds: Predictions, model: str = "") -> ModelResult:
    docs: dict[str, dict[str, DocCategoryResult]] = {}
    invalid: set[str] = set()
    missing: set[str] = set()
    for opis in gold.docs:
        facts = preds.facts.get(opis)
        if facts is None:
            missing.add(opis)
        if facts is None or is_invalid_output(facts):
            invalid.add(opis)
        docs[opis] = evaluate_document(opis, gold.items.get(opis, {}), facts,
                                       preds.model_inputs.get(opis))
    if missing:
        msg = f"{model}: {len(missing)} dokumentów GT bez rekordu wyjściowego — traktowane jak invalid_output"
        warnings.warn(msg, stacklevel=2)
        print(f"[OSTRZEŻENIE] {msg}")
    result = ModelResult(model, docs, invalid, missing, preds.duplicate_opis, preds.conflicting_opis)
    check_invariants(result, gold, preds)
    return result


_CACHE: dict[tuple, ModelResult] = {}


def evaluate_model_file(gold: GoldStandard, path: str | Path, model: str = "") -> ModelResult:
    """``evaluate_model`` with a cache (key: path, mtime, GT)."""
    p = Path(path)
    key = (str(p.resolve()), p.stat().st_mtime_ns, id(gold))
    if key not in _CACHE:
        _CACHE[key] = evaluate_model(gold, load_predictions(p), model or p.name)
    return _CACHE[key]


# ─── Assertions ────────────────────────────────────────────────────────────────

def check_invariants(result: ModelResult, gold: GoldStandard, preds: Predictions) -> None:
    """Aborts the run on any count mismatch."""
    def _fail(msg: str) -> None:
        raise AssertionError(f"[{result.model}] {msg}")

    for opis, cats in result.docs.items():
        facts = preds.facts.get(opis)
        n_pred_indep = count_predictions(facts) if facts is not None else {c: 0 for c in CATEGORIES}
        for cat, r in cats.items():
            where = f"{opis[:40]!r} / {cat}"
            gold_ids = {id(g) for g, _ in r.tp}
            pred_ids = {p.index for _, p in r.tp}
            if not (len(gold_ids) == len(pred_ids) == len(r.tp)):
                _fail(f"TP_gt ≠ TP_pred: {where}")
            if len(r.tp) + len(r.fn) != len(gold.items.get(opis, {}).get(cat, [])):
                _fail(f"TP + FN ≠ n_gt: {where}")
            if len(r.tp) + len(r.fp) != n_pred_indep[cat]:
                _fail(f"TP + FP ≠ n_pred: {where}")
            if any(l not in FN_LABELS for _, l in r.fn) or any(l not in FP_LABELS for _, l in r.fp):
                _fail(f"nieznana etykieta: {where}")
            if opis in result.invalid_docs and (r.tp or r.fp):
                _fail(f"dokument invalid_output ma TP/FP: {where}")


# ─── Counts ────────────────────────────────────────────────────────────────────

COUNT_COLUMNS = (["tp", "fp", "fn", "n_gt", "n_pred", "tp_ungrounded"]
                 + [f"fp_{l}" for l in FP_LABELS] + [f"fn_{l}" for l in FN_LABELS])


def counts_rows(result: ModelResult, patient: Optional[dict[str, str]] = None) -> list[dict]:
    """One row per (model, document, category) — a single source of counts for all tables."""
    rows: list[dict] = []
    for r in result.iter_results():
        fp_c = Counter(l for _, l in r.fp)
        fn_c = Counter(l for _, l in r.fn)
        row = {
            "model": result.model, "opis": r.opis,
            "patient": (patient or {}).get(r.opis, ""), "category": r.category,
            "invalid_output_doc": r.opis in result.invalid_docs,
            "tp": len(r.tp), "fp": len(r.fp), "fn": len(r.fn),
            "n_gt": r.n_gt, "n_pred": r.n_pred, "tp_ungrounded": len(r.tp_ungrounded),
        }
        row.update({f"fp_{l}": fp_c[l] for l in FP_LABELS})
        row.update({f"fn_{l}": fn_c[l] for l in FN_LABELS})
        rows.append(row)
    return rows


def prf(tp: int, fp: int, fn: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return p, r, f1


def write_counts_csv(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "patient", "opis", "category", "invalid_output_doc"] + COUNT_COLUMNS
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
