"""
Extraction pipeline: structured facts from hypertension clinic visit notes (schema v5).

For each visit note:
  Stage 1   — Neural extraction: the LLM (vLLM, JSON-schema-constrained decoding with
              xgrammar) returns the facts defined in FACT_SCHEMA.
  Stage 1.5 — Symbolic enrichment: each medication gets a drug class from DRUG_ONTOLOGY,
              and diagnoses implied by current medications are derived
              (DRUG_CLASS_TO_DIAGNOSIS → facts.diagnoses_from_medications).

Output: one JSON record per document (DOC_KEY, ID_PACJENTA, OPIS, OPIS_MODEL_INPUT, facts,
generation) and <output>_run_meta.json with package versions, sampling parameters, hashes of
the prompt and schema, and CLI arguments of every run.

The v4 validation stages (RuleEngine, independent regex extraction + agreement scoring,
longitudinal validation, self-consistency, LLM judge, stratification) were removed. They did
not change the extracted facts and were disabled (--skip-*) in the reported runs.

v5 changes relative to v4 (details: v5_changes.md, v5_preflight.md; in code: `# v5:`):
  - generation logging (finish_reason, tokens, parse_ok, raw_output) + <output>_run_meta.json
  - well-being enum without the concatenated value; medications: all (not only antihypertensives)
  - rule-E examples replaced with drugs absent from the data
  - OPIS_MODEL_INPUT in the record, --no-expand-abbreviations
  - Qwen3: enable_thinking=False; resuming by DOC_KEY
  - schema: no field_confidence, maxItems, maxLength; grammar without arbitrary whitespace
  - CLI: --max-tokens, --repetition-penalty, --max-docs, --max-patient-id
"""

import os
import re
import sys
import json
import hashlib
import argparse
import statistics
import importlib.metadata
from datetime import datetime
from typing import Any

import pandas as pd

# v5: torch / transformers / vllm are imported in main() so that the module (schema, prompt)
#     can be imported in tests without a GPU and without vllm.

from schema import MEDICAL_ABBREVIATIONS


# v5: schema/prompt version saved in run_meta
SCHEMA_VERSION = "v5"

# v5: array limits in FACT_SCHEMA (3b). Max in GT: 20 medications per document → 40 is safe.
MAX_ITEMS_MEDICATIONS = 40
MAX_ITEMS_DIAGNOSES_FROM_NOTES = 15
MAX_ITEMS_DIAGNOSES_FROM_MEDICATIONS = 15

# v5: string length limits (3c). xgrammar 0.1.29 enforces them, but a string with maxLength
#     disallows escape sequences and allows a raw TAB → parse with strict=False.
MAX_LEN_EVIDENCE = 300   # evidence_span, phrase
MAX_LEN_SHORT = 150      # details, side_effects, result_text, frequency

# v5: CLI defaults (3e, 3g)
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_PATIENT_ID = 121

# v5: well-being categories = GT labels without the artefact „średnieskargi na dolegliwości”
#     (3 GT rows; matching.py maps it to „średnie”, decision D3) — see v5_preflight.md §2.1
WELLBEING_CATEGORIES: list[str] = [
    "dobre", "dość dobre", "lepsze", "średnie", "gorsze",
    "skargi na dolegliwości", "złe",
]

# v5: drugs from the rule-E examples of the v4 prompt — for counting their share of FPs (2c)
V4_PROMPT_EXAMPLE_DRUGS: list[str] = [
    "telmizek", "telmisartan", "concor", "bisoprolol", "polpril", "ramipril",
    "acard", "roswera", "rozuwastatyna", "tertensif", "indapamid",
    "ozempic", "semaglutyd", "euthyrox", "lewotyroksyna",
    "xarelto", "rywaroksaban", "eliquis", "apiksaban",
]

# v5: new rule-E examples — 0 occurrences in notes, GT and drug mapping (v5_preflight.md §4)
V5_PROMPT_EXAMPLE_DRUGS: list[tuple[str, str]] = [
    ("Edarbi", "azylsartan"),
    ("Rasilez", "aliskiren"),
    ("Ranexa", "ranolazyna"),
    ("Lescol", "fluwastatyna"),
    ("Minipress", "prazosyna"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DRUG ONTOLOGY + DRUG → DIAGNOSIS MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

DRUG_ONTOLOGY: dict[str, str] = {
    # ── Antihypertensives ────────────────────────────────────────────────────
    # ACE-I
    "ramipril": "ACEI", "perindopril": "ACEI", "enalapril": "ACEI",
    "lisinopril": "ACEI", "captopril": "ACEI", "chinapril": "ACEI",
    "trandolapril": "ACEI", "cilazapril": "ACEI", "zofenopril": "ACEI",
    "polpril": "ACEI", "prenessa": "ACEI", "prestarium": "ACEI",
    "hartil": "ACEI", "vivace": "ACEI", "inhibace": "ACEI",
    "accupro": "ACEI", "enarenal": "ACEI", "gopten": "ACEI",
    "primacor": "ACEI", "tritace": "ACEI",
    # ARB
    "walsartan": "ARB", "losartan": "ARB", "telmisartan": "ARB",
    "kandesartan": "ARB", "irbesartan": "ARB", "olmesartan": "ARB",
    "eprosartan": "ARB", "walsacor": "ARB", "valsacor": "ARB",
    "cozaar": "ARB", "diovan": "ARB", "telmizek": "ARB",
    "micardis": "ARB", "atacand": "ARB", "aprovel": "ARB",
    "lorista": "ARB", "gisartan": "ARB", "axudan": "ARB",
    "valzek": "ARB", "olmecip": "ARB",
    # CCB
    "amlodypina": "CCB", "lacydypina": "CCB", "lerkanidypina": "CCB",
    "nitrendypina": "CCB", "felodypina": "CCB", "nifedypina": "CCB",
    "werapamil": "CCB", "diltiazem": "CCB",
    "amlessa": "CCB", "lacipil": "CCB", "lercan": "CCB",
    "plendil": "CCB", "cordaflex": "CCB", "isoptin": "CCB",
    "norvasc": "CCB",
    # Beta-blocker
    "bisoprolol": "BB", "metoprolol": "BB", "nebiwolol": "BB",
    "karwedilol": "BB", "atenolol": "BB", "betaksolol": "BB",
    "propranolol": "BB",
    "bisocard": "BB", "metocard": "BB", "ebivol": "BB",
    "concor": "BB", "betaloc": "BB", "lokren": "BB",
    "dilatrend": "BB", "avedol": "BB", "corvitol": "BB",
    # Diuretics — thiazide / thiazide-like
    "hydrochlorotiazyd": "DIURETIC_THIAZIDE",
    "indapamid": "DIURETIC_THIAZIDE",
    "chlortalidon": "DIURETIC_THIAZIDE",
    "tertensif": "DIURETIC_THIAZIDE",
    "hypothiazid": "DIURETIC_THIAZIDE",
    "indapen": "DIURETIC_THIAZIDE",
    # Diuretics — loop
    "torasemid": "DIURETIC_LOOP", "furosemid": "DIURETIC_LOOP",
    "torsemed": "DIURETIC_LOOP", "furosemidum": "DIURETIC_LOOP",
    # MRA
    "spironolakton": "MRA", "eplerenon": "MRA",
    "aldactone": "MRA", "inspra": "MRA", "spironol": "MRA",
    # Alpha-blocker
    "doksazosyna": "ALPHA", "terazosyna": "ALPHA",
    "cardura": "ALPHA", "doxar": "ALPHA", "kamiren": "ALPHA",
    # Central
    "klonidyna": "CENTRAL", "rilmenidyna": "CENTRAL",
    "moksonidyna": "CENTRAL", "iporel": "CENTRAL", "moxogamma": "CENTRAL",

    # ── Antidiabetics ─────────────────────────────────────────────────────
    "metformina": "ANTIDIABETIC", "metformax": "ANTIDIABETIC",
    "siofor": "ANTIDIABETIC", "glucophage": "ANTIDIABETIC",
    "gliklazyd": "ANTIDIABETIC", "diaprel": "ANTIDIABETIC",
    "glimepiryd": "ANTIDIABETIC", "amaryl": "ANTIDIABETIC",
    "empagliflozyna": "ANTIDIABETIC", "jardiance": "ANTIDIABETIC",
    "dapagliflozyna": "ANTIDIABETIC", "forxiga": "ANTIDIABETIC",
    "kanagliflozyna": "ANTIDIABETIC", "invokana": "ANTIDIABETIC",
    "liraglutyd": "ANTIDIABETIC", "victoza": "ANTIDIABETIC",
    "semaglutyd": "ANTIDIABETIC", "ozempic": "ANTIDIABETIC",
    "rybelsus": "ANTIDIABETIC", "trulicity": "ANTIDIABETIC",
    "sitagliptyna": "ANTIDIABETIC", "januvia": "ANTIDIABETIC",
    "insulina": "ANTIDIABETIC", "lantus": "ANTIDIABETIC",
    "humalog": "ANTIDIABETIC", "novomix": "ANTIDIABETIC",

    # ── Statins ───────────────────────────────────────────────────────────
    "atorwastatyna": "STATIN", "rozuwastatyna": "STATIN",
    "simwastatyna": "STATIN", "fluwastatyna": "STATIN",
    "prawastatyna": "STATIN",
    "torvacard": "STATIN", "roswera": "STATIN", "rosucard": "STATIN",
    "atoris": "STATIN", "tulip": "STATIN", "sortis": "STATIN",
    "zocor": "STATIN", "vasilip": "STATIN",

    # ── Antiplatelet drugs ──────────────────────────────────────────────
    "acard": "ANTIPLATELET", "polocard": "ANTIPLATELET",
    "kwas acetylosalicylowy": "ANTIPLATELET",  # v5: active substance of Acard (2c)
    "aspiryna": "ANTIPLATELET", "asa": "ANTIPLATELET",
    "klopidogrel": "ANTIPLATELET", "plavix": "ANTIPLATELET",
    "areplex": "ANTIPLATELET", "trombex": "ANTIPLATELET",
    "tikagrelor": "ANTIPLATELET", "brilique": "ANTIPLATELET",
    "prasugrel": "ANTIPLATELET", "efient": "ANTIPLATELET",

    # ── Anticoagulants ────────────────────────────────────────────────────
    "warfaryna": "ANTICOAGULANT", "acenokumarol": "ANTICOAGULANT",
    "sintrom": "ANTICOAGULANT",
    "rywaroksaban": "ANTICOAGULANT", "xarelto": "ANTICOAGULANT",
    "apiksaban": "ANTICOAGULANT", "eliquis": "ANTICOAGULANT",
    "edoksaban": "ANTICOAGULANT", "roteas": "ANTICOAGULANT",
    "lixiana": "ANTICOAGULANT",
    "dabigatran": "ANTICOAGULANT", "pradaxa": "ANTICOAGULANT",

    # ── Thyroid hormones ──────────────────────────────────────────────────
    "lewotyroksyna": "THYROID_HORMONE", "letrox": "THYROID_HORMONE",
    "euthyrox": "THYROID_HORMONE", "eltroxin": "THYROID_HORMONE",
    "l-tyroksyna": "THYROID_HORMONE",
    "tiamazol": "ANTITHYROID", "metizol": "ANTITHYROID",
    "thyrozol": "ANTITHYROID", "propylotiouracyl": "ANTITHYROID",

    # ── Asthma / COPD ─────────────────────────────────────────────────────
    "salbutamol": "BRONCHODILATOR", "ventolin": "BRONCHODILATOR",
    "formoterol": "BRONCHODILATOR", "salmeterol": "BRONCHODILATOR",
    "tiotropium": "BRONCHODILATOR", "spiriva": "BRONCHODILATOR",
    "indakaterol": "BRONCHODILATOR",
    "budesonid": "ICS", "flutikazon": "ICS", "beklometazon": "ICS",
    "pulmicort": "ICS", "seretide": "ICS", "symbicort": "ICS",
    "fostex": "ICS",
}

# Drug class → suggested diagnosis
DRUG_CLASS_TO_DIAGNOSIS: dict[str, str] = {
    "ANTIDIABETIC":     "diabetes",
    "STATIN":           "dyslipidemia",
    "ANTIPLATELET":     "coronary_artery_disease",
    "ANTICOAGULANT":    "atrial_fibrillation",
    "THYROID_HORMONE":  "hypothyroidism",
    "ANTITHYROID":      "hyperthyroidism",
    "BRONCHODILATOR":   "obstructive_lung_disease",
    "ICS":              "obstructive_lung_disease",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  JSON SCHEMAS — what the LLM must produce (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

BOOLEAN_FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {
            "type": ["boolean", "null"],
            "description": "true = tekst potwierdza, false = tekst zaprzecza, null = tekst nie wspomina."
        },
        "evidence_span": {
            "type": ["string", "null"],
            "description": "Dosłowny fragment tekstu źródłowego. Gdy present=true MUSI być niepusty.",
            "maxLength": MAX_LEN_EVIDENCE,  # v5: 3c
        },
        "details": {
            "type": ["string", "null"],
            "description": "Opcjonalny szczegół (np. wartość AHI, klasa NYHA, postać AF).",
            "maxLength": MAX_LEN_SHORT,  # v5: 3c
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["present", "evidence_span", "confidence"],
    "additionalProperties": False,
}

MEDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "name":       {"type": "string"},
        "dose_mg":    {"type": ["number", "null"]},
        "frequency":  {"type": ["string", "null"], "maxLength": MAX_LEN_SHORT},  # v5: 3c
        "is_current": {"type": "boolean"},
        "efficacy":   {"type": ["string", "null"],
                       "enum": ["effective", "partially_effective", "ineffective",
                                "unknown", None]},
        "tolerance":  {"type": ["string", "null"],
                       "enum": ["good", "poor", "unknown", None]},
        "side_effects": {"type": ["string", "null"], "maxLength": MAX_LEN_SHORT},  # v5: 3c
        "evidence_span": {"type": "string", "maxLength": MAX_LEN_EVIDENCE},  # v5: 3c
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["name", "is_current", "evidence_span", "confidence"],
    "additionalProperties": False,
}

ABPM_HOLTER_SCHEMA = {
    "type": "object",
    "properties": {
        "mentioned": {"type": "boolean"},
        "keyword": {"type": ["string", "null"]},
        "result_text": {"type": ["string", "null"], "maxLength": MAX_LEN_SHORT},  # v5: 3c
        "date": {"type": ["string", "null"]},
        "evidence_span": {"type": ["string", "null"], "maxLength": MAX_LEN_EVIDENCE},  # v5: 3c
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["mentioned", "confidence"],
    "additionalProperties": False,
}

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "visit_date": {"type": ["string", "null"]},
        "medications": {"type": "array", "items": MEDICATION_SCHEMA,
                        "maxItems": MAX_ITEMS_MEDICATIONS},  # v5: 3b
        "therapy_adherence": {
            "type": "object",
            "properties": {
                "status":   {"type": ["string", "null"],
                             "enum": ["good", "partial", "poor", "unknown", None]},
                "details":  {"type": ["string", "null"], "maxLength": MAX_LEN_SHORT},  # v5: 3c
                "evidence_span": {"type": ["string", "null"],
                                  "maxLength": MAX_LEN_EVIDENCE},  # v5: 3c
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["status", "confidence"],
            "additionalProperties": False,
        },
        "choroby_towarzyszace": {
            "type": "object",
            "properties": {
                "diabetes":                 BOOLEAN_FINDING_SCHEMA,
                "chronic_kidney_disease":   BOOLEAN_FINDING_SCHEMA,
                "sleep_apnea":              BOOLEAN_FINDING_SCHEMA,
                "coronary_artery_disease":  BOOLEAN_FINDING_SCHEMA,
                "stroke_or_tia":            BOOLEAN_FINDING_SCHEMA,
                "atrial_fibrillation":      BOOLEAN_FINDING_SCHEMA,
                "asthma":                   BOOLEAN_FINDING_SCHEMA,
                "copd":                     BOOLEAN_FINDING_SCHEMA,
                "thyroid_disease":          BOOLEAN_FINDING_SCHEMA,
                "obesity":                  BOOLEAN_FINDING_SCHEMA,
                "smoking":                  BOOLEAN_FINDING_SCHEMA,
                "weight_change":            BOOLEAN_FINDING_SCHEMA,
            },
            "required": [
                "diabetes", "chronic_kidney_disease", "sleep_apnea",
                "coronary_artery_disease", "stroke_or_tia", "atrial_fibrillation",
                "asthma", "copd", "thyroid_disease", "obesity", "smoking",
                "weight_change",
            ],
            "additionalProperties": False,
        },
        "abpm_holter": ABPM_HOLTER_SCHEMA,
        "diagnoses_from_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "diagnosis":  {"type": "string"},
                    "phrase":     {"type": "string", "maxLength": MAX_LEN_EVIDENCE},  # v5: 3c
                    "icd10":      {"type": ["string", "null"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["diagnosis", "phrase", "confidence"],
                "additionalProperties": False,
            },
            "maxItems": MAX_ITEMS_DIAGNOSES_FROM_NOTES,  # v5: 3b
        },
        "diagnoses_from_medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "diagnosis":         {"type": "string"},
                    "supporting_drug":   {"type": "string"},
                    "supporting_class":  {"type": "string"},
                },
                "required": ["diagnosis", "supporting_drug", "supporting_class"],
                "additionalProperties": False,
            },
            "maxItems": MAX_ITEMS_DIAGNOSES_FROM_MEDICATIONS,  # v5: 3b
        },
        "objawy": {
            "type": "object",
            "properties": {
                "dizziness":    BOOLEAN_FINDING_SCHEMA,
                "syncope":      BOOLEAN_FINDING_SCHEMA,
                "chest_pain":   BOOLEAN_FINDING_SCHEMA,
                "dyspnea":      BOOLEAN_FINDING_SCHEMA,
                "edema":        BOOLEAN_FINDING_SCHEMA,
                "palpitations": BOOLEAN_FINDING_SCHEMA,
            },
            "required": ["dizziness", "syncope", "chest_pain",
                         "dyspnea", "edema", "palpitations"],
            "additionalProperties": False,
        },
        "samopoczucie": {
            "type": "object",
            "properties": {
                # v5: 2a — in v4 a missing comma after "średnie" concatenated the literals
                "category": {
                    "type": ["string", "null"],
                    "enum": WELLBEING_CATEGORIES + [None],
                },
                "evidence_span": {"type": ["string", "null"],
                                  "maxLength": MAX_LEN_EVIDENCE},  # v5: 3c
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["category", "confidence"],
            "additionalProperties": False,
        },
        # v5: 3a — removed "field_confidence" (an object with arbitrary keys allowed
        #     the grammar to generate keys endlessly)
    },
    "required": [
        "medications", "therapy_adherence", "choroby_towarzyszace",
        "abpm_holter", "diagnoses_from_notes", "objawy", "samopoczucie",
    ],
    "additionalProperties": False,
}


def _check_wellbeing_enum() -> None:
    """v5: 2a — well-being enum = WELLBEING_CATEGORIES + null, without concatenated values."""
    enum = FACT_SCHEMA["properties"]["samopoczucie"]["properties"]["category"]["enum"]
    values = [v for v in enum if v is not None]
    assert None in enum, "enum samopoczucia musi zawierać null"
    assert values == WELLBEING_CATEGORIES, f"nieoczekiwany enum samopoczucia: {values}"
    assert len(set(values)) == len(values), "duplikaty w enumie samopoczucia"
    for v in values:
        for a in values:
            for b in values:
                assert v != a + b and v != a + " " + b, f"sklejona wartość w enumie: {v!r}"


_check_wellbeing_enum()

COMORBIDITY_KEYS = [
    "diabetes", "chronic_kidney_disease", "sleep_apnea",
    "coronary_artery_disease", "stroke_or_tia", "atrial_fibrillation",
    "asthma", "copd", "thyroid_disease", "obesity", "smoking",
    "weight_change",
]

SYMPTOM_KEYS = [
    "dizziness", "syncope", "chest_pain", "dyspnea", "edema", "palpitations",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  STAGE 1.5 — SYMBOLIC ENRICHMENT (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_medications_with_ontology(facts: dict,
                                      ontology: dict[str, str] | None = None) -> dict:
    ont = ontology or DRUG_ONTOLOGY
    for med in facts.get("medications") or []:
        name = (med.get("name") or "").lower().strip()
        med["drug_class"] = ont.get(name, "UNKNOWN")
    return facts


def derive_diagnoses_from_medications(facts: dict,
                                       ontology: dict[str, str] | None = None,
                                       class_to_dx: dict[str, str] | None = None) -> dict:
    ont = ontology or DRUG_ONTOLOGY
    mapping = class_to_dx or DRUG_CLASS_TO_DIAGNOSIS
    derived: dict[str, dict] = {}
    for med in facts.get("medications") or []:
        if not med.get("is_current"):
            continue
        name = (med.get("name") or "").lower().strip()
        cls = ont.get(name)
        if cls and cls in mapping:
            dx = mapping[cls]
            if dx not in derived:
                derived[dx] = {
                    "diagnosis": dx,
                    "supporting_drug": name,
                    "supporting_class": cls,
                }
    facts["diagnoses_from_medications"] = list(derived.values())
    return facts


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EXTRACTION PROMPT (Stage 1 — Neural; v5: items 2a, 2b, 2c, 3a)
# ═══════════════════════════════════════════════════════════════════════════════

# v5: 2a — the category list in the prompt is generated from the same constant as the schema enum
_WELLBEING_PROMPT_LIST = ", ".join(f'"{c}"' for c in WELLBEING_CATEGORIES)
# v5: 2c — examples of trade names absent from the data (V5_PROMPT_EXAMPLE_DRUGS)
_DRUG_EXAMPLES_PROMPT = ", ".join(f"{b} → {g}" for b, g in V5_PROMPT_EXAMPLE_DRUGS)

EXTRACTION_SYSTEM_PROMPT = """\
Jesteś systemem ekstrakcji USTRUKTURYZOWANYCH FAKTÓW z notatek wizyt w poradni nadciśnienia tętniczego. Twoim zadaniem jest ekstrakcja informacji zgodnie z poniższym ZAKRESEM.

## ZAKRES EKSTRAKCJI

1. PRZEBIEG NT (tylko leki + przestrzeganie terapii):
   - medications: lista WSZYSTKICH leków wymienionych w notatce, obecnych i przeszłych —
     nie tylko hipotensyjnych (także m.in. przeciwcukrzycowe, hipolipemizujące,
     przeciwpłytkowe, przeciwzakrzepowe, hormony tarczycy i wszystkie pozostałe).
     Każdy lek to osobny obiekt z polami: name (generic, małe litery), dose_mg,
     frequency, is_current, efficacy, tolerance, side_effects, evidence_span, confidence.
   - therapy_adherence: jak pacjent przestrzega terapii.

2. CHOROBY TOWARZYSZĄCE (12 pól, każde to obiekt {present, evidence_span, details, confidence}):
   - diabetes (cukrzyca)
   - chronic_kidney_disease (CKD / PChN)
   - sleep_apnea (OBS / OSA / bezdech senny; w details podaj AHI jeśli jest)
   - coronary_artery_disease (CAD / MI / CHW / CHNS / zawał / po PCI/CABG)
   - stroke_or_tia (udar / TIA)
   - atrial_fibrillation (AF / AFIB / AFL / migotanie / trzepotanie przedsionków)
   - asthma (astma)
   - copd (POChP / POCHP)
   - thyroid_disease (choroby tarczycy — nadczynność, niedoczynność, Hashimoto)
   - obesity (otyłość; BMI ≥ 30)
   - smoking (palenie tytoniu — obecne, przeszłe lub nigdy)
   - weight_change (znacząca zmiana masy ciała; w details: kierunek i wielkość)

3. ABPM / HOLTER CIŚNIENIOWY (informacja o badaniu wykonanym na danej wizycie):
   - mentioned: czy pojawia się hasło ABPM lub Holter ciśnieniowy w tekście
   - keyword: dosłowne wystąpienie hasła
   - result_text: krótki opis wyniku jeśli podany
   - date: data wykonania badania jeśli podana
   - evidence_span: dosłowny cytat

4. ROZPOZNANIA Z NOTATKI (diagnoses_from_notes): lista rozpoznań z sekcji
   "rozpoznania", "wywiad chorobowy" itp. Każde rozpoznanie ma:
   - diagnosis: ID z LISTY z punktu 2 (np. "diabetes", "coronary_artery_disease",
     "atrial_fibrillation"). Jeśli rozpoznania nie ma w liście, NIE dodawaj go.
   - phrase: dosłowny fragment tekstu
   - icd10 (opcjonalnie)
   - confidence
   UWAGA: diagnoses_from_medications zostawiaj jako pustą tablicę — to pole
   wypełnia warstwa symboliczna na podstawie ontologii leków.

5. OBJAWY (6 pól, format jak dla chorób towarzyszących):
   - dizziness (zawroty głowy)
   - syncope (omdlenia)
   - chest_pain (ból w klatce piersiowej)
   - dyspnea (duszność)
   - edema (obrzęki)
   - palpitations (kołatania serca)

6. SAMOPOCZUCIE: kategoria globalna pacjenta z listy:
   {wellbeing_list}
   Dopasuj do najbliższej. Jeśli tekst nie pozwala na zaklasyfikowanie, użyj null.

## ZASADY OGÓLNE

A. Wydobywaj TYLKO informacje EXPLICITE obecne w tekście — nie wnioskuj.
B. Dla pól boolean (present): true=tekst potwierdza, false=tekst zaprzecza,
   null=tekst nie wspomina.
C. EVIDENCE_SPAN: dla KAŻDEGO faktu z present=true MUSI być DOSŁOWNY fragment
   tekstu źródłowego skopiowany verbatim.
D. CONFIDENCE: "high" (jednoznaczne), "medium" (wymaga drobnej interpretacji),
   "low" (niejednoznaczne/pośrednie).
E. LEKI: nazwy handlowe konwertuj na generyczne (małe litery). Przykłady:
   {drug_examples}.

## SKRÓTY MEDYCZNE
NT = nadciśnienie tętnicze, BP/RR = ciśnienie, ABPM = ambulatoryjne monitorowanie
ciśnienia, AF/AFIB = migotanie przedsionków, AFL = trzepotanie, TIA = przejściowy
atak niedokrwienny, CAD/CHW/CHNS = choroba wieńcowa, MI = zawał, CKD/PChN/PCHN =
przewlekła choroba nerek, OBS/OSA = obturacyjny bezdech senny, AHI = wskaźnik
bezdechu, POChP/POCHP = przewlekła obturacyjna choroba płuc, DM = cukrzyca.

Odpowiedz WYŁĄCZNIE poprawnym JSON-em zgodnym z podanym schematem.
""".replace("{wellbeing_list}", _WELLBEING_PROMPT_LIST).replace(
    "{drug_examples}", _DRUG_EXAMPLES_PROMPT)  # v5: 2a, 2c


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  ABBREVIATION EXPANDER (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

def expand_abbreviations(text: str, abbreviations: dict | None = None) -> str:
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


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL PATHS (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATHS = {
    "Llama3.1-8":     "meta-llama/Llama-3.1-8B-Instruct",
    "Llama3.3-70":    "meta-llama/Llama-3.3-70B-Instruct",
    "Gemma3-12":      "google/gemma-3-12b-it",
    "Gemma4-12":      "google/gemma-4-12B-it",
    "Qwen3-14":       "Qwen/Qwen3-14B",
    "Qwen3.8-27":     "Qwen/Qwen3.8-27B",  # requires vLLM >= 0.17 (arch. qwen3_5), env llm-venv-vllm032
    "Bielik-v3-11":   "speakleash/Bielik-11B-v3.0-Instruct",
    # "Bielik-PL-v3-11":   "speakleash/Bielik-PL-11B-v3.0-Instruct",
    "Ministral-3-14": "mistralai/Ministral-3-14B-Instruct-2512",
    "PLLuM-v2-12": "CYFRAGOVPL/PLLuM-12B-instruct-2512",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def flush_results(output_path: str, results: list) -> None:
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []
    existing.extend(results)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(existing, out, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  v5: GENERATION, PARSING AND RUN-METADATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def empty_facts() -> dict:
    """v5: placeholder object on a failed parse (as in v4, without field_confidence)."""
    return {
        "medications": [],
        "therapy_adherence": {"status": None, "confidence": "low"},
        "choroby_towarzyszace": {
            k: {"present": None, "evidence_span": None,
                "details": None, "confidence": "low"}
            for k in COMORBIDITY_KEYS
        },
        "abpm_holter": {"mentioned": False, "confidence": "low"},
        "diagnoses_from_notes": [],
        "diagnoses_from_medications": [],
        "objawy": {
            k: {"present": None, "evidence_span": None,
                "details": None, "confidence": "low"}
            for k in SYMPTOM_KEYS
        },
        "samopoczucie": {"category": None, "confidence": "low"},
    }


def parse_facts_json(text: str) -> dict | None:
    """v5: parses the model output; None if it is not a valid JSON object.

    strict=False: a string with maxLength in xgrammar 0.1.29 allows a raw TAB, which
    json.loads(strict=True) does not accept (v5_preflight.md §1.2).
    """
    try:
        obj = json.loads(text, strict=False)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def generation_info(request_output, completion, parse_ok: bool,
                    keep_raw: bool = True) -> dict:
    """v5: 1.1 — metadane jednej generacji vLLM."""
    info = {
        "finish_reason": completion.finish_reason,
        "n_output_tokens": len(completion.token_ids or []),
        "n_prompt_tokens": len(request_output.prompt_token_ids or []),
        "parse_ok": parse_ok,
    }
    if not parse_ok and keep_raw:
        info["raw_output"] = completion.text
    return info


def is_qwen3(model_name: str) -> bool:
    """v5: 2e — enable_thinking=False only for Qwen3 (other templates may not accept it)."""
    return model_name.startswith("Qwen3")


def chat_template_kwargs(model_name: str) -> dict:
    return {"enable_thinking": False} if is_qwen3(model_name) else {}


def apply_chat_template(tokenizer, messages: list[dict], model_name: str) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **chat_template_kwargs(model_name),
    )


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(obj: Any) -> str:
    data = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sampling_params_dict(sp) -> dict:
    """v5: 1.3 — all SamplingParams fields in a JSON-serializable form."""
    fields = getattr(sp, "__struct_fields__", None) or list(vars(sp))
    out = {}
    for f in fields:
        v = getattr(sp, f, None)
        if f == "structured_outputs" and v is not None:
            v = {k: getattr(v, k, None) for k in ("disable_fallback", "disable_any_whitespace",
                                                   "whitespace_pattern")} | {
                "json_sha256": _sha256(v.json if isinstance(v.json, str) else v.json)
                if v.json is not None else None}
        out[f] = v
    return json.loads(json.dumps(out, default=str))


def run_meta_path(output_path: str) -> str:
    root, _ = os.path.splitext(output_path)
    return root + "_run_meta.json"


def write_run_meta(output_path: str, meta: dict) -> str:
    """v5: 1.3 — appends run metadata to <output>_run_meta.json (a "runs" list,
    because the results file may be produced over several resumed runs)."""
    path = run_meta_path(output_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {"runs": []}
    existing.setdefault("runs", []).append(meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    return path


def generation_summary(results: list[dict]) -> dict:
    """v5: 1.4 — share of parse_ok, share of finish_reason=="length", median/max tokens."""
    gens = [r["generation"] for r in results if r.get("generation")]
    if not gens:
        return {"n": 0}
    toks = [g["n_output_tokens"] for g in gens]
    return {
        "n": len(gens),
        "parse_ok_rate": sum(g["parse_ok"] for g in gens) / len(gens),
        "finish_length_rate": sum(g["finish_reason"] == "length" for g in gens) / len(gens),
        "fallback_used": sum(bool(g.get("fallback_used")) for g in gens),
        "median_output_tokens": statistics.median(toks),
        "max_output_tokens": max(toks),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN PIPELINE (v5: changes marked `# v5:`)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # v5: heavy imports moved here (see module header)
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    parser = argparse.ArgumentParser(
        description="Extraction of structured facts from hypertension clinic visit notes (schema v5)")
    parser.add_argument("--model", default="Bielik-v3-11", type=str)
    parser.add_argument("--input-csv",
                        default="Data/dedup_holter_wybrani_pacjenci_dane.csv")
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--output-file", default="Bielik-v3-11_neurosymbolic_v5_10_patients.json", type=str)
    parser.add_argument(
        "--tensor-parallel-size", default=None, type=int,
        help="Liczba GPU do tensor parallelism (duże modele, np. Llama3.3-70, nie mieszczą "
             "się na jednym GPU). Domyślnie: wszystkie widoczne GPU.",
    )
    # v5: new arguments
    parser.add_argument(
        "--no-expand-abbreviations", action="store_true",
        help="v5 (2d): nie rozwijaj skrótów przed podaniem tekstu modelowi "
             "(domyślnie rozwijanie włączone, jak w v4).",
    )
    parser.add_argument(
        "--max-tokens", default=DEFAULT_MAX_TOKENS, type=int,
        help=f"v5 (3e): max_tokens dla ekstrakcji "
             f"(domyślnie {DEFAULT_MAX_TOKENS}; v4: 4096).",
    )
    parser.add_argument(
        "--repetition-penalty", default=1.0, type=float,
        help="v5 (3f): repetition_penalty dla ekstrakcji "
             "(domyślnie 1.0 = bez zmiany względem v4).",
    )
    parser.add_argument(
        "--max-docs", default=None, type=int,
        help="v5 (3g): przetwórz tylko pierwsze N dokumentów w zakresie (przebiegi testowe).",
    )
    parser.add_argument(
        "--max-patient-id", default=DEFAULT_MAX_PATIENT_ID, type=int,
        help=f"v5 (3g): górna granica ID_PACJENTA_ANON (włącznie; domyślnie "
             f"{DEFAULT_MAX_PATIENT_ID}).",
    )
    args = parser.parse_args()
    expand = not args.no_expand_abbreviations  # v5: 2d

    model_name = args.model
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Nieznany model '{model_name}'. Dostępne: {list(MODEL_PATHS)}")
    model_path = MODEL_PATHS[model_name]

    # Visible GPUs (large models, e.g. Llama3.3-70B, require tensor parallelism across
    # several GPUs — a single 46GB GPU cannot hold ~140GB of weights in fp16/bf16).
    gpu_util = 0.90
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_devices:
        available_gpu_ids = [g.strip() for g in cuda_devices.split(",") if g.strip()]
    else:
        available_gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    if not available_gpu_ids:
        raise RuntimeError(
            "Brak dostępnych GPU — sprawdź CUDA_VISIBLE_DEVICES / instalację sterowników CUDA."
        )
    tensor_parallel_size = args.tensor_parallel_size or len(available_gpu_ids)

    # v5: 3d — in vLLM 0.16 xgrammar reads disable_any_whitespace only from the engine
    #     config (the field in StructuredOutputsParams is ignored), and the config rejects
    #     this option with backend="auto" → explicitly backend="xgrammar" (no silent fallback).
    structured_outputs_config = {"backend": "xgrammar", "disable_any_whitespace": True}

    print(f"[1/4] Ładowanie modelu ekstrakcji: {model_path} "
          f"(GPU {available_gpu_ids}, tensor_parallel_size={tensor_parallel_size})")
    if model_name == "Ministral-3-14":
        llm = LLM(model=model_path, tokenizer_mode="mistral",
                  max_model_len=32768, gpu_memory_utilization=gpu_util,
                  tensor_parallel_size=tensor_parallel_size,
                  structured_outputs_config=structured_outputs_config)  # v5: 3d
        tokenizer = llm.get_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model=model_path, max_model_len=32768, gpu_memory_utilization=gpu_util,
                  tensor_parallel_size=tensor_parallel_size,
                  structured_outputs_config=structured_outputs_config)  # v5: 3d

    # v5: 3d — disable_any_whitespace is also set per request (no effect in 0.16, the engine
    #     config above does the work; kept for versions that honour it)
    extraction_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,                    # v5: 3e
        repetition_penalty=args.repetition_penalty,    # v5: 3f
        structured_outputs=StructuredOutputsParams(json=json.dumps(FACT_SCHEMA),
                                                   disable_any_whitespace=True),
    )

    print("[2/4] Ładowanie danych")
    data_df = pd.read_csv(args.input_csv, delimiter=",", header=0)
    # v5: 2f — stable document key = row index in the CSV before sorting
    data_df["DOC_KEY"] = data_df.index.astype(int)
    # Sort by patient ID (stable sort preserves intra-patient chronological order)
    data_df = data_df.sort_values(
        ["ID_PACJENTA_ANON"], ascending=True, kind="stable"
    ).reset_index(drop=True)

    # v5: 3g — range: ID ≤ --max-patient-id (instead of the hard-coded `> 200`), non-empty OPIS,
    #     optionally the first --max-docs documents
    in_scope = data_df[data_df["ID_PACJENTA_ANON"].astype(int) <= args.max_patient_id]
    has_text = in_scope["OPIS"].apply(lambda t: isinstance(t, str) and bool(t.strip()))
    for _, _row in in_scope[~has_text].iterrows():
        print(f"  Pacjent {int(_row['ID_PACJENTA_ANON'])}: pominięto — brak tekstu w kolumnie "
              f"OPIS (DOC_KEY={int(_row['DOC_KEY'])}, wartość: {_row['OPIS']!r})")
    in_scope = in_scope[has_text]
    if args.max_docs is not None:
        in_scope = in_scope.head(args.max_docs)
    scope_keys = set(int(k) for k in in_scope["DOC_KEY"])
    print(f"  Dokumentów w zakresie: {len(in_scope)} "
          f"(ID_PACJENTA ≤ {args.max_patient_id}"
          f"{f', max-docs={args.max_docs}' if args.max_docs is not None else ''})")

    os.makedirs(args.output_dir, exist_ok=True)
    results: list[dict] = []

    checkpoint_file = os.path.join(
        args.output_dir, args.output_file
    )
    checkpoint_written = 0

    # v5: 1.3 — run metadata in <output>_run_meta.json
    run_meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__),
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "model_path": model_path,
        "versions": {p: _pkg_version(p) for p in
                     ("vllm", "xgrammar", "transformers", "tokenizers", "torch")},
        "python": sys.version.split()[0],
        "structured_outputs_config": structured_outputs_config,
        "chat_template_kwargs": {
            "extraction": chat_template_kwargs(model_name),
        },
        "sampling_params": {
            "extraction": sampling_params_dict(extraction_params),
        },
        "sha256": {
            "extraction_system_prompt": _sha256(EXTRACTION_SYSTEM_PROMPT),
            "fact_schema": _sha256(FACT_SCHEMA),
        },
        "expand_abbreviations": expand,
        "cli_args": vars(args),
        "argv": sys.argv,
        "input_csv": args.input_csv,
        "n_docs_in_scope": len(in_scope),
    }
    meta_path = write_run_meta(checkpoint_file, run_meta)
    print(f"  Metadane przebiegu → {meta_path}")

    # ── Resume: v5: 2f — by the set of processed DOC_KEYs (v4: by the last ID_PACJENTA) ──
    done_keys: set[int] = set()
    done_opis: set[str] = set()   # v5: fallback for records without DOC_KEY (v4 files)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as _f:
                _existing = json.load(_f)
            if _existing:
                for _rec in _existing:
                    if "DOC_KEY" in _rec:
                        done_keys.add(int(_rec["DOC_KEY"]))
                    else:
                        done_opis.add(_rec["OPIS"])
                print(
                    f"[RESUME] Znaleziono istniejący plik wyjściowy: "
                    f"{len(_existing)} rekordów ({len(done_keys)} z DOC_KEY). "
                    f"Pomijanie już przetworzonych dokumentów."
                )
                if done_opis:
                    print(f"[RESUME] ⚠ {len(done_opis)} rekordów bez DOC_KEY — dopasowanie po OPIS.")
        except (json.JSONDecodeError, KeyError, TypeError) as _e:
            print(f"[RESUME] Nie można wczytać istniejącego pliku ({_e}). "
                  "Uruchamianie od początku.")
            done_keys, done_opis = set(), set()

    print("[3/4] Przetwarzanie pacjentów")
    for _, row in in_scope.iterrows():
        patient_id = int(row["ID_PACJENTA_ANON"])
        doc_key = int(row["DOC_KEY"])
        raw_text = row["OPIS"]

        if doc_key in done_keys or raw_text in done_opis:  # v5: 2f
            continue

        text = expand_abbreviations(raw_text) if expand else raw_text  # v5: 2d
        print(f"  Pacjent {patient_id} (DOC_KEY={doc_key})...")

        # ── Stage 1: Neural extraction ─────────────────────────────────────
        print(f"[INFO] Pacjent {patient_id}: Stage 1 — ekstrakcja LLM")
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user",
             "content": "Wyekstrahuj ustrukturyzowane fakty z poniższego tekstu wizyty:\n\n"
                        + text},
        ]
        prompt = apply_chat_template(tokenizer, messages, model_name)  # v5: 2e
        outputs = llm.generate(prompt, extraction_params)
        completion = outputs[0].outputs[0]
        raw_json = completion.text.strip()

        facts = parse_facts_json(raw_json)  # v5: 3c (strict=False)
        generation = generation_info(outputs[0], completion, parse_ok=facts is not None)  # v5: 1.1
        if facts is None:
            print(f"    ⚠ JSON parse error dla pacjenta {patient_id} "
                  f"(finish_reason={completion.finish_reason}, "
                  f"tokens={generation['n_output_tokens']})")
            facts = empty_facts()
            generation["fallback_used"] = True  # v5: 1.1

        # ── Stage 1.5: Symbolic enrichment ────────────────────────────
        print(f"[INFO] Pacjent {patient_id}: Stage 1.5 — wzbogacanie ontologią")
        facts = enrich_medications_with_ontology(facts)
        facts = derive_diagnoses_from_medications(facts)

        # ── Record saving ──────────────────────────────────────────────────
        record = {
            "DOC_KEY": doc_key,                 # v5: 2f
            "ID_PACJENTA": patient_id,
            "OPIS": raw_text,
            "OPIS_MODEL_INPUT": text,           # v5: 2d — the text the model saw
            "facts": facts,
            "generation": generation,           # v5: 1.1
        }
        results.append(record)

        if patient_id % 10 == 0:
            flush_results(checkpoint_file, results[checkpoint_written:])
            checkpoint_written = len(results)
            print(f"    → Checkpoint: {checkpoint_written} rekordów → {checkpoint_file}")

        print(f"    → finish={generation['finish_reason']}, parse_ok={generation['parse_ok']}, "
              f"tokens={generation['n_output_tokens']}")

    flush_results(checkpoint_file, results[checkpoint_written:])
    print(f"    → Checkpoint (końcowy): {len(results)} rekordów → {checkpoint_file}")

    # v5: 2f — completeness check against the range
    with open(checkpoint_file, "r", encoding="utf-8") as _f:
        _all = json.load(_f)
    _all_keys = {int(r["DOC_KEY"]) for r in _all if "DOC_KEY" in r}
    n_done_in_scope = len(_all_keys & scope_keys) + sum(
        1 for t in in_scope["OPIS"] if t in done_opis)
    print(f"  Przetworzonych dokumentów w zakresie: {n_done_in_scope} / {len(in_scope)} "
          f"(w tym przebiegu: {len(results)}; rekordów w pliku: {len(_all)})")
    if n_done_in_scope != len(in_scope):
        print(f"  ⚠ OSTRZEŻENIE: liczba przetworzonych dokumentów ({n_done_in_scope}) "
              f"różni się od liczby dokumentów w zakresie ({len(in_scope)}).")

    # ── Summary ─────────────────────────────────────────────────────────────
    gen_sum = generation_summary(results)  # v5: 1.4
    print(f"\n{'='*65}")
    print(f"PODSUMOWANIE v5: {len(results)} dokumentów przetworzonych w tym przebiegu")
    # v5: 1.4 — Stage 1 generation statistics
    if gen_sum["n"]:
        print(f"  Generacja Stage 1 (n={gen_sum['n']}):")
        print(f"    parse_ok:                        {gen_sum['parse_ok_rate']:.1%}")
        print(f"    finish_reason == 'length':       {gen_sum['finish_length_rate']:.1%}")
        print(f"    fallback_used:                   {gen_sum['fallback_used']}")
        print(f"    n_output_tokens mediana / max:   "
              f"{gen_sum['median_output_tokens']} / {gen_sum['max_output_tokens']}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
