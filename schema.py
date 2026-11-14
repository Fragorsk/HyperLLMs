"""
Schema utilities for HTN clinic note extraction (v3).

This file contains:
  - MEDICAL_ABBREVIATIONS — dictionary of medical abbreviations for preprocessing
  - SYSTEM_PROMPT_PL — kept as a reference (not used by the v3 main)

The main extraction schema (FACT_SCHEMA) was moved to
main_neurosymbolic_v3.py together with the rest of the pipeline, because the extraction scope
changed relative to previous versions (see the comment in main_v3).
"""

# =============================================================================
# MEDICAL ABBREVIATIONS FOR PREPROCESSING
# =============================================================================
#
# Abbreviations are expanded in the visit text BEFORE extraction — e.g. "NT" →
# "nadciśnienie tętnicze (NT)". This helps smaller models recognize
# concepts for which they saw both abbreviations and full forms during training.
#
# Version v3: added abbreviations relevant to the new scope (CHW, AFIB, AFL, AHI,
# POCHP, BMI variants), all earlier ones kept for backward
# compatibility with other scripts.

MEDICAL_ABBREVIATIONS = {
    # Hypertension / blood pressure
    "NT": "nadciśnienie tętnicze",
    "nt": "nadciśnienie tętnicze",
    "BP": "ciśnienie tętnicze",
    "RR": "ciśnienie tętnicze",
    "HR": "częstość akcji serca",
    "ABPM": "ambulatoryjne monitorowanie ciśnienia",
    "HBPM": "domowe pomiary ciśnienia",

    # Fibrillation / arrhythmia
    "AF": "migotanie przedsionków",
    "FA": "migotanie przedsionków",
    "AFIB": "migotanie przedsionków",
    "AFL": "trzepotanie przedsionków",

    # Coronary artery disease / myocardial infarction
    "MI": "zawał mięśnia sercowego",
    "CAD": "choroba wieńcowa",
    "CHW": "choroba wieńcowa",
    "ChNS": "choroba niedokrwienna serca",
    "CHNS": "choroba niedokrwienna serca",
    "PCI": "przezskórna interwencja wieńcowa",
    "CABG": "pomostowanie aortalno-wieńcowe",

    # Heart failure
    "CHF": "niewydolność serca",
    "NS": "niewydolność serca",

    # Cerebrovascular diseases
    "TIA": "przejściowy atak niedokrwienny",

    # Kidney diseases
    "CKD": "przewlekła choroba nerek",
    "PChN": "przewlekła choroba nerek",
    "PCHN": "przewlekła choroba nerek",
    "AKI": "ostre uszkodzenie nerek",
    "GFR": "przesączanie kłębuszkowe",
    "eGFR": "szacunkowe przesączanie kłębuszkowe",

    # Sleep apnea
    "OBS": "obturacyjny bezdech senny",
    "OSA": "obturacyjny bezdech senny",
    "AHI": "wskaźnik bezdechu sennego",

    # Lung diseases
    "POCHP": "przewlekła obturacyjna choroba płuc",
    "POChP": "przewlekła obturacyjna choroba płuc",
    "COPD": "przewlekła obturacyjna choroba płuc",

    # Diabetes / metabolism
    "DM": "cukrzyca",
    "DM2": "cukrzyca typu 2",
    "HLP": "hiperlipidemia",
    "BMI": "wskaźnik masy ciała",

    # Medications — classes
    "NLPZ": "niesteroidowe leki przeciwzapalne",
    "NSAID": "niesteroidowe leki przeciwzapalne",
    "GKS": "glikokortykosteroidy",
    "HTZ": "hormonalna terapia zastępcza",
    "HRT": "hormonalna terapia zastępcza",
    "ACEi": "inhibitor konwertazy angiotensyny",
    "ARB": "bloker receptora angiotensynowego",
    "CCB": "bloker kanału wapniowego",
    "BB": "beta-bloker",
    "MRA": "antagonista receptora mineralokortykoidowego",

    # Other
    "UTI": "zakażenie układu moczowego",
    "ZUM": "zakażenie układu moczowego",
    "PAD": "choroba tętnic obwodowych",
}


# Old hierarchical text schema — kept as a reference for
# other scripts; v3 does not use it directly.
SYSTEM_PROMPT_PL = """Jesteś specjalistycznym systemem ekstrakcji danych medycznych z tekstów wizyt w poradni nadciśnienia tętniczego.

## ZASADY OGÓLNE

1. Wydobywaj TYLKO informacje explicite obecne w tekście.
2. Jeśli informacja nie jest obecna w tekście, wpisz null.
3. Pole confidence: high/medium/low.
4. Pole source_quote: dosłowny fragment tekstu.

## FORMAT ODPOWIEDZI

Odpowiedz WYŁĄCZNIE poprawnym JSON-em.
"""