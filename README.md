# Evaluating locally hosted large language models for structured information extraction from Polish-language hypertension clinic notes

This is the code used for experiments in paper of the same name. It extracts structured clinical information (medications, comorbidities, symptoms and other
visit facts) from Polish hypertension clinic visit notes with locally run LLMs. The extractions are evaluated
against a manually validated gold standard, followed by statistical comparison of models and error
analysis.

Models evaluated in the reported experiment:

| `--model` key | Hugging Face model | Output file key |
|---|---|---|
| `PLLuM-v2-12` | `CYFRAGOVPL/PLLuM-12B-instruct-2512` | `PLLuM-v2-12` |
| `Bielik-v3-11` | `speakleash/Bielik-11B-v3.0-Instruct` | `Bielik-v3-11` |
| `Qwen3-14` | `Qwen/Qwen3-14B` | `Qwen3-14` |
| `Qwen3.8-27` | `Qwen/Qwen3.8-27B` | `Qwen3_8-27` |
| `Ministral-3-14` | `mistralai/Ministral-3-14B-Instruct-2512` | `Ministral-3-14` |
| `Gemma3-12` | `google/gemma-3-12b-it` | `Gemma3-12` |

`Llama3.3-70` (`meta-llama/Llama-3.3-70B-Instruct`) is also supported by the extraction script but
is not part of the reported results.

---

## 1. Installation

Python 3.12 and NVIDIA GPUs are required. You need **two separate virtual environments**:
Qwen3.8-27B (architecture `qwen3_5`) needs vLLM ≥ 0.17, which is incompatible with the stack used
for the other models.

### Environment A: all other models + evaluation

```bash
python3.12 -m venv llm-venv
source llm-venv/bin/activate
pip install -r requirements.txt
```

Main pins: `vllm==0.16.0`, `torch==2.9.1`, `transformers==4.57.6`, `xgrammar==0.1.29`.
This environment is also used for all evaluation steps (sections 4–8).

### Environment B: Qwen3.8-27 inference only

```bash
python3.12 -m venv llm-venv-vllm032
source llm-venv-vllm032/bin/activate
pip install -r requirements-qwen3.8.txt
```

Main pins: `vllm==0.32.0+cu129`, `torch==2.13.0+cu129`, `transformers==5.19.0`, `xgrammar==0.2.8`.

- These are CUDA 12.9 builds, used because NVIDIA driver 550 does not support CUDA 13. If pip
  cannot resolve `vllm==0.32.0+cu129`, install the official vLLM wheel for CUDA 12.9 manually
  first.
- If `nvcc` is not available, set `VLLM_USE_FLASHINFER_SAMPLER=0` when running inference.

### Model weights

Models are loaded by their Hugging Face IDs (table above). Gated models (for example Gemma and
Llama) require accepting the licence on Hugging Face and logging in (`huggingface-cli login`) or
setting `HF_TOKEN`.

---

## 2. Input data

Run all commands from the repository root. The scripts expect these files (available from the corresponding author upon reasonable request):

| Path | Content |
|---|---|
| `Data/dedup_holter_wybrani_pacjenci_dane.csv` | Clinic visit notes (input to extraction) |
| `annotations/validated_annotations.csv` | Gold standard: 504 documents, 100 patients, 3,526 annotations. **Final; never modified by any step.** |
| `annotations/final_medications_mapping.csv` | Drug name → active substance (PL / INN) mapping |

---

## 3. Pipeline overview

| Step | Script | Environment | Output |
|---|---|---|---|
| 1 | `main_neurosymbolic.py` | A (B for Qwen3.8-27) | raw extractions per model |
| 2 | `utils.map_medication_names()` | A | medication names normalised in the model outputs |
| 3 | `validation.py` | A | P/R/F1, model comparisons, JSON validity |
| 4 | `run_error_analysis.py` | A | error classification, error patterns, figures |
| 5 | `table6_fields.py` | A | Table 6: per-field Se / PPV / F1 |
| 6 | `prompt_leakage.py` | A | Table 7: prompt-example leakage + sensitivity analysis |

Shared modules (imported, not run directly): `schema.py` (extraction schema, prompts, abbreviation
list), `matching.py` (TP/FP/FN definition used by every evaluation step) and `stats_clustered.py`
(patient-clustered bootstrap, Friedman/Wilcoxon tests, Holm correction, ICC). `error_analysis.py`
and `error_patterns.py` are called by `run_error_analysis.py`.

---

## 4. Step 1: Extraction (inference)

Each model runs in a separate process. For every note, the script runs two stages:
- **Stage 1, neural extraction:** the LLM returns the facts defined by the JSON schema, with
  constrained decoding.
- **Stage 1.5, symbolic enrichment:** each medication gets a drug class from the drug ontology,
  and diagnoses implied by current medications are added.

The reported runs were made with an earlier version of the script. That version also contained
optional validation stages (rule engine, regex cross-extraction, longitudinal checks,
self-consistency, LLM judge, stratification), which were switched off with `--skip-*` flags. These
stages have since been removed, because they did not change the extracted facts.

**Environment A** (one 11–14B model per GPU):

```bash
source llm-venv/bin/activate
for M in PLLuM-v2-12 Bielik-v3-11 Qwen3-14 Ministral-3-14 Gemma3-12; do
  CUDA_VISIBLE_DEVICES=0 python main_neurosymbolic.py --model $M \
      --output-file ${M}_200_patients.json
done
```

To run models in parallel, give each one its own GPU (`CUDA_VISIBLE_DEVICES=0`, `1`, …) and start
each with `nohup … &`.

**Environment B** (Qwen3.8-27, two GPUs):

```bash
source llm-venv-vllm032/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0,1 \
    python main_neurosymbolic.py --model Qwen3.8-27 \
    --output-file Qwen3_8-27_200_patients.json
```

Other useful options:

| Option | Default | Meaning |
|---|---|---|
| `--output-dir` | `outputs` | output directory |
| `--tensor-parallel-size` | all visible GPUs | number of GPUs for the extraction model |
| `--max-tokens` | 8192 | generation limit for extraction |
| `--max-docs N` | – | process only the first N documents (test runs) |
| `--no-expand-abbreviations` | off | give the model the raw note, without expanding abbreviations |

Each run writes:
- `<output-file>.json`: one record per document, with the extracted facts, the text given to the
  model (`OPIS_MODEL_INPUT`) and generation metadata (finish reason, token counts, parse status,
  raw output).
- `<output-file>_run_meta.json`: package versions, sampling parameters, schema hash and CLI
  arguments of every run.

If a run is interrupted, restart it with **the same command**. Documents already saved are skipped.

**Handing over to evaluation:** the evaluation scripts read `outputs_v5/{key}_200_patients.json`,
where `key` is the output file key from the model table (for example `Qwen3_8-27`). Copy the result
files and their `_run_meta.json` into `outputs_v5/`, or pass `--output-dir outputs_v5` in step 1.
The `200_patients` suffix is a historical name. The actual patient range is set by
`--max-patient-id`.

---

## 5. Step 2: Normalisation of medication names

This step maps the medication names returned by the models (brand names, misspellings, INN names) to
Polish active-substance names, as used in the gold standard. It changes **only the model outputs**.
The gold standard is not touched.

```bash
python -c "from utils import map_medication_names; map_medication_names()"
```

For each model in `outputs_v5/`:
- On the first run, it saves an untouched copy as `RAW_{key}_200_patients.json`.
- It rewrites `facts.medications[].name` in `{key}_200_patients.json`.

Re-running the step always maps from the `RAW_*` copy, so it gives the same result. Names that
cannot be mapped are printed to the console and left unchanged.

---

## 6. Step 3: Evaluation (`validation.py`)

```bash
python validation.py --outputs-dir outputs_v5 --results-dir results_v5_clustered_d9
```

This step computes, at entity level:
- precision, recall and F1 per model and per category, with 95% CIs from a patient-clustered
  bootstrap;
- Friedman and pairwise Wilcoxon tests with Holm correction;
- JSON validity rates;
- ICC of document-level F1 within patients.

The primary unit of analysis is the patient (`--unit patient`). By default, the document-level
analysis is also run as a sensitivity analysis (`--no-sensitivity` turns it off). Bootstrap
settings: `--n-boot 10000`, `--seed 42`.

Main outputs in `results_v5_clustered_d9/`:
- `evaluation_table_{patient,document}.{tex,csv}`
- `json_validity.{tex,csv}`
- `entity_accuracy.csv`
- `counts_per_document.csv`
- `icc_validation.csv`
- `unit_comparison.{csv,md}`
- `icc_report.csv`

---

## 7. Step 4: Error analysis (`run_error_analysis.py`)

```bash
python run_error_analysis.py --outputs-dir outputs_v5 \
    --output-dir results_v5_clustered_d9/error_analysis \
    --comparison-dir results_v5_clustered_d9
```

- **Level 1** creates one record per false positive and false negative, labelled with an error type
  (for example hallucinated entity, negation error, omission). The records are saved in
  `tables/error_analysis_results.csv`.
- **Level 2** aggregates the records into patterns: error type by model and by category, severity,
  most often hallucinated and omitted medications, per-field accuracy, Friedman / post-hoc tests,
  and Spearman correlations with note length and entity density. Results are written as tables
  (`tables/*.csv|tex`) and figures (`figures/*.pdf|png`).

Steps 3 and 4 both update `unit_comparison.{csv,md}` and `icc_report.csv` in `--comparison-dir`.
After both have run, these files contain the results of both steps, whichever ran first.

---

## 8. Steps 5–7: Additional analyses

**Table 6: per-field sensitivity, PPV and F1** for comorbidities and symptoms:

```bash
python table6_fields.py --outputs-dir outputs_v5 --results-dir results_v5_clustered_d9
```

- Outputs: `table6_fields_main.{tex,csv}` (main text) and `tableS_fields_full.{tex,csv}`
  (supplement, with 95% CIs and TP/FP/FN).
- The script stops with an error if its field totals do not match the category rows of the main
  evaluation table.

**Table 7: prompt-example leakage.** This counts predicted medications that match the example
substances given in the extraction prompt. None of these substances occurs in the data, so every
such prediction is a false positive. The script then recomputes the main P/R/F1 and Wilcoxon tables
without these predictions. It works on temporary copies and does not change `outputs_v5/`.

```bash
python prompt_leakage.py --outputs-dir outputs_v5 --results-dir results_v5_clustered_d9
```

- Outputs: `table7_prompt_leakage.{tex,csv}`, `leakage_by_substance.csv`,
  `leakage_predictions.csv`, `leakage_wilcoxon_changes.csv`, `leakage_report.md`.

---

## 9. Full evaluation run (after step 1)

```bash
source llm-venv/bin/activate
python -c "from utils import map_medication_names; map_medication_names()"
python validation.py
python run_error_analysis.py
python table6_fields.py
python prompt_leakage.py
```

The default arguments of all evaluation scripts point to `outputs_v5/`,
`annotations/validated_annotations.csv` and `results_v5_clustered_d9/`.
