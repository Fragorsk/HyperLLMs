import csv
import glob
import os

import pandas as pd

# Single source of truth: the model list of the evaluation (validation.py).
from validation import DEFAULT_MODELS  # noqa: E402


def save_drug_dicts_to_csv(output_dir: str = "Data/Ontology") -> None:
    """
    Writes DRUG_ONTOLOGY and DRUG_CLASS_TO_DIAGNOSIS from main_neurosymbolic.py
    (Stage 1.5) to separate CSV files with 2 columns each.
    """
    from main_neurosymbolic import DRUG_CLASS_TO_DIAGNOSIS, DRUG_ONTOLOGY

    drug_ontology, drug_class_to_diagnosis = DRUG_ONTOLOGY, DRUG_CLASS_TO_DIAGNOSIS
    os.makedirs(output_dir, exist_ok=True)

    # {str: str} — 2 columns (1 nesting level)
    with open(os.path.join(output_dir, "drug_ontology.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_name", "drug_class"])
        for drug_name, drug_class in drug_ontology.items():
            writer.writerow([drug_name, drug_class])

    # {str: str} — 2 columns (1 nesting level)
    with open(os.path.join(output_dir, "drug_class_to_diagnosis.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_class", "diagnosis"])
        for drug_class, diagnosis in drug_class_to_diagnosis.items():
            writer.writerow([drug_class, diagnosis])


def prepare_data():
    holter_data = pd.read_csv("Data/holter_wybrani_pacjenci_dane.csv")

    print("Records per patient:\n", holter_data.value_counts(["ID_PACJENTA_ANON"]))
    print("Note type counts:\n", holter_data.value_counts(["TYP_OPISU"]))

    holter_data["OPIS"] = holter_data["OPIS"].str.strip()
    dedup_holter_data = holter_data.drop_duplicates("OPIS")

    excluded_types = ["Zgoda na planowany zabieg", "Badania własne", "Wynik", "Zlecenie"]
    dedup_holter_data = dedup_holter_data[~dedup_holter_data["TYP_OPISU"].isin(excluded_types)]

    desc_number = len(holter_data["OPIS"])
    org_desc_number = len(dedup_holter_data.index)

    print("Total number of notes:\n", desc_number)
    print("Number of unique notes (after deduplication and type filtering):\n", org_desc_number)

    dedup_holter_data.to_csv("Data/dedup_holter_wybrani_pacjenci_dane.csv", index=False)


def split_annotation_json(input_path: str, output_dir: str | None = None, batch_size: int = 10) -> list[str]:
    """
    Splits a JSON file with annotation results into smaller files of `batch_size` patients.

    The input file must be a list of records with an 'ID_PACJENTA' field.
    The number of unique patients must be a multiple of `batch_size`.

    Output files are written next to the input file (or to `output_dir`)
    with the name: <stem>_<first_id>-<last_id>_patients.json
    """
    import json
    from pathlib import Path

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"File does not exist: {input_path}")

    with src.open(encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    if not isinstance(data, list):
        raise ValueError("The JSON file must contain a list of records.")

    # Unique patient IDs (preserves order of appearance)
    seen: dict[int | str, None] = {}
    for record in data:
        pid = record["ID_PACJENTA"]
        seen[pid] = None
    unique_ids = list(seen.keys())

    n = len(unique_ids)
    remainder = n % batch_size
    # Number of full batches; if there is a remainder, the last batch gets batch_size + remainder
    num_full_batches = n // batch_size if remainder == 0 else n // batch_size - 1

    dest_dir = Path(output_dir) if output_dir else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Cut the patient-count part of the name from the stem, e.g. "_100_patients"
    import re
    stem_clean = re.sub(r"_\d+_patients$", "", src.stem)

    written_paths: list[str] = []
    batch_starts = list(range(0, num_full_batches * batch_size, batch_size))
    # If there is a remainder, add a last enlarged batch (batch_size + remainder)
    if remainder != 0:
        batch_starts.append(num_full_batches * batch_size)

    for i in batch_starts:
        # Last batch with the remainder: extends to the end of the list
        if remainder != 0 and i == batch_starts[-1]:
            this_batch_size = batch_size + remainder
        else:
            this_batch_size = batch_size

        batch_ids = set(unique_ids[i : i + this_batch_size])
        batch_records = [r for r in data if r["ID_PACJENTA"] in batch_ids]

        first_id = unique_ids[i]
        last_id = unique_ids[i + this_batch_size - 1]
        out_name = f"{stem_clean}_{first_id}-{last_id}_patients.json"
        out_path = dest_dir / out_name

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(batch_records, f, ensure_ascii=False, indent=2)

        written_paths.append(str(out_path))
        print(f"  Saved {len(batch_records)} records ({this_batch_size} patients) → {out_path.name}")

    return written_paths


def merge_outputs(output_dir: str = "outputs") -> list[str]:
    """
    Joins pairs of files *_neurosymbolic_v4_10_patients.json
    and *_neurosymbolic_v4_11-200_patients.json into a single
    *_200_patients.json file for each model.

    Exceptions (SPECIAL_CASES): models for which the suffix list is different,
    e.g. PLLuM-v2-12 requires joining three files.

    Returns the list of written file paths.
    """
    import json
    from pathlib import Path

    src_dir = Path(output_dir)
    suffix_10 = "_neurosymbolic_v4_10_patients.json"
    suffix_rest = "_neurosymbolic_v4_11-200_patients.json"

    # model_name → list of suffixes to join (instead of the default pair)
    SPECIAL_CASES: dict[str, list[str]] = {
        "PLLuM-v2-12": [
            "_neurosymbolic_v4_10_patients.json",
            "_neurosymbolic_v4_50_patients.json",
            "_neurosymbolic_v4_60-200_patients.json",
        ],
    }

    written: list[str] = []
    for file_10 in sorted(src_dir.glob(f"*{suffix_10}")):
        model_name = file_10.name[: -len(suffix_10)]

        if model_name in SPECIAL_CASES:
            suffixes = SPECIAL_CASES[model_name]
        else:
            suffixes = [suffix_10, suffix_rest]

        parts: list[list] = []
        missing = False
        for suffix in suffixes:
            path = src_dir / f"{model_name}{suffix}"
            if not path.exists():
                print(f"  Skipped {model_name!r}: missing file {path.name}")
                missing = True
                break
            with path.open(encoding="utf-8") as f:
                parts.append(json.load(f))

        if missing:
            continue

        merged = [record for part in parts for record in part]
        counts = " + ".join(str(len(p)) for p in parts)
        out_path = src_dir / f"{model_name}_200_patients.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"  {model_name}: {counts} = {len(merged)} records → {out_path.name}")
        written.append(str(out_path))

    return written


def merge_annotation_csvs(annotations_dir: str = "annotations") -> pd.DataFrame:
    """
    Merges all annotation CSV files from the `annotations_dir` directory
    into a single `validated_annotations.csv` file (same columns as the
    source files), written to the same directory.

    Prints the number of unique patients (ID_PACJENTA), the number of unique
    original texts (OPIS) and the number of annotations (rows) in the merged file.
    """
    output_path = os.path.join(annotations_dir, "validated_annotations.csv")
    derived_paths = {
        os.path.abspath(output_path),
        os.path.abspath(os.path.join(annotations_dir, "medications.csv")),
    }

    csv_paths = sorted(
        path
        for path in glob.glob(os.path.join(annotations_dir, "*.csv"))
        if os.path.abspath(path) not in derived_paths
    )
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files in directory: {annotations_dir}")

    merged = pd.concat(
        (pd.read_csv(path, encoding="utf-8-sig") for path in csv_paths),
        ignore_index=True,
    )
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    n_patients = merged["ID_PACJENTA"].nunique()
    n_texts = merged["OPIS"].nunique()
    n_annotations = len(merged)

    print(f"Merged {len(csv_paths)} files: {[os.path.basename(p) for p in csv_paths]}")
    print(f"Patients: {n_patients}")
    print(f"Unique notes: {n_texts}")
    print(f"Annotations: {n_annotations}")
    print(f"Zapisano: {output_path}")

    return merged


def extract_unique_medications(
    input_path: str = "annotations/validated_annotations.csv",
    output_path: str = "annotations/medications.csv",
) -> pd.DataFrame:
    """
    Selects from the `Label` column of `input_path` all values starting
    with "Lek:", removes that prefix, reduces them to a list of unique values
    (sorted alphabetically) and writes it to `output_path` with a single
    `Medication` column.
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    labels = df["Label"].dropna().astype(str).str.strip()
    medications = (
        labels[labels.str.startswith("Lek:")]
        .str.removeprefix("Lek:")
        .str.strip()
    )
    medications = medications[medications != ""]

    result = pd.DataFrame({"Medication": sorted(medications.unique())})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"'Lek:' labels: {len(medications)}")
    print(f"Unique medications: {len(result)}")
    print(f"Zapisano: {output_path}")

    return result


def entity_distribution_to_latex(
    annotations_csv: str = "annotations/validated_annotations.csv",
    output_tex: str = "results/entity_distribution.tex",
    caption: str = "Distribution of annotated entities in the test set, by category.",
    label: str = "tab:entity_distribution",
) -> pd.DataFrame:
    """
    Counts the evaluated GT entities per category — exactly those that go into the
    recall denominator (matching.load_gold: after merging duplicates, without negated
    labels and labels outside the evaluation schema), and writes the result as a LaTeX table
    (Category, Count, % of total) + a CSV next to it.
    """
    from pathlib import Path

    import matching

    gold = matching.load_gold(annotations_csv)
    total = gold.n_gt()
    rows = [(cat, gold.n_gt(cat)) for cat in matching.CATEGORIES]

    output_path = Path(output_tex)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Category & Count & \\% of total \\\\",
        "\\midrule",
    ]
    for cat_en, n in rows:
        lines.append(f"{cat_en} & {n} & {n / total * 100:.1f}\\% \\\\")
    lines.append("\\midrule")
    lines.append(f"\\textbf{{Total}} & {total} & 100.0\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    result = pd.DataFrame(rows, columns=["Category", "Count"])
    result["Percent"] = result["Count"] / total * 100
    result.to_csv(output_path.with_suffix(".csv"), index=False)

    print(f"Saved entity distribution ({total} annotations, {len(rows)} categories) -> {output_path}")
    return result


DEFAULT_ANNOTATIONS_CSV = "annotations/validated_annotations.csv"
DEFAULT_DATA_CSV = "Data/dedup_holter_wybrani_pacjenci_dane.csv"


def count_patients(
    annotations_csv: str = DEFAULT_ANNOTATIONS_CSV,
    data_csv: str = DEFAULT_DATA_CSV,
) -> None:
    """Prints the ID_PACJENTA_ANON range for the texts in the GT annotation file,
    matching them against the source data file by the OPIS column."""
    import csv

    # Collect unique OPIS texts from the annotation file
    with open(annotations_csv, encoding="utf-8-sig", newline="") as f:
        ann_opis: set[str] = {row["OPIS"] for row in csv.DictReader(f)}

    print(f"Unique OPIS texts in annotations: {len(ann_opis)}")

    # Match against the source data by OPIS text
    matched_ids: list[int] = []
    matched_opis: set[str] = set()

    with open(data_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["OPIS"] in ann_opis:
                matched_ids.append(int(row["ID_PACJENTA_ANON"]))
                matched_opis.add(row["OPIS"])

    unmatched_count = len(ann_opis - matched_opis)

    if not matched_ids:
        print("No matches found between the files.")
        return

    matched_ids_sorted = sorted(set(matched_ids))
    print(f"Matched ID_PACJENTA_ANON: {len(matched_ids_sorted)}")
    print(f"Range: {matched_ids_sorted[0]} – {matched_ids_sorted[-1]}")
    print(f"Full list: {matched_ids_sorted}")
    if unmatched_count > 0:
        print(f"Warning: {unmatched_count} annotated OPIS texts not found in the source data.")


def response_type_stats(
    results_dir: str = "outputs",
    output_dir: str = "utils_outputs",
) -> pd.DataFrame:
    """
    Classifies the responses of every model for which the file
    <results_dir>/<model_name>_200_patients.json exists, and writes aggregate
    response-type statistics to <output_dir>/response_types_all_models.csv
    (long format: model, response_type, count, n_patients, percent).

    Note: the Divide_* pipeline handles parse errors per category
    (an empty category instead of the whole skeleton), so its parse errors
    end up in empty_valid / with_findings_*, not in parse_error.

    Response types:
      - parse_error   — the response was not valid JSON; the record contains
                        the default fallback skeleton from main_neurosymbolic_v4
                        (all fields null, field_confidence={})
      - empty_valid   — valid JSON, but without any positive findings
                        (no medications, diagnoses, present=True, ABPM, well-being,
                        therapy adherence status)
      - with_findings — valid JSON with at least one finding
                        (+ split: with medications / without medications)
    """
    import json
    from pathlib import Path

    suffix = "_200_patients.json"
    paths = sorted(Path(results_dir).glob(f"*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No *{suffix} files in directory: {results_dir}")

    from matching import is_invalid_output as is_parse_error

    UNKNOWN_VALUES = (None, "", "unknown")

    def has_findings(facts: dict) -> bool:
        entities = list(facts.get("choroby_towarzyszace", {}).values()) + \
                   list(facts.get("objawy", {}).values())
        return bool(
            facts.get("medications")
            or facts.get("diagnoses_from_notes")
            or any(e.get("present") is True for e in entities)
            or facts.get("abpm_holter", {}).get("mentioned") is True
            or facts.get("samopoczucie", {}).get("category") not in UNKNOWN_VALUES
            or facts.get("therapy_adherence", {}).get("status") not in UNKNOWN_VALUES
        )

    def classify(facts: dict) -> str:
        if is_parse_error(facts):
            return "parse_error"
        if not has_findings(facts):
            return "empty_valid"
        if facts.get("medications"):
            return "with_findings_with_medications"
        return "with_findings_no_medications"

    ORDER = ["parse_error", "empty_valid",
             "with_findings_with_medications", "with_findings_no_medications"]

    per_model: list[pd.DataFrame] = []
    for path in paths:
        model_name = path.name[: -len(suffix)]
        with path.open(encoding="utf-8") as f:
            records: list[dict] = json.load(f)

        df = pd.DataFrame(
            {"ID_PACJENTA": r["ID_PACJENTA"], "response_type": classify(r["facts"])}
            for r in records
        )

        total = len(df)
        stats = (
            df.groupby("response_type")
            .agg(count=("ID_PACJENTA", "size"), n_patients=("ID_PACJENTA", "nunique"))
            .reindex(ORDER, fill_value=0)
            .reset_index()
        )
        stats["percent"] = (stats["count"] / total * 100).round(2)
        stats.loc[len(stats)] = ["total", total, df["ID_PACJENTA"].nunique(), 100.0]
        stats.insert(0, "model", model_name)
        per_model.append(stats)

    result = pd.concat(per_model, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "response_types_all_models.csv")
    result.to_csv(output_path, index=False, encoding="utf-8")

    summary = result.pivot(index="model", columns="response_type", values="percent")
    print(f"Response types (% of records) for {len(paths)} models:")
    print(summary[ORDER].to_string())
    print(f"Zapisano: {output_path}")

    return result


def map_medication_names(
    results_dir: str = "outputs_v5",
    models: list[str] | None = None,
    mapping_path: str = "annotations/final_medications_mapping.csv",
    annotations_path: str = "annotations/validated_annotations.csv",
    extra_mapping_path: str | None = "annotations/medication_mapping_candidates_v5_validated.csv",
    extra_mapping_ratings: tuple[str, ...] = ("propozycja poprawna",),
    map_annotations: bool = False,
) -> dict[str, int]:
    """
    Normalizes drug names to active substances (the `substancja_PL` column)
    according to the mapping from `mapping_path`. The match sources are, in order:
    `nazwa_oryginalna`, `substancja_PL` itself and `substancja_INN`
    (the first match wins), and finally the verified mapping
    of names returned by models from `extra_mapping_path`: rows whose
    `ocena` column belongs to `extra_mapping_ratings` map
    `nazwa_oryginalna` → `substancja_PL_docelowa` (None = without this source).

    1. For each model in `models` (default DEFAULT_MODELS) creates a copy
       <results_dir>/RAW_<model>_200_patients.json, and then in the original
       file <model>_200_patients.json replaces `facts.medications[].name`.
    2. Only when `map_annotations=True`: creates a backup of `annotations_path`
       (RAW_<file_name>) and replaces the "Lek: <name>" label values
       in its `Label` column. Disabled by default — validated_annotations.csv is
       the final GT (medication labels are already in `substancja_PL` form) and must not change.

    Matching is case-insensitive and ignores leading/trailing whitespace.
    Names not in the mapping stay unchanged (they are printed to the console).
    If a RAW_* copy already exists, it is not overwritten and serves as the
    mapping source — so re-running gives the same result.

    Returns a dict {file: number of mapped names}.
    """
    import json
    import shutil
    from collections import Counter
    from pathlib import Path

    mapping_df = pd.read_csv(mapping_path, encoding="utf-8-sig")
    mapping = {
        str(orig).strip().lower(): str(target).strip()
        for orig, target in zip(mapping_df["nazwa_oryginalna"], mapping_df["substancja_PL"])
        if pd.notna(orig) and pd.notna(target)
    }
    # Names already matching substancja_PL are also treated as mapped (normalizes letter case)
    for target in mapping_df["substancja_PL"].dropna():
        mapping.setdefault(str(target).strip().lower(), str(target).strip())
    # INN names (substancja_INN) → substancja_PL; when ambiguous, pick the most frequent
    inn_df = mapping_df.dropna(subset=["substancja_INN", "substancja_PL"])
    inn_to_pl = (
        inn_df.assign(inn=inn_df["substancja_INN"].astype(str).str.strip().str.lower(),
                      pl=inn_df["substancja_PL"].astype(str).str.strip())
        .groupby("inn")["pl"]
        .agg(lambda s: s.value_counts().index[0])
    )
    for inn, target in inn_to_pl.items():
        mapping.setdefault(inn, target)
    # Verified names returned by models → substancja_PL_docelowa (selected ratings only)
    if extra_mapping_path is not None:
        extra_df = pd.read_csv(extra_mapping_path, encoding="utf-8-sig")
        extra_df = extra_df[extra_df["ocena"].astype(str).str.strip().isin(extra_mapping_ratings)]
        extra_df = extra_df.dropna(subset=["nazwa_oryginalna", "substancja_PL_docelowa"])
        n_added, conflicts = 0, []
        for orig, target in zip(extra_df["nazwa_oryginalna"], extra_df["substancja_PL_docelowa"]):
            key, target = str(orig).strip().lower(), str(target).strip()
            if key in mapping:
                if mapping[key] != target:
                    conflicts.append((key, mapping[key], target))
                continue
            mapping[key] = target
            n_added += 1
        print(f"Additional mapping ({Path(extra_mapping_path).name}, ratings {list(extra_mapping_ratings)}): "
              f"added {n_added} names")
        if conflicts:
            print(f"  Skipped {len(conflicts)} names already in {Path(mapping_path).name} "
                  f"with a different target (main mapping wins): {conflicts[:10]}")

    def map_name(name):
        if not isinstance(name, str):
            return name, False
        mapped = mapping.get(name.strip().lower())
        return (mapped, True) if mapped is not None else (name, False)

    def backup(path: Path) -> Path:
        raw_path = path.with_name(f"RAW_{path.name}")
        if raw_path.exists():
            print(f"  Copy {raw_path.name} already exists — using it as the source")
        else:
            shutil.copy2(path, raw_path)
            print(f"  Created copy: {raw_path.name}")
        return raw_path

    stats: dict[str, int] = {}

    # 1. Model responses
    for model in models or DEFAULT_MODELS:
        path = Path(results_dir) / f"{model}_200_patients.json"
        if not path.exists():
            print(f"Skipped {model!r}: missing file {path}")
            continue

        print(f"{model}:")
        raw_path = backup(path)
        with raw_path.open(encoding="utf-8") as f:
            records: list[dict] = json.load(f)

        n_total, n_mapped, unmapped = 0, 0, Counter()
        for record in records:
            for med in record.get("facts", {}).get("medications") or []:
                if not isinstance(med, dict) or "name" not in med:
                    continue
                n_total += 1
                med["name"], ok = map_name(med["name"])
                if ok:
                    n_mapped += 1
                else:
                    unmapped[med["name"]] += 1

        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        stats[str(path)] = n_mapped
        print(f"  Mapped {n_mapped}/{n_total} medication names, "
              f"unique unmapped: {len(unmapped)}")
        if unmapped:
            print(f"  Most frequent unmapped: {unmapped.most_common(15)}")

    # 2. GT annotations (skipped by default — GT is frozen)
    if not map_annotations:
        print(f"Skipped mapping of {Path(annotations_path).name} (map_annotations=False)")
        return stats

    path = Path(annotations_path)
    print(f"{path.name}:")
    raw_path = backup(path)
    df = pd.read_csv(raw_path, encoding="utf-8-sig")

    prefix = "Lek:"
    n_total, n_mapped, unmapped = 0, 0, Counter()

    def map_label(label):
        nonlocal n_total, n_mapped
        if not isinstance(label, str) or not label.strip().startswith(prefix):
            return label
        n_total += 1
        name = label.strip().removeprefix(prefix).strip()
        mapped, ok = map_name(name)
        if ok:
            n_mapped += 1
            return f"{prefix} {mapped}"
        unmapped[name] += 1
        return label

    df["Label"] = df["Label"].map(map_label)
    df.to_csv(path, index=False, encoding="utf-8-sig")

    stats[str(path)] = n_mapped
    print(f"  Mapped {n_mapped}/{n_total} '{prefix}' labels, "
          f"unique unmapped: {len(unmapped)}")
    if unmapped:
        print(f"  Unmapped: {unmapped.most_common(15)}")

    return stats


if __name__ == "__main__":
    map_medication_names()