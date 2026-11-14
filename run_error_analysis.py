"""Main entry point: run Level 1 error classification and Level 2 pattern
analysis end-to-end, and print a concise console summary.

Usage:
    python run_error_analysis.py [options]

Options:
    --gold-csv PATH        Gold standard CSV (default: annotations/validated_annotations.csv)
    --outputs-dir PATH     Directory with {model}_200_patients.json files (default: outputs)
    --output-dir PATH      Root output directory (default: output)
    --models M1 M2 ...     Subset of models to evaluate (default: all 6)
    --unit {patient,document}  Primary unit of analysis for tests/CIs (default: patient)
    --sensitivity / --no-sensitivity  Also compute the other unit (default: on)
    --comparison-dir PATH  Where unit_comparison.* and icc_report.csv go (default: results_v5_clustered_d9)
    --baseline-dir PATH    Pre-change tables for point-estimate assertions (default: none)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import stats_clustered as sc
from error_analysis import DEFAULT_CSV, DEFAULT_MODELS, run_level1
from error_patterns import run_level2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-level error analysis for clinical NLP model evaluation")
    parser.add_argument("--gold-csv", type=str, default=DEFAULT_CSV,
                        help="Path to gold standard annotations CSV")
    parser.add_argument("--outputs-dir", type=str, default="outputs_v5",
                        help="Directory containing {model}_200_patients.json files")
    parser.add_argument("--output-dir", type=str, default="results_v5_clustered_d9/error_analysis",
                        help="Root directory for tables/ and figures/ outputs")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Subset of model names to evaluate (default: all 6)")
    parser.add_argument("--unit", choices=sc.UNITS, default="patient",
                        help="Primary unit of analysis for tests and CIs")
    parser.add_argument("--sensitivity", action=argparse.BooleanOptionalAction, default=True,
                        help="Also compute the other unit of analysis (document when --unit patient)")
    parser.add_argument("--comparison-dir", type=str, default="results_v5_clustered_d9",
                        help="Directory for unit_comparison.csv/.md and icc_report.csv")
    parser.add_argument("--baseline-dir", type=str, default="",
                        help="Pre-change tables used to assert identical point estimates ('' = skip)")
    return parser.parse_args()


def print_console_summary(error_records, level2_results, output_dir: str = "output") -> None:
    print()
    print("=" * 70)
    print("  TWO-LEVEL ERROR ANALYSIS — SUMMARY")
    print("=" * 70)
    print(f"  Total classified error records: {len(error_records)}")
    print()

    by_model = level2_results["error_type_by_model"]
    print("  Errors per model:")
    for model in by_model.index:
        print(f"    {model:<20} {int(by_model.loc[model].sum()):5d}")

    print()
    sev = level2_results["severity_by_model"]
    print("  Severity breakdown per model (critical / moderate / minor):")
    for model in sev.index:
        row = sev.loc[model]
        c = int(row.get("critical", 0))
        m = int(row.get("moderate", 0))
        n = int(row.get("minor", 0))
        print(f"    {model:<20} {c:5d} / {m:5d} / {n:5d}")

    print()
    top_omitted = level2_results["top_omitted_medications"]
    if not top_omitted.empty:
        print("  Top 5 most-omitted GT medications:")
        for _, row in top_omitted.head(5).iterrows():
            print(f"    {row['drug']:<20} {int(row['error_count']):5d}")

    print()
    top_halluc = level2_results["top_hallucinated_medications"]
    if not top_halluc.empty:
        print("  Top 5 most-hallucinated medication names:")
        for _, row in top_halluc.head(5).iterrows():
            print(f"    {row['drug']:<20} {int(row['error_count']):5d}")

    print()
    corr = level2_results["spearman_note_length"]
    print(f"  Spearman correlation: error rate vs. note length (95% CI: {corr['unit'].iloc[0]} bootstrap)")
    for _, row in corr.iterrows():
        print(f"    {row['model_name']:<20} rho={row['spearman_rho']:+.3f}  "
              f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]  p_doc={row['p_value']:.4f}")

    print()
    print(f"  Full outputs written under: {Path(output_dir).resolve()}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    models = args.models if args.models is not None else DEFAULT_MODELS

    tables_dir = Path(args.output_dir) / "tables"
    error_csv = tables_dir / "error_analysis_results.csv"
    summary_json = tables_dir / "error_analysis_summary.json"
    notes_meta_csv = tables_dir / "notes_meta.csv"
    notes_model_meta_csv = tables_dir / "notes_model_meta.csv"

    print("Running Level 1: error classification...")
    error_records = run_level1(
        csv_path=args.gold_csv,
        outputs_dir=args.outputs_dir,
        models=models,
        output_csv=str(error_csv),
        output_summary_json=str(summary_json),
        output_notes_meta_csv=str(notes_meta_csv),
        output_notes_model_meta_csv=str(notes_model_meta_csv),
    )

    print("\nRunning Level 2: aggregate pattern analysis...")
    level2_results = run_level2(
        error_csv=str(error_csv),
        notes_meta_csv=str(notes_meta_csv),
        notes_model_meta_csv=str(notes_model_meta_csv),
        output_dir=args.output_dir,
        units=tuple([args.unit] + ([u for u in sc.UNITS if u != args.unit] if args.sensitivity else [])),
        comparison_dir=args.comparison_dir,
        baseline_tables_dir=args.baseline_dir or None,
    )

    print_console_summary(error_records, level2_results, args.output_dir)


if __name__ == "__main__":
    main()
