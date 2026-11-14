"""Level 2 error analysis: aggregate patterns, tables, and publication-quality
figures built from the Level 1 output (``error_analysis_results.csv`` +
``notes_meta.csv`` produced by ``error_analysis.py``).

All figures are saved as PNG (300 DPI) and PDF under ``output/figures/``;
all tables (including a LaTeX summary) under ``output/tables/``. Inferential
outputs (Friedman / post-hoc tests, Spearman CIs) are written per unit of
analysis with suffix ``_patient`` (primary) and ``_document`` (sensitivity);
all statistics come from ``stats_clustered.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Style ────────────────────────────────────────────────────────────────

PALETTE = sns.color_palette("colorblind")
FIG_WIDTH_IN = 6.7  # ~17cm, two-column journal width
DPI = 300

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})
sns.set_palette(PALETTE)

import matching
import stats_clustered as sc
from validation import MODEL_DISPLAY

# Error keys (type + side for two-sided types) in table order: FN, FP, TP flags.
# Rate denominators: FN → n_gt, FP → n_pred, TP → n_tp (sums from notes_model_meta.csv).
ERROR_TYPES_ORDER = matching.ERROR_KEYS
SIDE_DENOMINATOR = {"FN": "n_gt_entities", "FP": "n_model_entities", "TP": "n_tp"}
SEVERITY_ORDER = ["critical", "moderate", "minor"]
SEVERITY_COLORS = {"critical": PALETTE[3], "moderate": PALETTE[1], "minor": PALETTE[0]}
# Error type names in plot panel titles
ERROR_TYPE_DISPLAY = {
    "omission": "Omission",
    "invalid_output": "Invalid output",
    "negation_error_fn": "Negation error",
    "classification_error_fn": "Classification error FN",
    "hallucination_entity": "Hallucination entity",
    "duplicate": "Duplicate",
    "negation_error_fp": "Negation error FP",
    "classification_error_fp": "Classification error FP",
    "hallucination_evidence": "Hallucination evidence",
}


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _error_type_col(df: pd.DataFrame) -> pd.Series:
    """Error key used in Level 2: error_type, and for types occurring on both
    sides (negation_error, classification_error) with a side suffix (_fn / _fp)."""
    return pd.Series([matching.error_key(t, s) for t, s in zip(df["error_type"], df["side"])], index=df.index)


def _fp_fn_only(df: pd.DataFrame) -> pd.DataFrame:
    """FP/FN records (without TP quality flags) — errors in the TP/FP/FN sense."""
    return df[df["side"] != "TP"]


def load_level1_output(
    error_csv: str,
    notes_meta_csv: str,
    notes_model_meta_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(error_csv, keep_default_na=False, na_values=[""])
    df["error_type_fine"] = _error_type_col(df)
    notes_meta = pd.read_csv(notes_meta_csv)
    notes_model_meta = pd.read_csv(notes_model_meta_csv)
    return df, notes_meta, notes_model_meta


# ─── 1. Error type distribution per model ────────────────────────────────

def error_type_by_model(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    counts = df.groupby(["model_name", "error_type_fine"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[c for c in ERROR_TYPES_ORDER if c in counts.columns])
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    out = counts.copy()
    for c in counts.columns:
        out[c] = counts[c].astype(str) + " (" + pct[c].round(1).astype(str) + "%)"
    out.to_csv(tables_dir / "error_type_by_model.csv")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 4))
    counts.plot(kind="bar", stacked=True, ax=ax, color=PALETTE[:len(counts.columns)], width=0.7)
    ax.set_xlabel("Model")
    ax.set_ylabel("Error count")
    ax.set_title("Error type distribution per model")
    ax.legend(title="Error type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save_fig(fig, figures_dir, "error_type_by_model")
    return counts


# ─── 2. Error type distribution per category ─────────────────────────────

def error_type_by_category(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    """Error type distribution per extraction category (all models pooled).
    Written as count + row-wise percentage, e.g. "572 (23.4%)" — the percentage
    is of that category's own total errors, so it shows which error type
    concentrates in which category regardless of how many errors each
    category has overall (categories differ hugely in raw error volume:
    Medications alone has ~7300 while Symptoms has ~340)."""
    counts = df.groupby(["category", "error_type_fine"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[c for c in ERROR_TYPES_ORDER if c in counts.columns])
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    out = counts.copy()
    for c in counts.columns:
        out[c] = counts[c].astype(str) + " (" + pct[c].round(1).astype(str) + "%)"
    out.to_csv(tables_dir / "error_type_by_category.csv")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 4))
    counts.plot(kind="bar", stacked=True, ax=ax, color=PALETTE[:len(counts.columns)], width=0.7)
    ax.set_xlabel("Category")
    ax.set_ylabel("Error count")
    ax.set_title("Error type distribution per extraction category")
    ax.legend(title="Error type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save_fig(fig, figures_dir, "error_type_by_category")
    return counts


def error_type_by_category_grid(df: pd.DataFrame, figures_dir: Path) -> None:
    """Same data as error_type_by_category, split the other way: one panel
    per error type (2x3 grid, up to 6 slots) showing that error type's raw
    count across all 7 entity categories as a horizontal bar chart, sorted
    descending so the most-affected category is on top. Complements the
    stacked error_type_by_category chart (categories on the x-axis, error
    types stacked) by making it easy to read off a single error type's
    profile across categories at a glance."""
    counts = df.groupby(["category", "error_type_fine"]).size().unstack(fill_value=0)
    error_types = [e for e in ERROR_TYPES_ORDER if e in counts.columns]
    if not error_types:
        return

    n_cols = 3
    n_rows = -(-len(error_types) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.3, n_rows * 4.0), squeeze=False)
    axes_flat = axes.flatten()

    for i, et in enumerate(error_types[: n_rows * n_cols]):
        ax = axes_flat[i]
        vals = counts[et].sort_values(ascending=True)
        bars = ax.barh(vals.index, vals.values, color=PALETTE[i % len(PALETTE)])
        ax.bar_label(bars, padding=3, fontsize=7.5)
        ax.set_xlim(0, max(vals.values.max() * 1.18, 1))
        ax.set_title(ERROR_TYPE_DISPLAY.get(et, et), fontsize=11)
        ax.set_xlabel("Error count", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    for i in range(len(error_types), n_rows * n_cols):
        axes_flat[i].axis("off")

    fig.suptitle("Error type occurrence across entity categories", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "error_type_by_category_grid.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(figures_dir / "error_type_by_category_grid.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── 3. Heatmap: models x error types ─────────────────────────────────────

def heatmap_model_x_errortype(df: pd.DataFrame, figures_dir: Path) -> None:
    counts = df.groupby(["model_name", "error_type_fine"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[c for c in ERROR_TYPES_ORDER if c in counts.columns])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 3.5))
    sns.heatmap(counts, annot=True, fmt="d", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "Error count"}, linewidths=0.5)
    ax.set_xlabel("Error type")
    ax.set_ylabel("Model")
    ax.set_title("Error counts: models x error types")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save_fig(fig, figures_dir, "heatmap_model_x_errortype")


# ─── 4. Heatmap: models x fields (comorbidities/symptoms) ────────────────

def heatmap_model_x_field(df: pd.DataFrame, notes_meta: pd.DataFrame, figures_dir: Path) -> None:
    sub = _fp_fn_only(df)
    sub = sub[sub["category"].isin(["Comorbidities", "Symptoms"])]
    if sub.empty:
        return
    n_notes = notes_meta["note_id"].nunique()
    error_counts = sub.groupby(["field", "model_name"]).size().unstack(fill_value=0)
    rate = error_counts / n_notes  # error rate per note-opportunity

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, max(3.5, 0.32 * len(rate))))
    sns.heatmap(rate, annot=True, fmt=".2f", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "Error rate (errors / note)"}, linewidths=0.5)
    ax.set_xlabel("Model")
    ax.set_ylabel("Clinical field")
    ax.set_title("Error rate per clinical field: comorbidities & symptoms")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save_fig(fig, figures_dir, "heatmap_model_x_field")


FIELD_DISPLAY_NAMES: dict[str, str] = {
    "diabetes": "Diabetes",
    "coronary_artery_disease": "Coronary artery disease",
    "thyroid_disease": "Thyroid disease",
    "sleep_apnea": "Sleep apnea",
    "obesity": "Obesity",
    "atrial_fibrillation": "Atrial fibrillation",
    "stroke_or_tia": "Stroke / TIA",
    "copd": "COPD",
    "chronic_kidney_disease": "Chronic kidney disease",
    "weight_change": "Weight change",
    "smoking": "Smoking",
    "dyspnea": "Dyspnea",
    "chest_pain": "Chest pain",
    "palpitations": "Palpitations",
    "dizziness": "Dizziness",
    "edema": "Edema",
}


def field_accuracy_table(df: pd.DataFrame, notes_meta: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Per-field accuracy for Comorbidities and Symptoms (CSV + LaTeX), one
    row per clinical field showing every model's accuracy for that specific
    field. Accuracy = 1 - error_count/n_notes, which equals
    (TP+TN)/(TP+FP+FN+TN) since binary fields are evaluated on every note
    and generate at most one error record per (note, field, model) — see
    heatmap_model_x_field for the same computation as an error-rate heatmap.
    Rows are sorted by category then ascending mean accuracy (hardest field
    first), to show which specific comorbidities/symptoms are easiest/hardest
    to extract, per model. Only FP/FN records count (TP evidence flags are not
    errors); every evaluated field (matching.EVALUATED_FIELDS) gets a row."""
    sub = _fp_fn_only(df)
    sub = sub[sub["category"].isin(["Comorbidities", "Symptoms"])]
    if sub.empty:
        return pd.DataFrame()

    n_notes = notes_meta["note_id"].nunique()
    models = list(df["model_name"].unique())

    full_index = pd.MultiIndex.from_tuples(
        [(cat, f) for cat in ("Comorbidities", "Symptoms") for f in matching.EVALUATED_FIELDS[cat]],
        names=["category", "field"],
    )
    error_counts = (
        sub.groupby(["category", "field", "model_name"]).size()
        .unstack(fill_value=0).reindex(index=full_index, columns=models, fill_value=0)
    )
    acc = 1 - error_counts / n_notes
    acc["mean_accuracy"] = acc[models].mean(axis=1)
    acc = acc.sort_values(["category", "mean_accuracy"], ascending=[True, True]).reset_index()
    acc["field"] = acc["field"].map(lambda f: FIELD_DISPLAY_NAMES.get(f, f))
    acc = acc.rename(columns={"category": "Category", "field": "Field"})[
        ["Category", "Field"] + models + ["mean_accuracy"]
    ]
    acc.to_csv(tables_dir / "field_accuracy.csv", index=False)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Per-field accuracy (1 -- error rate) for comorbidities and "
        "symptoms, per model, sorted from hardest to easiest within each category.}",
        "\\label{tab:field_accuracy}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "r" * (len(models) + 1) + "}",
        "\\toprule",
        "Field & " + " & ".join(MODEL_DISPLAY.get(m, m) for m in models) + " & Mean \\\\",
        "\\midrule",
    ]
    for cat in ["Comorbidities", "Symptoms"]:
        cat_rows = acc[acc["Category"] == cat]
        if cat_rows.empty:
            continue
        lines.append(f"\\multicolumn{{{len(models) + 2}}}{{l}}{{\\textit{{{cat}}}}} \\\\")
        for _, row in cat_rows.iterrows():
            cells = " & ".join(f"{row[m] * 100:.1f}\\%" for m in models)
            lines.append(f"{row['Field']} & {cells} & {row['mean_accuracy'] * 100:.1f}\\% \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]

    tex_path = tables_dir / "field_accuracy.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return acc


# ─── 5. Error rate vs. note length ────────────────────────────────────────

def _note_error_rates(df: pd.DataFrame, notes_meta: pd.DataFrame) -> pd.DataFrame:
    """Per (note, model): n_errors (FP+FN records) / n_gold_entities, with note length,
    gold entity count and patient_id. Notes with zero gold entities (error_rate
    undefined) are dropped."""
    err_per_note_model = _fp_fn_only(df).groupby(["note_id", "model_name"]).size().rename("n_errors")
    idx = pd.MultiIndex.from_product(
        [notes_meta["note_id"].unique(), df["model_name"].unique()],
        names=["note_id", "model_name"],
    )
    full = err_per_note_model.reindex(idx, fill_value=0).reset_index()
    full = full.merge(notes_meta[["note_id", "patient_id", "note_length_chars", "n_gold_entities"]], on="note_id")
    full["error_rate"] = full["n_errors"] / full["n_gold_entities"].replace(0, np.nan)
    return full.dropna(subset=["error_rate"])


def _scatter_error_rate(full: pd.DataFrame, x_col: str, xlabel: str, title: str,
                        figures_dir: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 4.2))
    for i, model in enumerate(list(full["model_name"].unique())):
        m = full[full["model_name"] == model]
        ax.scatter(m[x_col], m["error_rate"], label=model, alpha=0.6,
                   s=22, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error rate (errors / gold entities)")
    ax.set_title(title)
    ax.legend(title="Model", fontsize=9, loc="upper left")
    _save_fig(fig, figures_dir, name)


def spearman_error_rate(full: pd.DataFrame, x_col: str, unit: str,
                        n_boot: int = sc.N_BOOT, seed: int = sc.SEED) -> pd.DataFrame:
    """Spearman's rho of per-note error rate vs. `x_col`, per model, computed on
    documents (the explanatory variables vary within patient). 95% CI from a
    cluster bootstrap over patients (unit='patient') or a document bootstrap
    (unit='document'); p_value = scipy spearmanr on documents (sensitivity)."""
    models = list(full["model_name"].unique())
    wide = full.pivot(index="note_id", columns="model_name", values="error_rate").reindex(columns=models)
    meta = full.drop_duplicates("note_id").set_index("note_id").loc[wide.index]
    out = sc.spearman_bootstrap(meta[x_col].to_numpy(), wide, meta["patient_id"].to_numpy(), unit, n_boot, seed)
    return out.assign(unit=unit)


def error_rate_vs_note_length(df: pd.DataFrame, notes_meta: pd.DataFrame,
                              tables_dir: Path, figures_dir: Path,
                              units: tuple[str, ...] = sc.UNITS) -> dict[str, pd.DataFrame]:
    """Spearman correlation of error RATE (errors / gold entities in that
    note, not raw error count) vs. note length, per model. Notes with zero
    gold entities (error_rate undefined) are excluded from both the plot and
    the correlation. One CSV per unit of the CI (spearman_note_length_{unit}.csv)."""
    full = _note_error_rates(df, notes_meta)
    _scatter_error_rate(full, "note_length_chars", "Note length (characters)",
                        "Error rate vs. note length", figures_dir, "error_rate_vs_note_length")
    out = {}
    for u in units:
        out[u] = spearman_error_rate(full, "note_length_chars", u)
        out[u].to_csv(tables_dir / f"spearman_note_length_{u}.csv", index=False)
    return out


# ─── 6. Error rate vs. entity density ─────────────────────────────────────

def error_rate_vs_entity_density(df: pd.DataFrame, notes_meta: pd.DataFrame,
                                 tables_dir: Path, figures_dir: Path,
                                 units: tuple[str, ...] = sc.UNITS) -> dict[str, pd.DataFrame]:
    """Spearman correlation of error RATE (errors / gold entities in that
    note, not raw error count) vs. gold entity density, per model. Notes
    with zero gold entities (error_rate undefined) are excluded.

    Caveat: error_rate's denominator is n_gold_entities, the same quantity
    plotted on the x-axis, so this is a ratio-vs-its-own-denominator
    correlation (a known artifact — rho here is biased, typically toward
    negative, independent of any real relationship). Read it as a sanity
    check / exploratory plot rather than a clean causal signal; unlike the
    note-length correlation above, this one doesn't have that confound."""
    full = _note_error_rates(df, notes_meta)
    _scatter_error_rate(full, "n_gold_entities", "Number of gold entities in note",
                        "Error rate vs. gold entity density", figures_dir, "error_rate_vs_entity_density")
    out = {}
    for u in units:
        out[u] = spearman_error_rate(full, "n_gold_entities", u)
        out[u].to_csv(tables_dir / f"spearman_entity_density_{u}.csv", index=False)
    return out


def _sig_stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def spearman_correlations_summary(
    corr_note_length: pd.DataFrame,
    corr_entity_density: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    unit: str = "patient",
) -> pd.DataFrame:
    """Grouped horizontal bar chart summarizing the two per-model Spearman
    correlations from error_rate_vs_note_length / error_rate_vs_entity_density
    side by side, with 95% CI whiskers. unit='patient': CI from the patient
    cluster bootstrap, '*' = CI excludes 0; unit='document' (sensitivity):
    CI from the document bootstrap, stars from the document-level spearmanr p
    (*** p<0.001, ** p<0.01, * p<0.05), as before. Written with suffix _{unit}."""
    merged = corr_note_length.merge(
        corr_entity_density.drop(columns=["unit"]), on="model_name", suffixes=("_length", "_density")
    ).sort_values("model_name", ascending=False).reset_index(drop=True)
    merged.to_csv(tables_dir / f"spearman_summary_{unit}.csv", index=False)

    models = merged["model_name"].tolist()
    y = np.arange(len(models))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, max(3.5, 0.7 * len(models))))
    for offset, sfx, color, lab in ((bar_h / 2, "length", PALETTE[0], "vs. note length"),
                                    (-bar_h / 2, "density", PALETTE[1], "vs. gold entity density")):
        rho = merged[f"spearman_rho_{sfx}"]
        err = np.vstack([rho - merged[f"ci_low_{sfx}"], merged[f"ci_high_{sfx}"] - rho])
        bars = ax.barh(y + offset, rho, height=bar_h, color=color, label=lab,
                       xerr=err, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "0.3"})
        for bar, r, p, sig, hi, lo in zip(bars, rho, merged[f"p_value_{sfx}"], merged[f"ci_excludes_zero_{sfx}"],
                                          merged[f"ci_high_{sfx}"], merged[f"ci_low_{sfx}"]):
            mark = ("*" if sig else "") if unit == "patient" else _sig_stars(p)
            x_txt = (hi + 0.02) if r >= 0 else (lo - 0.02)
            ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                    f"{r:.2f}{mark}", va="center", ha="left" if r >= 0 else "right", fontsize=8)

    ax.axvline(0, color="grey", linewidth=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in models])
    ax.set_xlim(-1.1, 1.15)
    if unit == "patient":
        ax.set_xlabel("Spearman's rho (error rate per note); 95% CI: patient cluster bootstrap, * CI excludes 0")
    else:
        ax.set_xlabel("Spearman's rho (error rate per note); 95% CI: document bootstrap, stars: document-level p")
    ax.set_title("Error rate correlations: note length vs. gold entity density")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    _save_fig(fig, figures_dir, f"spearman_correlations_summary_{unit}")
    return merged


# ─── 7. Per-entity medication error analysis ──────────────────────────────

MEDICATIONS_MAPPING_CSV = "annotations/final_medications_mapping.csv"

# Plot labels for names the mapping file does not cover (normalized name -> label)
DRUG_LABEL_OVERRIDES: dict[str, str] = {
    "insulina – rodzaj nieokreślony": "insulin",
    # Polish substance names not in the mapping file → INN
    "azylsartan": "azilsartan",
    "fluwastatyna": "fluvastatin",
    "prazosyna": "prazosin",
    "ranolazyna": "ranolazine",
    "nifedypina": "nifedipine",
    "tamsulozyna": "tamsulosin",
    "letrozol": "letrozole",
    "karnityna": "carnitine",
    "fluvastatyna": "fluvastatin",
    "lewofloksacyna": "levofloxacin",
    "dihydralazyna": "dihydralazine",
    "dihydroergotamina": "dihydroergotamine",
    "acenokumarol": "acenocoumarol",
    "mizolastyna": "mizolastine",
    "glipizyd": "glipizide",
    "dopamina": "dopamine",
    "luteina": "lutein",
    "azytromycyna": "azithromycin",
    "amitriptylina": "amitriptyline",
    "bromokryptyna": "bromocriptine",
    "fluoksetyna": "fluoxetine",
    "gabapentyna": "gabapentin",
    "indometacyna": "indometacin",
    "karbocystyna": "carbocisteine",
    "karbimazol": "carbimazole",
    "maprotylina": "maprotiline",
    "trimipramina": "trimipramine",
    "terazosyna": "terazosin",
    "winpocetyna": "vinpocetine",
    "famotydyna": "famotidine",
    "bezafibrat": "bezafibrate",
    "klofibrat": "clofibrate",
    "dipirydamol": "dipyridamole",
    "nikotyna": "nicotine",
    "lewozajmian tyroksyny": "levothyroxine",
    # drug group names
    "statyna": "statin",
    "statyny": "statin",
    "flozyna": "gliflozin",
    # descriptive GT labels (non-drugs / unspecified drugs)
    "insulina krótko-/szybkodziałająca – nieokreślona": "short-acting insulin (unspecified)",
    "lek moczopędny – nieokreślony": "diuretic (unspecified)",
    "witaminy – nieokreślone": "vitamins (unspecified)",
    "inne leki – nieokreślone": "other drugs (unspecified)",
    "paski do glukometru (wyrób medyczny)": "glucose test strips (device)",
    "zalecenie dietetyczne, nie lek": "dietary advice (not a drug)",
    "suplement diety (monakolina K, fitosterole, czosnek, CoQ10, wit. B)": "dietary supplement",
    "brak danych": "no data (GT entry)",
}


def _drug_key(name: str) -> str:
    """Lookup key for a drug name: matching._norm (lower case, '-'/'_' -> space),
    trailing dots and repeated spaces removed."""
    return " ".join(matching._norm(str(name)).strip(" .").split())


def _load_pl_to_inn_mapping(mapping_path: str = MEDICATIONS_MAPPING_CSV) -> dict[str, str]:
    """Normalized drug name -> substancja_INN from the medication mapping file,
    built from nazwa_oryginalna (brand / as-written names) and substancja_PL
    (Polish substance names; wins on conflict), plus every INN mapped to itself.
    If one name has several INN names, the most frequent one wins."""
    mapping_df = pd.read_csv(mapping_path, encoding="utf-8-sig").dropna(subset=["substancja_INN"])
    mapping_df = mapping_df.assign(inn=mapping_df["substancja_INN"].astype(str).str.strip())
    lookup: dict[str, str] = {}
    for col in ("substancja_INN", "nazwa_oryginalna", "substancja_PL"):
        sub = mapping_df.dropna(subset=[col])
        keys = sub[col].astype(str).map(_drug_key)
        lookup.update(sub.assign(key=keys).groupby("key")["inn"].agg(lambda s: s.value_counts().index[0]).to_dict())
    return lookup


def _drug_names_to_inn(names: pd.Series) -> pd.Series:
    """Lower-cased INN label for each drug name: mapping file (nazwa_oryginalna /
    substancja_PL -> substancja_INN), then DRUG_LABEL_OVERRIDES; unmapped names
    are kept as-is (normalized) and listed by _report_unmapped_drugs."""
    lookup = {**_load_pl_to_inn_mapping(), **{_drug_key(k): v for k, v in DRUG_LABEL_OVERRIDES.items()}}
    return names.map(lambda n: lookup.get(_drug_key(n), _drug_key(n)).lower())


def _report_unmapped_drugs(meds: pd.DataFrame, tables_dir: Path, csv_stem: str) -> None:
    """Writes {csv_stem}_unmapped.csv: drug labels that are not an INN from the
    mapping file or an override label after _drug_names_to_inn, with counts."""
    known = {v.lower() for v in _load_pl_to_inn_mapping().values()} | {v.lower() for v in DRUG_LABEL_OVERRIDES.values()}
    unmapped = meds.loc[~meds["drug"].isin(known), "drug"].value_counts()
    unmapped.rename_axis("drug").rename("error_count").to_csv(tables_dir / f"{csv_stem}_unmapped.csv")
    if not unmapped.empty:
        top = ", ".join(f"{d} ({n})" for d, n in unmapped.head(10).items())
        print(f"[UWAGA] {csv_stem}: {len(unmapped)} nazw bez INN ({int(unmapped.sum())} rekordów), najczęstsze: {top}")


def _top_medications_grid(
    meds: pd.DataFrame,
    top_n: int,
    suptitle: str,
    file_stem: str,
    figures_dir: Path,
    tables_dir: Path,
    csv_stem: str,
) -> pd.DataFrame:
    """Shared 2x3-grid-of-models plotting helper for the two medication-error
    breakdowns below: one horizontal bar panel per model, each showing that
    model's own top-N drugs for whatever `meds` subset was passed in (so the
    drug set can differ panel to panel), colored consistently with the rest
    of this module (PALETTE, sorted model order)."""
    if meds.empty:
        return pd.DataFrame()

    _report_unmapped_drugs(meds, tables_dir, csv_stem)
    counts = meds.groupby("drug").size().sort_values(ascending=False)
    counts.to_csv(tables_dir / f"{csv_stem}.csv", header=["error_count"])

    models = list(meds["model_name"].unique())
    by_model = meds.groupby(["model_name", "drug"]).size()

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.3, n_rows * 4.3))
    axes_flat = axes.flatten()

    per_model_frames: list[pd.DataFrame] = []
    for i, model in enumerate(models[: n_rows * n_cols]):
        ax = axes_flat[i]
        model_counts = by_model.loc[model].sort_values(ascending=False).head(top_n)

        model_df = model_counts.rename("error_count").rename_axis("drug").reset_index()
        model_df.insert(0, "model_name", model)
        per_model_frames.append(model_df)

        top = model_counts.iloc[::-1]
        bars = ax.barh(top.index, top.values, color=PALETTE[i % len(PALETTE)])
        ax.bar_label(bars, padding=3, fontsize=7.5)
        ax.set_xlim(0, top.values.max() * 1.18)
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11)
        ax.set_xlabel("Error count", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    for i in range(len(models), n_rows * n_cols):
        axes_flat[i].axis("off")

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{file_stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(figures_dir / f"{file_stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    by_model_df = pd.concat(per_model_frames, ignore_index=True)
    by_model_df.to_csv(tables_dir / f"{csv_stem}_by_model.csv", index=False)

    return counts.reset_index(name="error_count")


def top_omitted_medications(df: pd.DataFrame, tables_dir: Path, figures_dir: Path, top_n: int = 10) -> pd.DataFrame:
    """Top-N most-omitted GT medications per model: omission only (field = GT
    drug name). invalid_output FNs are excluded (the model returned no parsable
    output, not a drug-specific omission), and so are hallucination_evidence
    flags, which mark correctly extracted drugs."""
    meds = df[
        (df["category"] == "Medication")
        & (df["error_type_fine"] == "omission")
    ].copy()
    meds["drug"] = _drug_names_to_inn(meds["field"])
    return _top_medications_grid(
        meds, top_n,
        f"Top {top_n} most-omitted GT medications per model",
        "top_omitted_medications", figures_dir, tables_dir,
        "medication_omission_counts",
    )


def top_hallucinated_medications(df: pd.DataFrame, tables_dir: Path, figures_dir: Path, top_n: int = 10) -> pd.DataFrame:
    """Top-N most-hallucinated medication names per model: hallucination_entity
    only, i.e. drug names the model predicted that matched no GT entity in
    that note. Reported under the model's own predicted name (not a GT drug
    identity — see top_omitted_medications for the GT-anchored counterpart),
    so entries here can include non-drug artifacts (e.g. a model literally
    emitting the string "null")."""
    meds = df[
        (df["category"] == "Medication")
        & (df["error_type_fine"] == "hallucination_entity")
    ].copy()
    meds["drug"] = _drug_names_to_inn(meds["field"])
    return _top_medications_grid(
        meds, top_n,
        f"Top {top_n} most-hallucinated medication names per model",
        "top_hallucinated_medications", figures_dir, tables_dir,
        "medication_hallucination_counts",
    )


# ─── 8. Severity distribution per model ───────────────────────────────────

def severity_by_model(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    counts = df.groupby(["model_name", "severity"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[c for c in SEVERITY_ORDER if c in counts.columns])
    counts.to_csv(tables_dir / "severity_by_model.csv")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 4))
    colors = [SEVERITY_COLORS[c] for c in counts.columns]
    counts.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.6)
    ax.set_xlabel("Model")
    ax.set_ylabel("Error count")
    ax.set_title("Severity distribution per model")
    ax.legend(title="Severity", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save_fig(fig, figures_dir, "severity_by_model")
    return counts


# ─── 9. Summary statistics table -> LaTeX ─────────────────────────────────

def summary_table_latex(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    counts = df.groupby(["error_type_fine", "model_name"], sort=False).size().unstack(fill_value=0)
    counts = counts.reindex(index=[c for c in ERROR_TYPES_ORDER if c in counts.index])
    totals = counts.sum(axis=0)
    counts.loc["Total"] = totals

    print("[summary_table_latex COUNTS]\n", counts)
    
    display = counts.drop(index="Total").astype(str)
    for col in counts.columns:
        pct_col = (counts[col].drop("Total") / totals[col] * 100).round(1)
        display[col] = counts[col].drop("Total").astype(str) + " (" + pct_col.astype(str) + r"\%)"
    display.loc["Total"] = totals.astype(str)
    display.index.name = None
    display.columns.name = None

    print("[summary_table_latex DISPLAY]\n", display)
    tex_path = tables_dir / "summary_statistics.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(display.rename(columns=MODEL_DISPLAY).to_latex(
            caption="Error type counts (percent of that model's total errors) per model.",
            label="tab:error_type_summary",
            column_format="l" + "r" * len(display.columns),
            escape=False,
        ))
    display.to_csv(tables_dir / "summary_statistics.csv")
    return display


# ─── 10. Summary table with correct rate denominators -> LaTeX ─────────────

def summary_table_rates_latex(
    df: pd.DataFrame,
    notes_model_meta: pd.DataFrame,
    tables_dir: Path,
) -> pd.DataFrame:
    """Error rate table with one denominator per side of the confusion matrix.

    - FN types (omission, invalid_output, negation_error_fn, classification_error_fn):
      rate = count / n_gt; they sum to 1 − Recall.
    - FP types (hallucination_entity, duplicate, negation_error_fp, classification_error_fp):
      rate = count / n_pred (= TP + FP); they sum to 1 − Precision.
    - hallucination_evidence (flag on correct extractions): rate = count / TP.

    Each block is followed by its total; the identities above are asserted.
    """
    models = list(notes_model_meta["model_name"].unique())
    denom = {side: notes_model_meta.groupby("model_name")[col].sum() for side, col in SIDE_DENOMINATOR.items()}

    counts = (
        df.groupby(["error_type_fine", "model_name"]).size()
        .unstack(fill_value=0)
        .reindex(index=ERROR_TYPES_ORDER, columns=models, fill_value=0)
    )

    def _rate(n: int, d: int) -> float:
        return n / d if d > 0 else float("nan")

    rate_data: dict[str, dict[str, float]] = {}
    for et in ERROR_TYPES_ORDER:
        side = matching.error_key_side(et)
        rate_data[et] = {m: _rate(int(counts.loc[et, m]), int(denom[side].get(m, 0))) for m in models}
    rate_df = pd.DataFrame(rate_data).T.reindex(columns=models)
    rate_df.index.name = "error_type"

    # Σ FN-rates = 1 − R, Σ FP-rates = 1 − P (same TP/FP/FN as the P/R/F1 table)
    for m in models:
        n_gt, n_pred, n_tp = (int(denom[s].get(m, 0)) for s in ("FN", "FP", "TP"))
        fn_sum = rate_df.loc[matching.ERROR_KEYS_FN, m].sum()
        if n_gt and not np.isclose(fn_sum, 1 - n_tp / n_gt):
            raise AssertionError(f"{m}: Σ FN-rates {fn_sum:.6f} ≠ 1 − R {1 - n_tp / n_gt:.6f}")
        fp_sum = rate_df.loc[matching.ERROR_KEYS_FP, m].sum()
        if n_pred and not np.isclose(fp_sum, 1 - n_tp / n_pred):
            raise AssertionError(f"{m}: Σ FP-rates {fp_sum:.6f} ≠ 1 − P {1 - n_tp / n_pred:.6f}")

    def _fmt(v: float) -> str:
        return "---" if pd.isna(v) else f"{v * 100:.1f}\\%"

    def _esc(s: str) -> str:
        return s.replace("_", "\\_")

    caption = (
        "Error rates per model (\\%). FN block: unmatched gold entities by error type, "
        "relative to the number of gold entities ($n_{GT}$); the block total equals $1-R$. "
        "FP block: unmatched predictions by error type, relative to the number of model "
        "predictions ($n_{pred}=\\mathrm{TP}+\\mathrm{FP}$); the block total equals $1-P$. "
        "Each FN and each FP carries exactly one label. Last row: correct extractions (TP) whose "
        "evidence span is not found in the input text, relative to TP."
    )
    disp = [MODEL_DISPLAY.get(m, m) for m in models]
    lines = [
        "\\begin{table}",
        f"\\caption{{{caption}}}",
        "\\label{tab:error_rates}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "r" * len(models) + "}",
        "\\toprule",
        " & " + " & ".join(disp) + " \\\\",
        "\\midrule",
    ]
    for side, keys, total_label in (("FN", matching.ERROR_KEYS_FN, "Total FN ($1-R$)"),
                                    ("FP", matching.ERROR_KEYS_FP, "Total FP ($1-P$)")):
        lines.append(f"$n_{{{'GT' if side == 'FN' else 'pred'}}}$ & "
                     + " & ".join(str(int(denom[side].get(m, 0))) for m in models) + " \\\\")
        for et in keys:
            lines.append(f"\\quad {_esc(et)} & " + " & ".join(_fmt(rate_df.loc[et, m]) for m in models) + " \\\\")
        lines.append(f"\\textit{{{total_label}}} & "
                     + " & ".join(_fmt(rate_df.loc[keys, m].sum()) for m in models) + " \\\\")
        lines.append("\\midrule")
    lines.append("TP & " + " & ".join(str(int(denom["TP"].get(m, 0))) for m in models) + " \\\\")
    for et in matching.ERROR_KEYS_TP:
        lines.append(f"\\quad {_esc(et)} & " + " & ".join(_fmt(rate_df.loc[et, m]) for m in models) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]

    tex_path = tables_dir / "summary_rates.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    rate_df.to_csv(tables_dir / "summary_rates.csv")
    print(f"Wrote error rate table -> {tex_path}")
    return rate_df


# ─── 11. Unit-level (patient / document) statistical analysis ──────────────

UNIT_LABEL = {"patient": "patient", "document": "document"}


def _fmt_p_and_stars(p: float | None) -> tuple[str, str]:
    """Format a p-value for print and return its significance stars
    (*** p<0.001, ** p<0.01, * p<0.05), matching the convention already used
    for the Wilcoxon table in validation.py."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "---", ""
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    p_str = "$<0.001$" if p < 0.001 else f"{p:.3f}"
    return p_str, stars


def _unit_rate_text(unit: str) -> str:
    if unit == "patient":
        return ("patients as blocks (patient-level rate = $\\Sigma$ errors / $\\Sigma$ denominator over all "
                "visits of the patient; unit of analysis: patient)")
    return ("documents as blocks (unit of analysis: document; sensitivity analysis, visits of the same "
            "patient treated as independent)")


def _friedman_table_tex(stat_df: pd.DataFrame, n_models: int, unit: str = "patient",
                        n_units_total: int | None = None) -> str:
    """Compact one-row-per-error-type Friedman omnibus test table."""
    fr = stat_df[stat_df["test"] == "friedman"].set_index("error_type")
    ordered = [et for et in ERROR_TYPES_ORDER if et in fr.index]
    total_txt = f" of {n_units_total}" if n_units_total else ""

    lines = [
        "\\begin{table}",
        f"\\caption{{Friedman test per error type ({_unit_rate_text(unit)}; models "
        f"as groups, $df={n_models - 1}$). Rates use the same GT-/model-entity "
        "denominators as the error rate table; units with a zero denominator for a "
        f"given error type are excluded from that test ($N$ = units tested{total_txt}). "
        "$W$: Kendall's coefficient of concordance. Significance: "
        "*** $p<0.001$, ** $p<0.01$, * $p<0.05$.}",
        f"\\label{{tab:friedman_summary{'' if unit == 'patient' else '_' + unit}}}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Error type & $N$ & $\\chi^2$ & $W$ & $p$-value \\\\",
        "\\midrule",
    ]
    for et in ordered:
        row = fr.loc[et]
        p_str, stars = _fmt_p_and_stars(row["p_value"])
        lines.append(f"{et} & {int(row['N'])} & {row['statistic']:.2f} & "
                     f"{row['kendall_w']:.3f} & {p_str}{stars} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def _has_posthoc(stat_df: pd.DataFrame, et: str) -> bool:
    """True if at least one pair has a computed post-hoc test (≥10 units with a difference ≠ 0)."""
    pw = stat_df[(stat_df["error_type"] == et) & (stat_df["test"] == "wilcoxon_signed_rank")]
    return bool(pw["effect_size"].notna().any())


def _posthoc_matrices_tex(stat_df: pd.DataFrame, models: list[str], unit: str = "patient") -> str:
    """One model x model rank-biserial-correlation matrix per error type for
    which the Friedman test was significant. Cell (row, col) is positive when
    `row` has a higher per-unit error rate than `col`; stars reflect the
    Holm-Bonferroni-adjusted p-value. Pairs skipped for having fewer than 10
    non-tied units (see statistical_analysis_unit_level) show 'n/a'."""
    friedman_p = stat_df[stat_df["test"] == "friedman"].set_index("error_type")["p_value"]
    sens = "" if unit == "patient" else " Sensitivity analysis: unit of analysis document."

    blocks: list[str] = []
    for et in [e for e in ERROR_TYPES_ORDER if e in stat_df["error_type"].values]:
        if friedman_p.get(et, 1.0) >= 0.05 or not _has_posthoc(stat_df, et):
            continue

        pw = stat_df[(stat_df["error_type"] == et) & (stat_df["test"] == "wilcoxon_signed_rank")]
        r_lookup: dict[tuple[str, str], tuple[float | None, float | None]] = {
            (row["model_a"], row["model_b"]): (row["effect_size"], row["p_adjusted"])
            for _, row in pw.iterrows()
        }
        n_units = int(stat_df[(stat_df["error_type"] == et) & (stat_df["test"] == "friedman")]["N"].iloc[0])

        rows_tex: list[str] = []
        for m_row in models:
            cells: list[str] = []
            for m_col in models:
                if m_row == m_col:
                    cells.append("---")
                elif (m_row, m_col) in r_lookup:
                    r, p_adj = r_lookup[(m_row, m_col)]
                    cells.append("n/a" if r is None or pd.isna(r) else f"{r:+.2f}{_fmt_p_and_stars(p_adj)[1]}")
                elif (m_col, m_row) in r_lookup:
                    r, p_adj = r_lookup[(m_col, m_row)]
                    cells.append("n/a" if r is None or pd.isna(r) else f"{-r:+.2f}{_fmt_p_and_stars(p_adj)[1]}")
                else:
                    cells.append("n/a")
            rows_tex.append(f"{MODEL_DISPLAY.get(m_row, m_row)} & " + " & ".join(cells) + " \\\\")

        blocks.append("\n".join([
            "\\begin{table}",
            f"\\caption{{Post-hoc pairwise Wilcoxon signed-rank tests for "
            f"\\textit{{{et}}} (Holm-Bonferroni corrected), rank-biserial "
            "correlation as effect size. Positive: the row model's "
            f"per-{UNIT_LABEL[unit]} error rate is higher than the column model's "
            f"(unit of analysis: {UNIT_LABEL[unit]}, $N = {n_units}$). "
            "Significance (on the adjusted p-value): "
            "*** $p_{\\mathrm{adj}}<0.001$, ** $p_{\\mathrm{adj}}<0.01$, "
            f"* $p_{{\\mathrm{{adj}}}}<0.05$.{sens}}}",
            f"\\label{{tab:posthoc_{et}{'' if unit == 'patient' else '_' + unit}}}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l" + "r" * len(models) + "}",
            "\\toprule",
            " & " + " & ".join(MODEL_DISPLAY.get(m, m) for m in models) + " \\\\",
            "\\midrule",
            *rows_tex,
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
        ]))

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def per_doc_error_counts(df: pd.DataFrame, notes_meta: pd.DataFrame,
                         notes_model_meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, patient, note): error counts per error key and the
    three denominators (n_gt_entities, n_model_entities, n_tp)."""
    counts = (df.groupby(["note_id", "model_name", "error_type_fine"]).size()
              .unstack(fill_value=0).reindex(columns=ERROR_TYPES_ORDER, fill_value=0))
    base = notes_model_meta.merge(notes_meta[["note_id", "patient_id"]], on="note_id", how="left")
    if base["patient_id"].isna().any():
        raise AssertionError("notes_model_meta: notatki bez patient_id w notes_meta")
    base = base.join(counts, on=["note_id", "model_name"])
    base[ERROR_TYPES_ORDER] = base[ERROR_TYPES_ORDER].fillna(0).astype(int)
    if base[ERROR_TYPES_ORDER].to_numpy().sum() != len(df):
        raise AssertionError("per_doc_error_counts: Σ liczników ≠ liczba rekordów błędów")
    return base


def unit_rate_table(per_doc: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Per (model, unit) error rates rate_{error key} = Σ errors / Σ denominator
    (stats_clustered.unit_table; denominator 0 → NaN)."""
    rates = {f"rate_{et}": (et, SIDE_DENOMINATOR[matching.error_key_side(et)]) for et in ERROR_TYPES_ORDER}
    count_cols = ERROR_TYPES_ORDER + list(SIDE_DENOMINATOR.values())
    return sc.unit_table(per_doc, unit, group_cols=("model_name",), patient_col="patient_id",
                         doc_col="note_id", count_cols=count_cols, rates=rates)


def statistical_analysis_unit_level(
    per_doc: pd.DataFrame,
    tables_dir: Path,
    unit: str = "patient",
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Friedman test per error type (units — patients or documents — as blocks,
    models as groups), via stats_clustered.rank_tests.

    Per-unit rate = Σ errors / Σ denominator over the unit's notes, with the
    same denominators as summary_table_rates_latex (FN types / n_gt, FP types /
    n_pred, hallucination_evidence / TP). A unit whose denominator is 0 for any
    model is excluded from that error type's tests (N and n_excluded reported).

    When the Friedman test is significant (p < 0.05) the CSV/TeX also report
    pairwise Wilcoxon signed-rank post-hoc tests (zero_method='wilcox', pairs
    with < 10 non-zero differences skipped) with Holm-Bonferroni correction and
    rank-biserial correlation as effect size. Post-hoc results are computed for
    every error type and returned (second value) for the unit comparison.

    Output: statistical_analysis_{unit}.csv/.tex and friedman_mean_ranks_{unit}.csv.
    CSV columns: error_type, test, statistic, p_value, model_a, model_b,
    p_adjusted, effect_size, N, n_excluded, kendall_w.
    """
    models = sorted(per_doc["model_name"].unique())
    n_models = len(models)

    if n_models < 3:
        print("At least 3 models required for Friedman test; skipping statistical analysis.")
        return pd.DataFrame(), {}

    table = unit_rate_table(per_doc, unit)
    n_units_total = table["unit_id"].nunique()
    stat_rows: list[dict] = []
    rank_rows: list[dict] = []
    full: dict[str, dict] = {}

    for et in ERROR_TYPES_ORDER:
        X = table.pivot(index="unit_id", columns="model_name", values=f"rate_{et}").reindex(columns=models)
        keep = sc.complete_rows(X.to_numpy())
        if keep.sum() == 0 or X[keep].var().sum() == 0:
            continue
        try:
            res = sc.rank_tests(X.to_numpy(), models, unit, zero_method="wilcox",
                                sensitivity_methods=(), min_nonzero=10)
        except Exception:
            continue
        fr = res["friedman"]
        if np.isnan(fr["chi2"]):
            continue
        full[et] = res
        stat_rows.append({
            "error_type": et, "test": "friedman",
            "statistic": round(fr["chi2"], 4), "p_value": round(fr["p"], 6),
            "model_a": None, "model_b": None,
            "p_adjusted": None, "effect_size": None,
            "N": res["N"], "n_excluded": res["n_excluded"], "kendall_w": round(fr["kendall_w"], 4),
        })
        rank_rows += [{"error_type": et, "model_name": m, "mean_rank": v, "N": res["N"]}
                      for m, v in fr["mean_ranks"].items()]
        if res["n_excluded"]:
            print(f"  [{unit}] {et}: wykluczono {res['n_excluded']} z {res['n_total']} jednostek (mianownik 0)")

        if fr["p"] >= 0.05:
            continue
        for _, r in res["pairs"].iterrows():
            tested = r["p_raw"] is not None and not pd.isna(r["p_raw"])
            stat_rows.append({
                "error_type": et, "test": "wilcoxon_signed_rank",
                "statistic": round(r["W"], 4) if tested else None,
                "p_value": round(r["p_raw"], 6) if tested else None,
                "model_a": r["model_a"], "model_b": r["model_b"],
                "p_adjusted": round(r["p_holm"], 6) if tested else None,
                "effect_size": round(r["r_rb"], 4) if tested else None,
                "N": res["N"], "n_excluded": res["n_excluded"], "kendall_w": None,
            })

    stat_df = pd.DataFrame(stat_rows)
    if not stat_df.empty:
        stat_df.to_csv(tables_dir / f"statistical_analysis_{unit}.csv", index=False)
        pd.DataFrame(rank_rows).to_csv(tables_dir / f"friedman_mean_ranks_{unit}.csv", index=False)
        print(f"Wrote statistical analysis -> {tables_dir / f'statistical_analysis_{unit}.csv'}")

        tex = (_friedman_table_tex(stat_df, n_models, unit, n_units_total) + "\n"
               + _posthoc_matrices_tex(stat_df, models, unit))
        tex_path = tables_dir / f"statistical_analysis_{unit}.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Wrote statistical analysis LaTeX -> {tex_path}")
    return stat_df, full


def icc_error_rates(per_doc: pd.DataFrame, n_boot: int = sc.N_BOOT, seed: int = sc.SEED) -> pd.DataFrame:
    """ICC(1) of document-level error rates within patient, per error type and model
    (documents with a zero denominator are skipped), 95% CI from the patient cluster bootstrap."""
    doc = unit_rate_table(per_doc, "document")
    rows = []
    for et in ERROR_TYPES_ORDER:
        for model, g in doc.groupby("model_name", sort=True):
            icc = sc.icc_by_patient(g[f"rate_{et}"].to_numpy(), g["patient_id"].to_numpy(), n_boot, seed)
            rows.append({"metric": "document_error_rate", "error_type": et, "model": model, **icc})
    return pd.DataFrame(rows)


# ─── 12. Post-hoc effect-size heatmaps ─────────────────────────────────────

def posthoc_effect_size_heatmaps(stat_df: pd.DataFrame, models: list[str], figures_dir: Path,
                                 unit: str = "patient") -> None:
    """Heatmap version of the post-hoc pairwise matrices in statistical_analysis_{unit}.tex,
    combined into a single 2x3 panel grid (row 1: 3 error types, row 2: 2 error
    types + 1 empty cell) with one shared colorbar. One panel per error type
    with a significant Friedman test (up to 5). Color = rank-biserial r (row
    model's per-unit rate vs. column model's); cell text repeats the value
    with Holm-Bonferroni significance stars. Saved as posthoc_heatmaps_{unit}."""
    if stat_df.empty:
        return
    friedman_p = stat_df[stat_df["test"] == "friedman"].set_index("error_type")["p_value"]
    ets = [e for e in ERROR_TYPES_ORDER if e in stat_df["error_type"].values
           and friedman_p.get(e, 1.0) < 0.05 and _has_posthoc(stat_df, e)]
    if not ets:
        return

    n = len(models)
    n_cols = 3
    n_rows = -(-len(ets) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.3, n_rows * 4.1), squeeze=False)
    axes_flat = axes.flatten()
    cbar_ax = fig.add_axes((0.93, 0.15, 0.015, 0.7))

    for idx, et in enumerate(ets[: n_rows * n_cols]):
        ax = axes_flat[idx]
        pw = stat_df[(stat_df["error_type"] == et) & (stat_df["test"] == "wilcoxon_signed_rank")]
        r_lookup: dict[tuple[str, str], tuple[float | None, float | None]] = {
            (row["model_a"], row["model_b"]): (row["effect_size"], row["p_adjusted"])
            for _, row in pw.iterrows()
        }

        val = np.full((n, n), np.nan)
        annot = np.full((n, n), "", dtype=object)
        mask = np.zeros((n, n), dtype=bool)

        for i, m_row in enumerate(models):
            for j, m_col in enumerate(models):
                if i == j:
                    mask[i, j] = True
                    continue
                if (m_row, m_col) in r_lookup:
                    r, p_adj = r_lookup[(m_row, m_col)]
                elif (m_col, m_row) in r_lookup:
                    r0, p_adj = r_lookup[(m_col, m_row)]
                    r = None if r0 is None or pd.isna(r0) else -r0
                else:
                    r, p_adj = None, None

                if r is None or pd.isna(r):
                    annot[i, j] = "n/a"
                else:
                    val[i, j] = r
                    annot[i, j] = f"{r:+.2f}{_fmt_p_and_stars(p_adj)[1]}"

        sns.heatmap(
            val, mask=mask, annot=annot, fmt="", cmap="vlag", vmin=-1, vmax=1, center=0,
            xticklabels=models, yticklabels=models, ax=ax, linewidths=0.5, linecolor="white",
            annot_kws={"fontsize": 7},
            cbar=(idx == 0), cbar_ax=cbar_ax if idx == 0 else None,
            cbar_kws={"label": "Rank-biserial r (row vs. column)"} if idx == 0 else None,
        )
        ax.set_title(ERROR_TYPE_DISPLAY.get(et, et), fontsize=10.5)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7.5)
        plt.setp(ax.get_yticklabels(), fontsize=7.5)

    for idx in range(len(ets), n_rows * n_cols):
        axes_flat[idx].axis("off")

    fig.suptitle("Post-hoc pairwise comparisons (rank-biserial r, Holm-Bonferroni corrected; "
                 f"unit: {UNIT_LABEL[unit]})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.91, 0.95))
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"posthoc_heatmaps_{unit}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(figures_dir / f"posthoc_heatmaps_{unit}.pdf", bbox_inches="tight")
    plt.close(fig)


# ─── 13. Critical-difference diagrams ──────────────────────────────────────

def _find_cliques(order: list[str], nonsig: set[frozenset]) -> list[list[str]]:
    """Maximal contiguous runs (in rank order) where every pair inside the
    run is pairwise non-significant — the standard simplification used by
    critical-difference diagrams to draw connecting bars."""
    n = len(order)
    cliques: list[list[str]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and all(
            frozenset((order[a], order[j + 1])) in nonsig for a in range(i, j + 1)
        ):
            j += 1
        if j > i:
            cliques.append(order[i:j + 1])
        i += 1
    return [c for c in cliques if not any(set(c) < set(other) for other in cliques)]


def critical_difference_diagrams(
    mean_ranks: pd.DataFrame,
    stat_df: pd.DataFrame,
    figures_dir: Path,
    unit: str = "patient",
) -> None:
    """Critical-difference-style diagram per error type: average per-unit
    rank of each model (friedman_mean_ranks_{unit}.csv; lower rank = fewer
    errors), with a horizontal bar connecting each maximal run of models that
    are not significantly different (Holm-Bonferroni adjusted pairwise
    Wilcoxon, alpha=0.05). Reuses the pairwise results already computed by
    statistical_analysis_unit_level rather than a separate Nemenyi test, so
    bars reflect the same tests as the LaTeX tables and heatmaps. Same
    significant-Friedman gate as those outputs."""
    if stat_df.empty:
        return

    friedman_p = stat_df[stat_df["test"] == "friedman"].set_index("error_type")["p_value"]

    for et in [e for e in ERROR_TYPES_ORDER if e in stat_df["error_type"].values]:
        if friedman_p.get(et, 1.0) >= 0.05:
            continue

        avg_rank = mean_ranks[mean_ranks["error_type"] == et].set_index("model_name")["mean_rank"]
        order = avg_rank.sort_values().index.tolist()

        pw = stat_df[(stat_df["error_type"] == et) & (stat_df["test"] == "wilcoxon_signed_rank")]
        nonsig: set[frozenset] = set()
        for _, row in pw.iterrows():
            p_adj = row["p_adjusted"]
            if p_adj is None or pd.isna(p_adj) or p_adj >= 0.05:
                nonsig.add(frozenset((row["model_a"], row["model_b"])))

        cliques = _find_cliques(order, nonsig)

        # Stagger label heights for models whose ranks are too close together
        # to fit side by side at the same height (otherwise adjacent labels
        # overlap horizontally).
        xs = [avg_rank[m] for m in order]
        x_range = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
        min_gap = 0.12 * x_range if x_range > 0 else 0.3
        tiers: list[int] = []
        prev_x = None
        tier = 0
        for x in xs:
            tier = tier + 1 if (prev_x is not None and (x - prev_x) < min_gap) else 0
            tiers.append(tier)
            prev_x = x
        max_tier = max(tiers, default=0)

        n_cliques = len(cliques)
        fig_h = 1.6 + 0.42 * (max_tier + 1) + 0.28 * max(1, n_cliques)
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_h))

        ax.scatter(xs, [0] * len(order), color=PALETTE[0], zorder=3, s=45)
        for m, x, t in zip(order, xs, tiers):
            ax.annotate(f"{m}\n({x:.2f})", (x, 0), textcoords="offset points",
                        xytext=(0, 12 + t * 26), ha="center", fontsize=8.5)

        for k, clique in enumerate(cliques):
            xs_c = [avg_rank[m] for m in clique]
            y = -0.16 * (k + 1)
            ax.plot([min(xs_c), max(xs_c)], [y, y], color=PALETTE[3], linewidth=3.5,
                     solid_capstyle="butt")

        ax.axhline(0, color="grey", linewidth=0.8, zorder=1)
        ax.set_yticks([])
        xlabel = f"Average rank across {UNIT_LABEL[unit]}s (lower = fewer errors)"
        if cliques:
            xlabel += "\norange bar = not significantly different (Holm–Bonferroni-adjusted Wilcoxon, $\\alpha$=0.05)"
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(f"Critical-difference diagram: {et}")
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_ylim(-0.16 * (n_cliques + 1) - 0.12, 0.12 + 0.42 * (max_tier + 1))
        _save_fig(fig, figures_dir, f"cd_diagram_{et}_{unit}")


# ─── Unit comparison and assertions ────────────────────────────────────────

def unit_comparison_rows(full: dict[str, dict[str, dict]], spearman: dict[str, dict[str, pd.DataFrame]],
                         alpha: float = sc.ALPHA) -> list[dict]:
    """unit_comparison rows for the error analysis: Friedman per error type,
    post-hoc pairs (significant = Friedman significant AND p_Holm < alpha at that
    unit), Spearman correlations (document: spearmanr p < alpha as before;
    patient: cluster-bootstrap CI excludes 0)."""
    rows: list[dict] = []
    d_all, p_all = full["document"], full["patient"]
    for et in ERROR_TYPES_ORDER:
        d, p = d_all.get(et), p_all.get(et)
        if d is None and p is None:
            continue

        def _fr(res, key, default=np.nan):
            return res["friedman"][key] if res is not None else default

        fr_sig_d = d is not None and _fr(d, "p") < alpha
        fr_sig_p = p is not None and _fr(p, "p") < alpha
        rows.append({
            "source": "error_analysis", "test": "Friedman per error type (p unadjusted)", "item": et,
            "statistic_name": "chi2", "stat_document": _fr(d, "chi2"), "stat_patient": _fr(p, "chi2"),
            "p_holm_document": _fr(d, "p"), "p_holm_patient": _fr(p, "p"),
            "kendall_w_document": _fr(d, "kendall_w"), "kendall_w_patient": _fr(p, "kendall_w"),
            "N_document": d["N"] if d else np.nan, "N_patient": p["N"] if p else np.nan,
            "excluded_document": d["n_excluded"] if d else np.nan, "excluded_patient": p["n_excluded"] if p else np.nan,
            "significant_document": fr_sig_d, "significant_patient": fr_sig_p,
        })
        pd_ = d["pairs"].set_index(["model_a", "model_b"]) if d else None
        pp_ = p["pairs"].set_index(["model_a", "model_b"]) if p else None
        keys = list((pd_ if pd_ is not None else pp_).index)
        for key in keys:
            a = pd_.loc[key] if pd_ is not None else None
            b = pp_.loc[key] if pp_ is not None else None

            def _g(r, c):
                return np.nan if r is None or r[c] is None or pd.isna(r[c]) else r[c]

            rows.append({
                "source": "error_analysis", "test": f"Post-hoc Wilcoxon (Holm): {et}", "item": f"{key[0]} vs {key[1]}",
                "statistic_name": "W", "stat_document": _g(a, "W"), "stat_patient": _g(b, "W"),
                "p_holm_document": _g(a, "p_holm"), "p_holm_patient": _g(b, "p_holm"),
                "effect_document": _g(a, "r_rb"), "effect_patient": _g(b, "r_rb"),
                "r_rb_document": _g(a, "r_rb"), "r_rb_patient": _g(b, "r_rb"),
                "N_document": d["N"] if d else np.nan, "N_patient": p["N"] if p else np.nan,
                "excluded_document": d["n_excluded"] if d else np.nan,
                "excluded_patient": p["n_excluded"] if p else np.nan,
                "significant_document": bool(fr_sig_d and _g(a, "p_holm") < alpha),
                "significant_patient": bool(fr_sig_p and _g(b, "p_holm") < alpha),
            })
    for var, label in (("note_length", "note length"), ("entity_density", "gold entity density")):
        a = spearman[var]["document"].set_index("model_name")
        b = spearman[var]["patient"].set_index("model_name")
        for m in a.index:
            rows.append({
                "source": "error_analysis", "test": f"Spearman error rate vs {label}", "item": m,
                "statistic_name": "rho", "stat_document": a.loc[m, "spearman_rho"], "stat_patient": b.loc[m, "spearman_rho"],
                "p_holm_document": a.loc[m, "p_value"],
                "ci_document": sc.fmt_ci((a.loc[m, "ci_low"], a.loc[m, "ci_high"])),
                "ci_patient": sc.fmt_ci((b.loc[m, "ci_low"], b.loc[m, "ci_high"])),
                "N_document": a.loc[m, "n_docs"], "N_patient": b.loc[m, "n_patients"],
                "significant_document": bool(a.loc[m, "p_value"] < alpha),
                "significant_patient": bool(b.loc[m, "ci_excludes_zero"]),
            })
    return rows


def check_against_baseline(tables_dir: Path, baseline_tables_dir: Path | None, per_doc: pd.DataFrame,
                           full: dict[str, dict[str, dict]], n_patients: int) -> list[str]:
    """Phase-3 assertions: Σ over patients == Σ over documents, N = n_patients − exclusions,
    error count/rate tables identical to the pre-change run."""
    done: list[str] = []
    pat = unit_rate_table(per_doc, "patient")
    doc = unit_rate_table(per_doc, "document")
    cols = ERROR_TYPES_ORDER + list(SIDE_DENOMINATOR.values())
    if not (pat.groupby("model_name")[cols].sum().equals(doc.groupby("model_name")[cols].sum())):
        raise AssertionError("Σ liczników błędów/mianowników po pacjentach ≠ po dokumentach")
    done.append("Σ liczników błędów i mianowników: pacjenci == dokumenty")
    if pat["unit_id"].nunique() != n_patients:
        raise AssertionError(f"jednostek pacjentowych {pat['unit_id'].nunique()} ≠ {n_patients}")
    for et, res in full.get("patient", {}).items():
        if res["N"] + res["n_excluded"] != n_patients:
            raise AssertionError(f"{et}: N {res['N']} + wykluczone {res['n_excluded']} ≠ {n_patients}")
    excl = {et: r["n_excluded"] for et, r in full.get("patient", {}).items() if r["n_excluded"]}
    done.append(f"testy pacjentowe: N = {n_patients} − wykluczenia (niezerowe: {excl or 'brak'})")

    if baseline_tables_dir is None or not baseline_tables_dir.exists():
        print(f"[UWAGA] Brak wyników bazowych {baseline_tables_dir} — pominięto porównanie z wynikami sprzed zmiany")
        return done
    for name in ("summary_rates.csv", "summary_statistics.csv", "error_type_by_model.csv",
                 "severity_by_model.csv", "field_accuracy.csv"):
        base = baseline_tables_dir / name
        if base.exists():
            pd.testing.assert_frame_equal(pd.read_csv(tables_dir / name), pd.read_csv(base))
            done.append(f"{name} identyczny z {base}")
    base_sp = baseline_tables_dir / "spearman_summary.csv"
    if base_sp.exists():
        a = pd.read_csv(base_sp).set_index("model_name")
        for var, col in (("length", "note_length"), ("density", "entity_density")):
            b = pd.read_csv(tables_dir / f"spearman_{col}_document.csv").set_index("model_name").loc[a.index]
            np.testing.assert_allclose(b["spearman_rho"], a[f"spearman_rho_{var}"], rtol=1e-12)
            np.testing.assert_allclose(b["p_value"], a[f"p_value_{var}"], rtol=1e-9)
        done.append(f"ρ Spearmana i p dokumentowe identyczne z {base_sp}")
    base_st = baseline_tables_dir / "statistical_analysis.csv"
    if base_st.exists() and "document" in full:
        # error types without exclusions (denominator never 0) — same procedure as before the change
        a = pd.read_csv(base_st)
        b = pd.read_csv(tables_dir / "statistical_analysis_document.csv")
        same = [et for et, r in full["document"].items() if r["n_excluded"] == 0]
        cols = ["error_type", "test", "statistic", "p_value", "model_a", "model_b", "p_adjusted", "effect_size"]
        for et in same:
            x = a[a["error_type"] == et][cols].reset_index(drop=True)
            y = b[b["error_type"] == et][cols].reset_index(drop=True)
            pd.testing.assert_frame_equal(x, y, check_dtype=False)
        done.append(f"Friedman/post-hoc dokumentowe identyczne z bazą dla typów bez wykluczeń: {', '.join(same)}")
    return done


# ─── Runner ────────────────────────────────────────────────────────────────

def run_level2(
    error_csv: str = "output/tables/error_analysis_results.csv",
    notes_meta_csv: str = "output/tables/notes_meta.csv",
    notes_model_meta_csv: str = "output/tables/notes_model_meta.csv",
    output_dir: str = "output",
    units: tuple[str, ...] = ("patient", "document"),
    comparison_dir: str | None = None,
    baseline_tables_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Level 2 outputs. Descriptive tables/figures are written once; inferential
    ones (Friedman/post-hoc, Spearman CIs) once per unit with suffix _{unit}.
    With both units, writes unit_comparison_error_analysis.csv and
    icc_error_analysis.csv to `comparison_dir` and merges the combined reports there."""
    tables_dir = Path(output_dir) / "tables"
    figures_dir = Path(output_dir) / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df, notes_meta, notes_model_meta = load_level1_output(
        error_csv, notes_meta_csv, notes_model_meta_csv
    )
    print("[LOAD LEVEL 1 OUTPUT DF]\n", df)
    results: dict[str, pd.DataFrame] = {}
    results["error_type_by_model"] = error_type_by_model(df, tables_dir, figures_dir)
    results["error_type_by_category"] = error_type_by_category(df, tables_dir, figures_dir)
    error_type_by_category_grid(df, figures_dir)
    heatmap_model_x_errortype(df, figures_dir)
    heatmap_model_x_field(df, notes_meta, figures_dir)
    results["field_accuracy"] = field_accuracy_table(df, notes_meta, tables_dir)
    sp_len = error_rate_vs_note_length(df, notes_meta, tables_dir, figures_dir, units)
    sp_den = error_rate_vs_entity_density(df, notes_meta, tables_dir, figures_dir, units)
    for u in units:
        results[f"spearman_note_length_{u}"] = sp_len[u]
        results[f"spearman_entity_density_{u}"] = sp_den[u]
        results[f"spearman_summary_{u}"] = spearman_correlations_summary(sp_len[u], sp_den[u], tables_dir, figures_dir, u)
    results["spearman_note_length"] = sp_len[units[0]]
    results["top_omitted_medications"] = top_omitted_medications(df, tables_dir, figures_dir)
    results["top_hallucinated_medications"] = top_hallucinated_medications(df, tables_dir, figures_dir)
    results["severity_by_model"] = severity_by_model(df, tables_dir, figures_dir)
    results["summary_statistics"] = summary_table_latex(df, tables_dir)
    results["summary_rates"] = summary_table_rates_latex(df, notes_model_meta, tables_dir)

    per_doc = per_doc_error_counts(df, notes_meta, notes_model_meta)
    full: dict[str, dict[str, dict]] = {}
    for u in units:
        stat_df, full[u] = statistical_analysis_unit_level(per_doc, tables_dir, u)
        results[f"statistical_analysis_{u}"] = stat_df
        posthoc_effect_size_heatmaps(stat_df, list(notes_model_meta["model_name"].unique()), figures_dir, u)
        # critical_difference_diagrams(pd.read_csv(tables_dir / f"friedman_mean_ranks_{u}.csv"), stat_df, figures_dir, u)

    n_patients = notes_meta["patient_id"].nunique()
    checks = check_against_baseline(tables_dir, Path(baseline_tables_dir) if baseline_tables_dir else None,
                                    per_doc, full, n_patients)
    print("Asercje:\n  " + "\n  ".join(checks))

    icc = icc_error_rates(per_doc)
    results["icc"] = icc
    if comparison_dir is not None:
        cdir = Path(comparison_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        icc.to_csv(cdir / "icc_error_analysis.csv", index=False)
        if set(units) == set(sc.UNITS):
            sc.write_unit_comparison_part(
                unit_comparison_rows(full, {"note_length": sp_len, "entity_density": sp_den}),
                cdir / "unit_comparison_error_analysis.csv")
        sc.merge_unit_comparison(cdir, [cdir / "unit_comparison_validation.csv", cdir / "unit_comparison_error_analysis.csv"])
        sc.merge_icc_report(cdir, [cdir / "icc_validation.csv", cdir / "icc_error_analysis.csv"])

    print(f"Level 2 analysis complete. Tables -> {tables_dir}, figures -> {figures_dir}")
    return results


if __name__ == "__main__":
    run_level2()
