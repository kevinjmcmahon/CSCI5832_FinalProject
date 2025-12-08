"""
Generate label distribution bar charts for SemEval2026 subtasks.

This script aggregates label counts from the `train` splits of subtask1,
subtask2, and subtask3 for a specified set of languages, then saves one bar
chart per subtask under the chosen output directory.

Usage:
    python plot_label_distributions.py --output-dir figures

By default it uses languages: english, arabic, spanish, german, and chinese
(`eng`, `arb`, `spa`, `deu`, `zho`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


LANG_CODES = ["eng", "arb", "spa", "deu", "zho"]

SUBTASK_CONFIG: Dict[str, Dict[str, List[str]]] = {
    "subtask1": {"label_cols": ["polarization"]},
    "subtask2": {
        "label_cols": [
            "political",
            "racial/ethnic",
            "religious",
            "gender/sexual",
            "other",
        ]
    },
    "subtask3": {
        "label_cols": [
            "stereotype",
            "vilification",
            "dehumanization",
            "extreme_language",
            "lack_of_empathy",
            "invalidation",
        ]
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create label distribution bar charts for all subtasks."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="Path to the repository root containing subtask directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to save the generated figures.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Language codes to include (e.g., eng arb spa deu zho). Defaults to "
        "all languages with data in every subtask when omitted.",
    )
    return parser.parse_args()


def discover_languages(data_root: Path) -> List[str]:
    """Return languages that have train data for every subtask."""
    lang_sets = []
    for subtask in SUBTASK_CONFIG:
        train_dir = data_root / subtask / "train"
        langs = {p.stem for p in train_dir.glob("*.csv")}
        if langs:
            lang_sets.append(langs)
    if not lang_sets:
        raise FileNotFoundError("No training data found to detect languages.")
    intersection = set.intersection(*lang_sets)
    if intersection:
        return sorted(intersection)
    # Fall back to the union if the intersection is empty.
    union_langs = sorted(set.union(*lang_sets))
    print(
        "Warning: Not all subtask directories share the same languages. "
        "Using the union of available languages."
    )
    return union_langs


def load_frames_by_language(
    data_root: Path, subtask: str, languages: Iterable[str]
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for lang in languages:
        csv_path = data_root / subtask / "train" / f"{lang}.csv"
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue
        frames[lang] = pd.read_csv(csv_path)
    if not frames:
        raise FileNotFoundError(
            f"No data files found for {subtask} in languages: {', '.join(languages)}"
        )
    return frames


def count_labels_by_language(
    subtask: str, frames: Dict[str, pd.DataFrame]
) -> Dict[str, pd.Series]:
    label_cols = SUBTASK_CONFIG[subtask]["label_cols"]
    counts: Dict[str, pd.Series] = {}
    for lang, df in frames.items():
        if subtask == "subtask1":
            series = df[label_cols[0]].astype(int)
            lang_counts = series.value_counts().reindex([0, 1], fill_value=0)
            lang_counts.index = ["no", "yes"]
        else:
            lang_counts = pd.Series(
                {col: int(df[col].astype(int).sum()) for col in label_cols}
            )
        counts[lang] = lang_counts
    return counts


def plot_counts(
    subtask: str,
    per_lang_counts: Dict[str, pd.Series],
    label_order: List[str],
    languages: List[str],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.12
    x_positions = list(range(len(label_order)))

    for idx, lang in enumerate(languages):
        if lang not in per_lang_counts:
            continue
        offsets = [x + (idx - (len(languages) - 1) / 2) * width for x in x_positions]
        vals = [per_lang_counts[lang].get(label, 0) for label in label_order]
        ax.bar(offsets, vals, width=width, label=lang)

    totals = [sum(per_lang_counts[lang].get(label, 0) for lang in per_lang_counts) for label in label_order]
    for x, total in zip(x_positions, totals):
        ax.text(
            x,
            total,
            f"{total}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(label_order, rotation=25, ha="right")
    ax.set_title(f"{subtask} label distribution by language")
    ax.set_ylabel("Number of labels")
    ax.set_xlabel("Label")
    ax.legend(title="Language")
    max_total = max(totals) if totals else 0
    ax.set_ylim(0, max_total * 1.2 if max_total else 1)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subtask}_label_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_example_totals(
    frames_by_subtask: Dict[str, Dict[str, pd.DataFrame]],
    languages: List[str],
    output_dir: Path,
) -> Path:
    subtask_names = list(frames_by_subtask.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.12
    x_positions = list(range(len(subtask_names)))

    max_val = 0
    for idx, lang in enumerate(languages):
        offsets = [
            x + (idx - (len(languages) - 1) / 2) * width for x in x_positions
        ]
        vals = [
            len(frames_by_subtask[subtask].get(lang, pd.DataFrame()))
            if lang in frames_by_subtask[subtask]
            else 0
            for subtask in subtask_names
        ]
        max_val = max(max_val, max(vals) if vals else 0)
        bars = ax.bar(offsets, vals, width=width, label=lang)
        ax.bar_label(bars, padding=3, fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(subtask_names, rotation=0, ha="center")
    ax.set_title("Training samples by subtask and language")
    ax.set_ylabel("Number of examples")
    ax.set_xlabel("Subtask")
    ax.legend(title="Language")
    ax.set_ylim(0, max_val * 1.25 if max_val else 1)
    fig.tight_layout()

    # Save per-language totals as a small CSV table.
    totals_by_language = {
        lang: sum(
            len(frames_by_subtask[subtask].get(lang, pd.DataFrame()))
            for subtask in subtask_names
        )
        for lang in languages
    }
    totals_df = pd.DataFrame(
        [{"language": lang, "total_examples": totals_by_language[lang]} for lang in languages]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "train_examples_by_language.csv"
    totals_df.to_csv(table_path, index=False)

    output_path = output_dir / "train_examples_by_subtask.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved totals table to {table_path}")
    return output_path


def save_subtask2_language_table(
    frames: Dict[str, pd.DataFrame], output_dir: Path
) -> Path:
    """Write a table of subtask2 class counts per language."""
    label_cols = SUBTASK_CONFIG["subtask2"]["label_cols"]
    rows = []
    for lang, df in sorted(frames.items()):
        row = {"language": lang, "total_examples": len(df)}
        for col in label_cols:
            row[col] = int(df[col].astype(int).sum())
        rows.append(row)
    if not rows:
        raise ValueError("No subtask2 frames provided to save_subtask2_language_table.")

    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "subtask2_language_class_counts.csv"
    table.to_csv(path, index=False)
    print(f"subtask2: saved counts table to {path}")
    return path


def main() -> None:
    args = parse_args()
    languages = (
        discover_languages(args.data_root) if args.languages is None else args.languages
    )
    frames_by_subtask: Dict[str, Dict[str, pd.DataFrame]] = {}
    for subtask in SUBTASK_CONFIG:
        frames = load_frames_by_language(args.data_root, subtask, languages)
        frames_by_subtask[subtask] = frames
        per_lang_counts = count_labels_by_language(subtask, frames)
        label_order = SUBTASK_CONFIG[subtask]["label_cols"]
        if subtask == "subtask1":
            label_order = ["no", "yes"]
        output_path = plot_counts(
            subtask, per_lang_counts, label_order, languages, args.output_dir
        )
        print(f"{subtask}: saved figure to {output_path}")
        if subtask == "subtask2":
            save_subtask2_language_table(frames, args.output_dir)
    total_path = plot_example_totals(frames_by_subtask, languages, args.output_dir)
    print(f"Totals: saved figure to {total_path}")


if __name__ == "__main__":
    main()
