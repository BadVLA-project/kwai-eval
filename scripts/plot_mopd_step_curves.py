#!/usr/bin/env python3
"""Plot benchmark score curves for Qwen3-VL MOPD checkpoints.

The script scans a VLMEvalKit-style output directory, keeps benchmarks that are
present in the base model and every selected MOPD step, and draws a clean
paper-style line chart with step on the x-axis.

Example:
    python scripts/plot_mopd_step_curves.py \
        --work-dir /m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_final \
        --base-model Qwen3-VL-4B-Instruct \
        --out-dir ./mopd_step_plots
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCORE_SUFFIXES = (
    "_etbench_acc.csv",
    "_etbench_acc.xlsx",
    "_score.xlsx",
    "_acc.csv",
    "_acc.xlsx",
    "_rating.json",
    "_score.json",
)

PRIMARY_KEYS = (
    "AVG",
    "group_score",
    "overall/overall",
    "overall/score",
    "overall/acc",
    "overall/accuracy",
    "overall/mIoU",
    "overall",
    "Overall",
    "final_rating/total",
    "M-Avg",
    "Macro Accuracy",
    "Overall Consistency",
    "Final Score",
    "mIoU",
    "score",
    "accuracy",
    "acc",
)

BENCH_PRIMARY_OVERRIDES = {
    "vinoground": (
        "text_score",
        "overall/text_score",
        "Text",
        "text",
    ),
}

COUNT_KEYS = {
    "sample",
    "samples",
    "total",
    "failed",
    "count",
    "num",
    "number",
    "correct",
}

BENCH_SHORT_NAMES = {
    "AoTBench_QA_adaptive": "AoT-QA",
    "AoTBench_ReverseFilm_adaptive": "ReverseFilm",
    "AoTBench_Rtime_t2v_adaptive": "Rtime-t2v",
    "AoTBench_Rtime_v2t_adaptive": "Rtime-v2t",
    "AoTBench_UCF101_adaptive": "UCF101",
    "ETBench_adaptive": "ETBench",
    "MLVU_MCQ_adaptive": "MLVU",
    "MVBench_MP4_adaptive": "MVBench",
    "TempCompass_MCQ_adaptive": "TempCompass",
    "TimeLensBench_ActivityNet_adaptive": "TL-ActivityNet",
    "TimeLensBench_Charades_adaptive": "TL-Charades",
    "TimeLensBench_QVHighlights_adaptive": "TL-QVHighlights",
    "Video_Holmes_adaptive": "Video-Holmes",
    "Video-MME_adaptive": "Video-MME",
    "Vinoground_adaptive": "Vinoground-Text",
}

PALETTE = (
    "#D55E00",  # vermillion
    "#0072B2",  # blue
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8A63D2",  # purple
    "#6B7280",  # gray
    "#B45309",  # amber/brown
    "#0F766E",  # teal
    "#BE123C",  # rose
    "#2563EB",  # royal blue
)

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "*", "<", ">", "p")


@dataclass(frozen=True)
class MopdCurveData:
    """Scores for benchmarks shared by base and every selected MOPD step."""

    steps: list[int]
    models: list[str]
    benchmarks: list[str]
    scores: dict[str, list[float]]
    skipped_models: list[str]
    dropped_benchmarks: dict[str, list[str]]


@dataclass(frozen=True)
class MethodFamily:
    """A training method family mapped to checkpoint directory names."""

    label: str
    pattern: str


@dataclass(frozen=True)
class MethodComparisonData:
    """Scores for multiple training method families aligned by shared steps."""

    steps: list[int]
    families: list[str]
    benchmarks: list[str]
    scores: dict[str, dict[str, list[float]]]
    models: dict[str, list[str]]
    skipped_families: list[str]
    dropped_benchmarks: dict[str, list[str]]


DEFAULT_METHOD_FAMILIES = (
    MethodFamily("OPD", "{base}-MOPD-Step{step}"),
    MethodFamily("EMA-GRPO", "{base}-EMA-GRPO-Step{step}"),
)


def _is_count_like(key: str) -> bool:
    return str(key).strip().lower() in COUNT_KEYS


def _normalize_score(value) -> float:
    try:
        if isinstance(value, str):
            raw = value.strip()
            percent = raw.endswith("%")
            value = raw.rstrip("%")
        else:
            percent = False
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if math.isnan(v):
        return float("nan")
    if percent:
        return round(v, 4)
    return round(v * 100, 4) if abs(v) <= 1.0 else round(v, 4)


def _flatten_metrics(data, prefix: str = "") -> dict[str, float]:
    flat = {}
    if not isinstance(data, dict):
        return flat

    for key, value in data.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, full_key))
        elif isinstance(value, (int, float, str)):
            score = _normalize_score(value)
            if not math.isnan(score):
                flat[full_key] = score
        elif isinstance(value, list):
            parsed = _score_from_list(value)
            if not math.isnan(parsed):
                flat[full_key] = parsed
    return flat


def _score_from_list(value) -> float:
    if len(value) >= 3:
        return _normalize_score(value[2])
    if len(value) == 2:
        try:
            correct = float(value[0])
            total = float(value[1])
        except (TypeError, ValueError):
            return float("nan")
        if total > 0:
            return round(correct / total * 100, 4)
    return float("nan")


def _preferred_primary_keys(benchmark: str | None) -> tuple[str, ...]:
    if not benchmark:
        return ()
    lower = benchmark.lower()
    for token, keys in BENCH_PRIMARY_OVERRIDES.items():
        if token in lower:
            return keys
    return ()


def _score_from_dict(data: dict, benchmark: str | None = None) -> float:
    preferred = _preferred_primary_keys(benchmark)
    for key in preferred:
        if key in data and isinstance(data[key], (int, float, str)):
            return _normalize_score(data[key])

    for key in PRIMARY_KEYS:
        if key in data and isinstance(data[key], (int, float, str)):
            return _normalize_score(data[key])

    if "final_rating" in data and isinstance(data["final_rating"], dict):
        total = data["final_rating"].get("total")
        if total is not None:
            return _normalize_score(total)

    if "overall" in data:
        overall = data["overall"]
        if isinstance(overall, (int, float, str)):
            return _normalize_score(overall)
        if isinstance(overall, list):
            return _score_from_list(overall)
        if isinstance(overall, dict):
            for key in ("overall", "score", "acc", "accuracy", "mIoU"):
                if key in overall:
                    return _normalize_score(overall[key])

    if "total" in data and isinstance(data["total"], dict):
        for key in ("acc", "accuracy", "score"):
            if key in data["total"]:
                return _normalize_score(data["total"][key])

    list_scores = [
        _score_from_list(value)
        for value in data.values()
        if isinstance(value, list)
    ]
    list_scores = [score for score in list_scores if not math.isnan(score)]
    if list_scores:
        return round(sum(list_scores) / len(list_scores), 4)

    flat = _flatten_metrics(data)
    for key in preferred:
        if key in flat:
            return flat[key]

    for key in PRIMARY_KEYS:
        if key in flat:
            return flat[key]

    numeric_values = [
        value
        for key, value in flat.items()
        if not _is_count_like(key.split("/")[-1])
    ]
    if numeric_values:
        return round(sum(numeric_values) / len(numeric_values), 4)

    return float("nan")


def _frame_to_metrics(df) -> dict[str, float]:
    result = {}
    if df is None or df.empty:
        return result

    if "category" in df.columns and "accuracy" in df.columns:
        for _, row in df.iterrows():
            split = str(row.get("split", "")).strip()
            category = str(row["category"]).strip()
            key = f"{split}/{category}" if split and split.lower() != "nan" else category
            score = _normalize_score(row["accuracy"])
            if not math.isnan(score):
                result[key] = score
        return result

    if "metric" in df.columns and "value" in df.columns:
        for _, row in df.iterrows():
            score = _normalize_score(row["value"])
            if not math.isnan(score):
                result[str(row["metric"]).strip()] = score
        return result

    if "task" in df.columns and "acc" in df.columns:
        for _, row in df.iterrows():
            score = _normalize_score(row["acc"])
            if not math.isnan(score):
                result[str(row["task"]).strip()] = score
        return result

    if "Split" in df.columns:
        for _, row in df.iterrows():
            split = str(row["Split"]).strip()
            for column in df.columns:
                if column == "Split" or _is_count_like(column):
                    continue
                score = _normalize_score(row[column])
                if not math.isnan(score):
                    result[f"{split}/{column}"] = score
        return result

    if len(df) == 1:
        row = df.iloc[0]
        for column in df.columns:
            if _is_count_like(column):
                continue
            score = _normalize_score(row[column])
            if not math.isnan(score):
                result[str(column)] = score
    return result


def _score_from_frame(df, benchmark: str | None = None) -> float:
    metrics = _frame_to_metrics(df)
    for key in _preferred_primary_keys(benchmark):
        if key in metrics:
            return metrics[key]

    for key in PRIMARY_KEYS:
        if key in metrics:
            return metrics[key]

    for key in ("overall", "Overall", "/overall"):
        if key in metrics:
            return metrics[key]

    values = [
        value
        for key, value in metrics.items()
        if not _is_count_like(key.split("/")[-1])
    ]
    if values:
        return round(sum(values) / len(values), 4)
    return float("nan")


def _load_score(path: Path, benchmark: str | None = None) -> float:
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return _score_from_dict(data, benchmark)
            return float("nan")

        if path.suffix in {".csv", ".xlsx"}:
            import pandas as pd

            df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
            return _score_from_frame(df, benchmark)
    except Exception as exc:
        print(f"warning: failed to parse {path}: {exc}", file=sys.stderr)
    return float("nan")


def _extract_benchmark(model_name: str, filename: str) -> str | None:
    stem = filename
    for suffix in SCORE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        return None

    prefix = f"{model_name}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]

    match = re.search(rf"{re.escape(model_name)}_(.+)$", stem)
    if match:
        return match.group(1)
    return None


def _candidate_dirs(model_dir: Path) -> list[Path]:
    children = [path for path in model_dir.iterdir() if path.is_dir()]
    return [model_dir] + sorted(children, reverse=True)


def scan_model_scores(model_dir: Path) -> dict[str, float]:
    """Return {benchmark: primary_score} for the newest result per benchmark."""

    if not model_dir.is_dir():
        return {}

    scores = {}
    seen = set()
    model_name = model_dir.name

    for run_dir in _candidate_dirs(model_dir):
        for path in sorted(run_dir.iterdir()):
            if not path.is_file():
                continue
            bench = _extract_benchmark(model_name, path.name)
            if not bench or bench in seen:
                continue
            score = _load_score(path, bench)
            if not math.isnan(score):
                scores[bench] = score
                seen.add(bench)
    return scores


def discover_mopd_models(work_dir: Path, base_model: str) -> tuple[list[str], list[int], list[str]]:
    """Return selected model names and steps, with base represented as step 0."""

    models = [base_model]
    steps = [0]
    skipped = []
    pattern = re.compile(rf"^{re.escape(base_model)}-MOPD-Step(\d+)$")
    found = []

    for entry in sorted(work_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry.name))

    for step, name in sorted(found):
        models.append(name)
        steps.append(step)

    if not (work_dir / base_model).is_dir():
        skipped.append(base_model)

    return models, steps, skipped


def build_mopd_curve_data(
    work_dir: str | Path,
    base_model: str,
    selected_steps: list[int] | None = None,
    benchmarks: list[str] | None = None,
) -> MopdCurveData:
    """Collect score curves for benchmarks common to base and all MOPD models."""

    work_dir = Path(work_dir)
    models, steps, skipped_models = discover_mopd_models(work_dir, base_model)

    if selected_steps is not None:
        allowed = {0, *selected_steps}
        pairs = [(model, step) for model, step in zip(models, steps) if step in allowed]
        models = [model for model, _ in pairs]
        steps = [step for _, step in pairs]

    model_scores = {}
    for model in models:
        scores = scan_model_scores(work_dir / model)
        if scores:
            model_scores[model] = scores
        elif model not in skipped_models:
            skipped_models.append(model)

    valid_pairs = [
        (model, step)
        for model, step in zip(models, steps)
        if model in model_scores
    ]
    models = [model for model, _ in valid_pairs]
    steps = [step for _, step in valid_pairs]

    if base_model not in model_scores:
        raise FileNotFoundError(f"No parseable result files found for base model: {base_model}")
    if len(models) < 2:
        raise ValueError("Need the base model and at least one MOPD step with parseable results.")

    common = set(model_scores[models[0]])
    for model in models[1:]:
        common &= set(model_scores[model])

    if benchmarks:
        requested = list(dict.fromkeys(benchmarks))
        common &= set(requested)
        order = [bench for bench in requested if bench in common]
    else:
        order = sorted(common, key=_bench_sort_key)

    dropped = defaultdict(list)
    all_benchmarks = set().union(*(scores.keys() for scores in model_scores.values()))
    for bench in sorted(all_benchmarks):
        if benchmarks and bench not in benchmarks:
            continue
        missing = [model for model in models if bench not in model_scores[model]]
        if missing:
            dropped[bench] = missing

    if not order:
        raise ValueError("No benchmark intersection found across base and selected MOPD steps.")

    score_curves = {
        bench: [model_scores[model][bench] for model in models]
        for bench in order
    }

    return MopdCurveData(
        steps=steps,
        models=models,
        benchmarks=order,
        scores=score_curves,
        skipped_models=skipped_models,
        dropped_benchmarks=dict(dropped),
    )


def _family_model_name(base_model: str, family: MethodFamily, step: int) -> str:
    return family.pattern.format(base=base_model, step=step)


def _discover_family_steps(work_dir: Path, base_model: str, family: MethodFamily) -> list[int]:
    step_token = "__STEP__"
    template = re.escape(family.pattern.format(base=base_model, step=step_token))
    pattern = re.compile("^" + template.replace(re.escape(step_token), r"(\d+)") + "$")
    steps = []
    for entry in sorted(work_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def build_method_comparison_data(
    work_dir: str | Path,
    base_model: str,
    families: tuple[MethodFamily, ...] = DEFAULT_METHOD_FAMILIES,
    selected_steps: list[int] | None = None,
    benchmarks: list[str] | None = None,
) -> MethodComparisonData:
    """Collect aligned curves for multiple training method families.

    Steps are intersected across families, so each plotted point compares the
    same checkpoint step for every method. Base is always included at step 0.
    """

    work_dir = Path(work_dir)
    base_scores = scan_model_scores(work_dir / base_model)
    if not base_scores:
        raise FileNotFoundError(f"No parseable result files found for base model: {base_model}")

    family_steps = {}
    skipped_families = []
    for family in families:
        discovered = _discover_family_steps(work_dir, base_model, family)
        if selected_steps is not None:
            allowed = set(selected_steps)
            discovered = [step for step in discovered if step in allowed]
        if discovered:
            family_steps[family.label] = discovered
        else:
            skipped_families.append(family.label)

    active_families = [family for family in families if family.label in family_steps]
    if len(active_families) < 2:
        raise ValueError("Need at least two method families with parseable step directories.")

    candidate_steps = set(family_steps[active_families[0].label])
    for family in active_families[1:]:
        candidate_steps &= set(family_steps[family.label])
    candidate_steps = sorted(candidate_steps)
    if not candidate_steps:
        raise ValueError("No shared non-base steps found across method families.")

    step_model_scores = {}
    usable_steps = []
    for step in candidate_steps:
        by_family = {}
        for family in active_families:
            model = _family_model_name(base_model, family, step)
            model_scores = scan_model_scores(work_dir / model)
            if not model_scores:
                break
            by_family[family.label] = (model, model_scores)
        if len(by_family) == len(active_families):
            step_model_scores[step] = by_family
            usable_steps.append(step)

    if not usable_steps:
        raise ValueError("No shared method steps have parseable scores for every family.")

    all_steps = [0] + usable_steps
    usable_families = active_families
    family_models = {
        family.label: [
            base_model,
            *[step_model_scores[step][family.label][0] for step in usable_steps],
        ]
        for family in usable_families
    }
    family_model_scores = {
        family.label: {
            base_model: base_scores,
            **{
                step_model_scores[step][family.label][0]: step_model_scores[step][family.label][1]
                for step in usable_steps
            },
        }
        for family in usable_families
    }

    common_benchmarks = set(base_scores)
    for family in usable_families:
        for model in family_models[family.label]:
            common_benchmarks &= set(family_model_scores[family.label][model])

    if benchmarks:
        requested = list(dict.fromkeys(benchmarks))
        common_benchmarks &= set(requested)
        ordered_benchmarks = [bench for bench in requested if bench in common_benchmarks]
    else:
        ordered_benchmarks = sorted(common_benchmarks, key=_bench_sort_key)

    if not ordered_benchmarks:
        raise ValueError("No benchmark intersection found across method families.")

    dropped = defaultdict(list)
    all_benchmarks = set(base_scores)
    for family in usable_families:
        for model in family_models[family.label]:
            all_benchmarks |= set(family_model_scores[family.label][model])

    for bench in sorted(all_benchmarks):
        if benchmarks and bench not in benchmarks:
            continue
        missing = []
        for family in usable_families:
            for model in family_models[family.label]:
                if bench not in family_model_scores[family.label][model]:
                    missing.append(f"{family.label}:{model}")
        if missing:
            dropped[bench] = missing

    scores = {}
    for family in usable_families:
        label = family.label
        scores[label] = {}
        for bench in ordered_benchmarks:
            scores[label][bench] = [
                family_model_scores[label][model][bench]
                for model in family_models[label]
            ]

    return MethodComparisonData(
        steps=all_steps,
        families=[family.label for family in usable_families],
        benchmarks=ordered_benchmarks,
        scores=scores,
        models=family_models,
        skipped_families=skipped_families,
        dropped_benchmarks=dict(dropped),
    )


def _bench_sort_key(name: str) -> tuple[int, str]:
    order = [
        "AoTBench",
        "ETBench",
        "MVBench",
        "MLVU",
        "Video-MME",
        "TempCompass",
        "TimeLensBench",
        "Vinoground",
        "Video_Holmes",
    ]
    for idx, prefix in enumerate(order):
        if name.startswith(prefix):
            return idx, name
    return len(order), name


def short_bench(name: str) -> str:
    if name in BENCH_SHORT_NAMES:
        return BENCH_SHORT_NAMES[name]
    label = name
    for token in ("_adaptive", "_16frame", "_1fps"):
        label = label.replace(token, "")
    label = label.replace("AoTBench_", "AoT-")
    label = label.replace("TimeLensBench_", "TL-")
    label = label.replace("_", " ")
    return label


def write_score_tables(data: MopdCurveData, out_dir: Path, base_model: str) -> tuple[Path, Path]:
    import pandas as pd

    rows = []
    for bench in data.benchmarks:
        base_score = data.scores[bench][0]
        for model, step, score in zip(data.models, data.steps, data.scores[bench]):
            rows.append(
                {
                    "benchmark": bench,
                    "benchmark_short": short_bench(bench),
                    "step": step,
                    "model": model,
                    "score": score,
                    "delta_vs_base": round(score - base_score, 4),
                }
            )

    scores_path = out_dir / "mopd_step_scores.csv"
    pd.DataFrame(rows).to_csv(scores_path, index=False)

    delta_rows = []
    for bench in data.benchmarks:
        values = data.scores[bench]
        delta_rows.append(
            {
                "benchmark": bench,
                "benchmark_short": short_bench(bench),
                "base_model": base_model,
                "base_score": values[0],
                "best_step": data.steps[int(np.nanargmax(values))],
                "best_score": float(np.nanmax(values)),
                "final_step": data.steps[-1],
                "final_score": values[-1],
                "final_delta": round(values[-1] - values[0], 4),
                "best_delta": round(float(np.nanmax(values)) - values[0], 4),
            }
        )

    deltas_path = out_dir / "mopd_step_deltas.csv"
    pd.DataFrame(delta_rows).to_csv(deltas_path, index=False)
    return scores_path, deltas_path


def write_method_comparison_tables(data: MethodComparisonData, out_dir: Path) -> tuple[Path, Path]:
    import pandas as pd

    score_rows = []
    for family in data.families:
        for bench in data.benchmarks:
            base_score = data.scores[family][bench][0]
            for model, step, score in zip(data.models[family], data.steps, data.scores[family][bench]):
                score_rows.append(
                    {
                        "family": family,
                        "benchmark": bench,
                        "benchmark_short": short_bench(bench),
                        "step": step,
                        "model": model,
                        "score": score,
                        "delta_vs_base": round(score - base_score, 4),
                    }
                )

    scores_path = out_dir / "method_comparison_scores.csv"
    pd.DataFrame(score_rows).to_csv(scores_path, index=False)

    delta_rows = []
    for family in data.families:
        for bench in data.benchmarks:
            values = data.scores[family][bench]
            delta_rows.append(
                {
                    "family": family,
                    "benchmark": bench,
                    "benchmark_short": short_bench(bench),
                    "base_score": values[0],
                    "best_step": data.steps[int(np.nanargmax(values))],
                    "best_score": float(np.nanmax(values)),
                    "final_step": data.steps[-1],
                    "final_score": values[-1],
                    "final_delta": round(values[-1] - values[0], 4),
                    "best_delta": round(float(np.nanmax(values)) - values[0], 4),
                }
            )

    deltas_path = out_dir / "method_comparison_deltas.csv"
    pd.DataFrame(delta_rows).to_csv(deltas_path, index=False)
    return scores_path, deltas_path


def plot_curves(
    data: MopdCurveData,
    out_path: Path,
    title: str,
    normalized: bool = False,
    annotate_final: bool = True,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.0,
            "savefig.facecolor": "white",
        }
    )

    fig_width = max(8.0, min(13.5, 6.8 + len(data.benchmarks) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, 5.4))
    x = np.array(data.steps)

    plotted_values = []
    for idx, bench in enumerate(data.benchmarks):
        y = np.array(data.scores[bench], dtype=float)
        if normalized:
            y = y - y[0]
        plotted_values.extend(y.tolist())
        color = PALETTE[idx % len(PALETTE)]
        marker = MARKERS[idx % len(MARKERS)]
        label = short_bench(bench)
        ax.plot(
            x,
            y,
            marker=marker,
            markersize=6,
            linewidth=2.2,
            color=color,
            label=label,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

        if annotate_final:
            delta = data.scores[bench][-1] - data.scores[bench][0]
            suffix = f"{delta:+.1f}"
            ann_y = y[-1]
            ax.annotate(
                suffix,
                xy=(x[-1], ann_y),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8.5,
                color=color,
                fontweight="bold",
            )

    if normalized:
        ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.55)
        ax.set_ylabel("Score gain vs. base")
    else:
        ax.set_ylabel("Score")

    ax.set_xlabel("MOPD training step")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Base" if step == 0 else str(step) for step in data.steps])
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color="#C9CED6", alpha=0.72)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, color="#D6DAE0", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    valid = np.array([value for value in plotted_values if not math.isnan(value)])
    if len(valid):
        lo, hi = float(valid.min()), float(valid.max())
        pad = max(1.0, (hi - lo) * 0.15)
        if normalized:
            ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.set_ylim(max(0.0, lo - pad), min(105.0, hi + pad))

    legend_cols = 2 if len(data.benchmarks) <= 8 else 3
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=legend_cols,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.4,
    )

    fig.text(
        0.01,
        0.01,
        "Only benchmarks available for base and every selected MOPD step are plotted.",
        color="#555555",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")

    for ext in (".pdf", ".svg"):
        fig.savefig(out_path.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def plot_small_multiples(data: MopdCurveData, out_path: Path, title: str) -> None:
    n = len(data.benchmarks)
    cols = 3 if n > 4 else min(2, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4.2, rows * 3.0),
        sharex=True,
    )
    axes = np.array(axes).reshape(-1)
    x = np.array(data.steps)

    for idx, bench in enumerate(data.benchmarks):
        ax = axes[idx]
        y = np.array(data.scores[bench], dtype=float)
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        for step, score in zip(x, y):
            ax.text(step, score, f"{score:.1f}", ha="center", va="bottom", fontsize=7.8, color="#333333")
        delta = y[-1] - y[0]
        ax.set_title(f"{short_bench(bench)}  ({delta:+.1f})", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        pad = max(0.8, (hi - lo) * 0.2)
        ax.set_ylim(max(0.0, lo - pad), min(105.0, hi + pad))

    for ax in axes[n:]:
        ax.set_visible(False)

    for ax in axes[:n]:
        ax.set_xticks(x)
        ax.set_xticklabels(["Base" if step == 0 else str(step) for step in data.steps])

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.supxlabel("MOPD training step", y=0.02, fontsize=12)
    fig.supylabel("Score", x=0.02, fontsize=12)
    fig.tight_layout(rect=(0.03, 0.04, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    for ext in (".pdf", ".svg"):
        fig.savefig(out_path.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def plot_method_mean_gain(data: MethodComparisonData, out_path: Path, title: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.array(data.steps)
    plotted = []
    family_styles = {
        "OPD": {"color": "#D55E00", "marker": "o"},
        "EMA-GRPO": {"color": "#0072B2", "marker": "s"},
    }

    for idx, family in enumerate(data.families):
        matrix = np.array([data.scores[family][bench] for bench in data.benchmarks], dtype=float)
        gains = matrix - matrix[:, [0]]
        mean_gain = gains.mean(axis=0)
        plotted.extend(mean_gain.tolist())
        style = family_styles.get(
            family,
            {"color": PALETTE[idx % len(PALETTE)], "marker": MARKERS[idx % len(MARKERS)]},
        )
        ax.plot(
            x,
            mean_gain,
            label=family,
            color=style["color"],
            marker=style["marker"],
            markersize=7,
            linewidth=2.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )
        for step, value in zip(x[1:], mean_gain[1:]):
            ax.text(step, value, f"{value:+.1f}", ha="center", va="bottom", fontsize=9, color=style["color"])

    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean score gain vs. base")
    ax.set_xticks(x)
    ax.set_xticklabels(["Base" if step == 0 else str(step) for step in x])
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color="#C9CED6", alpha=0.72)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, color="#D6DAE0", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    valid = np.array(plotted)
    if len(valid):
        lo, hi = float(valid.min()), float(valid.max())
        pad = max(0.6, (hi - lo) * 0.2)
        ax.set_ylim(lo - pad, hi + pad)
    ax.legend(frameon=False, loc="upper left")
    fig.text(
        0.01,
        0.01,
        f"Mean over shared benchmarks: {', '.join(short_bench(b) for b in data.benchmarks)}.",
        color="#555555",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    for ext in (".pdf", ".svg"):
        fig.savefig(out_path.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def plot_method_benchmark_gains(data: MethodComparisonData, out_path: Path, title: str) -> None:
    n = len(data.benchmarks)
    cols = 3 if n > 4 else min(2, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 3.15), sharex=True)
    axes = np.array(axes).reshape(-1)
    x = np.array(data.steps)
    family_styles = {
        "OPD": {"color": "#D55E00", "marker": "o"},
        "EMA-GRPO": {"color": "#0072B2", "marker": "s"},
    }

    for idx, bench in enumerate(data.benchmarks):
        ax = axes[idx]
        valid_values = []
        for fi, family in enumerate(data.families):
            y = np.array(data.scores[family][bench], dtype=float)
            gains = y - y[0]
            valid_values.extend(gains.tolist())
            style = family_styles.get(
                family,
                {"color": PALETTE[fi % len(PALETTE)], "marker": MARKERS[fi % len(MARKERS)]},
            )
            ax.plot(
                x,
                gains,
                label=family,
                color=style["color"],
                marker=style["marker"],
                markersize=5.5,
                linewidth=2.2,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
            ax.text(
                x[-1],
                gains[-1],
                f"{gains[-1]:+.1f}",
                ha="left",
                va="center",
                fontsize=8,
                color=style["color"],
                fontweight="bold",
            )
        ax.axhline(0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.5)
        ax.set_title(short_bench(bench), fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        lo, hi = float(np.nanmin(valid_values)), float(np.nanmax(valid_values))
        pad = max(0.5, (hi - lo) * 0.25)
        ax.set_ylim(lo - pad, hi + pad)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[:n]:
        ax.set_xticks(x)
        ax.set_xticklabels(["Base" if step == 0 else str(step) for step in data.steps])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(data.families), frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.supxlabel("Training step", y=0.06, fontsize=12)
    fig.supylabel("Score gain vs. base", x=0.02, fontsize=12)
    fig.tight_layout(rect=(0.03, 0.1, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    for ext in (".pdf", ".svg"):
        fig.savefig(out_path.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot base vs. Qwen3-VL MOPD step score curves on shared benchmarks. "
            "If EMA-GRPO step directories are present, also writes OPD vs. EMA-GRPO comparison plots."
        )
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Directory containing model result subdirectories.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen3-VL-4B-Instruct",
        help="Base model directory name. Default: Qwen3-VL-4B-Instruct",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=None,
        help="Optional MOPD steps to include, e.g. --steps 50 100 150. Base is always included.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional benchmark names to plot. By default uses the full intersection.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <work-dir>/mopd_step_plots",
    )
    parser.add_argument(
        "--title",
        default="MOPD Step Gains on Shared Benchmarks",
        help="Figure title.",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Retained for compatibility. Gain curves are now always written.",
    )
    parser.add_argument(
        "--small-multiples",
        action="store_true",
        help="Write compact per-benchmark zoomed plots. This is now enabled by default.",
    )
    parser.add_argument(
        "--no-small-multiples",
        action="store_true",
        help="Disable the compact per-benchmark zoomed plot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir) if args.out_dir else work_dir / "mopd_step_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_mopd_curve_data(
        work_dir=work_dir,
        base_model=args.base_model,
        selected_steps=args.steps,
        benchmarks=args.benchmarks,
    )

    scores_path, deltas_path = write_score_tables(data, out_dir, args.base_model)

    plot_path = out_dir / "mopd_step_curves.png"
    plot_curves(data, plot_path, args.title, normalized=False)

    gain_path = out_dir / "mopd_step_gain_curves.png"
    plot_curves(
        data,
        gain_path,
        "MOPD Score Gains vs. Base",
        normalized=True,
    )

    if not args.no_small_multiples:
        plot_small_multiples(
            data,
            out_dir / "mopd_step_small_multiples.png",
            args.title,
        )

    method_outputs = []
    method_skip_reason = None
    try:
        method_data = build_method_comparison_data(
            work_dir=work_dir,
            base_model=args.base_model,
            selected_steps=args.steps,
            benchmarks=args.benchmarks,
        )
    except (FileNotFoundError, ValueError) as exc:
        method_skip_reason = str(exc)
        method_data = None

    if method_data is not None:
        method_scores_path, method_deltas_path = write_method_comparison_tables(method_data, out_dir)
        mean_gain_path = out_dir / "method_comparison_mean_gain.png"
        benchmark_gain_path = out_dir / "method_comparison_benchmark_gains.png"
        plot_method_mean_gain(
            method_data,
            mean_gain_path,
            "OPD vs. EMA-GRPO Mean Gains",
        )
        plot_method_benchmark_gains(
            method_data,
            benchmark_gain_path,
            "OPD vs. EMA-GRPO Benchmark Gains",
        )
        method_outputs = [
            method_scores_path,
            method_deltas_path,
            mean_gain_path,
            benchmark_gain_path,
        ]

    print(f"Models: {', '.join(data.models)}")
    print(f"Steps: {', '.join(map(str, data.steps))}")
    print(f"Benchmarks ({len(data.benchmarks)}): {', '.join(data.benchmarks)}")
    if data.skipped_models:
        print(f"Skipped models without parseable scores: {', '.join(data.skipped_models)}")
    if data.dropped_benchmarks:
        print(f"Dropped non-intersection benchmarks: {len(data.dropped_benchmarks)}")
    print(f"Saved scores: {scores_path}")
    print(f"Saved deltas: {deltas_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved gain plot: {gain_path}")
    if not args.no_small_multiples:
        print(f"Saved zoomed plot: {out_dir / 'mopd_step_small_multiples.png'}")
    if method_data is not None:
        print(f"Method families: {', '.join(method_data.families)}")
        print(f"Method comparison steps: {', '.join(map(str, method_data.steps))}")
        print(f"Method comparison benchmarks ({len(method_data.benchmarks)}): "
              f"{', '.join(method_data.benchmarks)}")
        for output in method_outputs:
            print(f"Saved method comparison: {output}")
    elif method_skip_reason:
        print(f"Skipped method comparison: {method_skip_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
