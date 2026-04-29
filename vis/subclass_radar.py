"""Build lightweight per-benchmark subclass radar reports from aggregate scores."""

from __future__ import annotations

import json
import math
import os
import os.path as osp
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import ResultLoader


DEFAULT_BENCHMARKS = ["MVBench", "VideoMME", "AoTBench", "MLVU"]

DIMENSION_PREFIXES = {
    "VideoMME": [
        ("overall/task_type/", "overall/task_type"),
        ("overall/domain/", "overall/domain"),
        ("overall/sub_category/", "overall/sub_category"),
    ],
}

SUMMARY_KEYS = {
    "AVG",
    "Average",
    "Overall",
    "overall",
    "overall/overall",
    "score",
    "acc",
    "accuracy",
    "total",
    "count",
    "samples",
    "failed",
}


def _norm_name(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _bench_matches(bench: str, family: str) -> bool:
    return _norm_name(bench).startswith(_norm_name(family))


def _strip_family_prefix(bench: str, family: str) -> str:
    label = bench
    for prefix in [family, family.replace("VideoMME", "Video-MME")]:
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break
    label = label.strip("_-/ ")
    return label or bench


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _score(loader: ResultLoader, value: Any) -> float | None:
    val = loader._normalize_score(value)
    if isinstance(val, float) and math.isnan(val):
        return None
    return float(val)


def _extract_dimensions(
    loader: ResultLoader,
    family: str,
    breakdown: dict[str, Any] | None,
) -> tuple[str, dict[str, float]]:
    if not isinstance(breakdown, dict) or not breakdown:
        return "", {}

    for prefix, dimension_name in DIMENSION_PREFIXES.get(family, []):
        rows = {}
        for key, value in breakdown.items():
            if not str(key).startswith(prefix):
                continue
            score = _score(loader, value)
            if score is not None:
                rows[str(key)[len(prefix) :]] = score
        if len(rows) >= 2:
            return dimension_name, rows

    rows = {}
    for key, value in breakdown.items():
        key = str(key)
        if key in SUMMARY_KEYS or key.endswith("/overall"):
            continue
        if key.count("/") > 2:
            continue
        score = _score(loader, value)
        if score is not None:
            rows[key] = score
    if len(rows) >= 2:
        return "subclass", rows
    return "", {}


def _matching_benchmarks(loader: ResultLoader, family: str) -> list[str]:
    benches = [bench for bench in loader.benchmarks if _bench_matches(bench, family)]
    benches.sort(key=lambda bench: (0 if _norm_name(bench) == _norm_name(family) else 1, len(bench), bench))
    return benches


def _collect_family(loader: ResultLoader, family: str) -> dict[str, Any] | None:
    benches = _matching_benchmarks(loader, family)
    if not benches:
        return None

    models = loader.models
    per_model: dict[str, dict[str, float]] = {model: {} for model in models}
    source_benchmarks: list[str] = []
    dimension_name = ""

    exact = [bench for bench in benches if _norm_name(bench) == _norm_name(family)]
    candidate_benches = exact or benches

    for bench in candidate_benches:
        bench_dimensions: dict[str, dict[str, float]] = {}
        bench_dimension_name = ""
        for model in models:
            breakdown = loader.load_breakdown(model, bench)
            dim_name, dims = _extract_dimensions(loader, family, breakdown)
            if dims:
                bench_dimensions[model] = dims
                bench_dimension_name = dim_name

        if bench_dimensions:
            source_benchmarks.append(bench)
            dimension_name = bench_dimension_name or dimension_name
            multi_source = len(candidate_benches) > 1 and not exact
            for model, dims in bench_dimensions.items():
                for dim, value in dims.items():
                    label = f"{_strip_family_prefix(bench, family)}/{dim}" if multi_source else dim
                    per_model[model][label] = value
            continue

        if not exact:
            values = {}
            for model in models:
                value = loader.load_score(model, bench)
                if not isinstance(value, float) or not math.isnan(value):
                    values[model] = float(value)
            if values:
                source_benchmarks.append(bench)
                dimension_name = "sub_benchmark"
                dim = _strip_family_prefix(bench, family)
                for model, value in values.items():
                    per_model[model][dim] = value

    dimensions = sorted({dim for dims in per_model.values() for dim in dims})
    if len(dimensions) < 2:
        return {
            "bench": family,
            "source_benchmarks": source_benchmarks,
            "dimension": dimension_name,
            "models": models,
            "dimensions": dimensions,
            "scores": per_model,
            "skipped": True,
            "reason": "fewer than two subclasses",
        }

    return {
        "bench": family,
        "source_benchmarks": source_benchmarks,
        "dimension": dimension_name or "subclass",
        "models": models,
        "dimensions": dimensions,
        "scores": per_model,
        "skipped": False,
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_scores_csv(bench_report: dict[str, Any], path: str) -> None:
    rows = []
    for model, scores in bench_report["scores"].items():
        for dim in bench_report["dimensions"]:
            rows.append(
                {
                    "bench": bench_report["bench"],
                    "model": model,
                    "dimension": dim,
                    "score": scores.get(dim),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _short_label(label: str, limit: int = 24) -> str:
    text = str(label)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _plot_radar(bench_report: dict[str, Any], path: str) -> None:
    dimensions = bench_report["dimensions"]
    models = bench_report["models"]
    if len(dimensions) < 2:
        return

    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_rlabel_position(180)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_short_label(dim) for dim in dimensions], fontsize=8)

    for idx, model in enumerate(models):
        scores = bench_report["scores"].get(model, {})
        values = [scores.get(dim, 0.0) or 0.0 for dim in dimensions]
        if not any(value > 0 for value in values):
            continue
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=model)
        ax.fill(angles, values, alpha=0.05)

    ax.set_title(f"{bench_report['bench']} subclass radar", y=1.08, fontsize=13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_index(report: dict[str, Any], out_dir: str) -> None:
    sections = []
    for bench in report["benchmarks"]:
        bench_report = report["by_bench"].get(bench)
        if not bench_report or bench_report.get("skipped"):
            sections.append(f"<h2>{bench}</h2><p>No subclass breakdown found.</p>")
            continue
        image_path = f"by_bench/{bench}/radar.png"
        csv_path = f"by_bench/{bench}/subclass_scores.csv"
        sections.append(
            "\n".join(
                [
                    f"<h2>{bench}</h2>",
                    f"<p>Dimension: {bench_report.get('dimension', 'subclass')}</p>",
                    f'<p><a href="{csv_path}">subclass_scores.csv</a></p>',
                    f'<img src="{image_path}" alt="{bench} radar">',
                ]
            )
        )

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Benchmark Subclass Radar</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }
h1 { font-size: 22px; }
h2 { font-size: 18px; margin-top: 28px; }
img { max-width: 960px; width: 100%; background: white; border: 1px solid #e5e7eb; border-radius: 6px; }
a { color: #2563eb; }
</style>
</head>
<body>
<h1>Benchmark Subclass Radar</h1>
__SECTIONS__
</body>
</html>
""".replace("__SECTIONS__", "\n".join(sections))
    with open(osp.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def build_subclass_radar_report(
    work_dir: str,
    out_dir: str,
    benchmarks: list[str] | None = None,
    models: list[str] | None = None,
) -> dict[str, Any]:
    loader = ResultLoader(work_dir)
    if models:
        keep = set(models)
        loader._models = [model for model in loader.models if model in keep]

    requested = benchmarks or DEFAULT_BENCHMARKS
    _ensure_dir(out_dir)
    by_bench_dir = osp.join(out_dir, "by_bench")
    _ensure_dir(by_bench_dir)

    report: dict[str, Any] = {
        "work_dir": osp.abspath(work_dir),
        "benchmarks": [],
        "models": loader.models,
        "by_bench": {},
    }

    for bench in requested:
        bench_report = _collect_family(loader, bench)
        if bench_report is None:
            bench_report = {
                "bench": bench,
                "source_benchmarks": [],
                "dimension": "",
                "models": loader.models,
                "dimensions": [],
                "scores": {model: {} for model in loader.models},
                "skipped": True,
                "reason": "benchmark not found",
            }
        report["benchmarks"].append(bench)
        report["by_bench"][bench] = bench_report

        bench_dir = osp.join(by_bench_dir, bench)
        _ensure_dir(bench_dir)
        _write_json(bench_report, osp.join(bench_dir, "subclass_scores.json"))
        _write_scores_csv(bench_report, osp.join(bench_dir, "subclass_scores.csv"))
        if not bench_report.get("skipped"):
            _plot_radar(bench_report, osp.join(bench_dir, "radar.png"))

    _write_json(report, osp.join(out_dir, "summary.json"))
    _write_index(report, out_dir)
    return json.loads(json.dumps(report, default=_json_safe))


def build_subclass_radar_payload(
    loader: ResultLoader,
    benchmarks: list[str] | None = None,
) -> dict[str, Any]:
    """Build dashboard-ready subclass radar data without writing files."""
    requested = benchmarks or DEFAULT_BENCHMARKS
    report: dict[str, Any] = {
        "benchmarks": [],
        "models": loader.models,
        "by_bench": {},
    }
    for bench in requested:
        bench_report = _collect_family(loader, bench)
        if bench_report is None:
            bench_report = {
                "bench": bench,
                "source_benchmarks": [],
                "dimension": "",
                "models": loader.models,
                "dimensions": [],
                "scores": {model: {} for model in loader.models},
                "skipped": True,
                "reason": "benchmark not found",
            }
        report["benchmarks"].append(bench)
        report["by_bench"][bench] = bench_report
    return json.loads(json.dumps(report, default=_json_safe))
