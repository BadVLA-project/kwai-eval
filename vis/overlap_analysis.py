"""Row-level benchmark overlap analysis for multiple evaluated models."""

from __future__ import annotations

import glob
import json
import math
import os
import os.path as osp
import tempfile
from itertools import combinations
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", osp.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import numpy as np
import pandas as pd


GROUP_PRIORITY = [
    "category",
    "source",
    "split",
    "area",
    "reasoning",
    "tag",
    "task",
    "task_type",
    "question_type",
    "subtask",
    "sub_task",
    "subcategory",
    "sub_category",
    "subject",
    "domain",
    "type",
]

DISPLAY_COLUMNS = [
    "question",
    "answer",
    "video",
    "video_name",
    "video_path",
    "image",
    "image_path",
]

EXCLUDED_GROUP_COLUMNS = {
    "index",
    "question_id",
    "id",
    "_case_id",
    "question",
    "answer",
    "prediction",
    "thinking",
    "extra_records",
    "image",
    "image_path",
    "video",
    "video_name",
    "video_path",
    "frames",
    "frame_paths",
    "candidates",
    "choices",
    "extracted_answer",
    "pred_id",
    "score",
    "correct",
    "iou",
    "_metric",
    "_correct",
}

METRIC_COLUMNS = ["correct", "score", "iou"]
SCORE_SUFFIXES = [".json", ".jsonl", ".csv", ".tsv", ".xlsx"]


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _shorten(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = " ".join(str(value).strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _load_file(path: str) -> Any:
    suffix = osp.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported score file: {path}")


def _to_dataframe(data: Any) -> pd.DataFrame | None:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        if all(isinstance(row, dict) for row in data):
            return pd.DataFrame(data)
        return None
    if isinstance(data, dict):
        if "columns" in data and "data" in data:
            return pd.DataFrame(data["data"], columns=data["columns"])
        values = list(data.values())
        if values and all(isinstance(value, list) for value in values):
            lengths = {len(value) for value in values}
            if len(lengths) == 1:
                return pd.DataFrame(data)
    return None


def load_row_score_file(path: str) -> pd.DataFrame | None:
    try:
        df = _to_dataframe(_load_file(path))
    except Exception:
        return None
    if df is None or df.empty:
        return None
    id_col = next((col for col in ["index", "question_id", "id"] if col in df.columns), None)
    if id_col is None:
        return None
    metric_col = infer_metric_column(df)
    if metric_col is None:
        return None
    df = df.copy()
    df["_case_id"] = df[id_col].astype(str)
    return df


def infer_metric_column(df: pd.DataFrame) -> str | None:
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            return col
    return None


def metric_scale(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return 100.0
    return 100.0 if valid.min() >= 0 and valid.max() <= 1 else 1.0


def normalize_metric_dataframe(
    df: pd.DataFrame,
    group_columns: list[str],
    correct_threshold: float,
) -> pd.DataFrame:
    metric_col = infer_metric_column(df)
    if metric_col is None:
        raise ValueError("No metric column found")

    metric = pd.to_numeric(df[metric_col], errors="coerce")
    valid = metric.notna() & (metric >= 0)

    out = df.copy()
    out["_metric"] = metric.where(valid)
    if metric_col in {"correct", "score"}:
        unique = set(metric.dropna().unique().tolist())
        if unique.issubset({0, 1, 0.0, 1.0}):
            correct = metric > 0
        else:
            correct = metric >= correct_threshold
    else:
        correct = metric >= correct_threshold
    out["_correct"] = pd.Series(np.nan, index=out.index, dtype=object)
    out.loc[valid, "_correct"] = correct.loc[valid].astype(bool)

    keep = ["_case_id", "_metric", "_correct", "prediction"]
    keep += [col for col in DISPLAY_COLUMNS + group_columns if col in out.columns]
    keep = list(dict.fromkeys(keep))
    return out[[col for col in keep if col in out.columns]].copy()


def benchmark_from_score_path(model: str, path: str) -> str | None:
    stem = osp.splitext(osp.basename(path))[0]
    prefix = f"{model}_"
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix) :]
    if not tail.endswith("_score"):
        return None
    bench = tail[: -len("_score")]
    return bench or None


def discover_model_score_files(work_dir: str, models: list[str] | None = None) -> dict[str, dict[str, str]]:
    if not osp.isdir(work_dir):
        return {}

    model_names = models or [
        entry
        for entry in sorted(os.listdir(work_dir))
        if not entry.startswith(".") and osp.isdir(osp.join(work_dir, entry))
    ]
    discovered: dict[str, dict[str, str]] = {}

    for model in model_names:
        model_dir = osp.join(work_dir, model)
        if not osp.isdir(model_dir):
            continue
        files = []
        for suffix in SCORE_SUFFIXES:
            files.extend(glob.glob(osp.join(model_dir, f"{model}_*_score{suffix}")))
            files.extend(glob.glob(osp.join(model_dir, "T*", f"{model}_*_score{suffix}")))

        by_dataset: dict[str, list[str]] = {}
        for path in sorted(set(files)):
            dataset = benchmark_from_score_path(model, path)
            if dataset is not None:
                by_dataset.setdefault(dataset, []).append(path)

        usable: dict[str, str] = {}
        for dataset, paths in by_dataset.items():
            good_paths = [path for path in paths if load_row_score_file(path) is not None]
            if good_paths:
                usable[dataset] = sorted(good_paths)[-1]
        if usable:
            discovered[model] = usable

    return discovered


def discover_group_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if col in EXCLUDED_GROUP_COLUMNS or col.endswith("_base") or col.endswith("_cand"):
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        as_text = series.astype(str)
        nunique = as_text.nunique()
        if nunique < 2 or nunique > 40:
            continue
        if nunique > len(series) * 0.5:
            continue
        if as_text.map(len).mean() > 60:
            continue
        candidates.append(col)

    ordered = [col for col in GROUP_PRIORITY if col in candidates]
    rest = sorted([col for col in candidates if col not in ordered], key=lambda col: (df[col].nunique(), col))
    return ordered + rest


def _resolve_group_columns(frames: list[pd.DataFrame], explicit: list[str] | None) -> list[str]:
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if explicit:
        return [col for col in explicit if col in combined.columns]
    return discover_group_columns(combined)


def _score_percent(values: pd.Series) -> float | None:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean() * metric_scale(valid))


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(lambda value: bool(value) if pd.notna(value) else False).astype(bool)


def _safe_div(num: int | float, denom: int | float) -> float | None:
    return float(num / denom) if denom else None


def _sort_ids_by_reference(reference: pd.DataFrame, ids: set[str]) -> list[str]:
    order = []
    seen = set()
    for case_id in reference["_case_id"].astype(str):
        if case_id in ids and case_id not in seen:
            order.append(case_id)
            seen.add(case_id)
    return order + sorted(ids - seen)


def _first_non_empty(frames: list[pd.DataFrame], case_id: str, column: str, limit: int) -> str:
    for frame in frames:
        if column not in frame.columns:
            continue
        hit = frame[frame["_case_id"] == case_id]
        if hit.empty:
            continue
        value = hit.iloc[0].get(column)
        text = _shorten(value, limit)
        if text:
            return text
    return ""


def _group_tags(frames: list[pd.DataFrame], case_id: str, group_columns: list[str]) -> dict[str, Any]:
    tags = {}
    for col in group_columns:
        for frame in frames:
            if col not in frame.columns:
                continue
            hit = frame[frame["_case_id"] == case_id]
            if hit.empty:
                continue
            value = _json_safe(hit.iloc[0].get(col))
            if value is not None and str(value) != "":
                tags[col] = value
                break
    return tags


def build_overlap_analysis(
    work_dir: str,
    models: list[str] | None = None,
    baseline: str | None = None,
    data: list[str] | None = None,
    group_columns: list[str] | None = None,
    min_group_size: int = 1,
    correct_threshold: float = 0.5,
    max_case_matrix: int | None = None,
    all_baselines: bool = False,
) -> dict[str, Any]:
    """Build a dashboard-friendly multi-model answer overlap summary."""

    score_files = discover_model_score_files(work_dir, models=models)
    model_names = [model for model in (models or sorted(score_files)) if model in score_files]
    if baseline is None and model_names:
        baseline = model_names[0]

    dataset_set = set(data or [])
    if not dataset_set:
        for by_dataset in score_files.values():
            dataset_set.update(by_dataset)
    datasets = sorted(dataset_set)

    pairwise_overlap: list[dict[str, Any]] = []
    model_dataset_summary: list[dict[str, Any]] = []
    dataset_deltas: list[dict[str, Any]] = []
    group_deltas: list[dict[str, Any]] = []
    case_matrix: list[dict[str, Any]] = []

    for dataset in datasets:
        raw_frames: dict[str, pd.DataFrame] = {}
        for model in model_names:
            path = score_files.get(model, {}).get(dataset)
            if not path:
                continue
            df = load_row_score_file(path)
            if df is not None and not df.empty:
                raw_frames[model] = df

        active_models = [model for model in model_names if model in raw_frames]
        if len(active_models) < 2:
            continue

        resolved_groups = _resolve_group_columns([raw_frames[m] for m in active_models], group_columns)
        frames = {
            model: normalize_metric_dataframe(raw_frames[model], resolved_groups, correct_threshold)
            for model in active_models
        }

        for model in active_models:
            frame = frames[model]
            valid = frame["_metric"].notna()
            model_dataset_summary.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "score": _score_percent(frame.loc[valid, "_metric"]),
                    "correct_count": int(_bool_series(frame.loc[valid, "_correct"]).sum()),
                    "valid_cases": int(valid.sum()),
                    "score_file": score_files[model][dataset],
                }
            )

        for model_a, model_b in combinations(active_models, 2):
            left = frames[model_a][["_case_id", "_metric", "_correct"]].rename(
                columns={"_metric": "metric_a", "_correct": "correct_a"}
            )
            right = frames[model_b][["_case_id", "_metric", "_correct"]].rename(
                columns={"_metric": "metric_b", "_correct": "correct_b"}
            )
            paired = pd.merge(left, right, on="_case_id", how="inner")
            paired = paired[paired["metric_a"].notna() & paired["metric_b"].notna()]
            if paired.empty:
                continue

            a_correct = _bool_series(paired["correct_a"])
            b_correct = _bool_series(paired["correct_b"])
            both_correct = int((a_correct & b_correct).sum())
            model_a_only = int((a_correct & ~b_correct).sum())
            model_b_only = int((~a_correct & b_correct).sum())
            both_wrong = int((~a_correct & ~b_correct).sum())
            shared = int(len(paired))
            correct_union = both_correct + model_a_only + model_b_only
            score_a = float(a_correct.mean() * 100)
            score_b = float(b_correct.mean() * 100)

            pairwise_overlap.append(
                {
                    "dataset": dataset,
                    "model_a": model_a,
                    "model_b": model_b,
                    "shared_cases": shared,
                    "score_a": score_a,
                    "score_b": score_b,
                    "delta_b_minus_a": score_b - score_a,
                    "both_correct": both_correct,
                    "model_a_only": model_a_only,
                    "model_b_only": model_b_only,
                    "both_wrong": both_wrong,
                    "both_correct_rate": _safe_div(both_correct, shared),
                    "jaccard_correct": _safe_div(both_correct, correct_union),
                    "disagreement_count": model_a_only + model_b_only,
                    "disagreement_rate": _safe_div(model_a_only + model_b_only, shared),
                }
            )

        baseline_models = active_models if all_baselines else ([baseline] if baseline in active_models else [])
        for base_model in baseline_models:
            base_frame = frames[base_model]
            for candidate in active_models:
                if candidate == base_model:
                    continue
                cand_frame = frames[candidate]
                merged = pd.merge(
                    base_frame,
                    cand_frame,
                    on="_case_id",
                    how="inner",
                    suffixes=("_base", "_cand"),
                )
                merged = merged[merged["_metric_base"].notna() & merged["_metric_cand"].notna()]
                if merged.empty:
                    continue

                base_correct = _bool_series(merged["_correct_base"])
                cand_correct = _bool_series(merged["_correct_cand"])
                fixes = int((~base_correct & cand_correct).sum())
                drops = int((base_correct & ~cand_correct).sum())
                base_score = float(base_correct.mean() * 100)
                cand_score = float(cand_correct.mean() * 100)

                dataset_deltas.append(
                    {
                        "dataset": dataset,
                        "baseline_model": base_model,
                        "candidate_model": candidate,
                        "shared_cases": int(len(merged)),
                        "baseline_score": base_score,
                        "candidate_score": cand_score,
                        "delta": cand_score - base_score,
                        "fixes": fixes,
                        "drops": drops,
                        "stable_correct": int((base_correct & cand_correct).sum()),
                        "stable_wrong": int((~base_correct & ~cand_correct).sum()),
                    }
                )

                for col in resolved_groups:
                    group_col = col if col in merged.columns else f"{col}_cand"
                    if group_col not in merged.columns:
                        group_col = f"{col}_base"
                    if group_col not in merged.columns:
                        continue
                    for value, group_df in merged.groupby(group_col, dropna=True):
                        if len(group_df) < min_group_size:
                            continue
                        g_base = _bool_series(group_df["_correct_base"])
                        g_cand = _bool_series(group_df["_correct_cand"])
                        g_fixes = int((~g_base & g_cand).sum())
                        g_drops = int((g_base & ~g_cand).sum())
                        g_base_score = float(g_base.mean() * 100)
                        g_cand_score = float(g_cand.mean() * 100)
                        group_deltas.append(
                            {
                                "dataset": dataset,
                                "baseline_model": base_model,
                                "candidate_model": candidate,
                                "group_column": col,
                                "group_value": _json_safe(value),
                                "label": f"{col}={value}",
                                "sample_count": int(len(group_df)),
                                "baseline_score": g_base_score,
                                "candidate_score": g_cand_score,
                                "delta": g_cand_score - g_base_score,
                                "fixes": g_fixes,
                                "drops": g_drops,
                                "stable_correct": int((g_base & g_cand).sum()),
                                "stable_wrong": int((~g_base & ~g_cand).sum()),
                            }
                        )

        common_ids = set(frames[active_models[0]]["_case_id"].astype(str))
        for model in active_models[1:]:
            common_ids &= set(frames[model]["_case_id"].astype(str))
        if common_ids:
            ordered_ids = _sort_ids_by_reference(frames[active_models[0]], common_ids)
            for case_id in ordered_ids:
                if max_case_matrix is not None and len(case_matrix) >= max_case_matrix:
                    break
                model_metrics = {}
                model_correct = {}
                model_predictions = {}
                for model in active_models:
                    row = frames[model][frames[model]["_case_id"] == case_id].iloc[0]
                    metric_val = _json_safe(row.get("_metric"))
                    model_metrics[model] = float(metric_val) if metric_val is not None else None
                    correct_val = row.get("_correct")
                    model_correct[model] = bool(correct_val) if pd.notna(correct_val) else None
                    model_predictions[model] = _shorten(row.get("prediction", ""), 300)

                frame_list = [frames[model] for model in active_models]
                case_matrix.append(
                    {
                        "dataset": dataset,
                        "case_id": case_id,
                        "question": _first_non_empty(frame_list, case_id, "question", 500),
                        "answer": _first_non_empty(frame_list, case_id, "answer", 240),
                        "group_tags": _group_tags(frame_list, case_id, resolved_groups),
                        "model_metrics": model_metrics,
                        "model_correct": model_correct,
                        "model_predictions": model_predictions,
                    }
                )

    top_group_gains = sorted(group_deltas, key=lambda row: (row["delta"], row["sample_count"]), reverse=True)[:50]
    top_group_drops = sorted(group_deltas, key=lambda row: (row["delta"], -row["sample_count"]))[:50]

    return {
        "work_dir": osp.abspath(work_dir),
        "models": model_names,
        "baseline": baseline,
        "datasets": sorted({row["dataset"] for row in pairwise_overlap}),
        "score_files": score_files,
        "model_dataset_summary": model_dataset_summary,
        "dataset_deltas": dataset_deltas,
        "pairwise_overlap": pairwise_overlap,
        "group_deltas": group_deltas,
        "top_group_gains": top_group_gains,
        "top_group_drops": top_group_drops,
        "case_matrix": case_matrix,
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dump_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _dump_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _dump_csv(rows: list[dict[str, Any]], path: str) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_pairwise_heatmap(analysis: dict[str, Any], out_path: str) -> None:
    rows = analysis.get("pairwise_overlap", [])
    models = analysis.get("models", [])
    if len(models) < 2 or not rows:
        return
    matrix = pd.DataFrame(np.nan, index=models, columns=models)
    for model in models:
        matrix.loc[model, model] = 0.0
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        rate = row.get("disagreement_rate")
        if rate is None:
            continue
        key = (row["model_a"], row["model_b"])
        grouped.setdefault(key, []).append(float(rate) * 100)
    for (model_a, model_b), vals in grouped.items():
        value = sum(vals) / len(vals)
        matrix.loc[model_a, model_b] = value
        matrix.loc[model_b, model_a] = value

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.8), max(5, len(models) * 0.7)))
    im = ax.imshow(matrix.values.astype(float), cmap="YlOrRd", vmin=0, vmax=np.nanmax(matrix.values) or 1)
    ax.set_xticks(range(len(models)), labels=models, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), labels=models, fontsize=8)
    ax.set_title("Pairwise disagreement rate (%)")
    for i in range(len(models)):
        for j in range(len(models)):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_dataset_delta(analysis: dict[str, Any], out_path: str) -> None:
    rows = analysis.get("dataset_deltas", [])
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.sort_values("delta")
    labels = [f"{row.dataset}\n{row.candidate_model}" for row in df.itertuples()]
    values = df["delta"].tolist()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, min(18, len(df) * 0.6)), max(4, min(12, len(df) * 0.25))))
    colors = ["#55A868" if val >= 0 else "#C44E52" for val in values]
    ax.barh(range(len(values)), values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    ax.set_xlabel("Accuracy delta vs baseline")
    ax.set_title("Dataset deltas")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_group_delta(analysis: dict[str, Any], out_path: str) -> None:
    rows = analysis.get("group_deltas", [])
    if not rows:
        return
    picked = sorted(rows, key=lambda row: (abs(row["delta"]), row["sample_count"]), reverse=True)[:20]
    picked = sorted(picked, key=lambda row: row["delta"])
    labels = [
        f"{row['dataset']} | {row['candidate_model']}\n{row['label']} (n={row['sample_count']})"
        for row in picked
    ]
    values = [row["delta"] for row in picked]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, max(5, len(picked) * 0.35)))
    colors = ["#55A868" if val >= 0 else "#C44E52" for val in values]
    ax.barh(range(len(values)), values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    ax.set_xlabel("Accuracy delta vs baseline")
    ax.set_title("Top subgroup deltas")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_markdown_report(analysis: dict[str, Any], out_path: str) -> None:
    lines = [
        "# Benchmark Overlap Report",
        "",
        f"- Work dir: `{analysis.get('work_dir', '')}`",
        f"- Baseline: `{analysis.get('baseline', '')}`",
        f"- Models: {', '.join(f'`{m}`' for m in analysis.get('models', []))}",
        f"- Datasets: {', '.join(f'`{d}`' for d in analysis.get('datasets', []))}",
        "",
        "## Pairwise Overlap",
        "",
        "| Dataset | Model A | Model B | Shared | Both correct | A only | B only | Both wrong | Disagree | Delta B-A |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis.get("pairwise_overlap", [])[:80]:
        disagree = row.get("disagreement_rate")
        lines.append(
            "| {dataset} | {model_a} | {model_b} | {shared_cases} | {both_correct} | "
            "{model_a_only} | {model_b_only} | {both_wrong} | {disagree} | {delta:.2f} |".format(
                disagree="" if disagree is None else f"{disagree * 100:.1f}%",
                delta=row.get("delta_b_minus_a", 0.0),
                **row,
            )
        )

    lines += [
        "",
        "## Top Subgroup Gains",
        "",
        "| Dataset | Candidate | Group | N | Base | Cand | Delta | Fixes | Drops |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis.get("top_group_gains", [])[:30]:
        lines.append(
            f"| {row['dataset']} | {row['candidate_model']} | {row['label']} | "
            f"{row['sample_count']} | {row['baseline_score']:.2f} | "
            f"{row['candidate_score']:.2f} | {row['delta']:.2f} | {row['fixes']} | {row['drops']} |"
        )

    lines += [
        "",
        "## Top Subgroup Drops",
        "",
        "| Dataset | Candidate | Group | N | Base | Cand | Delta | Fixes | Drops |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis.get("top_group_drops", [])[:30]:
        lines.append(
            f"| {row['dataset']} | {row['candidate_model']} | {row['label']} | "
            f"{row['sample_count']} | {row['baseline_score']:.2f} | "
            f"{row['candidate_score']:.2f} | {row['delta']:.2f} | {row['fixes']} | {row['drops']} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_ai_prompt(analysis: dict[str, Any], out_path: str) -> None:
    prompt = {
        "task": "Analyze what capabilities improved or regressed after training.",
        "reading_order": [
            "dataset_deltas.csv",
            "group_deltas.csv",
            "pairwise_overlap.csv",
            "case_matrix.jsonl",
        ],
        "focus": [
            "subgroups with large positive delta and enough samples",
            "subgroups with regressions",
            "whether gains come from fixing baseline-wrong cases or from unstable tradeoffs",
            "cases where models disagree despite similar aggregate scores",
        ],
        "baseline": analysis.get("baseline"),
        "models": analysis.get("models", []),
        "datasets": analysis.get("datasets", []),
    }
    _dump_json(prompt, out_path)


def write_overlap_bundle(analysis: dict[str, Any], out_dir: str) -> dict[str, str]:
    """Write the full overlap bundle and return artifact paths."""

    _ensure_dir(out_dir)
    artifacts = {
        "summary": osp.join(out_dir, "summary.json"),
        "pairwise_overlap": osp.join(out_dir, "pairwise_overlap.csv"),
        "model_dataset_summary": osp.join(out_dir, "model_dataset_summary.csv"),
        "dataset_deltas": osp.join(out_dir, "dataset_deltas.csv"),
        "group_deltas": osp.join(out_dir, "group_deltas.csv"),
        "case_matrix": osp.join(out_dir, "case_matrix.jsonl"),
        "report": osp.join(out_dir, "report.md"),
        "ai_prompt": osp.join(out_dir, "ai_prompt.json"),
    }
    _dump_json(analysis, artifacts["summary"])
    _dump_csv(analysis.get("pairwise_overlap", []), artifacts["pairwise_overlap"])
    _dump_csv(analysis.get("model_dataset_summary", []), artifacts["model_dataset_summary"])
    _dump_csv(analysis.get("dataset_deltas", []), artifacts["dataset_deltas"])
    _dump_csv(analysis.get("group_deltas", []), artifacts["group_deltas"])
    _dump_jsonl(analysis.get("case_matrix", []), artifacts["case_matrix"])
    _write_markdown_report(analysis, artifacts["report"])
    _write_ai_prompt(analysis, artifacts["ai_prompt"])

    chart_paths = {
        "pairwise_heatmap": osp.join(out_dir, "01_pairwise_disagreement_heatmap.png"),
        "dataset_delta": osp.join(out_dir, "02_dataset_delta.png"),
        "group_delta": osp.join(out_dir, "03_group_delta.png"),
    }
    _plot_pairwise_heatmap(analysis, chart_paths["pairwise_heatmap"])
    _plot_dataset_delta(analysis, chart_paths["dataset_delta"])
    _plot_group_delta(analysis, chart_paths["group_delta"])
    artifacts.update({key: path for key, path in chart_paths.items() if osp.exists(path)})
    return artifacts
