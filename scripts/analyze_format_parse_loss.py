#!/usr/bin/env python3
"""Analyze benchmark score loss caused by answer-format parsing failures.

The script works on row-level VLMEvalKit score files.  It separates missing/API
prediction failures from format parsing failures, estimates the maximum possible
drop from parse failures, and tries a conservative MCQ re-parse to identify
cases that were likely correct but scored as invalid because of answer format.

Example:
    python scripts/analyze_format_parse_loss.py \
        --work-dir /path/to/eval_outputs \
        --model Qwen3-VL-4B-Instruct \
        --out-dir ./format_loss_report
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import os.path as osp
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd

try:
    from vlmeval.utils.matching_util import can_infer, extract_answer_from_cot, parse_options_from_question
except Exception:  # pragma: no cover - fallback for standalone use outside VLMEvalKit
    can_infer = None
    parse_options_from_question = None

    def extract_answer_from_cot(text: str, valid_options: str = "ABCD") -> str:
        if not isinstance(text, str):
            return ""
        option_pat = f"[{re.escape(valid_options)}{re.escape(valid_options.lower())}]"
        patterns = [
            rf"<answer>\s*({option_pat})\b",
            rf"final\s+answer\s*[:：]?\s*({option_pat})\b",
            rf"answer\s*[:：]?\s*({option_pat})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        stripped = text.strip()
        if len(stripped) <= 5:
            match = re.search(option_pat, stripped)
            if match:
                return match.group(0).upper()
        return ""


OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".json", ".jsonl"}
SCORE_SUFFIXES = ("_score", "_acc")

PREDICTION_FAIL_PATTERNS = (
    "Failed to obtain answer",
    "Failed to obtain response",
    "Failed to get answer",
)

METRIC_PRIORITY = (
    "score",
    "hit",
    "correct",
    "f1",
    "F1",
    "iou",
    "IoU",
    "accuracy",
    "acc",
)

PARSE_TEXT_COLUMNS = (
    "extracted_answer",
    "extract_answer",
    "parsed_answer",
    "parsed_prediction",
    "pred_answer",
)

PARSE_ID_COLUMNS = (
    "pred_id",
    "prediction_id",
    "parsed_id",
)

CASE_TEXT_COLUMNS = (
    "question",
    "answer",
    "prediction",
    "extracted_answer",
    "category",
    "task_type",
    "sub_task",
    "subtask",
    "domain",
    "video",
    "video_path",
    "image",
    "image_path",
)


@dataclass
class MetricSpec:
    key: str
    kind: str
    scale: float
    max_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-benchmark score loss from format parsing errors."
    )
    parser.add_argument("--work-dir", required=True, help="Evaluation result root directory")
    parser.add_argument("--model", required=True, help="Model directory/name prefix")
    parser.add_argument("--out-dir", required=True, help="Directory for summary and case exports")
    parser.add_argument("--data", nargs="*", default=None, help="Optional benchmark name filters")
    parser.add_argument(
        "--score-files",
        nargs="*",
        default=None,
        help="Optional explicit row-level score files. Overrides automatic discovery.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search the whole work-dir recursively if direct/model-dir discovery finds nothing.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=500,
        help="Maximum parse-failure cases written to JSONL; use 0 for all cases.",
    )
    return parser.parse_args()


def _is_nan(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _clean_text(value: Any) -> str:
    if value is None or _is_nan(value):
        return ""
    return " ".join(str(value).strip().split())


def _shorten(value: Any, limit: int = 240) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_file(path: str | Path) -> Any:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported file type: {path}")


def _to_dataframe(obj: Any) -> pd.DataFrame | None:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, list):
        if not obj:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in obj):
            return pd.DataFrame(obj)
        return None
    if isinstance(obj, dict):
        if "columns" in obj and "data" in obj:
            return pd.DataFrame(obj["data"], columns=obj["columns"])
        values = list(obj.values())
        if values and all(isinstance(value, list) for value in values):
            lengths = {len(value) for value in values}
            if len(lengths) == 1:
                return pd.DataFrame(obj)
    return None


def load_row_level_score(path: str | Path) -> pd.DataFrame | None:
    try:
        data = load_file(path)
    except Exception:
        return None
    df = _to_dataframe(data)
    if df is None or df.empty:
        return None
    if "prediction" not in df.columns:
        return None
    id_col = next((col for col in ("index", "question_id", "id") if col in df.columns), None)
    if id_col is None:
        return None
    df = df.copy()
    df["_case_id"] = df[id_col].astype(str)
    return df


def _nice_metric_max(max_valid: float) -> float:
    if not math.isfinite(max_valid) or max_valid <= 1:
        return 1.0
    for upper in (5.0, 10.0, 100.0):
        if max_valid <= upper:
            return upper
    return float(max_valid)


def infer_metric_spec(df: pd.DataFrame) -> MetricSpec | None:
    for key in METRIC_PRIORITY:
        if key not in df.columns:
            continue
        values = pd.to_numeric(df[key], errors="coerce")
        if values.notna().sum() == 0:
            continue
        valid = values[values >= 0].dropna()
        unique_valid = set(valid.unique().tolist())
        if key in {"hit", "correct"} or unique_valid.issubset({0, 1, 0.0, 1.0}):
            return MetricSpec(key=key, kind="binary", scale=100.0, max_value=1.0)
        if len(valid) == 0 or float(valid.max()) <= 1.0:
            return MetricSpec(key=key, kind="continuous", scale=100.0, max_value=1.0)
        return MetricSpec(
            key=key,
            kind="continuous",
            scale=1.0,
            max_value=_nice_metric_max(float(valid.max())),
        )
    return None


def metric_values(df: pd.DataFrame, metric: MetricSpec) -> pd.Series:
    return pd.to_numeric(df[metric.key], errors="coerce")


def clipped_metric_values(values: pd.Series, metric: MetricSpec) -> pd.Series:
    return values.fillna(0).clip(lower=0, upper=metric.max_value)


def is_missing_prediction(value: Any) -> bool:
    if value is None or _is_nan(value):
        return True
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return True
    return any(pattern in text for pattern in PREDICTION_FAIL_PATTERNS)


def parse_text_failure_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in PARSE_TEXT_COLUMNS:
        if col not in df.columns:
            continue
        values = df[col]
        empty = values.isna() | values.astype(str).str.strip().str.lower().isin(
            {"", "nan", "none", "null"}
        )
        failure_text = values.astype(str).str.contains(
            r"failed to parse|parse failed|invalid format|format error",
            case=False,
            na=False,
            regex=True,
        )
        mask |= empty | failure_text
    return mask


def parse_id_failure_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in PARSE_ID_COLUMNS:
        if col not in df.columns:
            continue
        ids = pd.to_numeric(df[col], errors="coerce")
        mask |= ids.isna() | (ids < 0)
    return mask


def parse_failure_mask(df: pd.DataFrame, metric: MetricSpec) -> tuple[pd.Series, pd.Series]:
    values = metric_values(df, metric)
    inference_fail = df["prediction"].map(is_missing_prediction)
    metric_fail = values.isna() | (values < 0)
    parse_signal = metric_fail | parse_text_failure_mask(df) | parse_id_failure_mask(df)
    return parse_signal & ~inference_fail, inference_fail


def benchmark_from_path(model: str, path: str | Path) -> str | None:
    stem = Path(path).stem
    prefix = f"{model}_"
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix):]
    for suffix in SCORE_SUFFIXES:
        if tail.endswith(suffix):
            return tail[: -len(suffix)]
    return None


def discover_score_files(work_dir: str | Path, model: str, recursive: bool = False) -> list[Path]:
    root = Path(work_dir)
    candidates: set[Path] = set()
    patterns = [f"{model}_*{suffix}.*" for suffix in SCORE_SUFFIXES]

    for pattern in patterns:
        candidates.update(root.glob(pattern))

    model_dir = root / model
    if model_dir.is_dir():
        for pattern in patterns:
            candidates.update(model_dir.rglob(pattern))

    if recursive and not candidates:
        for pattern in patterns:
            candidates.update(root.rglob(pattern))

    return sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _normalize_for_match(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"^\(?[a-z]\)?[.)：:\s-]*", "", text)
    return re.sub(r"\s+", " ", text).strip(" .,:;!?()[]{}")


def _parse_candidates(value: Any) -> list[str]:
    if value is None or _is_nan(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
        if isinstance(parsed, dict):
            return [str(parsed[key]) for key in sorted(parsed)]
    return []


def options_from_row(row: pd.Series) -> dict[str, str]:
    options: dict[str, str] = {}
    for letter in OPTION_LETTERS[:10]:
        if letter in row and not _is_nan(row[letter]) and _clean_text(row[letter]):
            options[letter] = _clean_text(row[letter])

    if not options:
        candidates = _parse_candidates(row.get("candidates", None))
        options = {OPTION_LETTERS[idx]: text for idx, text in enumerate(candidates)}

    if not options and parse_options_from_question is not None:
        try:
            parsed = parse_options_from_question(str(row.get("question", "")))
        except Exception:
            parsed = {}
        options = {str(key).upper(): str(value) for key, value in parsed.items()}

    return {
        letter: text
        for letter, text in options.items()
        if letter in OPTION_LETTERS and _clean_text(text)
    }


def _letter_from_text(value: Any, valid_options: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    match = re.match(r"^\(?([A-Z])\)?(?:[.)：:\s-]|$)", text, flags=re.IGNORECASE)
    if match and match.group(1).upper() in valid_options:
        return match.group(1).upper()
    if len(text) <= 5:
        match = re.search(rf"[{re.escape(valid_options)}]", text, flags=re.IGNORECASE)
        if match:
            return match.group(0).upper()
    return ""


def gold_answer_letter(row: pd.Series, options: dict[str, str]) -> str:
    valid_options = "".join(options) or OPTION_LETTERS[:10]

    for id_col in ("answer_id", "label_id", "gt_id"):
        if id_col in row and not _is_nan(row[id_col]):
            try:
                idx = int(row[id_col])
            except Exception:
                continue
            if 0 <= idx < len(OPTION_LETTERS):
                return OPTION_LETTERS[idx]

    for answer_col in ("answer", "gt", "label", "gold", "target"):
        if answer_col not in row:
            continue
        value = row[answer_col]
        letter = _letter_from_text(value, valid_options)
        if letter:
            return letter
        normalized = _normalize_for_match(value)
        for option_letter, option_text in options.items():
            if normalized and normalized == _normalize_for_match(option_text):
                return option_letter
    return ""


def recover_mcq_answer(row: pd.Series, options: dict[str, str]) -> str:
    prediction = _clean_text(row.get("prediction", ""))
    if not prediction:
        return ""
    valid_options = "".join(options) or OPTION_LETTERS[:10]

    recovered = extract_answer_from_cot(prediction, valid_options=valid_options)
    if recovered in valid_options:
        return recovered

    if options and can_infer is not None:
        try:
            inferred = can_infer(prediction, dict(options))
        except Exception:
            inferred = ""
        if isinstance(inferred, str) and inferred in valid_options:
            return inferred

    return _letter_from_text(prediction, valid_options)


def parse_failure_reasons(row: pd.Series, metric: MetricSpec) -> list[str]:
    reasons = []
    value = pd.to_numeric(pd.Series([row.get(metric.key)]), errors="coerce").iloc[0]
    if pd.isna(value):
        reasons.append(f"{metric.key}=nan")
    elif value < 0:
        reasons.append(f"{metric.key}<0")

    for col in PARSE_ID_COLUMNS:
        if col in row:
            parsed = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.isna(parsed):
                reasons.append(f"{col}=nan")
            elif parsed < 0:
                reasons.append(f"{col}<0")

    for col in PARSE_TEXT_COLUMNS:
        if col in row:
            text = _clean_text(row.get(col))
            if not text:
                reasons.append(f"{col}=empty")
            elif re.search(r"failed to parse|parse failed|invalid format|format error", text, flags=re.I):
                reasons.append(f"{col}=parse_error")

    return list(dict.fromkeys(reasons)) or ["parse_signal"]


def _case_record(
    dataset: str,
    path: Path,
    row: pd.Series,
    metric: MetricSpec,
    metric_value: float,
) -> dict[str, Any]:
    options = options_from_row(row)
    gold = gold_answer_letter(row, options)
    recovered = recover_mcq_answer(row, options)
    recovery_status = "not_mcq_or_unparsed"
    if recovered and gold:
        recovery_status = "recovered_correct" if recovered == gold else "recovered_wrong"
    elif recovered:
        recovery_status = "recovered_no_gold"

    record: dict[str, Any] = {
        "dataset": dataset,
        "score_file": str(path),
        "case_id": str(row.get("_case_id", "")),
        "metric_key": metric.key,
        "metric_value": None if pd.isna(metric_value) else float(metric_value),
        "parse_failure_reasons": parse_failure_reasons(row, metric),
        "gold_answer": gold,
        "recovered_answer": recovered,
        "recovery_status": recovery_status,
        "prediction": _shorten(row.get("prediction", ""), 500),
    }
    for col in CASE_TEXT_COLUMNS:
        if col in row and col not in record:
            record[col] = _shorten(row.get(col, ""), 300)
    return record


def analyze_score_file(path: Path, model: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    dataset = benchmark_from_path(model, path)
    if dataset is None:
        return None, []
    df = load_row_level_score(path)
    if df is None:
        return None, []
    metric = infer_metric_spec(df)
    if metric is None:
        return None, []

    values = metric_values(df, metric)
    clipped = clipped_metric_values(values, metric)
    parse_fail, inference_fail = parse_failure_mask(df, metric)

    total = len(df)
    parse_fail_rows = int(parse_fail.sum())
    inference_fail_rows = int(inference_fail.sum())

    recovered_correct_rows = 0
    recovered_wrong_rows = 0
    recovered_no_gold_rows = 0
    cases = []
    recoverable_gain = 0.0

    for idx, row in df[parse_fail].iterrows():
        metric_value = values.loc[idx]
        record = _case_record(dataset, path, row, metric, metric_value)
        cases.append(record)

        current_value = float(clipped.loc[idx]) if not pd.isna(clipped.loc[idx]) else 0.0
        if record["recovery_status"] == "recovered_correct":
            recovered_correct_rows += 1
            recoverable_gain += max(metric.max_value - current_value, 0.0)
        elif record["recovery_status"] == "recovered_wrong":
            recovered_wrong_rows += 1
        elif record["recovery_status"] == "recovered_no_gold":
            recovered_no_gold_rows += 1

    raw_score = float(clipped.sum() / total * metric.scale) if total else float("nan")
    upper_gain = parse_fail_rows * metric.max_value / total * metric.scale if total else float("nan")
    recoverable_gain_points = recoverable_gain / total * metric.scale if total else float("nan")

    summary = {
        "dataset": dataset,
        "score_file": str(path),
        "metric_key": metric.key,
        "metric_kind": metric.kind,
        "metric_scale": metric.scale,
        "max_item_score": metric.max_value,
        "total_rows": total,
        "valid_metric_rows": int(((values >= 0) & values.notna()).sum()),
        "parse_fail_rows": parse_fail_rows,
        "inference_fail_rows": inference_fail_rows,
        "parse_fail_rate": round(parse_fail_rows / total, 6) if total else 0.0,
        "inference_fail_rate": round(inference_fail_rows / total, 6) if total else 0.0,
        "recovered_correct_rows": recovered_correct_rows,
        "recovered_wrong_rows": recovered_wrong_rows,
        "recovered_no_gold_rows": recovered_no_gold_rows,
        "raw_score_points": round(raw_score, 6),
        "parse_fail_upper_bound_points": round(float(upper_gain), 6),
        "score_if_parse_fail_oracle_points": round(raw_score + float(upper_gain), 6),
        "recoverable_parse_loss_points": round(float(recoverable_gain_points), 6),
        "score_if_recovered_points": round(raw_score + float(recoverable_gain_points), 6),
    }
    return summary, cases


def _passes_data_filter(dataset: str, filters: list[str] | None) -> bool:
    if not filters:
        return True
    return any(dataset == item or item in dataset for item in filters)


def analyze_model_format_loss(
    work_dir: str | Path,
    model: str,
    data: list[str] | None = None,
    score_files: list[str | Path] | None = None,
    recursive: bool = False,
) -> dict[str, Any]:
    paths = [Path(path) for path in score_files] if score_files else discover_score_files(work_dir, model, recursive)
    summaries: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for path in sorted(paths):
        dataset = benchmark_from_path(model, path)
        if dataset is None:
            skipped.append({"score_file": str(path), "reason": "filename does not match model_score suffix"})
            continue
        if not _passes_data_filter(dataset, data):
            continue
        summary, file_cases = analyze_score_file(path, model)
        if summary is None:
            skipped.append({"score_file": str(path), "reason": "not a row-level parseable score file"})
            continue
        summaries.append(summary)
        cases.extend(file_cases)

    summaries = sorted(
        summaries,
        key=lambda row: (
            -float(row["parse_fail_upper_bound_points"]),
            -int(row["parse_fail_rows"]),
            str(row["dataset"]),
        ),
    )

    totals = {
        "model": model,
        "score_files": len(paths),
        "benchmarks": len(summaries),
        "total_rows": int(sum(row["total_rows"] for row in summaries)),
        "parse_fail_rows": int(sum(row["parse_fail_rows"] for row in summaries)),
        "inference_fail_rows": int(sum(row["inference_fail_rows"] for row in summaries)),
        "recovered_correct_rows": int(sum(row["recovered_correct_rows"] for row in summaries)),
        "recovered_wrong_rows": int(sum(row["recovered_wrong_rows"] for row in summaries)),
    }
    totals["parse_fail_rate"] = (
        round(totals["parse_fail_rows"] / totals["total_rows"], 6)
        if totals["total_rows"]
        else 0.0
    )

    return {
        "model": model,
        "work_dir": str(work_dir),
        "summary": summaries,
        "cases": cases,
        "totals": totals,
        "skipped": skipped,
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_analysis_outputs(analysis: dict[str, Any], out_dir: str | Path, max_cases: int = 500) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "format_parse_loss_summary.csv"
    cases_path = out / "format_parse_loss_cases.jsonl"
    json_path = out / "format_parse_loss_summary.json"
    skipped_path = out / "format_parse_loss_skipped.json"

    pd.DataFrame(analysis["summary"]).to_csv(summary_path, index=False)

    cases = analysis["cases"]
    if max_cases > 0:
        cases = cases[:max_cases]
    write_jsonl(cases, cases_path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": analysis["model"],
                "work_dir": analysis["work_dir"],
                "totals": analysis["totals"],
                "summary": analysis["summary"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with skipped_path.open("w", encoding="utf-8") as f:
        json.dump(analysis["skipped"], f, ensure_ascii=False, indent=2)

    return {
        "summary_csv": str(summary_path),
        "cases_jsonl": str(cases_path),
        "summary_json": str(json_path),
        "skipped_json": str(skipped_path),
    }


def print_console_summary(analysis: dict[str, Any], artifacts: dict[str, str]) -> None:
    totals = analysis["totals"]
    print(f"Model: {analysis['model']}")
    print(f"Benchmarks analyzed: {totals['benchmarks']}")
    print(f"Rows: {totals['total_rows']}")
    print(
        "Format parse failures: "
        f"{totals['parse_fail_rows']} ({totals['parse_fail_rate'] * 100:.2f}%)"
    )
    print(f"Inference/API failures: {totals['inference_fail_rows']}")
    print(f"Recoverable MCQ correct cases: {totals['recovered_correct_rows']}")
    print()
    if analysis["summary"]:
        display_cols = [
            "dataset",
            "metric_key",
            "total_rows",
            "parse_fail_rows",
            "inference_fail_rows",
            "raw_score_points",
            "parse_fail_upper_bound_points",
            "recoverable_parse_loss_points",
        ]
        print(pd.DataFrame(analysis["summary"])[display_cols].to_string(index=False))
        print()
    print(f"Saved summary: {artifacts['summary_csv']}")
    print(f"Saved cases: {artifacts['cases_jsonl']}")


def main() -> None:
    args = parse_args()
    analysis = analyze_model_format_loss(
        work_dir=args.work_dir,
        model=args.model,
        data=args.data,
        score_files=args.score_files,
        recursive=args.recursive,
    )
    artifacts = write_analysis_outputs(analysis, args.out_dir, max_cases=args.max_cases)
    print_console_summary(analysis, artifacts)


if __name__ == "__main__":
    main()
