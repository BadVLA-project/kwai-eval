import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vis.overlap_analysis import build_overlap_analysis
from vis.data_loader import ResultLoader
from vis.web import export_data


def _write_score(root, model, dataset, rows):
    model_dir = root / model
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{model}_{dataset}_score.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def test_build_overlap_analysis_counts_pairwise_correctness(tmp_path):
    rows_a = [
        {"index": 1, "question": "q1", "answer": "A", "prediction": "A", "correct": 1, "category": "motion"},
        {"index": 2, "question": "q2", "answer": "B", "prediction": "C", "correct": 0, "category": "motion"},
        {"index": 3, "question": "q3", "answer": "C", "prediction": "C", "correct": 1, "category": "object"},
        {"index": 4, "question": "q4", "answer": "D", "prediction": "A", "correct": 0, "category": "object"},
    ]
    rows_b = [
        {"index": 1, "question": "q1", "answer": "A", "prediction": "A", "correct": 1, "category": "motion"},
        {"index": 2, "question": "q2", "answer": "B", "prediction": "B", "correct": 1, "category": "motion"},
        {"index": 3, "question": "q3", "answer": "C", "prediction": "B", "correct": 0, "category": "object"},
        {"index": 4, "question": "q4", "answer": "D", "prediction": "A", "correct": 0, "category": "object"},
    ]
    rows_c = [
        {"index": 1, "question": "q1", "answer": "A", "prediction": "B", "correct": 0, "category": "motion"},
        {"index": 2, "question": "q2", "answer": "B", "prediction": "C", "correct": 0, "category": "motion"},
        {"index": 3, "question": "q3", "answer": "C", "prediction": "C", "correct": 1, "category": "object"},
        {"index": 4, "question": "q4", "answer": "D", "prediction": "D", "correct": 1, "category": "object"},
    ]
    _write_score(tmp_path, "base", "MiniBench", rows_a)
    _write_score(tmp_path, "cand", "MiniBench", rows_b)
    _write_score(tmp_path, "alt", "MiniBench", rows_c)

    analysis = build_overlap_analysis(str(tmp_path), models=["base", "cand", "alt"], baseline="base")

    pair = next(
        row
        for row in analysis["pairwise_overlap"]
        if row["dataset"] == "MiniBench" and row["model_a"] == "base" and row["model_b"] == "cand"
    )
    assert pair["shared_cases"] == 4
    assert pair["both_correct"] == 1
    assert pair["model_a_only"] == 1
    assert pair["model_b_only"] == 1
    assert pair["both_wrong"] == 1
    assert pair["jaccard_correct"] == 1 / 3
    assert pair["disagreement_rate"] == 0.5

    deltas = {
        (row["candidate_model"], row["group_column"], row["group_value"]): row
        for row in analysis["group_deltas"]
    }
    assert deltas[("cand", "category", "motion")]["delta"] == 50.0
    assert deltas[("cand", "category", "object")]["delta"] == -50.0


def test_build_overlap_analysis_exports_case_matrix(tmp_path):
    _write_score(
        tmp_path,
        "base",
        "MiniBench",
        [
            {"index": "a", "question": "qA", "answer": "yes", "prediction": "yes", "score": 1, "task": "order"},
            {"index": "b", "question": "qB", "answer": "no", "prediction": "yes", "score": 0, "task": "order"},
        ],
    )
    _write_score(
        tmp_path,
        "cand",
        "MiniBench",
        [
            {"index": "a", "question": "qA", "answer": "yes", "prediction": "yes", "score": 1, "task": "order"},
            {"index": "b", "question": "qB", "answer": "no", "prediction": "no", "score": 1, "task": "order"},
        ],
    )

    analysis = build_overlap_analysis(
        str(tmp_path),
        models=["base", "cand"],
        baseline="base",
        group_columns=["task"],
    )

    assert analysis["datasets"] == ["MiniBench"]
    assert analysis["case_matrix"] == [
        {
            "dataset": "MiniBench",
            "case_id": "a",
            "question": "qA",
            "answer": "yes",
            "group_tags": {"task": "order"},
            "model_metrics": {"base": 1.0, "cand": 1.0},
            "model_correct": {"base": True, "cand": True},
            "model_predictions": {"base": "yes", "cand": "yes"},
        },
        {
            "dataset": "MiniBench",
            "case_id": "b",
            "question": "qB",
            "answer": "no",
            "group_tags": {"task": "order"},
            "model_metrics": {"base": 0.0, "cand": 1.0},
            "model_correct": {"base": False, "cand": True},
            "model_predictions": {"base": "yes", "cand": "no"},
        },
    ]


def test_build_overlap_report_cli_writes_bundle(tmp_path):
    _write_score(
        tmp_path,
        "base",
        "MiniBench",
        [
            {"index": 1, "question": "q1", "answer": "A", "prediction": "A", "correct": 1, "category": "motion"},
            {"index": 2, "question": "q2", "answer": "B", "prediction": "C", "correct": 0, "category": "motion"},
        ],
    )
    _write_score(
        tmp_path,
        "cand",
        "MiniBench",
        [
            {"index": 1, "question": "q1", "answer": "A", "prediction": "A", "correct": 1, "category": "motion"},
            {"index": 2, "question": "q2", "answer": "B", "prediction": "B", "correct": 1, "category": "motion"},
        ],
    )
    out_dir = tmp_path / "overlap_report"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "build_overlap_report.py"),
            "--work-dir",
            str(tmp_path),
            "--models",
            "base",
            "cand",
            "--baseline",
            "base",
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "pairwise_overlap.csv").is_file()
    assert (out_dir / "case_matrix.jsonl").is_file()


def test_dashboard_export_skips_overlap_by_default(tmp_path):
    _write_score(
        tmp_path,
        "base",
        "MiniBench",
        [
            {"index": 1, "question": "q1", "answer": "A", "prediction": "A", "correct": 1, "category": "motion"},
            {"index": 2, "question": "q2", "answer": "B", "prediction": "C", "correct": 0, "category": "object"},
        ],
    )
    _write_score(
        tmp_path,
        "cand",
        "MiniBench",
        [
            {"index": 1, "question": "q1", "answer": "A", "prediction": "A", "correct": 1, "category": "motion"},
            {"index": 2, "question": "q2", "answer": "B", "prediction": "B", "correct": 1, "category": "object"},
        ],
    )

    payload = export_data(ResultLoader(str(tmp_path)))

    assert "overlap" not in payload


def test_dashboard_export_can_include_overlap_when_requested(tmp_path):
    _write_score(
        tmp_path,
        "base",
        "MiniBench",
        [
            {"index": 1, "question": "q1", "answer": "A", "prediction": "A", "correct": 1, "category": "motion"},
            {"index": 2, "question": "q2", "answer": "B", "prediction": "C", "correct": 0, "category": "object"},
        ],
    )
    _write_score(
        tmp_path,
        "cand",
        "MiniBench",
        [
            {"index": 1, "question": "q1", "answer": "A", "prediction": "A", "correct": 1, "category": "motion"},
            {"index": 2, "question": "q2", "answer": "B", "prediction": "B", "correct": 1, "category": "object"},
        ],
    )

    payload = export_data(ResultLoader(str(tmp_path)), include_overlap=True)

    assert payload["overlap"]["datasets"] == ["MiniBench"]
    assert payload["overlap"]["pairwise_overlap"][0]["model_a"] == "base"
    assert payload["overlap"]["pairwise_overlap"][0]["model_b"] == "cand"
