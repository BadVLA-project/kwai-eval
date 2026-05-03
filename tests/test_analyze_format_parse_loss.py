import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_format_parse_loss import analyze_model_format_loss


MODEL = "ToyModel"


def _model_dir(root):
    path = root / MODEL
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_mcq_score(root):
    path = _model_dir(root) / f"{MODEL}_ToyMCQ_score.csv"
    pd.DataFrame(
        [
            {
                "index": 0,
                "question": "Pick one.\nA. red\nB. blue\nC. green\nD. yellow",
                "answer": "B",
                "prediction": "B",
                "score": 1,
            },
            {
                "index": 1,
                "question": "Pick one.\nA. red\nB. blue\nC. green\nD. yellow",
                "answer": "C",
                "prediction": "My reasoning is long. Final answer: C.",
                "score": -1,
            },
            {
                "index": 2,
                "question": "Pick one.\nA. red\nB. blue\nC. green\nD. yellow",
                "answer": "A",
                "prediction": "Final answer: D",
                "score": -1,
            },
            {
                "index": 3,
                "question": "Pick one.\nA. red\nB. blue\nC. green\nD. yellow",
                "answer": "D",
                "prediction": "",
                "score": -1,
            },
        ]
    ).to_csv(path, index=False)


def _write_continuous_score(root):
    path = root / f"{MODEL}_DreamBench_score.xlsx"
    pd.DataFrame(
        [
            {"index": 0, "prediction": "good", "f1": 0.5},
            {"index": 1, "prediction": "judge returned invalid json", "f1": -1},
        ]
    ).to_excel(path, index=False)


def test_analyze_model_format_loss_counts_recoverable_mcq_and_upper_bound(tmp_path):
    _write_mcq_score(tmp_path)
    _write_continuous_score(tmp_path)

    analysis = analyze_model_format_loss(tmp_path, MODEL)
    by_dataset = {row["dataset"]: row for row in analysis["summary"]}

    mcq = by_dataset["ToyMCQ"]
    assert mcq["total_rows"] == 4
    assert mcq["parse_fail_rows"] == 2
    assert mcq["inference_fail_rows"] == 1
    assert mcq["recovered_correct_rows"] == 1
    assert mcq["recovered_wrong_rows"] == 1
    assert mcq["raw_score_points"] == 25.0
    assert mcq["parse_fail_upper_bound_points"] == 50.0
    assert mcq["recoverable_parse_loss_points"] == 25.0

    dream = by_dataset["DreamBench"]
    assert dream["metric_key"] == "f1"
    assert dream["total_rows"] == 2
    assert dream["parse_fail_rows"] == 1
    assert dream["raw_score_points"] == 25.0
    assert dream["parse_fail_upper_bound_points"] == 50.0
    assert dream["recoverable_parse_loss_points"] == 0.0


def test_analyze_format_parse_loss_cli_writes_summary_and_cases(tmp_path):
    _write_mcq_score(tmp_path)
    out_dir = tmp_path / "format_loss"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "analyze_format_parse_loss.py"),
            "--work-dir",
            str(tmp_path),
            "--model",
            MODEL,
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    summary_path = out_dir / "format_parse_loss_summary.csv"
    cases_path = out_dir / "format_parse_loss_cases.jsonl"
    assert summary_path.is_file()
    assert cases_path.is_file()

    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "dataset"] == "ToyMCQ"
    assert summary.loc[0, "recoverable_parse_loss_points"] == 25.0

    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines()]
    assert {case["case_id"] for case in cases} == {"1", "2"}
    assert cases[0]["recovered_answer"] == "C"
