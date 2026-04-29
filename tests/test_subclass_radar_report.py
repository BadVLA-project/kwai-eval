import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vis.subclass_radar import build_subclass_radar_report


def _model_dir(root, model):
    path = root / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_mvbench(root, model, rows):
    path = _model_dir(root, model) / f"{model}_MVBench_MP4_adaptive_acc.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_videomme(root, model, task_scores):
    payload = {
        "overall": {
            "overall": 0.60,
            "task_type": task_scores,
        }
    }
    path = _model_dir(root, model) / f"{model}_Video-MME_adaptive_rating.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_aot(root, model, subset, score):
    path = _model_dir(root, model) / f"{model}_AoTBench_{subset}_adaptive_score.json"
    path.write_text(json.dumps({"score": score}), encoding="utf-8")


def _write_mlvu(root, model, scores):
    path = _model_dir(root, model) / f"{model}_MLVU_MCQ_score.json"
    path.write_text(json.dumps(scores), encoding="utf-8")


def _write_four_bench_fixture(root):
    for model, offset in [("base", 0), ("cand", 5)]:
        _write_mvbench(
            root,
            model,
            [
                {"category": "action_sequence", "accuracy": 60 + offset},
                {"category": "object_shuffle", "accuracy": 40 + offset},
            ],
        )
        _write_videomme(
            root,
            model,
            {
                "Action Recognition": 0.50 + offset / 100,
                "Temporal Reasoning": 0.40 + offset / 100,
            },
        )
        _write_aot(root, model, "Rtime_t2v", 0.55 + offset / 100)
        _write_aot(root, model, "Rtime_v2t", 0.45 + offset / 100)
        _write_mlvu(root, model, {"plotQA": [6 + offset, 10], "summary": [4 + offset, 10]})


def test_build_subclass_radar_report_writes_by_bench_outputs(tmp_path):
    _write_four_bench_fixture(tmp_path)
    out_dir = tmp_path / "subclass_report"

    report = build_subclass_radar_report(
        str(tmp_path),
        str(out_dir),
        benchmarks=["MVBench", "VideoMME", "AoTBench", "MLVU"],
    )

    assert set(report["benchmarks"]) == {"MVBench", "VideoMME", "AoTBench", "MLVU"}
    for bench in ["MVBench", "VideoMME", "AoTBench", "MLVU"]:
        bench_dir = out_dir / "by_bench" / bench
        assert (bench_dir / "subclass_scores.csv").is_file()
        assert (bench_dir / "subclass_scores.json").is_file()
        assert (bench_dir / "radar.png").is_file()
    assert (out_dir / "index.html").is_file()
    assert report["by_bench"]["VideoMME"]["dimension"] == "overall/task_type"


def test_build_subclass_radar_report_cli(tmp_path):
    _write_four_bench_fixture(tmp_path)
    out_dir = tmp_path / "subclass_report"

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "build_subclass_radar_report.py"),
            "--work-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--benchmarks",
            "MVBench",
            "VideoMME",
            "AoTBench",
            "MLVU",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "by_bench" / "MVBench" / "radar.png").is_file()
