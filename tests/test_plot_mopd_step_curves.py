import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_mopd_step_curves import build_mopd_curve_data


BASE = "Qwen3-VL-4B-Instruct"


def _model_dir(root, model):
    path = root / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_acc_csv(root, model, bench, score):
    path = _model_dir(root, model) / f"{model}_{bench}_acc.csv"
    pd.DataFrame([{"category": "overall", "accuracy": score}]).to_csv(path, index=False)


def _write_rating_json(root, model, bench, score):
    path = _model_dir(root, model) / f"{model}_{bench}_rating.json"
    path.write_text(json.dumps({"overall": {"overall": score}}), encoding="utf-8")


def _write_score_json(root, model, bench, score):
    path = _model_dir(root, model) / f"{model}_{bench}_score.json"
    path.write_text(json.dumps({"score": score}), encoding="utf-8")


def test_build_mopd_curve_data_orders_steps_and_keeps_intersection(tmp_path):
    step50 = f"{BASE}-MOPD-Step50"
    step100 = f"{BASE}-MOPD-Step100"
    step150 = f"{BASE}-MOPD-Step150"

    for model, mv_score, mme_score in [
        (BASE, 50.0, 0.60),
        (step100, 57.5, 0.64),
        (step50, 55.0, 0.62),
    ]:
        _write_acc_csv(tmp_path, model, "MVBench_MP4_adaptive", mv_score)
        _write_rating_json(tmp_path, model, "Video-MME_adaptive", mme_score)

    _write_acc_csv(tmp_path, step150, "MVBench_MP4_adaptive", 58.0)
    _write_score_json(tmp_path, BASE, "BaseOnly_adaptive", 42.0)

    data = build_mopd_curve_data(tmp_path, BASE)

    assert data.steps == [0, 50, 100, 150]
    assert data.models == [BASE, step50, step100, step150]
    assert data.benchmarks == ["MVBench_MP4_adaptive"]
    assert data.scores["MVBench_MP4_adaptive"] == [50.0, 55.0, 57.5, 58.0]


def test_plot_mopd_step_curves_cli_writes_plot_and_csv(tmp_path):
    step50 = f"{BASE}-MOPD-Step50"
    for model, score in [(BASE, 0.50), (step50, 0.55)]:
        _write_score_json(tmp_path, model, "AoTBench_QA_adaptive", score)

    out_dir = tmp_path / "plots"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "plot_mopd_step_curves.py"),
            "--work-dir",
            str(tmp_path),
            "--base-model",
            BASE,
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "mopd_step_scores.csv").is_file()
    assert (out_dir / "mopd_step_deltas.csv").is_file()
    assert (out_dir / "mopd_step_curves.png").is_file()
