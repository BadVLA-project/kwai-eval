import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_mopd_step_curves import build_method_comparison_data, build_mopd_curve_data


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


def _write_unprefixed_score_json(root, model, bench, score):
    path = _model_dir(root, model) / f"{bench}_score.json"
    path.write_text(json.dumps({"score": score}), encoding="utf-8")


def _write_flat_score_json(root, model, bench, score):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{model}_{bench}_score.json"
    path.write_text(json.dumps({"score": score}), encoding="utf-8")


def _write_etbench_acc(root, model, avg):
    path = _model_dir(root, model) / f"{model}_ETBench_adaptive_etbench_acc.csv"
    pd.DataFrame([{"metric": "AVG", "value": avg}]).to_csv(path, index=False)


def _write_mvbench_rating(root, model, score):
    path = _model_dir(root, model) / f"{model}_MVBench_MP4_adaptive_rating.json"
    path.write_text(json.dumps({"overall": [10, 20, f"{score:.2f}%"]}), encoding="utf-8")


def _write_vinoground_score(root, model, text_score, video_score, group_score):
    path = _model_dir(root, model) / f"{model}_Vinoground_adaptive_score.json"
    path.write_text(
        json.dumps(
            {
                "text_score": text_score,
                "video_score": video_score,
                "group_score": group_score,
            }
        ),
        encoding="utf-8",
    )


def test_build_mopd_curve_data_orders_steps_and_auto_fills_missing_benchmarks(tmp_path):
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
    assert data.benchmarks == ["MVBench_MP4_adaptive", "Video-MME_adaptive", "BaseOnly_adaptive"]
    assert data.scores["MVBench_MP4_adaptive"] == [50.0, 55.0, 57.5, 58.0]
    assert data.scores["Video-MME_adaptive"][:3] == [60.0, 62.0, 64.0]
    assert pd.isna(data.scores["Video-MME_adaptive"][3])
    assert data.scores["BaseOnly_adaptive"][0] == 42.0
    assert all(pd.isna(value) for value in data.scores["BaseOnly_adaptive"][1:])


def test_build_mopd_curve_data_auto_adds_benchmarks_with_missing_steps(tmp_path):
    step50 = f"{BASE}-MOPD-Step50"
    step100 = f"{BASE}-MOPD-Step100"

    for model, score in [(BASE, 0.50), (step50, 0.55), (step100, 0.57)]:
        _write_score_json(tmp_path, model, "AoTBench_QA_adaptive", score)

    _write_score_json(tmp_path, step50, "NewBench_adaptive", 0.61)
    _write_score_json(tmp_path, step100, "NewBench_adaptive", 0.63)

    data = build_mopd_curve_data(tmp_path, BASE)

    assert data.benchmarks == ["AoTBench_QA_adaptive", "NewBench_adaptive"]
    assert data.scores["AoTBench_QA_adaptive"] == [50.0, 55.0, 57.0]
    assert pd.isna(data.scores["NewBench_adaptive"][0])
    assert data.scores["NewBench_adaptive"][1:] == [61.0, 63.0]
    assert data.dropped_benchmarks["NewBench_adaptive"] == [BASE]


def test_build_mopd_curve_data_merges_same_layout_work_dirs(tmp_path):
    shard0 = tmp_path / "eval_direct_final"
    shard1 = tmp_path / "eval_direct_final_1"
    step50 = f"{BASE}-MOPD-Step50"

    _write_score_json(shard0, BASE, "AoTBench_QA_adaptive", 0.50)
    _write_score_json(shard0, step50, "AoTBench_QA_adaptive", 0.55)
    _write_score_json(shard1, BASE, "NewBench_adaptive", 0.61)
    _write_score_json(shard1, step50, "NewBench_adaptive", 0.66)

    data = build_mopd_curve_data([shard0, shard1], BASE)

    assert data.steps == [0, 50]
    assert data.benchmarks == ["AoTBench_QA_adaptive", "NewBench_adaptive"]
    assert data.scores["AoTBench_QA_adaptive"] == [50.0, 55.0]
    assert data.scores["NewBench_adaptive"] == [61.0, 66.0]


def test_build_mopd_curve_data_accepts_unprefixed_vlmevalkit_result_files(tmp_path):
    step50 = f"{BASE}-MOPD-Step50"

    _write_unprefixed_score_json(tmp_path, BASE, "AoTBench_QA_adaptive", 0.50)
    _write_unprefixed_score_json(tmp_path, step50, "AoTBench_QA_adaptive", 0.55)

    data = build_mopd_curve_data(tmp_path, BASE)

    assert data.benchmarks == ["AoTBench_QA_adaptive"]
    assert data.scores["AoTBench_QA_adaptive"] == [50.0, 55.0]


def test_build_mopd_curve_data_accepts_flat_model_result_work_dirs(tmp_path):
    base_dir = tmp_path / "eval_direct_final"
    step_dir = tmp_path / "eval_direct_final_1"
    step50 = f"{BASE}-MOPD-Step50"

    _write_flat_score_json(base_dir, BASE, "AoTBench_QA_adaptive", 0.50)
    _write_flat_score_json(step_dir, step50, "AoTBench_QA_adaptive", 0.55)

    data = build_mopd_curve_data([base_dir, step_dir], BASE)

    assert data.steps == [0, 50]
    assert data.models == [BASE, step50]
    assert data.benchmarks == ["AoTBench_QA_adaptive"]
    assert data.scores["AoTBench_QA_adaptive"] == [50.0, 55.0]


def test_build_mopd_curve_data_uses_etbench_avg_mvbench_overall_and_vinoground_text(tmp_path):
    step50 = f"{BASE}-MOPD-Step50"
    for model, etbench, mvbench, text, video, group in [
        (BASE, 31.5, 52.0, 11.4, 22.0, 33.0),
        (step50, 34.0, 55.5, 12.9, 25.0, 36.0),
    ]:
        _write_etbench_acc(tmp_path, model, etbench)
        _write_mvbench_rating(tmp_path, model, mvbench)
        _write_vinoground_score(tmp_path, model, text, video, group)

    data = build_mopd_curve_data(tmp_path, BASE)

    assert data.benchmarks == [
        "ETBench_adaptive",
        "MVBench_MP4_adaptive",
        "Vinoground_adaptive",
    ]
    assert data.scores["ETBench_adaptive"] == [31.5, 34.0]
    assert data.scores["MVBench_MP4_adaptive"] == [52.0, 55.5]
    assert data.scores["Vinoground_adaptive"] == [11.4, 12.9]


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
    assert (out_dir / "mopd_step_small_multiples.png").is_file()
    assert not (out_dir / "mopd_step_curves.png").exists()
    assert not (out_dir / "mopd_step_gain_curves.png").exists()


def test_build_method_comparison_data_aligns_opd_and_ema_grpo_steps(tmp_path):
    for model, etbench, vino in [
        (BASE, 30.0, 10.0),
        (f"{BASE}-MOPD-Step50", 32.0, 12.0),
        (f"{BASE}-MOPD-Step100", 34.0, 14.0),
        (f"{BASE}-MOPD-Step150", 36.0, 16.0),
        (f"{BASE}-EMA-GRPO-Step50", 31.0, 11.0),
        (f"{BASE}-EMA-GRPO-Step100", 33.0, 13.0),
    ]:
        _write_etbench_acc(tmp_path, model, etbench)
        _write_vinoground_score(tmp_path, model, vino, video_score=20.0, group_score=30.0)

    data = build_method_comparison_data(tmp_path, BASE)

    assert data.families == ["OPD", "EMA-GRPO"]
    assert data.steps == [0, 50, 100]
    assert data.benchmarks == ["ETBench_adaptive", "Vinoground_adaptive"]
    assert data.scores["OPD"]["ETBench_adaptive"] == [30.0, 32.0, 34.0]
    assert data.scores["EMA-GRPO"]["ETBench_adaptive"] == [30.0, 31.0, 33.0]
    assert data.scores["OPD"]["Vinoground_adaptive"] == [10.0, 12.0, 14.0]
    assert data.scores["EMA-GRPO"]["Vinoground_adaptive"] == [10.0, 11.0, 13.0]


def test_build_method_comparison_data_auto_adds_benchmarks_with_missing_family_scores(tmp_path):
    for model, score in [
        (BASE, 0.50),
        (f"{BASE}-MOPD-Step50", 0.55),
        (f"{BASE}-EMA-GRPO-Step50", 0.53),
    ]:
        _write_score_json(tmp_path, model, "AoTBench_QA_adaptive", score)

    _write_score_json(tmp_path, f"{BASE}-MOPD-Step50", "MopdOnly_adaptive", 0.61)
    _write_score_json(tmp_path, f"{BASE}-EMA-GRPO-Step50", "EmaOnly_adaptive", 0.62)

    data = build_method_comparison_data(tmp_path, BASE)

    assert data.benchmarks == ["AoTBench_QA_adaptive", "EmaOnly_adaptive", "MopdOnly_adaptive"]
    assert data.scores["OPD"]["AoTBench_QA_adaptive"] == [50.0, 55.0]
    assert data.scores["EMA-GRPO"]["AoTBench_QA_adaptive"] == [50.0, 53.0]
    assert pd.isna(data.scores["OPD"]["EmaOnly_adaptive"][0])
    assert pd.isna(data.scores["OPD"]["EmaOnly_adaptive"][1])
    assert pd.isna(data.scores["EMA-GRPO"]["EmaOnly_adaptive"][0])
    assert data.scores["EMA-GRPO"]["EmaOnly_adaptive"][1] == 62.0
    assert data.scores["OPD"]["MopdOnly_adaptive"][1] == 61.0
    assert pd.isna(data.scores["EMA-GRPO"]["MopdOnly_adaptive"][1])


def test_build_method_comparison_data_merges_same_layout_work_dirs(tmp_path):
    shard0 = tmp_path / "eval_direct_final"
    shard1 = tmp_path / "eval_direct_final_1"

    for root, model, bench, score in [
        (shard0, BASE, "AoTBench_QA_adaptive", 0.50),
        (shard0, f"{BASE}-MOPD-Step50", "AoTBench_QA_adaptive", 0.55),
        (shard0, f"{BASE}-EMA-GRPO-Step50", "AoTBench_QA_adaptive", 0.53),
        (shard1, BASE, "NewBench_adaptive", 0.60),
        (shard1, f"{BASE}-MOPD-Step50", "NewBench_adaptive", 0.64),
        (shard1, f"{BASE}-EMA-GRPO-Step50", "NewBench_adaptive", 0.62),
    ]:
        _write_score_json(root, model, bench, score)

    data = build_method_comparison_data([shard0, shard1], BASE)

    assert data.steps == [0, 50]
    assert data.benchmarks == ["AoTBench_QA_adaptive", "NewBench_adaptive"]
    assert data.scores["OPD"]["NewBench_adaptive"] == [60.0, 64.0]
    assert data.scores["EMA-GRPO"]["NewBench_adaptive"] == [60.0, 62.0]


def test_build_method_comparison_data_accepts_flat_model_result_work_dirs(tmp_path):
    base_dir = tmp_path / "eval_direct_final"
    opd_dir = tmp_path / "eval_direct_final_1"
    ema_dir = tmp_path / "eval_direct_final_2"

    _write_flat_score_json(base_dir, BASE, "AoTBench_QA_adaptive", 0.50)
    _write_flat_score_json(opd_dir, f"{BASE}-MOPD-Step50", "AoTBench_QA_adaptive", 0.55)
    _write_flat_score_json(ema_dir, f"{BASE}-EMA-GRPO-Step50", "AoTBench_QA_adaptive", 0.53)

    data = build_method_comparison_data([base_dir, opd_dir, ema_dir], BASE)

    assert data.steps == [0, 50]
    assert data.scores["OPD"]["AoTBench_QA_adaptive"] == [50.0, 55.0]
    assert data.scores["EMA-GRPO"]["AoTBench_QA_adaptive"] == [50.0, 53.0]


def test_build_method_comparison_data_skips_unscored_shared_steps(tmp_path):
    for model, score in [
        (BASE, 0.50),
        (f"{BASE}-MOPD-Step50", 0.55),
        (f"{BASE}-MOPD-Step100", 0.57),
        (f"{BASE}-EMA-GRPO-Step50", 0.53),
    ]:
        _write_score_json(tmp_path, model, "AoTBench_QA_adaptive", score)

    (tmp_path / f"{BASE}-EMA-GRPO-Step100").mkdir()

    data = build_method_comparison_data(tmp_path, BASE)

    assert data.steps == [0, 50]
    assert data.scores["OPD"]["AoTBench_QA_adaptive"] == [50.0, 55.0]
    assert data.scores["EMA-GRPO"]["AoTBench_QA_adaptive"] == [50.0, 53.0]


def test_plot_mopd_step_curves_cli_writes_method_comparison_when_ema_exists(tmp_path):
    for model, score in [
        (BASE, 0.50),
        (f"{BASE}-MOPD-Step50", 0.55),
        (f"{BASE}-EMA-GRPO-Step50", 0.53),
    ]:
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
    assert (out_dir / "method_comparison_scores.csv").is_file()
    assert (out_dir / "method_comparison_deltas.csv").is_file()
    assert (out_dir / "method_comparison_benchmark_gains.png").is_file()
    assert not (out_dir / "method_comparison_mean_gain.png").exists()


def test_plot_mopd_step_curves_cli_accepts_multiple_work_dirs(tmp_path):
    shard0 = tmp_path / "eval_direct_final"
    shard1 = tmp_path / "eval_direct_final_1"
    step50 = f"{BASE}-MOPD-Step50"

    _write_score_json(shard0, BASE, "AoTBench_QA_adaptive", 0.50)
    _write_score_json(shard0, step50, "AoTBench_QA_adaptive", 0.55)
    _write_score_json(shard1, BASE, "NewBench_adaptive", 0.60)
    _write_score_json(shard1, step50, "NewBench_adaptive", 0.65)

    out_dir = tmp_path / "plots"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "plot_mopd_step_curves.py"),
            "--work-dir",
            str(shard0),
            str(shard1),
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
    scores = pd.read_csv(out_dir / "mopd_step_scores.csv")
    assert set(scores["benchmark"]) == {"AoTBench_QA_adaptive", "NewBench_adaptive"}
    assert "Work dirs:" in result.stdout
