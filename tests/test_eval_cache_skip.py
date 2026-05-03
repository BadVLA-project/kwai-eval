import json
import os
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
vlmeval_pkg = types.ModuleType('vlmeval')
vlmeval_pkg.__path__ = [str(ROOT / 'vlmeval')]
smp_pkg = types.ModuleType('vlmeval.smp')
smp_pkg.__path__ = [str(ROOT / 'vlmeval' / 'smp')]
misc_mod = types.ModuleType('vlmeval.smp.misc')
misc_mod.toliststr = lambda x: x if isinstance(x, list) else [x]
misc_mod.timestr = lambda _: '260503000000'
vlm_mod = types.ModuleType('vlmeval.smp.vlm')
vlm_mod.decode_base64_to_image_file = lambda *args, **kwargs: None
validators_mod = types.ModuleType('validators')
validators_mod.url = lambda _: False
for name, module in {
    'vlmeval': vlmeval_pkg,
    'vlmeval.smp': smp_pkg,
    'vlmeval.smp.misc': misc_mod,
    'vlmeval.smp.vlm': vlm_mod,
    'validators': validators_mod,
}.items():
    sys.modules[name] = module

import vlmeval.smp.file as file_mod
from vlmeval.smp.file import find_cached_eval_result, prepare_reuse_files


def test_find_cached_eval_result_prefers_run_summary_score_json(tmp_path):
    pred = tmp_path / 'Model_ETBench_adaptive.xlsx'
    pred.write_text('prediction placeholder', encoding='utf-8')
    score = tmp_path / 'Model_ETBench_adaptive_score.json'
    score.write_text(json.dumps({'AVG': 42.5}), encoding='utf-8')

    cached, path = find_cached_eval_result(str(pred), dataset_name='ETBench_adaptive')

    assert path == str(score)
    assert cached == {'AVG': 42.5}


def test_find_cached_eval_result_accepts_etbench_acc_csv(tmp_path):
    pred = tmp_path / 'Model_ETBench_adaptive.xlsx'
    pred.write_text('prediction placeholder', encoding='utf-8')
    acc = tmp_path / 'Model_ETBench_adaptive_etbench_acc.csv'
    pd.DataFrame([{'metric': 'AVG', 'value': 42.5}]).to_csv(acc, index=False)

    cached, path = find_cached_eval_result(str(pred), dataset_name='ETBench_adaptive')

    assert path == str(acc)
    assert list(cached['metric']) == ['AVG']
    assert list(cached['value']) == [42.5]


def test_find_cached_eval_result_ignores_stale_score_files(tmp_path):
    pred = tmp_path / 'Model_ETBench_adaptive.xlsx'
    pred.write_text('prediction placeholder', encoding='utf-8')
    score = tmp_path / 'Model_ETBench_adaptive_score.json'
    score.write_text(json.dumps({'AVG': 42.5}), encoding='utf-8')
    os.utime(score, (10, 10))
    os.utime(pred, (20, 20))

    cached, path = find_cached_eval_result(str(pred), dataset_name='ETBench_adaptive')

    assert cached is None
    assert path is None


def test_run_py_checks_eval_cache_before_evaluating():
    script = (ROOT / 'run.py').read_text(encoding='utf-8')

    cache_check = script.index('find_cached_eval_result(')
    evaluate_call = script.index('dataset.evaluate(result_file')

    assert cache_check < evaluate_call


def test_run_py_can_skip_fresh_eval_cache_before_dataset_build():
    script = (ROOT / 'run.py').read_text(encoding='utf-8')

    prebuild_cache_check = script.index('prebuild_cached_eval_path')
    dataset_build = script.index('if use_config:', script.index('for _, dataset_name in enumerate(args.data):'))

    assert prebuild_cache_check < dataset_build


def test_run_py_prepares_reuse_before_prebuild_cache_check():
    script = (ROOT / 'run.py').read_text(encoding='utf-8')
    dataset_loop = script.index('for _, dataset_name in enumerate(args.data):')

    early_reuse = script.index('prepare_reuse_files(', dataset_loop)
    prebuild_cache_check = script.index('prebuild_cached_eval_path', dataset_loop)
    dataset_build = script.index('if use_config:', dataset_loop)

    assert early_reuse < prebuild_cache_check < dataset_build


def test_prepare_reuse_files_falls_back_to_model_root_summary_files(tmp_path, monkeypatch):
    model_dir = tmp_path / 'Qwen3-VL-4B-Instruct'
    current_dir = model_dir / 'T20260503'
    current_dir.mkdir(parents=True)

    pred = model_dir / 'Qwen3-VL-4B-Instruct_VideoMMMU_adaptive.xlsx'
    pred.write_text('prediction placeholder', encoding='utf-8')
    score = model_dir / 'Qwen3-VL-4B-Instruct_VideoMMMU_adaptive_score.json'
    score.write_text(json.dumps({'Overall': {'acc': 100.0}}), encoding='utf-8')
    monkeypatch.setattr(file_mod, 'fetch_aux_files', lambda _: [str(score)])

    prepare_reuse_files(
        pred_root_meta=str(model_dir),
        eval_id='T20260503',
        model_name='Qwen3-VL-4B-Instruct',
        dataset_name='VideoMMMU_adaptive',
        reuse=True,
        reuse_aux=True,
    )

    copied_pred = current_dir / pred.name
    copied_score = current_dir / score.name
    assert copied_pred.read_text(encoding='utf-8') == 'prediction placeholder'
    assert json.loads(copied_score.read_text(encoding='utf-8')) == {'Overall': {'acc': 100.0}}
