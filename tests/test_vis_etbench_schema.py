import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vis.data_loader import ResultLoader


def test_etbench_threshold_columns_are_visible(tmp_path):
    model = 'Qwen3-VL-4B-Instruct'
    model_dir = tmp_path / model
    model_dir.mkdir()
    score_path = model_dir / f'{model}_ETBench_adaptive_score.json'
    score_path.write_text(json.dumps({
        'REF/Acc': 51.8,
        'GND/F1': 34.59,
        'GND/F1@0.1': 55.1,
        'GND/F1@0.3': 42.2,
        'GND/F1@0.5': 28.3,
        'GND/F1@0.7': 12.4,
        'TVG/F1@0.1': 60.0,
        'EPM/F1@0.1': 50.0,
        'TAL/F1@0.1': 55.3,
        'CAP/F1': 30.99,
        'CAP/F1@0.1': 44.4,
        'CAP/F1@0.3': 33.3,
        'CAP/F1@0.5': 22.2,
        'CAP/F1@0.7': 11.1,
        'DVC/F1@0.1': 40.0,
        'SLC/F1@0.1': 48.8,
        'CAP/SentSim': 19.66,
        'COM/mRec': 14.66,
        'AVG': 33.01,
    }), encoding='utf-8')

    loader = ResultLoader(str(tmp_path))

    etbench_group = next(group for group in loader.table_groups if group['id'] == 'ETBench_adaptive')
    labels = [col['label'] for col in etbench_group['columns']]
    assert labels == [
        'REF',
        'GND', 'GND@.1', 'GND@.3', 'GND@.5', 'GND@.7',
        'CAP(F1)', 'CAP@.1', 'CAP@.3', 'CAP@.5', 'CAP@.7',
        'CAP(Sim)', 'COM', 'AVG',
    ]

    column_by_label = {col['label']: col['id'] for col in etbench_group['columns']}
    assert loader.load_column_score(model, column_by_label['GND@.1']) == 55.1
    assert loader.load_column_score(model, column_by_label['CAP@.1']) == 44.4

    gnd_breakdown = loader.load_column_breakdown(model, column_by_label['GND@.1'])
    assert gnd_breakdown == {
        'TVG/F1@0.1': 60.0,
        'EPM/F1@0.1': 50.0,
        'TAL/F1@0.1': 55.3,
    }


def test_loader_skips_broken_score_symlinks(tmp_path):
    model = 'Qwen3-VL-4B-Instruct'
    model_dir = tmp_path / model
    model_dir.mkdir()
    valid_score = model_dir / f'{model}_ETBench_adaptive_score.json'
    valid_score.write_text(json.dumps({
        'REF/Acc': 50.0,
        'GND/F1': 40.0,
        'CAP/F1': 30.0,
        'CAP/SentSim': 20.0,
        'COM/mRec': 10.0,
        'AVG': 30.0,
    }), encoding='utf-8')
    broken_score = model_dir / f'{model}_TimeLensBench_Charades_adaptive_score.json'
    broken_score.symlink_to(model_dir / 'deleted_score.json')

    loader = ResultLoader(str(tmp_path))

    assert 'ETBench_adaptive' in loader.benchmarks
    assert 'TimeLensBench_Charades_adaptive' not in loader.benchmarks
    assert loader.load_breakdown(model, 'TimeLensBench_Charades_adaptive') is None
