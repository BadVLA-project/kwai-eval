import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.patch_etbench_thresholds import patch_prediction_file


def _write_jsonl(path, rows):
    path.write_text(
        '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding='utf-8',
    )


def test_patch_prediction_file_updates_only_fast_etbench_metrics(tmp_path):
    pred = tmp_path / 'Model_ETBench_adaptive.jsonl'
    rows = [
        {
            'task': 'tvg',
            'source': 'src',
            'video': 'v_tvg',
            'answer': json.dumps({'tgt': [[0, 10]]}),
            'prediction': 'The event happens from 0 - 10 seconds.',
        },
        {
            'task': 'epm',
            'source': 'src',
            'video': 'v_epm',
            'answer': json.dumps({'tgt': [[0, 10]]}),
            'prediction': 'The event happens from 0 - 2 seconds.',
        },
        {
            'task': 'dvc',
            'source': 'src',
            'video': 'v_dvc',
            'answer': json.dumps({'tgt': [[0, 10]], 'g': ['event a']}),
            'prediction': '0 - 10 seconds, event a.',
        },
        {
            'task': 'slc',
            'source': 'src',
            'video': 'v_slc',
            'answer': json.dumps({'tgt': [[0, 10]], 'g': ['event b']}),
            'prediction': '0 - 2 seconds, event b.',
        },
    ]
    _write_jsonl(pred, rows)

    score = tmp_path / 'Model_ETBench_adaptive_score.json'
    score.write_text(json.dumps({
        'REF/Acc': 88.8,
        'CAP/SentSim': 77.7,
        'AVG': 55.5,
    }), encoding='utf-8')

    updated, _ = patch_prediction_file(str(pred), backup=False)

    assert updated['REF/Acc'] == 88.8
    assert updated['CAP/SentSim'] == 77.7
    assert updated['GND/F1@0.1'] == 100.0
    assert updated['GND/F1@0.3'] == 50.0
    assert updated['GND/F1@0.5'] == 50.0
    assert updated['GND/F1@0.7'] == 50.0
    assert updated['CAP/F1@0.1'] == 100.0
    assert updated['CAP/F1@0.3'] == 50.0
    assert updated['CAP/F1@0.5'] == 50.0
    assert updated['CAP/F1@0.7'] == 50.0

    persisted = json.loads(score.read_text(encoding='utf-8'))
    assert persisted['GND/F1@0.1'] == 100.0
    assert persisted['CAP/F1@0.7'] == 50.0
    assert (tmp_path / 'Model_ETBench_adaptive_etbench_acc.csv').exists()
