import importlib.util
import os
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_vdc_utils():
    module_path = ROOT / 'vlmeval' / 'dataset' / 'vdc_utils.py'
    spec = importlib.util.spec_from_file_location('vdc_utils_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_hf_snapshot_path_accepts_cache_repo_root(tmp_path):
    utils = load_vdc_utils()
    repo_root = tmp_path / 'datasets--Enxin--VLMEval-VDC'
    snapshot = repo_root / 'snapshots' / 'abc123'
    snapshot.mkdir(parents=True)
    (snapshot / 'VDC.tsv').write_text('index\tvideo\n', encoding='utf-8')

    assert utils.resolve_hf_snapshot_path(str(repo_root)) == str(snapshot)


def test_resolve_vdc_local_path_uses_hf_home_root_cache(tmp_path):
    utils = load_vdc_utils()
    repo_root = tmp_path / 'datasets--Enxin--VLMEval-VDC'
    snapshot = repo_root / 'snapshots' / 'abc123'
    snapshot.mkdir(parents=True)
    (snapshot / 'VDC.tsv').write_text('index\tvideo\n', encoding='utf-8')

    assert utils.resolve_vdc_local_path(env={'HF_HOME': str(tmp_path)}) == str(snapshot)


def test_resolve_hf_snapshot_path_accepts_snapshot_without_tsv(tmp_path):
    utils = load_vdc_utils()
    repo_root = tmp_path / 'datasets--Enxin--VLMEval-VDC'
    old_snapshot = repo_root / 'snapshots' / 'old'
    new_snapshot = repo_root / 'snapshots' / 'new'
    old_snapshot.mkdir(parents=True)
    (new_snapshot / 'short_test').mkdir(parents=True)
    os.utime(old_snapshot, (1, 1))
    os.utime(new_snapshot, (2, 2))

    assert utils.resolve_hf_snapshot_path(str(repo_root)) == str(new_snapshot)


def test_ensure_vdc_data_file_builds_tsv_from_official_splits(tmp_path):
    utils = load_vdc_utils()
    split = tmp_path / 'short_test'
    split.mkdir()
    pd.DataFrame([
        {
            'video_name': 'sample-video',
            'caption': 'A person runs across a field.',
            'qa_list': [{'question': 'What is happening?', 'answer': 'running'}],
        }
    ]).to_parquet(split / 'data-00000-of-00001.parquet')

    data_file = utils.ensure_vdc_data_file(str(tmp_path))
    data = pd.read_csv(data_file, sep='\t')
    questions = json.loads(data.loc[0, 'question'])

    assert Path(data_file).name == 'VDC.tsv'
    assert data.loc[0, 'video'] == 'sample-video.mp4'
    assert data.loc[0, 'caption_type'] == 'short'
    assert data.loc[0, 'caption'] == 'A person runs across a field.'
    assert questions == [{'question': 'What is happening?', 'answer': 'running'}]


def test_find_vdc_video_root_prefers_videos_dir(tmp_path):
    utils = load_vdc_utils()
    dataset = tmp_path / 'snapshot'
    videos = dataset / 'videos'
    video = dataset / 'video'
    videos.mkdir(parents=True)
    video.mkdir()
    (videos / 'sample.mp4').write_bytes(b'')
    (video / 'sample.mp4').write_bytes(b'')

    assert utils.find_vdc_video_root(str(dataset)) == str(videos)
