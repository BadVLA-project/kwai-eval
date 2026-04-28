import importlib.util
from pathlib import Path


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
