import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_videomme_utils():
    module_path = ROOT / 'vlmeval' / 'dataset' / 'videomme_utils.py'
    spec = importlib.util.spec_from_file_location('videomme_utils_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_videomme_local_dir_prefers_env(tmp_path):
    utils = load_videomme_utils()
    local_dir = tmp_path / 'Video-MME'
    local_dir.mkdir()

    assert utils.resolve_videomme_local_dir(env={'VIDEO_MME_DIR': str(local_dir)}) == str(local_dir)


def test_find_videomme_video_dir_accepts_videos_layout(tmp_path):
    utils = load_videomme_utils()
    dataset = tmp_path / 'Video-MME'
    videos = dataset / 'videos'
    videos.mkdir(parents=True)
    (videos / 'sample.mp4').write_bytes(b'')

    assert utils.find_videomme_video_dir(str(dataset)) == str(videos)
    assert utils.resolve_videomme_video_path(str(dataset), 'sample') == str(videos / 'sample.mp4')


def test_videomme_video_relpath_uses_existing_videos_dir(tmp_path):
    utils = load_videomme_utils()
    dataset = tmp_path / 'Video-MME'
    videos = dataset / 'videos'
    videos.mkdir(parents=True)
    (videos / 'sample.mp4').write_bytes(b'')

    assert utils.videomme_video_relpath(str(dataset), 'sample') == './videos/sample.mp4'
