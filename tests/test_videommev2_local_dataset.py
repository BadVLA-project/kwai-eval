import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_videommev2_utils():
    module_path = ROOT / 'vlmeval' / 'dataset' / 'videomme_v2_utils.py'
    spec = importlib.util.spec_from_file_location('videomme_v2_utils_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_videommev2_paths_keep_artifacts_out_of_readonly_source(tmp_path):
    utils = load_videommev2_utils()
    source = tmp_path / 'Video-MME-v2-source'
    artifact = tmp_path / 'Video-MME-v2-artifacts'
    (source / 'videos').mkdir(parents=True)
    (source / 'test.parquet').write_bytes(b'parquet')
    (source / 'subtitle.zip').write_bytes(b'zip')

    paths = utils.resolve_videommev2_paths(
        env={
            'VIDEO_MME_V2_DIR': str(source),
            'VIDEO_MME_V2_ARTIFACT_DIR': str(artifact),
        }
    )

    assert paths.source_root == str(source)
    assert paths.artifact_root == str(artifact)
    assert paths.tsv_file == str(artifact / 'Video-MME-v2.tsv')
    assert paths.parquet_file == str(source / 'test.parquet')
    assert paths.video_dir == str(source / 'videos')
    assert paths.subtitle_zip == str(source / 'subtitle.zip')
    assert paths.subtitle_dir == str(artifact / 'subtitle')


def test_videommev2_video_path_prefers_videos_layout(tmp_path):
    utils = load_videommev2_utils()
    source = tmp_path / 'Video-MME-v2'
    videos = source / 'videos'
    videos.mkdir(parents=True)
    (videos / 'abc.mp4').write_bytes(b'')

    assert utils.resolve_videommev2_video_path(str(source), 'abc') == str(videos / 'abc.mp4')
    assert utils.videommev2_video_relpath(str(source), 'abc') == './videos/abc.mp4'
