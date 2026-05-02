import importlib.util
import sys
import types
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    module_path = ROOT / 'vlmeval' / 'dataset' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'{name}_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vcrbench_resolves_env_local_dir(tmp_path):
    utils = load_module('vcrbench_utils')
    local_root = tmp_path / 'VCR-Bench'
    video_dir = local_root / 'v1' / 'videos' / 'video'
    video_dir.mkdir(parents=True)
    (local_root / 'VCR-Bench.tsv').write_text('index\tvideo\tvideo_path\tquestion\n', encoding='utf-8')

    assert utils.resolve_vcrbench_dir(env={'VCRBENCH_DIR': str(local_root)}) == str(local_root)


def test_vcrbench_default_candidates_include_benchmarks_dir():
    utils = load_module('vcrbench_utils')

    assert '/m2v_intern/xuboshen/zgw/Benchmarks/VCR-Bench' in utils.DEFAULT_VCRBENCH_DIRS


def test_vcrbench_uses_video_zip_when_local_videos_are_not_extracted(tmp_path):
    utils = load_module('vcrbench_utils')
    local_root = tmp_path / 'VCR-Bench'
    videos_root = local_root / 'v1' / 'videos'
    videos_root.mkdir(parents=True)
    (local_root / 'VCR-Bench.tsv').write_text(
        'index\tvideo\tvideo_path\tquestion\n'
        '0\tsample\tvideo/sample.mp4\tWhat happened?\n',
        encoding='utf-8',
    )
    with zipfile.ZipFile(videos_root / 'video.zip', 'w') as zf:
        zf.writestr('video/sample.mp4', b'video')

    assert utils.ensure_vcrbench_videos_available(str(local_root))
    assert (videos_root / 'video' / 'sample.mp4').exists()
    assert utils.vcrbench_local_integrity(str(local_root))


def test_vcrbench_resolves_existing_v1_videos_layout(tmp_path):
    utils = load_module('vcrbench_utils')
    local_root = tmp_path / 'VCR-Bench'
    video_dir = local_root / 'v1' / 'videos'
    video_dir.mkdir(parents=True)
    (local_root / 'VCR-Bench.tsv').write_text(
        'index\tvideo\tvideo_path\tquestion\n'
        '0\tsample\tv1/videos/sample.mp4\tWhat happened?\n',
        encoding='utf-8',
    )
    (video_dir / 'sample.mp4').write_bytes(b'video')

    assert utils.resolve_vcrbench_video_path(str(local_root), 'v1/videos/sample.mp4') == str(video_dir / 'sample.mp4')
    assert utils.resolve_vcrbench_video_path(str(local_root), 'video/sample.mp4') == str(video_dir / 'sample.mp4')
    assert utils.vcrbench_local_integrity(str(local_root))


def test_vcrbench_build_prompt_uses_resolved_video_path(tmp_path, monkeypatch):
    vlmeval_pkg = types.ModuleType('vlmeval')
    vlmeval_pkg.__path__ = [str(ROOT / 'vlmeval')]
    dataset_pkg = types.ModuleType('vlmeval.dataset')
    dataset_pkg.__path__ = [str(ROOT / 'vlmeval' / 'dataset')]
    smp_stub = types.ModuleType('vlmeval.smp')
    smp_file_stub = types.ModuleType('vlmeval.smp.file')
    video_base_stub = types.ModuleType('vlmeval.dataset.video_base')
    utils_stub = types.ModuleType('vlmeval.dataset.utils')
    vutils = load_module('vcrbench_utils')
    track_stub = types.ModuleType('vlmeval.utils')
    hf_stub = types.ModuleType('huggingface_hub')

    class VideoBaseDataset:
        def make_video_struct(self, video_path, video_id=None):
            return {'type': 'video', 'value': video_path, 'video_id': video_id}

    video_base_stub.VideoBaseDataset = VideoBaseDataset
    utils_stub.build_judge = lambda *args, **kwargs: None
    utils_stub.DEBUG_MESSAGE = ''
    track_stub.track_progress_rich = lambda *args, **kwargs: None
    hf_stub.snapshot_download = lambda *args, **kwargs: None
    smp_file_stub.get_intermediate_file_path = lambda *args, **kwargs: ''
    smp_file_stub.get_file_extension = lambda path: str(path).rsplit('.', 1)[-1]

    for name, module in {
        'vlmeval': vlmeval_pkg,
        'vlmeval.dataset': dataset_pkg,
        'vlmeval.smp': smp_stub,
        'vlmeval.smp.file': smp_file_stub,
        'vlmeval.dataset.video_base': video_base_stub,
        'vlmeval.dataset.utils': utils_stub,
        'vlmeval.dataset.vcrbench_utils': vutils,
        'vlmeval.utils': track_stub,
        'huggingface_hub': hf_stub,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = ROOT / 'vlmeval' / 'dataset' / 'vcrbench.py'
    spec = importlib.util.spec_from_file_location('vlmeval.dataset.vcrbench', module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    local_root = tmp_path / 'VCR-Bench'
    video_dir = local_root / 'v1' / 'videos' / 'video'
    video_dir.mkdir(parents=True)
    (video_dir / 'sample.mp4').write_bytes(b'video')

    dataset = mod.VCRBench.__new__(mod.VCRBench)
    dataset.video_path = str(local_root)

    prompt = dataset.build_prompt(
        {'question': 'What happened?', 'video_path': 'video/sample.mp4', 'video': 'sample'},
        video_llm=True,
    )

    assert prompt[-1]['value'] == str(video_dir / 'sample.mp4')
