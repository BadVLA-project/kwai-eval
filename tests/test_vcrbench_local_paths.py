import importlib.util
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
