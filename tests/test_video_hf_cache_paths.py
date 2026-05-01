import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    module_path = ROOT / 'vlmeval' / 'dataset' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'{name}_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_longvideobench_resolves_hf_cache_repo_root_snapshot(tmp_path):
    utils = load_module('longvideobench_utils')
    repo_root = tmp_path / 'datasets--longvideobench--LongVideoBench'
    snapshot = repo_root / 'snapshots' / 'abc123'
    (snapshot / 'videos').mkdir(parents=True)
    (snapshot / 'lvb_val.json').write_text('[]')

    assert utils.resolve_longvideobench_dir(
        env={'LONGVIDEOBENCH_DIR': str(repo_root)}
    ) == str(snapshot)


def test_videommmu_resolves_hf_cache_repo_root_snapshot(tmp_path):
    utils = load_module('videommmu_utils')
    repo_root = tmp_path / 'datasets--lmms-lab--VideoMMMU'
    snapshot = repo_root / 'snapshots' / 'abc123'
    category = snapshot / 'Validation'
    category.mkdir(parents=True)
    (category / 'test-00000-of-00001.parquet').write_bytes(b'')

    assert utils.resolve_videommmu_dir(
        env={'VIDEOMMMU_DIR': str(repo_root)}
    ) == str(snapshot)


def test_tempcompass_resolves_hf_cache_repo_root_snapshot(tmp_path):
    utils = load_module('tempcompass_utils')
    repo_root = tmp_path / 'datasets--lmms-lab--TempCompass'
    snapshot = repo_root / 'snapshots' / 'abc123'
    (snapshot / 'multi-choice').mkdir(parents=True)
    (snapshot / 'multi-choice' / 'test-00000-of-00001.parquet').write_bytes(b'')

    assert utils.resolve_tempcompass_dir(
        env={'TEMPCOMPASS_DIR': str(repo_root)}
    ) == str(snapshot)


def test_vcrbench_resolves_hf_cache_repo_root_snapshot(tmp_path):
    utils = load_module('vcrbench_utils')
    repo_root = tmp_path / 'datasets--VLM-Reasoning--VCR-Bench'
    snapshot = repo_root / 'snapshots' / 'abc123'
    (snapshot / 'v1' / 'videos' / 'video').mkdir(parents=True)
    (snapshot / 'VCR-Bench.tsv').write_text('index\tvideo\tvideo_path\tquestion\n', encoding='utf-8')

    assert utils.resolve_vcrbench_dir(
        env={'VCRBENCH_DIR': str(repo_root)}
    ) == str(snapshot)
