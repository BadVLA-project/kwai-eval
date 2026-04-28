import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_dream_utils():
    module_path = ROOT / 'vlmeval' / 'dataset' / 'dream_utils.py'
    spec = importlib.util.spec_from_file_location('dream_utils_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_dream_local_dir_prefers_env(tmp_path):
    utils = load_dream_utils()
    local_dir = tmp_path / 'DREAM-1K'
    local_dir.mkdir()

    assert utils.resolve_dream_local_dir(env={'DREAM1K_DIR': str(local_dir)}) == str(local_dir)


def test_resolve_dream_video_source_accepts_hf_download_zip_layout(tmp_path):
    utils = load_dream_utils()
    local_dir = tmp_path / 'DREAM-1K'
    zip_dir = local_dir / 'video'
    zip_dir.mkdir(parents=True)
    zip_path = zip_dir / 'video.zip'
    zip_path.write_bytes(b'not-a-real-zip')

    root, found_zip = utils.resolve_dream_video_source(str(local_dir), '/tmp/default')

    assert root == str(zip_dir)
    assert found_zip == str(zip_path)


def test_resolve_dream_video_source_prefers_extracted_mp4s(tmp_path):
    utils = load_dream_utils()
    local_dir = tmp_path / 'DREAM-1K'
    video_dir = local_dir / 'video'
    video_dir.mkdir(parents=True)
    (video_dir / '1.mp4').write_bytes(b'')

    root, found_zip = utils.resolve_dream_video_source(str(local_dir), '/tmp/default')

    assert root == str(video_dir)
    assert found_zip is None


def test_resolve_dream_tsv_accepts_explicit_env(tmp_path):
    utils = load_dream_utils()
    tsv = tmp_path / 'DREAM-1K.tsv'
    tsv.write_text('index\tvideo\tquestion\tanswer\n', encoding='utf-8')

    assert utils.resolve_dream_annotation_file('/tmp/lmu', env={'DREAM1K_TSV': str(tsv)}) == str(tsv)


def test_resolve_dream_annotation_prefers_local_bench_jsonl(tmp_path):
    utils = load_dream_utils()
    local_dir = tmp_path / 'DREAM-1K'
    local_dir.mkdir()
    jsonl = local_dir / 'dream1k_bench.jsonl'
    jsonl.write_text('{}\n', encoding='utf-8')

    assert utils.resolve_dream_annotation_file('/tmp/lmu', local_dir=str(local_dir)) == str(jsonl)


def test_convert_dream_jsonl_to_tsv_normalizes_generic_rows(tmp_path):
    utils = load_dream_utils()
    jsonl = tmp_path / 'dream1k_bench.jsonl'
    jsonl.write_text(
        '{"idx": 7, "video": "video/7.mp4", "prompt": "Describe.", '
        '"response": "A person walks.", "events": ["A person walks."]}\n',
        encoding='utf-8',
    )
    tsv = tmp_path / 'dream1k_bench.vlmeval.tsv'

    utils.convert_dream_jsonl_to_tsv(str(jsonl), str(tsv))

    text = tsv.read_text(encoding='utf-8')
    assert 'index\tvideo\tquestion\tanswer\tevents' in text
    assert '7\tvideo/7.mp4\tDescribe.\tA person walks.' in text


def test_resolve_dream_converted_tsv_uses_cache_dir(tmp_path):
    utils = load_dream_utils()
    cache_dir = tmp_path / 'Benchmarks'

    path = utils.resolve_dream_converted_tsv(env={'DREAM1K_CACHE_DIR': str(cache_dir)})

    assert path == str(cache_dir / 'DREAM-1K.from_jsonl.tsv')
