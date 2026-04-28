import importlib.util
import ast
import csv
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


def test_convert_dream_jsonl_to_tsv_accepts_official_metadata_rows(tmp_path):
    utils = load_dream_utils()
    jsonl = tmp_path / 'dream1k_bench.jsonl'
    jsonl.write_text(
        '{"idx": 0, "video_file": "video/0.mp4", '
        '"description": "A person picks up a cup.", '
        '"events": ["A person picks up a cup."]}\n',
        encoding='utf-8',
    )
    tsv = tmp_path / 'dream1k_bench.vlmeval.tsv'

    utils.convert_dream_jsonl_to_tsv(str(jsonl), str(tsv))

    text = tsv.read_text(encoding='utf-8')
    assert '0\tvideo/0.mp4\tDescribe the video in detail.\tA person picks up a cup.' in text
    with tsv.open(encoding='utf-8') as f:
        row = next(csv.DictReader(f, delimiter='\t'))
    assert row['events'] == '["A person picks up a cup."]'


def test_convert_dream_jsonl_to_tsv_accepts_gt_frame_dir_rows(tmp_path):
    utils = load_dream_utils()
    jsonl = tmp_path / 'dream1k_bench.jsonl'
    jsonl.write_text(
        '{"video_frame_dir": "/pfs/Datasets/DREAM-1K/frames/2.mp4", '
        '"GT_description": "One character raises their hand.", '
        '"GT_events": ["One character raises their hand."], '
        '"index": 1}\n',
        encoding='utf-8',
    )
    tsv = tmp_path / 'dream1k_bench.vlmeval.tsv'

    utils.convert_dream_jsonl_to_tsv(str(jsonl), str(tsv))

    with tsv.open(encoding='utf-8') as f:
        row = next(csv.DictReader(f, delimiter='\t'))
    assert row['index'] == '1'
    assert row['video'] == 'video/2.mp4'
    assert row['answer'] == 'One character raises their hand.'
    assert row['events'] == '["One character raises their hand."]'


def test_parse_dream_events_response_rejects_unparseable_text():
    utils = load_dream_utils()

    try:
        utils.parse_dream_events_response('not json', source='predicted events')
    except ValueError as err:
        assert 'predicted events' in str(err)
    else:
        raise AssertionError('Expected invalid event extraction response to fail')


def test_resolve_dream_converted_tsv_uses_cache_dir(tmp_path):
    utils = load_dream_utils()
    cache_dir = tmp_path / 'Benchmarks'

    path = utils.resolve_dream_converted_tsv(env={'DREAM1K_CACHE_DIR': str(cache_dir)})

    assert path == str(cache_dir / 'DREAM-1K.from_jsonl.tsv')


def test_dream_local_prompt_style_matches_event_caption_granularity():
    utils = load_dream_utils()

    prompt = utils.resolve_dream_prompt('Describe the video in detail.', prompt_style='local')

    assert prompt == utils.DREAM_LOCAL_PROMPT
    assert prompt == (
        'Describe the video as a chronological list of visible events. '
        'Focus only on actions, object interactions, and character movements. '
        'Do not describe mood, atmosphere, intentions, or background unless needed to explain an event.'
    )
    assert 'mood, atmosphere, intentions' in prompt
    assert utils.resolve_dream_prompt(
        'Describe the video in detail.',
        prompt_style='official',
    ) == 'Describe the video in detail.'


def _dict_entries_from_assignment(path, assignment_name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == assignment_name for target in node.targets):
                return dict((ast.literal_eval(key), value) for key, value in zip(node.value.keys, node.value.values))
    raise AssertionError(f'{assignment_name} assignment not found')


def _keyword_value(call_node, keyword_name):
    for keyword in call_node.keywords:
        if keyword.arg == keyword_name:
            return ast.literal_eval(keyword.value)
    raise AssertionError(f'{keyword_name} keyword not found')


def test_dream_local_adaptive_dataset_is_registered_with_local_prompt():
    config_path = ROOT / 'vlmeval' / 'dataset' / 'video_dataset_config.py'
    entries = _dict_entries_from_assignment(config_path, 'adaptive_dataset')

    local_call = entries['DREAM_local_adaptive']

    assert isinstance(local_call, ast.Call)
    assert _keyword_value(local_call, 'dataset') == 'DREAM-1K'
    assert _keyword_value(local_call, 'adaptive') is True
    assert _keyword_value(local_call, 'prompt_style') == 'local'
    assert _keyword_value(local_call, 'dataset_name_alias') == 'DREAM_local_adaptive'


def test_dream1k_direct_script_defaults_to_local_prompt_dataset():
    script = (ROOT / 'scripts' / 'run_dream1k_direct.sh').read_text(encoding='utf-8')

    assert 'export DATA="${DATA:-DREAM_local_adaptive}"' in script
