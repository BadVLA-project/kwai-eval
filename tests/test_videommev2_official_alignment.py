import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path):
    return (ROOT / relative_path).read_text(encoding='utf-8')


def _dict_assignment(name):
    tree = ast.parse(_source('vlmeval/dataset/video_dataset_config.py'))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise AssertionError(f'{name} assignment not found')


def _dict_keys(dict_node):
    return {key.value for key in dict_node.keys if isinstance(key, ast.Constant)}


def test_videommev2_official_dataset_variants_match_upstream():
    keys = _dict_keys(_dict_assignment('videommev2_dataset'))

    expected = {
        'Video-MME-v2_64frame',
        'Video-MME-v2_1fps',
        'Video-MME-v2_64frame_subs',
        'Video-MME-v2_1fps_subs',
        'Video-MME-v2_64frame_subs_interleave',
        'Video-MME-v2_1fps_subs_interleave',
        'Video-MME-v2_64frame_reasoning',
        'Video-MME-v2_64frame_reasoning_subs',
        'Video-MME-v2_64frame_reasoning_subs_interleave',
        'Video-MME-v2_64frame_resize',
        'Video-MME-v2_1fps_resize',
        'Video-MME-v2_64frame_resize_subs',
        'Video-MME-v2_64frame_resize_subs_interleave',
        'Video-MME-v2_64frame_resize_reasoning',
    }

    assert keys == expected


def test_videommev2_legacy_dataset_variable_alias_is_kept():
    source = _source('vlmeval/dataset/video_dataset_config.py')

    assert 'videomme_v2_dataset = videommev2_dataset' in source


def test_videommev2_constructor_accepts_official_keywords_and_legacy_aliases():
    source = _source('vlmeval/dataset/videomme_v2.py')

    for text in [
        'with_subtitle=False',
        'subtitle_interleave=False',
        'resize_target_area=False',
        'use_subtitle=None',
        'subtitle_mode=None',
        'adaptive=False',
    ]:
        assert text in source


def test_videommev2_official_module_name_is_available():
    shim = _source('vlmeval/dataset/videommev2.py')
    init_source = _source('vlmeval/dataset/__init__.py')

    assert 'from .videomme_v2 import VideoMMEv2' in shim
    assert 'from .videommev2 import VideoMMEv2' in init_source


def test_videommev2_video_llm_struct_matches_official_resize_and_sampling_shape():
    source = _source('vlmeval/dataset/videomme_v2.py')

    assert 'resized_height' in source
    assert 'resized_width' in source
    assert "if self.nframe > 0:" in source
    assert "video_msg['nframes'] = self.nframe" in source
    assert "if self.fps > 0:" in source
    assert "video_msg['fps'] = self.fps" in source
    assert 'self.make_video_struct(' in source
