import importlib.util
import logging
import os
import os.path as osp
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_videommev2_utils():
    module_path = ROOT / 'vlmeval' / 'dataset' / 'videomme_v2_utils.py'
    spec = importlib.util.spec_from_file_location('videomme_v2_utils_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_videommev2_class(monkeypatch):
    class NoopLock:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    vlmeval_pkg = types.ModuleType('vlmeval')
    vlmeval_pkg.__path__ = [str(ROOT / 'vlmeval')]
    dataset_pkg = types.ModuleType('vlmeval.dataset')
    dataset_pkg.__path__ = [str(ROOT / 'vlmeval' / 'dataset')]

    smp_stub = types.ModuleType('vlmeval.smp')
    smp_stub.os = os
    smp_stub.osp = osp
    smp_stub.np = np
    smp_stub.pd = pd
    smp_stub.portalocker = types.SimpleNamespace(Lock=NoopLock)
    smp_stub.logging = logging
    smp_stub.LMUDataRoot = lambda: ''
    smp_stub.get_cache_path = lambda repo_id: None
    smp_stub.modelscope_flag_set = lambda: False
    smp_stub.load = lambda path: pd.DataFrame()

    smp_file_stub = types.ModuleType('vlmeval.smp.file')
    smp_file_stub.get_intermediate_file_path = lambda *args, **kwargs: ''
    smp_file_stub.get_file_extension = lambda path: Path(path).suffix

    dataset_utils_stub = types.ModuleType('vlmeval.dataset.utils')
    dataset_utils_stub.build_judge = lambda *args, **kwargs: None
    dataset_utils_stub.DEBUG_MESSAGE = ''

    hf_stub = types.ModuleType('huggingface_hub')
    hf_stub.snapshot_download = lambda *args, **kwargs: ''

    for name, module in {
        'vlmeval': vlmeval_pkg,
        'vlmeval.dataset': dataset_pkg,
        'vlmeval.smp': smp_stub,
        'vlmeval.smp.file': smp_file_stub,
        'vlmeval.dataset.utils': dataset_utils_stub,
        'huggingface_hub': hf_stub,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = ROOT / 'vlmeval' / 'dataset' / 'videomme_v2.py'
    spec = importlib.util.spec_from_file_location('vlmeval.dataset.videomme_v2_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VideoMMEv2


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


def test_prepare_dataset_reuses_existing_local_tsv_without_rewriting(tmp_path, monkeypatch):
    cls = load_videommev2_class(monkeypatch)

    source = tmp_path / 'Video-MME-v2-source'
    artifact = tmp_path / 'Video-MME-v2-artifacts'
    source.mkdir()
    artifact.mkdir()
    (source / 'test.parquet').write_bytes(b'parquet')
    (source / 'videos').mkdir()
    (source / 'videos' / '001.mp4').write_bytes(b'video')

    existing_tsv = artifact / 'Video-MME-v2.tsv'
    existing_tsv.write_text(
        'index\tvideo\tvideo_path\tquestion\toptions\tanswer\n'
        "0\t001\t./videos/001.mp4\tQuestion?\t['A', 'B']\tA\n",
        encoding='utf-8',
    )
    before = existing_tsv.read_text(encoding='utf-8')

    monkeypatch.setenv('VIDEO_MME_V2_DIR', str(source))
    monkeypatch.setenv('VIDEO_MME_V2_ARTIFACT_DIR', str(artifact))

    def fail_if_rewritten(*_args, **_kwargs):
        raise AssertionError('existing Video-MME-v2.tsv should not be regenerated')

    monkeypatch.setattr(cls.prepare_dataset.__globals__['pd'], 'read_parquet', fail_if_rewritten)

    dataset = object.__new__(cls)
    ret = dataset.prepare_dataset('Video-MME-v2')

    assert ret == {'data_file': str(existing_tsv), 'root': str(source)}
    assert existing_tsv.read_text(encoding='utf-8') == before


def test_prepare_dataset_regenerates_corrupt_local_tsv(tmp_path, monkeypatch):
    cls = load_videommev2_class(monkeypatch)

    source = tmp_path / 'Video-MME-v2-source'
    artifact = tmp_path / 'Video-MME-v2-artifacts'
    source.mkdir()
    artifact.mkdir()
    (source / 'test.parquet').write_bytes(b'parquet')
    (source / 'videos').mkdir()
    (source / 'videos' / '001.mp4').write_bytes(b'video')

    existing_tsv = artifact / 'Video-MME-v2.tsv'
    existing_tsv.write_text(
        'index\tvideo\tvideo_path\tquestion\toptions\tanswer\n'
        '0\t001\t./videos/001.mp4\t"unterminated question\t["A", "B"]\tA\n',
        encoding='utf-8',
    )

    monkeypatch.setenv('VIDEO_MME_V2_DIR', str(source))
    monkeypatch.setenv('VIDEO_MME_V2_ARTIFACT_DIR', str(artifact))

    def fake_read_parquet(_path):
        return pd.DataFrame({
            'video_id': ['001'],
            'question': ['Question?'],
            'options': [['A', 'B']],
            'answer': ['A'],
            'level': ['easy'],
        })

    monkeypatch.setattr(cls.prepare_dataset.__globals__['pd'], 'read_parquet', fake_read_parquet)

    dataset = object.__new__(cls)
    ret = dataset.prepare_dataset('Video-MME-v2')

    data = pd.read_csv(ret['data_file'], sep='\t')
    assert data[['video', 'video_path', 'question', 'answer']].iloc[0].to_dict() == {
        'video': 1,
        'video_path': './videos/001.mp4',
        'question': 'Question?',
        'answer': 'A',
    }


def test_build_prompt_accepts_numeric_video_ids_from_tsv(tmp_path, monkeypatch):
    cls = load_videommev2_class(monkeypatch)

    class FakeVideo:
        def __len__(self):
            return 1

        def get_avg_fps(self):
            return 30.0

    monkeypatch.setitem(
        sys.modules,
        'decord',
        types.SimpleNamespace(VideoReader=lambda path: FakeVideo()),
    )

    dataset = object.__new__(cls)
    dataset.data_root = str(tmp_path)
    dataset.frame_root = str(tmp_path / 'frames')
    dataset.frame_tmpl_adaptive = 'frame-{}-of-{}-adaptive.jpg'
    dataset.adaptive = True
    dataset.nframe = 0
    dataset.fps = -1
    dataset.resize_target_area = False
    dataset.use_subtitle = False
    dataset.reasoning = False
    dataset.SYS = ''

    frame_path = tmp_path / 'frames' / '3187' / 'frame-1-of-1-adaptive.jpg'
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b'cached frame')

    message = dataset.build_prompt(
        {
            'video': np.int64(3187),
            'question': 'What happens?',
            'options': "['one', 'two', 'three', 'four']",
        },
        video_llm=False,
    )

    image_values = [part['value'] for part in message if part['type'] == 'image']
    assert image_values == [str(frame_path)]


def test_build_prompt_accepts_loose_option_strings(tmp_path, monkeypatch):
    cls = load_videommev2_class(monkeypatch)

    class FakeVideo:
        def __len__(self):
            return 1

        def get_avg_fps(self):
            return 30.0

    monkeypatch.setitem(
        sys.modules,
        'decord',
        types.SimpleNamespace(VideoReader=lambda path: FakeVideo()),
    )

    dataset = object.__new__(cls)
    dataset.data_root = str(tmp_path)
    dataset.frame_root = str(tmp_path / 'frames')
    dataset.frame_tmpl_adaptive = 'frame-{}-of-{}-adaptive.jpg'
    dataset.adaptive = True
    dataset.nframe = 0
    dataset.fps = -1
    dataset.resize_target_area = False
    dataset.use_subtitle = False
    dataset.reasoning = False
    dataset.SYS = ''

    frame_path = tmp_path / 'frames' / '436' / 'frame-1-of-1-adaptive.jpg'
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b'cached frame')

    message = dataset.build_prompt(
        {
            'video': np.int64(436),
            'question': 'What is shown?',
            'options': '[red object, blue object, green object, yellow object]',
        },
        video_llm=False,
    )

    question_texts = [part['value'] for part in message if part['type'] == 'text']
    assert any('A. red object' in text for text in question_texts)
    assert any('D. yellow object' in text for text in question_texts)


def test_videommev2_parse_options_handles_numpy_string_reprs(monkeypatch):
    cls = load_videommev2_class(monkeypatch)

    assert cls._parse_options("[np.str_('red object'), np.str_('blue object')]") == [
        'red object',
        'blue object',
    ]
    assert cls._parse_options("['red object' 'blue object']") == [
        'red object',
        'blue object',
    ]


def test_build_prompt_prefers_video_path_when_video_id_loses_padding(tmp_path, monkeypatch):
    cls = load_videommev2_class(monkeypatch)
    opened = []

    class FakeVideo:
        def __len__(self):
            return 1

        def get_avg_fps(self):
            return 30.0

    def fake_video_reader(path):
        opened.append(path)
        return FakeVideo()

    monkeypatch.setitem(
        sys.modules,
        'decord',
        types.SimpleNamespace(VideoReader=fake_video_reader),
    )

    dataset = object.__new__(cls)
    dataset.data_root = str(tmp_path)
    dataset.frame_root = str(tmp_path / 'frames')
    dataset.frame_tmpl_adaptive = 'frame-{}-of-{}-adaptive.jpg'
    dataset.adaptive = True
    dataset.nframe = 0
    dataset.fps = -1
    dataset.resize_target_area = False
    dataset.use_subtitle = False
    dataset.reasoning = False
    dataset.SYS = ''

    video_path = tmp_path / 'videos' / '098.mp4'
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b'video')

    frame_path = tmp_path / 'frames' / '98' / 'frame-1-of-1-adaptive.jpg'
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b'cached frame')

    dataset.build_prompt(
        {
            'video': np.int64(98),
            'video_path': './videos/098.mp4',
            'question': 'What happens?',
            'options': "['one', 'two', 'three', 'four']",
        },
        video_llm=False,
    )

    assert opened == [str(video_path)]
