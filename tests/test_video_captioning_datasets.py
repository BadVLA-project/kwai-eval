import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def load_video_captioning_module():
    root = Path(__file__).resolve().parents[1]

    vlmeval_pkg = types.ModuleType('vlmeval')
    vlmeval_pkg.__path__ = [str(root / 'vlmeval')]
    dataset_pkg = types.ModuleType('vlmeval.dataset')
    dataset_pkg.__path__ = [str(root / 'vlmeval' / 'dataset')]
    smp_stub = types.ModuleType('vlmeval.smp')
    smp_stub.dump = lambda data, path, **kwargs: None
    smp_stub.load = lambda path, **kwargs: None
    smp_stub.get_cache_path = lambda repo_id: None
    smp_stub.get_intermediate_file_path = lambda eval_file, suffix, target_format=None: str(eval_file)
    smp_stub.md5 = lambda path: ''
    smp_stub.modelscope_flag_set = lambda: False

    video_base_stub = types.ModuleType('vlmeval.dataset.video_base')

    class VideoBaseDataset:
        def __init__(self, *args, **kwargs):
            pass

    video_base_stub.VideoBaseDataset = VideoBaseDataset
    sys.modules['vlmeval'] = vlmeval_pkg
    sys.modules['vlmeval.dataset'] = dataset_pkg
    sys.modules['vlmeval.smp'] = smp_stub
    sys.modules['vlmeval.dataset.video_base'] = video_base_stub

    spec = importlib.util.spec_from_file_location(
        'vlmeval.dataset.video_captioning',
        root / 'vlmeval' / 'dataset' / 'video_captioning.py',
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['vlmeval.dataset.video_captioning'] = module
    spec.loader.exec_module(module)
    return module


def test_youcook2_tsv_uses_lmms_eval_hf_schema(tmp_path):
    mod = load_video_captioning_module()
    data_dir = tmp_path / 'data'
    video_dir = tmp_path / 'YouCookIIVideos'
    data_dir.mkdir()
    video_dir.mkdir()
    (video_dir / 'clip.mp4').write_bytes(b'fake mp4')

    pd.DataFrame([
        {
            'youtube_id': 'abc123',
            'segment': [1.0, 4.0],
            'video_path': 'YouCookIIVideos/clip.mp4',
            'sentence': 'A person mixes the ingredients.',
        }
    ]).to_parquet(data_dir / 'val-00000-of-00001.parquet')

    data_file = mod.YouCook2Caption._generate_tsv(tmp_path, force=True)
    data = pd.read_csv(data_file, sep='\t')

    assert data.loc[0, 'video'] == 'abc123_1.0_4.0'
    assert data.loc[0, 'video_path'] == 'YouCookIIVideos/clip.mp4'
    assert data.loc[0, 'question'] == mod.YouCook2Caption.PROMPT
    assert data.loc[0, 'answer'] == 'A person mixes the ingredients.'


def test_temporalbench_caption_tsv_uses_short_caption_split(tmp_path):
    mod = load_video_captioning_module()
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (tmp_path / 'tb_clip.mp4').write_bytes(b'fake mp4')

    pd.DataFrame([
        {
            'idx': 7,
            'video_name': 'tb_clip.mp4',
            'question': 'Describe the main temporal change.',
            'GT': 'The ball rolls from left to right.',
            'dataset': 'synthetic',
            'category': 'Motion Direction/Orientation',
        }
    ]).to_parquet(data_dir / 'test_short_caption-00000-of-00001.parquet')

    data_file = mod.TemporalBenchCaption._generate_tsv(tmp_path, force=True)
    data = pd.read_csv(data_file, sep='\t')

    assert data.loc[0, 'index'] == 7
    assert data.loc[0, 'video'] == 'tb_clip'
    assert data.loc[0, 'video_path'] == 'tb_clip.mp4'
    assert data.loc[0, 'question'] == 'Describe the main temporal change.'
    assert data.loc[0, 'answer'] == 'The ball rolls from left to right.'


def test_caption_dataset_names_are_declared():
    mod = load_video_captioning_module()

    assert mod.YouCook2Caption.HF_REPO_ID == 'lmms-lab/YouCook2'
    assert mod.TemporalBenchCaption.HF_REPO_ID == 'microsoft/TemporalBench'
    assert 'YouCook2' in mod.YouCook2Caption.supported_datasets()
    assert 'TemporalBench_Captioning' in mod.TemporalBenchCaption.supported_datasets()
