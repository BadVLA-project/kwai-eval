import importlib.util
import os
import sys
import types
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_cvbench_module(monkeypatch):
    vlmeval_pkg = types.ModuleType('vlmeval')
    vlmeval_pkg.__path__ = [str(ROOT / 'vlmeval')]
    dataset_pkg = types.ModuleType('vlmeval.dataset')
    dataset_pkg.__path__ = [str(ROOT / 'vlmeval' / 'dataset')]

    smp_stub = types.ModuleType('vlmeval.smp')
    smp_stub.osp = os.path
    smp_stub.os = os
    smp_stub.pd = pd
    smp_stub.get_cache_path = lambda *_args, **_kwargs: None
    smp_stub.modelscope_flag_set = lambda: False
    smp_stub.load = lambda path: pd.read_csv(path, sep='\t')
    smp_stub.dump = lambda data, path: data.to_csv(path, sep='\t', index=False)

    smp_file_stub = types.ModuleType('vlmeval.smp.file')
    smp_file_stub.get_file_extension = lambda path: str(path).rsplit('.', 1)[-1]
    smp_file_stub.get_intermediate_file_path = lambda eval_file, suffix, target_format=None: str(eval_file)

    video_base_stub = types.ModuleType('vlmeval.dataset.video_base')

    class VideoBaseDataset:
        def __init__(self, *args, **kwargs):
            pass

    video_base_stub.VideoBaseDataset = VideoBaseDataset
    hf_stub = types.ModuleType('huggingface_hub')
    hf_stub.snapshot_download = lambda *args, **kwargs: None

    for name, module in {
        'vlmeval': vlmeval_pkg,
        'vlmeval.dataset': dataset_pkg,
        'vlmeval.smp': smp_stub,
        'vlmeval.smp.file': smp_file_stub,
        'vlmeval.dataset.video_base': video_base_stub,
        'huggingface_hub': hf_stub,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = ROOT / 'vlmeval' / 'dataset' / 'cvbench_video.py'
    spec = importlib.util.spec_from_file_location('vlmeval.dataset.cvbench_video', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cvbench_video_root_prefers_official_nested_video_r1_layout(tmp_path, monkeypatch):
    module = load_cvbench_module(monkeypatch)
    data_root = tmp_path / 'CVBench'
    nested_video_root = data_root / 'Video-R1' / 'src' / 'r1-v' / 'Evaluation' / 'CVBench'
    nested_video_root.mkdir(parents=True)

    dataset = module.CVBenchVideo.__new__(module.CVBenchVideo)

    assert dataset._resolve_video_root(str(data_root)) == str(nested_video_root)
