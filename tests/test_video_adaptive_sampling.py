import importlib.util
import sys
import types
from pathlib import Path


def load_video_base_dataset():
    root = Path(__file__).resolve().parents[1]
    vlmeval_pkg = types.ModuleType('vlmeval')
    vlmeval_pkg.__path__ = [str(root / 'vlmeval')]
    dataset_pkg = types.ModuleType('vlmeval.dataset')
    dataset_pkg.__path__ = [str(root / 'vlmeval' / 'dataset')]
    smp_stub = types.ModuleType('vlmeval.smp')
    sys.modules['vlmeval'] = vlmeval_pkg
    sys.modules['vlmeval.dataset'] = dataset_pkg
    sys.modules['vlmeval.smp'] = smp_stub

    spec = importlib.util.spec_from_file_location(
        'vlmeval.dataset.video_base',
        root / 'vlmeval' / 'dataset' / 'video_base.py',
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['vlmeval.dataset.video_base'] = module
    spec.loader.exec_module(module)
    return module.VideoBaseDataset


VideoBaseDataset = load_video_base_dataset()


class FakeVideo:
    def __init__(self, duration, fps=30.0):
        self.duration = duration
        self.fps = fps
        self.frames = int(duration * fps)

    def __len__(self):
        return self.frames

    def get_avg_fps(self):
        return self.fps


def test_adaptive_indices_use_60s_and_256s_thresholds():
    dataset = object.__new__(VideoBaseDataset)

    short_indices = VideoBaseDataset.compute_adaptive_indices(dataset, FakeVideo(60))
    assert len(short_indices) == 120
    assert dataset._last_adaptive_strategy.startswith('2fps ')

    medium_indices = VideoBaseDataset.compute_adaptive_indices(dataset, FakeVideo(61))
    assert len(medium_indices) == 61
    assert dataset._last_adaptive_strategy.startswith('1fps ')

    long_indices = VideoBaseDataset.compute_adaptive_indices(dataset, FakeVideo(300))
    assert len(long_indices) == 256
    assert dataset._last_adaptive_strategy.startswith('uniform-256 ')


def test_adaptive_video_kwargs_match_frame_extraction_policy(monkeypatch):
    dataset = object.__new__(VideoBaseDataset)
    dataset.frame_info = {}
    videos = {
        'short.mp4': FakeVideo(60),
        'medium.mp4': FakeVideo(61),
        'long.mp4': FakeVideo(300),
    }
    fake_decord = types.SimpleNamespace(VideoReader=lambda path: videos[path])
    monkeypatch.setitem(sys.modules, 'decord', fake_decord)

    assert VideoBaseDataset.get_adaptive_video_kwargs(dataset, 'short.mp4', 'short') == {'fps': 2.0}
    assert dataset.frame_info['short']['num_frames'] == 120

    assert VideoBaseDataset.get_adaptive_video_kwargs(dataset, 'medium.mp4', 'medium') == {'fps': 1.0}
    assert dataset.frame_info['medium']['num_frames'] == 61

    assert VideoBaseDataset.get_adaptive_video_kwargs(dataset, 'long.mp4', 'long') == {'nframes': 256}
    assert dataset.frame_info['long']['num_frames'] == 256
