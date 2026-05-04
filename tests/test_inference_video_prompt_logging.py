import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_inference_video(monkeypatch):
    torch_mod = types.ModuleType('torch')
    torch_dist_mod = types.ModuleType('torch.distributed')
    torch_mod.distributed = torch_dist_mod
    config_mod = types.ModuleType('vlmeval.config')
    config_mod.supported_VLM = {}
    utils_mod = types.ModuleType('vlmeval.utils')
    utils_mod.track_progress_rich = lambda *args, **kwargs: None
    utils_mod.shard_items = lambda items, rank, world_size: list(items)[rank::world_size]
    smp_mod = types.ModuleType('vlmeval.smp')
    smp_mod.__all__ = []

    monkeypatch.setitem(sys.modules, 'torch', torch_mod)
    monkeypatch.setitem(sys.modules, 'torch.distributed', torch_dist_mod)
    monkeypatch.setitem(sys.modules, 'vlmeval.config', config_mod)
    monkeypatch.setitem(sys.modules, 'vlmeval.utils', utils_mod)
    monkeypatch.setitem(sys.modules, 'vlmeval.smp', smp_mod)

    spec = importlib.util.spec_from_file_location(
        'inference_video_for_test',
        ROOT / 'vlmeval' / 'inference_video.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeModel:
    post_prompt = 'Answer with the option letter only.'

    def _should_apply_post_prompt(self, dataset):
        return dataset == 'Video-MME_adaptive'


def test_prompt_logging_uses_model_post_prompt_policy(monkeypatch):
    module = load_inference_video(monkeypatch)
    struct = [
        {'type': 'video', 'value': '/tmp/1.mp4'},
        {'type': 'text', 'value': 'Describe the video in detail.'},
    ]

    assert module._prompt_text_for_logging(FakeModel(), 'DREAM-1K_adaptive', struct) == (
        '[video]\nDescribe the video in detail.'
    )
    assert module._prompt_text_for_logging(FakeModel(), 'Video-MME_adaptive', struct).endswith(
        'Answer with the option letter only.'
    )
