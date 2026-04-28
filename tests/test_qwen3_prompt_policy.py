import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_qwen_model(monkeypatch):
    vlmeval_pkg = types.ModuleType('vlmeval')
    vlmeval_pkg.__path__ = []
    vlm_pkg = types.ModuleType('vlmeval.vlm')
    vlm_pkg.__path__ = []
    qwen_pkg = types.ModuleType('vlmeval.vlm.qwen3_vl')
    qwen_pkg.__path__ = []

    base_mod = types.ModuleType('vlmeval.vlm.base')

    class BaseModel:
        pass

    base_mod.BaseModel = BaseModel

    prompt_mod = types.ModuleType('vlmeval.vlm.qwen3_vl.prompt')

    class Qwen3VLPromptMixin:
        pass

    prompt_mod.Qwen3VLPromptMixin = Qwen3VLPromptMixin

    smp_mod = types.ModuleType('vlmeval.smp')
    smp_mod.get_gpu_memory = lambda: 0
    smp_mod.listinstr = lambda needles, haystack: any(x in haystack for x in needles)

    dataset_mod = types.ModuleType('vlmeval.dataset')
    dataset_types = {
        'DREAM-1K_adaptive': 'DREAM-1K',
        'Video-MME_adaptive': 'MCQ',
    }
    dataset_mod.DATASET_TYPE = lambda dataset, default='MCQ': dataset_types.get(dataset, default)

    torch_mod = types.ModuleType('torch')
    torch_mod.cuda = types.SimpleNamespace(empty_cache=lambda: None)

    for name, module in {
        'vlmeval': vlmeval_pkg,
        'vlmeval.vlm': vlm_pkg,
        'vlmeval.vlm.qwen3_vl': qwen_pkg,
        'vlmeval.vlm.base': base_mod,
        'vlmeval.vlm.qwen3_vl.prompt': prompt_mod,
        'vlmeval.smp': smp_mod,
        'vlmeval.dataset': dataset_mod,
        'torch': torch_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = ROOT / 'vlmeval' / 'vlm' / 'qwen3_vl' / 'model.py'
    spec = importlib.util.spec_from_file_location('vlmeval.vlm.qwen3_vl.model_for_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_direct_model(cls):
    model = cls.__new__(cls)
    model.min_pixels = None
    model.max_pixels = None
    model.total_pixels = None
    model.fps = None
    model.nframe = None
    model.FRAME_FACTOR = 2
    model.post_prompt = 'Answer with the option letter only.'
    model.post_prompt_mode = 'mcq_direct'
    return model


def test_direct_option_prompt_is_not_appended_to_dream_caption(monkeypatch):
    module = load_qwen_model(monkeypatch)
    model = make_direct_model(module.Qwen3VLChat)

    content = model._prepare_content(
        [{'type': 'text', 'value': 'Describe the video in detail.'}],
        dataset='DREAM-1K_adaptive',
    )

    texts = [item['text'] for item in content if item.get('type') == 'text']
    assert texts == ['Describe the video in detail.']


def test_direct_option_prompt_is_still_appended_to_mcq(monkeypatch):
    module = load_qwen_model(monkeypatch)
    model = make_direct_model(module.Qwen3VLChat)

    content = model._prepare_content(
        [{'type': 'text', 'value': 'Question: What happened?'}],
        dataset='Video-MME_adaptive',
    )

    texts = [item['text'] for item in content if item.get('type') == 'text']
    assert texts[-1] == 'Answer with the option letter only.'
