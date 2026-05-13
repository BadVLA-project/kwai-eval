import ast
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_qwen2_model(monkeypatch):
    vlmeval_pkg = types.ModuleType('vlmeval')
    vlmeval_pkg.__path__ = []
    vlm_pkg = types.ModuleType('vlmeval.vlm')
    vlm_pkg.__path__ = []
    qwen_pkg = types.ModuleType('vlmeval.vlm.qwen2_vl')
    qwen_pkg.__path__ = []

    base_mod = types.ModuleType('vlmeval.vlm.base')

    class BaseModel:
        pass

    base_mod.BaseModel = BaseModel

    prompt_mod = types.ModuleType('vlmeval.vlm.qwen2_vl.prompt')

    class Qwen2VLPromptMixin:
        pass

    prompt_mod.Qwen2VLPromptMixin = Qwen2VLPromptMixin

    dataset_mod = types.ModuleType('vlmeval.dataset')
    dataset_mod.DATASET_MODALITY = lambda dataset, default=None: 'VIDEO'

    smp_mod = types.ModuleType('vlmeval.smp')
    smp_mod.get_gpu_memory = lambda: 0
    smp_mod.listinstr = lambda needles, haystack: any(x in haystack for x in needles)

    torch_mod = types.ModuleType('torch')
    torch_mod.cuda = types.SimpleNamespace(empty_cache=lambda: None)

    transformers_mod = types.ModuleType('transformers')

    class StoppingCriteria:
        pass

    transformers_mod.StoppingCriteria = StoppingCriteria

    for name, module in {
        'vlmeval': vlmeval_pkg,
        'vlmeval.vlm': vlm_pkg,
        'vlmeval.vlm.qwen2_vl': qwen_pkg,
        'vlmeval.vlm.base': base_mod,
        'vlmeval.vlm.qwen2_vl.prompt': prompt_mod,
        'vlmeval.dataset': dataset_mod,
        'vlmeval.smp': smp_mod,
        'torch': torch_mod,
        'transformers': transformers_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = ROOT / 'vlmeval' / 'vlm' / 'qwen2_vl' / 'model.py'
    spec = importlib.util.spec_from_file_location('vlmeval.vlm.qwen2_vl.model_for_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_qwen2_model(cls):
    model = cls.__new__(cls)
    model.min_pixels = 1280 * 28 * 28
    model.max_pixels = 16384 * 28 * 28
    model.video_min_pixels = 3136
    model.video_max_pixels = 65536
    model.total_pixels = None
    model.fps = 2
    model.nframe = 128
    model.FRAME_FACTOR = 2
    model.limit_mm_per_prompt = 128
    return model


def _qwen2vl_series_entries():
    tree = ast.parse((ROOT / 'vlmeval' / 'config.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'qwen2vl_series':
                    return node.value
    raise AssertionError('qwen2vl_series assignment not found')


def _literal_keyword(call_node, name):
    for keyword in call_node.keywords:
        if keyword.arg == name:
            return ast.literal_eval(keyword.value)
    raise AssertionError(f'{name} keyword not found')


def test_qwen2_video_content_uses_video_pixel_limits(monkeypatch, tmp_path):
    module = load_qwen2_model(monkeypatch)
    model = make_qwen2_model(module.Qwen2VLChat)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'')

    content = model._prepare_content(
        [{'type': 'video', 'value': str(video), 'fps': 1.0}],
        dataset='Video-MME_adaptive',
    )

    assert content == [{
        'type': 'video',
        'video': f'file://{video}',
        'min_pixels': 3136,
        'max_pixels': 65536,
        'fps': 1.0,
    }]


def test_qwen2_vllm_video_content_uses_video_pixel_limits(monkeypatch, tmp_path):
    module = load_qwen2_model(monkeypatch)
    model = make_qwen2_model(module.Qwen2VLChat)
    video = tmp_path / 'long.mp4'
    video.write_bytes(b'')

    content = model._prepare_content_vllm(
        [{'type': 'video', 'value': str(video), 'nframes': 256}],
        dataset='LongVideoBench_adaptive',
    )

    assert content == [{
        'type': 'video',
        'video': f'file://{video}',
        'min_pixels': 3136,
        'max_pixels': 65536,
        'nframes': 256,
    }]


def test_qwen25_3b_and_7b_configure_qwen3_style_video_pixels():
    entries = _qwen2vl_series_entries()
    calls = {
        key.value: value
        for key, value in zip(entries.keys, entries.values)
        if isinstance(key, ast.Constant)
    }

    for model_name in ('Qwen2.5-VL-3B-Instruct', 'Qwen2.5-VL-7B-Instruct'):
        assert _literal_keyword(calls[model_name], 'video_min_pixels') == 3136
        assert _literal_keyword(calls[model_name], 'video_max_pixels') == 65536
