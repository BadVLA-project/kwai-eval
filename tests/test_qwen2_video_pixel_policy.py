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
    model.generate_kwargs = {
        'temperature': 0,
        'top_p': 0.001,
        'top_k': 1,
        'repetition_penalty': 1.0,
    }
    model.max_new_tokens = 1024
    model.model_path = 'Qwen2.5-VL-7B-Instruct'
    model.system_prompt = None
    model.use_audio_in_video = False
    model.post_process = False
    model.verbose = False
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


def test_qwen2_generate_batch_vllm_batches_requests(monkeypatch, tmp_path):
    module = load_qwen2_model(monkeypatch)

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm_mod = types.ModuleType('vllm')
    vllm_mod.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, 'vllm', vllm_mod)

    seen_content = []

    def process_vision_info(messages, return_video_kwargs=False):
        content = messages[-1]['content']
        seen_content.append(content)
        video_item = next(item for item in content if item['type'] == 'video')
        assert video_item['min_pixels'] == 3136
        assert video_item['max_pixels'] == 65536
        video_kwargs = {}
        for key in ('fps', 'nframes'):
            if key in video_item:
                video_kwargs[key] = [video_item[key]]
        return None, [f"decoded:{video_item['video']}"], video_kwargs

    qwen_utils = types.ModuleType('qwen_vl_utils')
    qwen_utils.process_vision_info = process_vision_info
    monkeypatch.setitem(sys.modules, 'qwen_vl_utils', qwen_utils)

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"prompt:{messages[-1]['content'][-1]['text']}"

    class FakeOutput:
        def __init__(self, text):
            self.outputs = [types.SimpleNamespace(text=text)]

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, reqs, sampling_params=None, use_tqdm=False):
            self.calls.append((reqs, sampling_params, use_tqdm))
            assert isinstance(reqs, list)
            return [FakeOutput(f'answer-{i}') for i, _ in enumerate(reqs)]

    model = make_qwen2_model(module.Qwen2VLChat)
    model.processor = FakeProcessor()
    model.llm = FakeLLM()

    short_video = tmp_path / 'short.mp4'
    long_video = tmp_path / 'long.mp4'
    short_video.write_bytes(b'')
    long_video.write_bytes(b'')

    responses = model.generate_batch_vllm(
        [
            [
                {'type': 'video', 'value': str(short_video), 'fps': 1.0},
                {'type': 'text', 'value': 'first'},
            ],
            [
                {'type': 'video', 'value': str(long_video), 'nframes': 256},
                {'type': 'text', 'value': 'second'},
            ],
        ],
        dataset='TimeLensBench_QVHighlights_adaptive',
        chunk_size=2,
    )

    assert responses == ['answer-0', 'answer-1']
    assert len(model.llm.calls) == 1
    reqs, sampling_params, use_tqdm = model.llm.calls[0]
    assert len(reqs) == 2
    assert reqs[0]['prompt'] == 'prompt:first'
    assert reqs[1]['prompt'] == 'prompt:second'
    assert reqs[0]['mm_processor_kwargs'] == {'fps': [1.0]}
    assert reqs[1]['mm_processor_kwargs'] == {'nframes': [256]}
    assert sampling_params.kwargs['temperature'] == 0.0
    assert sampling_params.kwargs['max_tokens'] == 1024
    assert use_tqdm is False
    assert len(seen_content) == 2
