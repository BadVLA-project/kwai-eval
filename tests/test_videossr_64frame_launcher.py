import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


VIDEO_SSR_64FRAME_DATASETS = [
    'TimeLensBench_QVHighlights_64frame',
    'TimeLensBench_ActivityNet_64frame',
    'TimeLensBench_Charades_64frame',
    'Video_Holmes_64frame',
    'VideoMMMU_64frame',
    'Video_TT_64frame',
    'ETBench_64frame',
    'MVBench_MP4_64frame',
    'TempCompass_MCQ_64frame',
    'AoTBench_ReverseFilm_64frame',
    'AoTBench_UCF101_64frame',
    'AoTBench_Rtime_t2v_64frame',
    'AoTBench_Rtime_v2t_64frame',
    'AoTBench_QA_64frame',
    'Vinoground_64frame',
    'Video-MME_64frame',
    'MLVU_MCQ_64frame',
    'LongVideoBench_64frame',
]


def _source(relative_path):
    return (ROOT / relative_path).read_text(encoding='utf-8')


def _dataset_keys_from_config():
    tree = ast.parse(_source('vlmeval/dataset/video_dataset_config.py'))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


def _default_dataset_block(script):
    match = re.search(r'DATASETS=\(\n(?P<body>.*?)\n\s*\)', script, flags=re.S)
    assert match is not None
    return [line.strip() for line in match.group('body').splitlines() if line.strip()]


def test_videossr_64frame_datasets_are_registered():
    keys = _dataset_keys_from_config()

    assert set(VIDEO_SSR_64FRAME_DATASETS) <= keys


def test_videossr_launcher_defaults_to_64frame_video_ssr_settings():
    script = _source('run_videossr_64frame.sh')
    datasets = _default_dataset_block(script)

    assert datasets == VIDEO_SSR_64FRAME_DATASETS
    assert 'VideoSSR-8B' in script
    assert 'MODEL_LIST=(' in script
    assert '--use-vllm' in script
    assert 'export USE_COT="${USE_COT:-0}"' in script
    assert 'export TEMPERATURE="${TEMPERATURE:-0}"' in script
    assert 'export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"' in script
    assert 'export VIDEO_SSR_FRAMES="${VIDEO_SSR_FRAMES:-64}"' in script
    assert 'DRY_RUN' in script
    assert '_adaptive' not in '\n'.join(datasets)
