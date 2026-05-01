import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path):
    return (ROOT / relative_path).read_text(encoding='utf-8')


def _adaptive_dataset_entries():
    tree = ast.parse(_source('vlmeval/dataset/video_dataset_config.py'))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'adaptive_dataset':
                    return node.value
    raise AssertionError('adaptive_dataset assignment not found')


def _dict_keys(dict_node):
    return {key.value for key in dict_node.keys if isinstance(key, ast.Constant)}


def test_missing_video_benchmarks_have_explicit_adaptive_registrations():
    keys = _dict_keys(_adaptive_dataset_entries())

    expected = {
        'VCRBench_adaptive',
        'CVBench_adaptive',
        'Video-MME-v2_adaptive',
        'Video-MME-v2_adaptive_subs',
        'VideoMMMU_adaptive',
        'LongVideoBench_adaptive',
        'LongVideoBench_adaptive_subs',
        'TempCompass_MCQ_adaptive',
    }

    assert expected <= keys


def test_new_video_benchmark_constructors_accept_adaptive_flag():
    expected_signatures = {
        'vlmeval/dataset/vcrbench.py': "def __init__(self, dataset='VCR-Bench', pack=False, nframe=0, fps=-1, adaptive=False):",
        'vlmeval/dataset/videomme_v2.py': "adaptive=False",
        'vlmeval/dataset/longvideobench.py': (
            "def __init__(self, dataset='LongVideoBench', use_subtitle=False, nframe=0, fps=-1, adaptive=False):"
        ),
        'vlmeval/dataset/tempcompass.py': "def __init__(self, dataset='TempCompass_MCQ', nframe=0, fps=-1, adaptive=False):",
    }

    for path, text in expected_signatures.items():
        assert text in _source(path)


def test_new_video_benchmark_constructors_forward_adaptive_flag():
    expected_calls = {
        'vlmeval/dataset/vcrbench.py': (
            'super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps, adaptive=adaptive)'
        ),
        'vlmeval/dataset/videomme_v2.py': (
            'super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)'
        ),
        'vlmeval/dataset/longvideobench.py': (
            'super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)'
        ),
        'vlmeval/dataset/tempcompass.py': (
            'super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)'
        ),
    }

    for path, text in expected_calls.items():
        assert text in _source(path)


def test_video_llm_prompt_paths_use_adaptive_video_structs():
    expected_snippets = {
        'vlmeval/dataset/vcrbench.py': 'self.make_video_struct(',
        'vlmeval/dataset/longvideobench.py': 'self.make_video_struct(osp.join(self.data_root, line[\'video_path\'])',
        'vlmeval/dataset/tempcompass.py': 'self.make_video_struct(video_path, video_id=line[\'video\'])',
        'vlmeval/dataset/videommmu.py': 'self.make_video_struct(osp.join(self.data_root, line[\'video\']), video_id=line[\'id\'])',
    }

    for path, text in expected_snippets.items():
        assert text in _source(path)

    videomme_v2 = _source('vlmeval/dataset/videomme_v2.py')
    assert 'self.make_video_struct(' in videomme_v2
    assert 'self._resolve_video_path(line)' in videomme_v2


def test_missing_video_launcher_defaults_to_adaptive_mcq_only_tempcompass():
    script = _source('run_missing_video_benchmarks.sh')

    expected_defaults = [
        'VCRBench_adaptive',
        'CVBench_adaptive',
        'Video-MME-v2_adaptive',
        'VideoMMMU_adaptive',
        'LongVideoBench_adaptive',
        'TempCompass_MCQ_adaptive',
    ]

    for dataset_name in expected_defaults:
        assert dataset_name in script

    assert 'TempCompass_64frame' not in script


def test_run_py_defaults_tempcompass_mcq_to_exact_matching():
    run_py = _source('run.py')
    exact_branch = re.search(
        r"listinstr\(\[(.*?)\], dataset_name\):\n\s+judge_kwargs\['model'\] = 'exact_matching'",
        run_py,
        flags=re.S,
    )

    assert exact_branch is not None
    assert "'TempCompass_MCQ'" in exact_branch.group(1)
