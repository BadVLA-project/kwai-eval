from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_direct_does_not_force_exact_matching_judge():
    script = (ROOT / 'run_direct.sh').read_text(encoding='utf-8')

    assert '--judge exact_matching' not in script
    assert 'JUDGE="${JUDGE:-}"' in script
    assert 'CMD+=(--judge "${JUDGE}")' in script
    assert 'CMD+=(--judge-args "${JUDGE_ARGS}")' in script


def test_run_direct_defaults_to_1024_generation_tokens():
    script = (ROOT / 'run_direct.sh').read_text(encoding='utf-8')

    assert 'export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"' in script


def test_run_direct_sets_readonly_videomme_cache_default():
    script = (ROOT / 'run_direct.sh').read_text(encoding='utf-8')

    assert 'export VIDEO_MME_DIR="${VIDEO_MME_DIR:-/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/datasets--lmms-lab--Video-MME}"' in script
