from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_direct_does_not_force_exact_matching_judge():
    script = (ROOT / 'run_direct.sh').read_text(encoding='utf-8')

    assert '--judge exact_matching' not in script
    assert 'JUDGE="${JUDGE:-}"' in script
    assert 'CMD+=(--judge "${JUDGE}")' in script
    assert 'CMD+=(--judge-args "${JUDGE_ARGS}")' in script
