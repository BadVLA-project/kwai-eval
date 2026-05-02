from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_compare_mopd_has_resilient_pair_timeout_mode():
    script = (ROOT / 'run_compare_mopd.sh').read_text(encoding='utf-8')

    assert 'RESILIENT="${RESILIENT:-0}"' in script
    assert 'PAIR_TIMEOUT="${PAIR_TIMEOUT:-7200}"' in script
    assert 'timeout --kill-after=60s "${PAIR_TIMEOUT}"' in script
