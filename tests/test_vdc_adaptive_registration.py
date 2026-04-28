from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vdc_adaptive_dataset_is_registered():
    config = (ROOT / 'vlmeval' / 'dataset' / 'video_dataset_config.py').read_text(encoding='utf-8')

    assert "'VDC_adaptive': partial(VDC, dataset='VDC', adaptive=True)" in config
