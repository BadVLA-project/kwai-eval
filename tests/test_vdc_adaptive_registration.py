from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vdc_adaptive_dataset_is_registered():
    config = (ROOT / 'vlmeval' / 'dataset' / 'video_dataset_config.py').read_text(encoding='utf-8')

    assert "'VDC_adaptive': partial(VDC, dataset='VDC', adaptive=True)" in config


def test_vdc_constructor_accepts_and_forwards_adaptive_flag():
    vdc_py = (ROOT / 'vlmeval' / 'dataset' / 'vdc.py').read_text(encoding='utf-8')

    assert "adaptive=False" in vdc_py
    assert "super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps, adaptive=adaptive)" in vdc_py


def test_vdc_evaluate_uses_judge_key_for_response_map():
    vdc_py = (ROOT / 'vlmeval' / 'dataset' / 'vdc.py').read_text(encoding='utf-8')

    assert "data_un['pred_response'] = [pred_map.get(idx, FAIL_MSG) for idx in indices]" in vdc_py
    assert "for idx in data_un['index']" not in vdc_py
