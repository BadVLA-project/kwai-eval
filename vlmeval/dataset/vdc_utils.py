import os
import os.path as osp
from glob import glob


DEFAULT_VDC_CACHE_ROOT = '/m2v_intern/xuboshen/zgw/hf_cache_temp/datasets--Enxin--VLMEval-VDC'
VDC_ENV_KEYS = ('VDC_DIR', 'VDC_CACHE_DIR', 'VLMEVAL_VDC_DIR')


def first_env(env, keys):
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def has_media_files(path):
    if not osp.isdir(path):
        return False
    media_exts = ('.mp4', '.mkv', '.mov', '.webm')
    return any(name.lower().endswith(media_exts) for name in os.listdir(path))


def resolve_hf_snapshot_path(path):
    if not path:
        return None
    if osp.exists(osp.join(path, 'VDC.tsv')):
        return path
    snapshots = glob(osp.join(path, 'snapshots', '*'))
    snapshots = [p for p in snapshots if osp.isdir(p)]
    with_tsv = [p for p in snapshots if osp.exists(osp.join(p, 'VDC.tsv'))]
    if with_tsv:
        with_tsv.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return with_tsv[0]
    return path if osp.exists(path) else None


def vdc_cache_candidates(env=None):
    env = os.environ if env is None else env
    candidates = []
    explicit = first_env(env, VDC_ENV_KEYS)
    if explicit:
        candidates.append(explicit)
    for root_key in ('HF_HOME', 'HUGGINGFACE_HUB_CACHE'):
        root = env.get(root_key)
        if not root:
            continue
        candidates.extend([
            osp.join(root, 'datasets--Enxin--VLMEval-VDC'),
            osp.join(root, 'hub', 'datasets--Enxin--VLMEval-VDC'),
        ])
    candidates.append(DEFAULT_VDC_CACHE_ROOT)
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_vdc_local_path(env=None):
    for candidate in vdc_cache_candidates(env=env):
        resolved = resolve_hf_snapshot_path(candidate)
        if resolved and osp.exists(resolved):
            return resolved
    return None


def find_vdc_data_file(dataset_path, dataset_name='VDC'):
    candidates = [
        osp.join(dataset_path, f'{dataset_name}.tsv'),
        osp.join(dataset_path, 'VDC.tsv'),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return candidates[0]


def find_vdc_video_root(dataset_path):
    candidates = [
        osp.join(dataset_path, 'videos'),
        osp.join(dataset_path, 'video'),
    ]
    for candidate in candidates:
        if has_media_files(candidate):
            return candidate
    for candidate in candidates:
        if osp.isdir(candidate):
            return candidate
    return candidates[0]
