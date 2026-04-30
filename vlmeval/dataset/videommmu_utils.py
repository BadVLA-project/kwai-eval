import os
import os.path as osp
from glob import glob


DEFAULT_VIDEOMMMU_DIRS = (
    '/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/datasets--lmms-lab--VideoMMMU',
    '/m2v_intern/xuboshen/zgw/Benchmarks/VideoMMMU',
)

VIDEOMMMU_DIR_ENV_KEYS = ('VIDEOMMMU_DIR', 'VIDEO_MMMU_DIR', 'VIDEOMMMU_CACHE_DIR')


def first_env(env, keys):
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def has_videommmu_metadata(path):
    if osp.exists(osp.join(path, 'VideoMMMU.tsv')):
        return True
    return bool(glob(osp.join(path, '**', '*.parquet'), recursive=True))


def resolve_hf_snapshot_path(path):
    if not path:
        return None

    snapshots = glob(osp.join(path, 'snapshots', '*'))
    snapshots = [p for p in snapshots if osp.isdir(p)]
    with_metadata = [p for p in snapshots if has_videommmu_metadata(p)]
    if with_metadata:
        with_metadata.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return with_metadata[0]
    if snapshots:
        snapshots.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return snapshots[0]
    if has_videommmu_metadata(path):
        return path
    return path if osp.exists(path) else None


def videommmu_candidates(env=None, default_dirs=DEFAULT_VIDEOMMMU_DIRS):
    env = os.environ if env is None else env
    candidates = []
    explicit = first_env(env, VIDEOMMMU_DIR_ENV_KEYS)
    if explicit:
        candidates.append(explicit)
    candidates.extend(default_dirs)

    seen = set()
    unique = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_videommmu_dir(env=None, default_dirs=DEFAULT_VIDEOMMMU_DIRS):
    for candidate in videommmu_candidates(env=env, default_dirs=default_dirs):
        resolved = resolve_hf_snapshot_path(candidate)
        if resolved and osp.exists(resolved):
            return resolved
    return None
