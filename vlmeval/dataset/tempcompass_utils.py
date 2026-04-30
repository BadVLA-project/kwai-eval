import os
import os.path as osp
from glob import glob


DEFAULT_TEMPCOMPASS_DIRS = (
    '/m2v_intern/xuboshen/zgw/hf_cache_temp/datasets--lmms-lab--TempCompass',
    '/m2v_intern/xuboshen/zgw/Benchmarks/TempCompass',
)

TEMPCOMPASS_DIR_ENV_KEYS = ('TEMPCOMPASS_DIR', 'TEMP_COMPASS_DIR', 'TEMPCOMPASS_CACHE_DIR')


def first_env(env, keys):
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def has_tempcompass_metadata(path):
    if osp.exists(osp.join(path, 'TempCompass_MCQ.tsv')):
        return True
    task_dirs = ('multi-choice', 'caption_matching', 'captioning', 'yes_no')
    return any(
        osp.exists(osp.join(path, task_dir, 'test-00000-of-00001.parquet'))
        for task_dir in task_dirs
    )


def resolve_hf_snapshot_path(path):
    if not path:
        return None

    snapshots = glob(osp.join(path, 'snapshots', '*'))
    snapshots = [p for p in snapshots if osp.isdir(p)]
    with_metadata = [p for p in snapshots if has_tempcompass_metadata(p)]
    if with_metadata:
        with_metadata.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return with_metadata[0]
    if snapshots:
        snapshots.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return snapshots[0]
    if has_tempcompass_metadata(path):
        return path
    return path if osp.exists(path) else None


def tempcompass_candidates(env=None, default_dirs=DEFAULT_TEMPCOMPASS_DIRS):
    env = os.environ if env is None else env
    candidates = []
    explicit = first_env(env, TEMPCOMPASS_DIR_ENV_KEYS)
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


def resolve_tempcompass_dir(env=None, default_dirs=DEFAULT_TEMPCOMPASS_DIRS):
    for candidate in tempcompass_candidates(env=env, default_dirs=default_dirs):
        resolved = resolve_hf_snapshot_path(candidate)
        if resolved and osp.exists(resolved):
            return resolved
    return None
