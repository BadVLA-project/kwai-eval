import os
import os.path as osp
from glob import glob


DEFAULT_VCRBENCH_DIRS = (
    '/m2v_intern/xuboshen/zgw/Benchmarks/VCR-Bench',
    '/m2v_intern/xuboshen/zgw/LMUData/datasets/VCR-Bench',
)

VCRBENCH_ENV_KEYS = ('VCRBENCH_DIR', 'VCR_BENCH_DIR')


def first_env(env, keys):
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def has_vcrbench_metadata(path, dataset_name='VCR-Bench'):
    return osp.exists(osp.join(path, f'{dataset_name}.tsv'))


def resolve_hf_snapshot_path(path, dataset_name='VCR-Bench'):
    if not path:
        return None
    if has_vcrbench_metadata(path, dataset_name=dataset_name):
        return path

    snapshots = glob(osp.join(path, 'snapshots', '*'))
    snapshots = [p for p in snapshots if osp.isdir(p)]
    with_metadata = [p for p in snapshots if has_vcrbench_metadata(p, dataset_name=dataset_name)]
    if with_metadata:
        with_metadata.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return with_metadata[0]
    if snapshots:
        snapshots.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return snapshots[0]
    return path if osp.exists(path) else None


def vcrbench_candidates(env=None, default_dirs=DEFAULT_VCRBENCH_DIRS):
    env = os.environ if env is None else env
    candidates = []
    explicit = first_env(env, VCRBENCH_ENV_KEYS)
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


def resolve_vcrbench_dir(env=None, default_dirs=DEFAULT_VCRBENCH_DIRS, dataset_name='VCR-Bench'):
    for candidate in vcrbench_candidates(env=env, default_dirs=default_dirs):
        resolved = resolve_hf_snapshot_path(candidate, dataset_name=dataset_name)
        if resolved and osp.exists(resolved):
            return resolved
    return None
