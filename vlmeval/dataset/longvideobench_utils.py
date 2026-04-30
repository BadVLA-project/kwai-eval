import os
import os.path as osp
from glob import glob


DEFAULT_LONGVIDEOBENCH_DIRS = (
    '/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/datasets--longvideobench--LongVideoBench',
    '/m2v_intern/xuboshen/zgw/Benchmarks/LongVideoBench',
)

LONGVIDEOBENCH_DIR_ENV_KEYS = ('LONGVIDEOBENCH_DIR', 'LONG_VIDEO_BENCH_DIR', 'LONGVIDEOBENCH_CACHE_DIR')


def first_env(env, keys):
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def has_longvideobench_metadata(path):
    return (
        osp.exists(osp.join(path, 'LongVideoBench.tsv'))
        or osp.exists(osp.join(path, 'lvb_val.json'))
    )


def resolve_hf_snapshot_path(path):
    if not path:
        return None
    if has_longvideobench_metadata(path):
        return path

    snapshots = glob(osp.join(path, 'snapshots', '*'))
    snapshots = [p for p in snapshots if osp.isdir(p)]
    with_metadata = [p for p in snapshots if has_longvideobench_metadata(p)]
    if with_metadata:
        with_metadata.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return with_metadata[0]
    if snapshots:
        snapshots.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return snapshots[0]
    return path if osp.exists(path) else None


def longvideobench_candidates(env=None, default_dirs=DEFAULT_LONGVIDEOBENCH_DIRS):
    env = os.environ if env is None else env
    candidates = []
    explicit = first_env(env, LONGVIDEOBENCH_DIR_ENV_KEYS)
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


def resolve_longvideobench_dir(env=None, default_dirs=DEFAULT_LONGVIDEOBENCH_DIRS):
    for candidate in longvideobench_candidates(env=env, default_dirs=default_dirs):
        resolved = resolve_hf_snapshot_path(candidate)
        if resolved and osp.exists(resolved):
            return resolved
    return None
