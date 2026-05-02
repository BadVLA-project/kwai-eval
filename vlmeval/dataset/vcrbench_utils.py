import os
import os.path as osp
import zipfile
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


def resolve_vcrbench_video_path(root, video_path):
    if not root or not video_path:
        return None
    rel = str(video_path).strip()
    if not rel or rel.lower() == 'nan':
        return None
    if rel.startswith('./'):
        rel = rel[2:]
    if osp.isabs(rel):
        return rel if osp.exists(rel) else None

    candidates = [
        osp.join(root, rel),
        osp.join(root, 'v1', 'videos', rel),
        osp.join(root, 'v1', 'videos', 'video', rel),
    ]
    for prefix in ('v1/videos/video/', 'v1/videos/', 'videos/video/', 'videos/', 'video/'):
        if rel.startswith(prefix):
            tail = rel[len(prefix):]
            candidates.extend([
                osp.join(root, 'v1', 'videos', tail),
                osp.join(root, 'v1', 'videos', 'video', tail),
                osp.join(root, tail),
            ])

    seen = set()
    for candidate in candidates:
        norm = osp.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if osp.exists(norm):
            return norm
    return None


def ensure_vcrbench_videos_available(path):
    if not path:
        return False
    video_dir = osp.join(path, 'v1', 'videos', 'video')
    if osp.isdir(video_dir):
        return True
    video_zip = osp.join(path, 'v1', 'videos', 'video.zip')
    if not osp.exists(video_zip):
        return False
    try:
        with zipfile.ZipFile(video_zip) as zf:
            zf.extractall(osp.join(path, 'v1', 'videos'))
    except Exception:
        return False
    return osp.isdir(video_dir)


def vcrbench_local_integrity(path, dataset_name='VCR-Bench', max_rows=20):
    if not path or not has_vcrbench_metadata(path, dataset_name=dataset_name):
        return False
    data_file = osp.join(path, f'{dataset_name}.tsv')
    try:
        import pandas as pd
        data = pd.read_csv(data_file, sep='\t')
    except Exception:
        return False
    if 'video_path' not in data:
        return False
    ensure_vcrbench_videos_available(path)
    for video_pth in data['video_path'].head(max_rows):
        if resolve_vcrbench_video_path(path, video_pth) is None:
            return False
    return True


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
