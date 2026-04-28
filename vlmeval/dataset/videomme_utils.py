import os
import os.path as osp
from glob import glob


DEFAULT_VIDEOMME_DIRS = (
    '/m2v_intern/xuboshen/zgw/LMUData/datasets/lmms-lab/Video-MME',
    '/m2v_intern/xuboshen/zgw/Benchmarks/Video-MME',
)

VIDEOMME_DIR_ENV_KEYS = ('VIDEO_MME_DIR', 'VIDEOMME_DIR')
VIDEOMME_CACHE_ROOT_ENV_KEYS = ('HUGGINGFACE_HUB_CACHE', 'HF_HOME')


def first_env(env, keys):
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def has_videomme_metadata(path):
    return (
        osp.exists(osp.join(path, 'Video-MME.tsv'))
        or osp.exists(osp.join(path, 'videomme', 'test-00000-of-00001.parquet'))
        or osp.exists(osp.join(path, 'test-00000-of-00001.parquet'))
    )


def resolve_hf_snapshot_path(path):
    if not path:
        return None
    if has_videomme_metadata(path):
        return path

    snapshots = glob(osp.join(path, 'snapshots', '*'))
    snapshots = [p for p in snapshots if osp.isdir(p)]
    with_metadata = [p for p in snapshots if has_videomme_metadata(p)]
    if with_metadata:
        with_metadata.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return with_metadata[0]
    if snapshots:
        snapshots.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return snapshots[0]
    return path if osp.exists(path) else None


def videomme_cache_candidates(env=None):
    env = os.environ if env is None else env
    candidates = []
    explicit = first_env(env, VIDEOMME_DIR_ENV_KEYS)
    if explicit:
        candidates.append(explicit)

    for root_key in VIDEOMME_CACHE_ROOT_ENV_KEYS:
        root = env.get(root_key)
        if not root:
            continue
        candidates.extend([
            osp.join(root, 'datasets--lmms-lab--Video-MME'),
            osp.join(root, 'hub', 'datasets--lmms-lab--Video-MME'),
        ])

    candidates.extend(DEFAULT_VIDEOMME_DIRS)
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_videomme_local_dir(env=None, default_dirs=DEFAULT_VIDEOMME_DIRS):
    env = os.environ if env is None else env
    candidates = videomme_cache_candidates(env)
    dirs = default_dirs if isinstance(default_dirs, (tuple, list)) else (default_dirs,)
    for candidate in dirs:
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        resolved = resolve_hf_snapshot_path(candidate)
        if resolved and osp.exists(resolved):
            return resolved
    return None


def is_hf_snapshot_path(path):
    parts = osp.normpath(path).split(os.sep)
    return 'snapshots' in parts and any(part.startswith('datasets--') for part in parts)


def find_videomme_video_dir(root):
    candidates = [
        osp.join(root, 'video'),
        osp.join(root, 'videos'),
    ]
    for candidate in candidates:
        if osp.isdir(candidate) and os.listdir(candidate):
            return candidate
    for candidate in candidates:
        if osp.isdir(candidate):
            return candidate
    return candidates[0]


def resolve_videomme_video_path(root, video_id):
    video_id = str(video_id)
    if video_id.lower().endswith('.mp4'):
        stem = video_id[:-4]
    else:
        stem = video_id

    candidates = [
        osp.join(root, 'video', f'{stem}.mp4'),
        osp.join(root, 'videos', f'{stem}.mp4'),
        osp.join(root, 'video', f'{stem}.MP4'),
        osp.join(root, 'videos', f'{stem}.MP4'),
        osp.join(root, video_id),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return candidates[0]


def videomme_video_relpath(root, video_id):
    video_path = resolve_videomme_video_path(root, video_id)
    try:
        rel = osp.relpath(video_path, root)
    except ValueError:
        rel = osp.join('video', f'{video_id}.mp4')
    return './' + rel.replace(os.sep, '/')
