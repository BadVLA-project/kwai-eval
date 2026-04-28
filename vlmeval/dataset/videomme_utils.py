import os
import os.path as osp


DEFAULT_VIDEOMME_DIRS = (
    '/m2v_intern/xuboshen/zgw/LMUData/datasets/lmms-lab/Video-MME',
    '/m2v_intern/xuboshen/zgw/Benchmarks/Video-MME',
)

VIDEOMME_DIR_ENV_KEYS = ('VIDEO_MME_DIR', 'VIDEOMME_DIR')


def first_env(env, keys):
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def resolve_videomme_local_dir(env=None, default_dirs=DEFAULT_VIDEOMME_DIRS):
    env = os.environ if env is None else env
    local_dir = first_env(env, VIDEOMME_DIR_ENV_KEYS)
    if local_dir:
        return local_dir

    dirs = default_dirs if isinstance(default_dirs, (tuple, list)) else (default_dirs,)
    for candidate in dirs:
        if candidate and osp.exists(candidate):
            return candidate
    return None


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
