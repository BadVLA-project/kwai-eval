import os
import os.path as osp
from dataclasses import dataclass
from glob import glob


DEFAULT_VIDEOMME_V2_SOURCE_DIRS = (
    '/ytech_m2v5_hdd/workspace/kling_mm/Datasets/Video-MME-v2',
    '/m2v_intern/xuboshen/zgw/Benchmarks/Video-MME-v2',
)

DEFAULT_VIDEOMME_V2_ARTIFACT_DIR = '/m2v_intern/xuboshen/zgw/Benchmarks/Video-MME-v2'

VIDEOMME_V2_SOURCE_ENV_KEYS = ('VIDEO_MME_V2_DIR', 'VIDEOMME_V2_DIR')
VIDEOMME_V2_ARTIFACT_ENV_KEYS = ('VIDEO_MME_V2_ARTIFACT_DIR', 'VIDEOMME_V2_ARTIFACT_DIR')


@dataclass(frozen=True)
class VideoMMEv2Paths:
    source_root: str
    artifact_root: str
    tsv_file: str
    parquet_file: str
    video_dir: str
    subtitle_zip: str
    subtitle_dir: str


def first_env(env, keys):
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def has_videommev2_metadata(path):
    return (
        osp.exists(osp.join(path, 'Video-MME-v2.tsv'))
        or osp.exists(osp.join(path, 'test.parquet'))
        or osp.exists(osp.join(path, 'test-00000-of-00001.parquet'))
        or osp.exists(osp.join(path, 'videommev2', 'test-00000-of-00001.parquet'))
        or osp.exists(osp.join(path, 'data', 'test-00000-of-00001.parquet'))
    )


def resolve_hf_snapshot_path(path):
    if not path:
        return None
    if has_videommev2_metadata(path):
        return path

    snapshots = glob(osp.join(path, 'snapshots', '*'))
    snapshots = [p for p in snapshots if osp.isdir(p)]
    with_metadata = [p for p in snapshots if has_videommev2_metadata(p)]
    if with_metadata:
        with_metadata.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return with_metadata[0]
    if snapshots:
        snapshots.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return snapshots[0]
    return path if osp.exists(path) else None


def videommev2_source_candidates(env=None, default_dirs=DEFAULT_VIDEOMME_V2_SOURCE_DIRS):
    env = os.environ if env is None else env
    candidates = []
    explicit = first_env(env, VIDEOMME_V2_SOURCE_ENV_KEYS)
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


def resolve_videommev2_source_dir(env=None, default_dirs=DEFAULT_VIDEOMME_V2_SOURCE_DIRS):
    for candidate in videommev2_source_candidates(env=env, default_dirs=default_dirs):
        resolved = resolve_hf_snapshot_path(candidate)
        if resolved and osp.exists(resolved):
            return resolved
    return None


def resolve_videommev2_artifact_dir(env=None, default_dir=DEFAULT_VIDEOMME_V2_ARTIFACT_DIR):
    env = os.environ if env is None else env
    return first_env(env, VIDEOMME_V2_ARTIFACT_ENV_KEYS) or default_dir


def find_videommev2_parquet(root):
    candidates = [
        osp.join(root, 'test.parquet'),
        osp.join(root, 'test-00000-of-00001.parquet'),
        osp.join(root, 'videommev2', 'test-00000-of-00001.parquet'),
        osp.join(root, 'data', 'test-00000-of-00001.parquet'),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate

    for walk_root, _dirs, files in os.walk(root):
        for filename in files:
            if filename.endswith('.parquet'):
                return osp.join(walk_root, filename)
    return None


def find_videommev2_video_dir(root):
    candidates = [
        osp.join(root, 'videos'),
        osp.join(root, 'video'),
    ]
    for candidate in candidates:
        if osp.isdir(candidate) and os.listdir(candidate):
            return candidate
    for candidate in candidates:
        if osp.isdir(candidate):
            return candidate
    return candidates[0]


def resolve_videommev2_video_path(root, video_id):
    video_id = str(video_id)
    stem = video_id[:-4] if video_id.lower().endswith('.mp4') else video_id
    candidates = [
        osp.join(root, 'videos', f'{stem}.mp4'),
        osp.join(root, 'video', f'{stem}.mp4'),
        osp.join(root, 'videos', f'{stem}.MP4'),
        osp.join(root, 'video', f'{stem}.MP4'),
        osp.join(root, video_id),
    ]
    for candidate in candidates:
        if osp.exists(candidate):
            return candidate
    return candidates[0]


def videommev2_video_relpath(root, video_id):
    video_path = resolve_videommev2_video_path(root, video_id)
    try:
        rel = osp.relpath(video_path, root)
    except ValueError:
        rel = osp.join('videos', f'{video_id}.mp4')
    return './' + rel.replace(os.sep, '/')


def resolve_videommev2_paths(env=None, dataset_name='Video-MME-v2'):
    env = os.environ if env is None else env
    source_root = resolve_videommev2_source_dir(env=env)
    if source_root is None:
        return None

    artifact_root = resolve_videommev2_artifact_dir(env=env)
    parquet_file = find_videommev2_parquet(source_root)
    subtitle_zip = osp.join(source_root, 'subtitle.zip')

    return VideoMMEv2Paths(
        source_root=source_root,
        artifact_root=artifact_root,
        tsv_file=osp.join(artifact_root, f'{dataset_name}.tsv'),
        parquet_file=parquet_file,
        video_dir=find_videommev2_video_dir(source_root),
        subtitle_zip=subtitle_zip if osp.exists(subtitle_zip) else '',
        subtitle_dir=osp.join(artifact_root, 'subtitle'),
    )
