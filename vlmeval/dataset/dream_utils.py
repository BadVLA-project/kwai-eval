import os
import os.path as osp
import csv
import json
import ast


DEFAULT_DREAM1K_DIRS = (
    '/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DREAM-1K',
    '/m2v_intern/xuboshen/zgw/Benchmarks/DREAM-1K',
)
DEFAULT_DREAM1K_DIR = DEFAULT_DREAM1K_DIRS[0]
DEFAULT_DREAM1K_CACHE_DIR = '/m2v_intern/xuboshen/zgw/Benchmarks'

DREAM_DIR_ENV_KEYS = ('DREAM1K_DIR', 'DREAM_1K_DIR', 'DREAM_DIR')
DREAM_TSV_ENV_KEYS = ('DREAM1K_TSV', 'DREAM_1K_TSV')
DREAM_ANN_ENV_KEYS = ('DREAM1K_ANN', 'DREAM_1K_ANN', 'DREAM1K_JSONL')
DREAM_CACHE_ENV_KEYS = ('DREAM1K_CACHE_DIR', 'DREAM_1K_CACHE_DIR')


def first_env(env, keys):
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def has_mp4_files(path):
    return osp.isdir(path) and any(x.lower().endswith('.mp4') for x in os.listdir(path))


def resolve_dream_local_dir(env=None, default_dir=DEFAULT_DREAM1K_DIRS):
    env = os.environ if env is None else env
    local_dir = first_env(env, DREAM_DIR_ENV_KEYS)
    if local_dir:
        return local_dir
    default_dirs = default_dir if isinstance(default_dir, (tuple, list)) else (default_dir,)
    for candidate in default_dirs:
        if osp.exists(candidate):
            return candidate
    return None


def resolve_dream_annotation_file(lmu_root, local_dir=None, env=None):
    env = os.environ if env is None else env
    explicit = first_env(env, DREAM_TSV_ENV_KEYS + DREAM_ANN_ENV_KEYS)
    candidates = []
    if explicit:
        candidates.append(explicit)
    if local_dir:
        candidates.extend([
            osp.join(local_dir, 'dream1k_bench.jsonl'),
            osp.join(local_dir, 'DREAM-1K.jsonl'),
            osp.join(local_dir, 'DREAM-1k.jsonl'),
            osp.join(local_dir, 'annotations', 'dream1k_bench.jsonl'),
            osp.join(local_dir, 'annotation', 'dream1k_bench.jsonl'),
            osp.join(local_dir, 'DREAM-1K.tsv'),
            osp.join(local_dir, 'DREAM-1k.tsv'),
            osp.join(local_dir, 'annotations', 'DREAM-1K.tsv'),
            osp.join(local_dir, 'annotation', 'DREAM-1K.tsv'),
        ])
    candidates.append(osp.join(lmu_root, 'DREAM-1K.tsv'))

    for candidate in candidates:
        if candidate and osp.exists(candidate):
            return candidate
    return candidates[-1]


def resolve_dream_tsv(lmu_root, local_dir=None, env=None):
    return resolve_dream_annotation_file(lmu_root, local_dir=local_dir, env=env)


def resolve_dream_cache_dir(lmu_root=None, env=None):
    env = os.environ if env is None else env
    return first_env(env, DREAM_CACHE_ENV_KEYS) or DEFAULT_DREAM1K_CACHE_DIR or lmu_root


def resolve_dream_converted_tsv(lmu_root=None, env=None):
    cache_dir = resolve_dream_cache_dir(lmu_root=lmu_root, env=env)
    return osp.join(cache_dir, 'DREAM-1K.from_jsonl.tsv')


def _text_from_messages(row, role, prefer_reference=False):
    for message in row.get('messages', []):
        if message.get('role') != role:
            continue
        for content in message.get('content', []):
            if content.get('type') != 'text':
                continue
            if prefer_reference and content.get('reference'):
                return content.get('reference')
            if content.get('text'):
                return content.get('text')
            if content.get('reference'):
                return content.get('reference')
    return ''


def _clean_text(value):
    if value is None:
        return ''
    value = str(value)
    if value in {'', '<placeholder>', 'placeholder', 'None', 'nan'}:
        return ''
    return value


def _first_nonempty(*values):
    for value in values:
        if isinstance(value, str):
            value = _clean_text(value)
        if value is not None and value != '':
            return value
    return ''


def _first_nested_value(obj, keys):
    if isinstance(obj, dict):
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str):
                value = _clean_text(value)
            if value is not None and value != '':
                return value
        for value in obj.values():
            found = _first_nested_value(value, keys)
            if found != '':
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _first_nested_value(item, keys)
            if found != '':
                return found
    return ''


def _normalize_video_ref(value):
    value = _clean_text(value)
    if not value:
        return ''
    value = value.replace('\\', '/')
    if value.startswith('video/'):
        return value
    if value.lower().endswith('.mp4'):
        return f'video/{osp.basename(value)}'
    return value


def _video_from_messages(row):
    for message in row.get('messages', []):
        for content in message.get('content', []):
            if content.get('type') != 'video':
                continue
            video = content.get('video')
            if isinstance(video, dict):
                return _first_nonempty(video.get('video_file'), video.get('path'), video.get('value'))
            if isinstance(video, str):
                return _clean_text(video)
    return ''


def _serialize_events(events):
    if isinstance(events, (list, dict)):
        return json.dumps(events, ensure_ascii=False)
    if events is None:
        return ''
    if isinstance(events, str):
        events = events.strip()
        if not events:
            return ''
        if events[0] in '[{':
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(events)
                    if isinstance(parsed, (list, dict)):
                        return json.dumps(parsed, ensure_ascii=False)
                except Exception:
                    pass
        return events
    return str(events)


def normalize_dream_jsonl_row(row, row_idx=0):
    idx = row.get('index', row.get('idx', row.get('vid', row_idx)))
    video = _normalize_video_ref(_first_nonempty(
        row.get('video'),
        row.get('video_file'),
        row.get('video_name'),
        row.get('video_path'),
        row.get('video_frame_dir'),
        row.get('frame_dir'),
        row.get('file'),
        row.get('path'),
        _video_from_messages(row),
        f'video/{row.get("vid", idx)}.mp4',
    ))
    question = _first_nonempty(
        row.get('question'),
        row.get('prompt'),
        row.get('instruction'),
        row.get('query'),
        _text_from_messages(row, 'user'),
        'Describe the video in detail.',
    )
    answer = _first_nonempty(
        row.get('answer'),
        row.get('response'),
        row.get('caption'),
        row.get('GT_description'),
        row.get('gt_description'),
        row.get('description'),
        row.get('video_description'),
        row.get('detailed_caption'),
        row.get('gt_caption'),
        _text_from_messages(row, 'assistant', prefer_reference=True),
        _first_nested_value(
            row,
            (
                'answer',
                'response',
                'caption',
                'GT_description',
                'gt_description',
                'description',
                'video_description',
                'detailed_caption',
                'gt_caption',
                'reference',
            ),
        ),
    )
    events = _first_nonempty(
        row.get('events'),
        row.get('event'),
        row.get('event_list'),
        row.get('key_events'),
        row.get('GT_events'),
        row.get('gt_events'),
        row.get('extra_info', {}).get('events') if isinstance(row.get('extra_info'), dict) else None,
        _first_nested_value(row, ('events', 'event_list', 'key_events', 'GT_events', 'gt_events')),
    )

    return {
        'index': idx,
        'video': video,
        'question': question,
        'answer': answer,
        'events': _serialize_events(events),
    }


def convert_dream_jsonl_to_tsv(jsonl_path, tsv_path):
    os.makedirs(osp.dirname(osp.abspath(tsv_path)), exist_ok=True)
    with open(jsonl_path, encoding='utf-8') as fin, open(tsv_path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=['index', 'video', 'question', 'answer', 'events'],
            delimiter='\t',
        )
        writer.writeheader()
        count = 0
        for row_idx, line in enumerate(fin):
            if not line.strip():
                continue
            writer.writerow(normalize_dream_jsonl_row(json.loads(line), row_idx=row_idx))
            count += 1
    return tsv_path, count


def resolve_dream_video_source(local_dir, default_root):
    if local_dir:
        root_candidates = [
            osp.join(local_dir, 'video'),
            osp.join(local_dir, 'videos'),
            osp.join(local_dir, 'DREAM-1K'),
            local_dir,
        ]
        for root in root_candidates:
            if has_mp4_files(root):
                return root, None

        zip_candidates = [
            osp.join(local_dir, 'video', 'video.zip'),
            osp.join(local_dir, 'videos', 'video.zip'),
            osp.join(local_dir, 'video.zip'),
        ]
        for zip_path in zip_candidates:
            if osp.exists(zip_path):
                extract_root = osp.dirname(zip_path)
                return extract_root, zip_path

        return osp.join(local_dir, 'video'), osp.join(local_dir, 'video', 'video.zip')

    return default_root, osp.join(default_root, 'video.zip')
