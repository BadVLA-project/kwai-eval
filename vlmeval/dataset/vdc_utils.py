import os
import os.path as osp
import json
from glob import glob


DEFAULT_VDC_CACHE_ROOT = '/m2v_intern/xuboshen/zgw/hf_cache_temp/datasets--Enxin--VLMEval-VDC'
VDC_ENV_KEYS = ('VDC_DIR', 'VDC_CACHE_DIR', 'VLMEVAL_VDC_DIR')
VDC_SPLIT_TYPES = {
    'short_test': 'short',
    'detailed_test': 'detailed',
    'background_test': 'background',
    'main_object_test': 'main_object',
    'camera_test': 'camera',
}


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
    if snapshots:
        snapshots.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return snapshots[0]
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


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def _read_split_table(split_path):
    import pandas as pd

    if osp.isdir(split_path):
        parquet_files = sorted(glob(osp.join(split_path, '**', '*.parquet'), recursive=True))
        if parquet_files:
            return pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)

        jsonl_files = sorted(glob(osp.join(split_path, '**', '*.jsonl'), recursive=True))
        if jsonl_files:
            frames = [pd.read_json(path, lines=True) for path in jsonl_files]
            return pd.concat(frames, ignore_index=True)

        if osp.exists(osp.join(split_path, 'dataset_info.json')) or osp.exists(osp.join(split_path, 'state.json')):
            try:
                from datasets import load_from_disk
                return load_from_disk(split_path).to_pandas()
            except Exception as err:
                raise FileNotFoundError(f'Cannot read VDC split metadata under {split_path}: {err}') from err

        json_files = sorted(glob(osp.join(split_path, '**', '*.json'), recursive=True))
        if json_files:
            frames = [pd.read_json(path) for path in json_files]
            return pd.concat(frames, ignore_index=True)

        csv_files = sorted(glob(osp.join(split_path, '**', '*.csv'), recursive=True))
        if csv_files:
            frames = [pd.read_csv(path) for path in csv_files]
            return pd.concat(frames, ignore_index=True)

        tsv_files = sorted(glob(osp.join(split_path, '**', '*.tsv'), recursive=True))
        if tsv_files:
            frames = [pd.read_csv(path, sep='\t') for path in tsv_files]
            return pd.concat(frames, ignore_index=True)

    suffix = osp.splitext(split_path)[1].lower()
    if suffix == '.parquet':
        return pd.read_parquet(split_path)
    if suffix == '.jsonl':
        return pd.read_json(split_path, lines=True)
    if suffix == '.json':
        return pd.read_json(split_path)
    if suffix == '.csv':
        return pd.read_csv(split_path)
    if suffix == '.tsv':
        return pd.read_csv(split_path, sep='\t')
    raise FileNotFoundError(f'Unsupported VDC split metadata file: {split_path}')


def _row_value(row, keys):
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        try:
            import pandas as pd
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if value is not None:
            return value
    return None


def _normalize_video_name(value):
    value = '' if value is None else str(value)
    value = value.strip()
    if not value:
        return value
    value = value.replace('\\', '/')
    if value.startswith('videos/'):
        value = value[len('videos/'):]
    if value.startswith('video/'):
        value = value[len('video/'):]
    if not osp.splitext(value)[1]:
        value = f'{value}.mp4'
    return value


def _serialize_questions(value):
    if value is None:
        return '[]'
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def build_vdc_tsv_from_splits(dataset_path, dataset_name='VDC', output_file=None):
    import pandas as pd

    rows = []
    for split_name, caption_type in VDC_SPLIT_TYPES.items():
        split_path = osp.join(dataset_path, split_name)
        if not osp.exists(split_path):
            continue
        data = _read_split_table(split_path)
        for _, row in data.iterrows():
            video = _normalize_video_name(_row_value(row, ['video', 'video_name', 'video_path', 'vid']))
            questions = _row_value(row, ['question', 'qa_list', 'questions', 'qa'])
            caption = _row_value(row, ['caption', 'answer', 'description'])
            rows.append({
                'index': len(rows),
                'video': video,
                'question': _serialize_questions(questions),
                'caption': '' if caption is None else str(caption),
                'caption_type': caption_type,
            })

    if not rows:
        split_list = ', '.join(VDC_SPLIT_TYPES)
        raise FileNotFoundError(
            f'Cannot build VDC TSV: no official split metadata ({split_list}) found under {dataset_path}'
        )

    output_file = output_file or osp.join(dataset_path, f'{dataset_name}.tsv')
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_file, sep='\t', index=False)
    return output_file


def ensure_vdc_data_file(dataset_path, dataset_name='VDC'):
    data_file = find_vdc_data_file(dataset_path, dataset_name)
    if osp.exists(data_file):
        return data_file
    return build_vdc_tsv_from_splits(dataset_path, dataset_name=dataset_name, output_file=data_file)


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
