#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp


SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
DEFAULT_TARSIER_ANN = osp.abspath(
    osp.join(SCRIPT_DIR, '..', '..', 'tarsier', 'data', 'annotations', 'DREAM-1k.jsonl')
)
DEFAULT_DREAM1K_DIR = '/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DREAM-1K'
LEGACY_DREAM1K_DIR = '/m2v_intern/xuboshen/zgw/Benchmarks/DREAM-1K'


def video_id(value):
    value = str(value).replace('\\', '/')
    value = value.split('/')[-1]
    if value.lower().endswith('.mp4'):
        value = value[:-4]
    return value


def find_video(root, vid):
    if isinstance(vid, str) and osp.exists(vid):
        return vid
    vid = video_id(vid)
    names = [f'{vid}.mp4', f'{vid}.MP4']
    dirs = [
        osp.join(root, 'video'),
        osp.join(root, 'videos'),
        root,
    ]
    for directory in dirs:
        for name in names:
            path = osp.join(directory, name)
            if osp.exists(path):
                return path
    raise FileNotFoundError(f'Cannot find video for vid={vid} under {root}')


def row_video_ref(row):
    return (
        row.get('vid')
        or row.get('video')
        or row.get('video_name')
        or row.get('video_path')
        or row.get('idx')
    )


def row_prompt(row):
    if row.get('question'):
        return row['question']
    if row.get('prompt'):
        return row['prompt']
    for message in row.get('messages', []):
        if message.get('role') == 'user':
            for content in message.get('content', []):
                if content.get('type') == 'text' and content.get('text'):
                    return content['text']
    return 'Describe the video in detail.'


def row_answer(row):
    for key in ('answer', 'response', 'caption'):
        if row.get(key):
            return row[key]
    for message in row.get('messages', []):
        if message.get('role') == 'assistant':
            for content in message.get('content', []):
                if content.get('reference'):
                    return content['reference']
                if content.get('text'):
                    return content['text']
    return ''


def ensure_tarsier_row(row):
    if 'messages' in row:
        return row
    idx = row.get('idx', row.get('index', row.get('vid')))
    extra_info = row.get('extra_info', {})
    if row.get('events') and 'events' not in extra_info:
        extra_info = dict(extra_info)
        extra_info['events'] = row['events']
    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': {'video_file': '<placeholder>'}, 'text': None, 'reference': None},
                    {'type': 'text', 'text': row_prompt(row), 'video': None, 'reference': None},
                ],
            },
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': '', 'reference': row_answer(row), 'video': None}],
            },
        ],
        'dataset': row.get('dataset', 'DREAM-1K'),
        'task': row.get('task', 'video/caption'),
        'idx': idx,
        'vid': row.get('vid', idx),
        'extra_info': extra_info,
    }


def patch_row(row, video_path):
    patched = json.loads(json.dumps(ensure_tarsier_row(row), ensure_ascii=False))
    for message in patched.get('messages', []):
        for content in message.get('content', []):
            if content.get('type') == 'video':
                content['video'] = {'video_file': video_path}
    patched.setdefault('extra_info', {})['dream_video_path'] = video_path
    return patched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tarsier-ann',
        default=None,
        help='Original Tarsier DREAM-1k jsonl with <placeholder> video fields.',
    )
    parser.add_argument(
        '--dream-root',
        default=os.environ.get('DREAM1K_DIR', DEFAULT_DREAM1K_DIR),
        help='Local DREAM-1K root downloaded by huggingface-cli.',
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=-1)
    args = parser.parse_args()

    if args.tarsier_ann is None:
        local_ann = osp.join(args.dream_root, 'dream1k_bench.jsonl')
        args.tarsier_ann = local_ann if osp.exists(local_ann) else DEFAULT_TARSIER_ANN
    if not osp.exists(args.dream_root) and osp.exists(LEGACY_DREAM1K_DIR):
        args.dream_root = LEGACY_DREAM1K_DIR

    rows = []
    with open(args.tarsier_ann, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[:args.limit]

    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for row in rows:
            vid = row_video_ref(row)
            video_path = find_video(args.dream_root, vid)
            f.write(json.dumps(patch_row(row, video_path), ensure_ascii=False) + '\n')

    print(f'Wrote {len(rows)} Tarsier DREAM-1K samples to {args.output}')


if __name__ == '__main__':
    main()
