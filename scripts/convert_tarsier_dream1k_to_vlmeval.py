#!/usr/bin/env python3
import argparse
import csv
import json
import os
import os.path as osp


def video_id_from_value(value):
    value = str(value).replace('\\', '/')
    value = value.split('/')[-1]
    if value.lower().endswith('.mp4'):
        value = value[:-4]
    return value


def load_index_map(annotation_path):
    index_map = {}
    if not annotation_path or not osp.exists(annotation_path):
        return index_map
    if annotation_path.endswith('.jsonl'):
        with open(annotation_path, encoding='utf-8') as f:
            for row_num, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = row.get('index', row.get('idx', row.get('vid', row_num)))
                video = row.get('video') or row.get('video_name') or row.get('video_path') or row.get('vid') or idx
                index_map[video_id_from_value(video)] = idx
    else:
        with open(annotation_path, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row_num, row in enumerate(reader):
                idx = row.get('index', row_num)
                video = row.get('video', '')
                if video:
                    index_map[video_id_from_value(video)] = idx
    return index_map


def extract_prediction(row):
    messages = row.get('messages', [])
    if not messages:
        return ''
    for message in reversed(messages):
        if message.get('role') == 'assistant':
            for content in reversed(message.get('content', [])):
                if content.get('type') == 'text':
                    return content.get('text', '')
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-jsonl', required=True)
    parser.add_argument(
        '--dream-ann',
        default=os.environ.get(
            'DREAM1K_ANN',
            '/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DREAM-1K/dream1k_bench.jsonl',
        ),
    )
    parser.add_argument('--output-tsv', required=True)
    args = parser.parse_args()

    index_map = load_index_map(args.dream_ann)
    os.makedirs(osp.dirname(osp.abspath(args.output_tsv)), exist_ok=True)

    rows = []
    with open(args.pred_jsonl, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            vid = str(row.get('vid', row.get('idx', '')))
            rows.append({
                'index': index_map.get(vid, row.get('idx', vid)),
                'prediction': extract_prediction(row),
            })

    with open(args.output_tsv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'prediction'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} VLMEval DREAM-1K predictions to {args.output_tsv}')


if __name__ == '__main__':
    main()
