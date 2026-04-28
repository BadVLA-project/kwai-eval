#!/usr/bin/env python3
import argparse
import csv
import os
import os.path as osp
import sys


SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
DEFAULT_AURORA_DIR = osp.abspath(osp.join(SCRIPT_DIR, '..', '..', 'aurora'))
DEFAULT_DREAM_TSV = '/m2v_intern/xuboshen/zgw/LMUData/datasets/DREAM-1K.tsv'
DEFAULT_DREAM_ROOT = '/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DREAM-1K'


def video_id_from_value(value):
    value = str(value).replace('\\', '/').split('/')[-1]
    if value.lower().endswith('.mp4'):
        value = value[:-4]
    return value


def find_video(root, video_value):
    video_id = video_id_from_value(video_value)
    dirs = [osp.join(root, 'video'), osp.join(root, 'videos'), root]
    names = [f'{video_id}.mp4', f'{video_id}.MP4']
    for directory in dirs:
        for name in names:
            path = osp.join(directory, name)
            if osp.exists(path):
                return path
    raise FileNotFoundError(f'Cannot find video={video_value} under {root}')


def process_text(inputs, tokenizer):
    import torch

    from src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    return torch.tensor(ids).cuda().unsqueeze(0)


def text_from_messages(row, role, prefer_reference=False):
    for message in row.get('messages', []):
        if message.get('role') != role:
            continue
        for content in message.get('content', []):
            if content.get('type') != 'text':
                continue
            if prefer_reference and content.get('reference'):
                return content['reference']
            if content.get('text'):
                return content['text']
            if content.get('reference'):
                return content['reference']
    return ''


def normalize_jsonl_row(row, row_idx):
    idx = row.get('index', row.get('idx', row.get('vid', row_idx)))
    video = row.get('video') or row.get('video_name') or row.get('video_path') or f'video/{row.get("vid", idx)}.mp4'
    question = row.get('question') or row.get('prompt') or text_from_messages(row, 'user') or 'Describe the video in detail.'
    answer = row.get('answer') or row.get('response') or row.get('caption') or text_from_messages(row, 'assistant', True)
    return {'index': idx, 'video': video, 'question': question, 'answer': answer}


def load_rows(annotation_path, limit):
    rows = []
    if annotation_path.endswith('.jsonl'):
        with open(annotation_path, encoding='utf-8') as f:
            for row_num, line in enumerate(f):
                if not line.strip():
                    continue
                import json
                rows.append(normalize_jsonl_row(json.loads(line), row_num))
                if limit > 0 and len(rows) >= limit:
                    break
    else:
        with open(annotation_path, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row_num, row in enumerate(reader):
                row.setdefault('index', str(row_num))
                rows.append(row)
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aurora-dir', default=os.environ.get('AURORA_DIR', DEFAULT_AURORA_DIR))
    parser.add_argument('--model-path', default=os.environ.get('AURORA_MODEL', 'wchai/AuroraCap-7B-VID-xtuner'))
    parser.add_argument('--dream-root', default=os.environ.get('DREAM1K_DIR', DEFAULT_DREAM_ROOT))
    parser.add_argument('--dream-ann', default=os.environ.get('DREAM1K_ANN', ''))
    parser.add_argument('--output-tsv', required=True)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--num-frm', type=int, default=8)
    parser.add_argument('--token-kept-ratio', type=float, default=0.8)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()

    sys.path.insert(0, args.aurora_dir)

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor

    from src.xtuner.xtuner.model.aurora import AuroraEncoder, AuroraModel
    from src.xtuner.xtuner.tools.load_video import read_video_pyav
    from src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, PROMPT_TEMPLATE

    pretrained_pth = snapshot_download(repo_id=args.model_path) if not osp.isdir(args.model_path) else args.model_path
    pretrained_vit = osp.join(pretrained_pth, 'visual_encoder')
    projector_path = osp.join(pretrained_pth, 'projector')

    auroracap = AuroraModel(
        llm=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_pth,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ),
        visual_encoder=AuroraEncoder.from_pretrained(
            pretrained_model_name_or_path=pretrained_vit,
            torch_dtype=torch.float16,
        ),
    ).cuda()
    auroracap.projector = AutoModel.from_pretrained(
        projector_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        trust_remote_code=True,
        size=378,
        crop_size=378,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_pth,
        trust_remote_code=True,
        padding_side='right',
    )
    auroracap.eval()

    dream_ann = args.dream_ann or osp.join(args.dream_root, 'dream1k_bench.jsonl')
    if not osp.exists(dream_ann):
        dream_ann = os.environ.get('DREAM1K_TSV', DEFAULT_DREAM_TSV)
    rows = load_rows(dream_ann, args.limit)
    os.makedirs(osp.dirname(osp.abspath(args.output_tsv)), exist_ok=True)
    with open(args.output_tsv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'prediction'], delimiter='\t')
        writer.writeheader()
        for row in rows:
            video_path = find_video(args.dream_root, row['video'])
            prompt = row.get('question') or 'Describe the video in detail.'
            video_frames = read_video_pyav(video_path, args.num_frm)
            image_tensor = image_processor(video_frames, return_tensors='pt')['pixel_values']
            image_tensor = [_image.to(dtype=torch.float16).cuda() for _image in image_tensor]
            image_tokens = ' '.join([DEFAULT_IMAGE_TOKEN] * len(video_frames))
            text_input = image_tokens + '\n' + prompt
            prompt_text = PROMPT_TEMPLATE.vicuna['INSTRUCTION'].format(input=text_input, round=1)
            data = {
                'pixel_values': torch.stack(image_tensor).unsqueeze(0),
                'input_ids': process_text(prompt_text, tokenizer).cuda(),
            }
            auroracap.visual_encoder.reset_tome_r(args.token_kept_ratio)
            with torch.inference_mode():
                output = auroracap(data, mode='inference')
                cont = auroracap.llm.generate(
                    **output,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                )
            prediction = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            writer.writerow({'index': row['index'], 'prediction': prediction})
            f.flush()

    print(f'Wrote Aurora DREAM-1K predictions to {args.output_tsv}')


if __name__ == '__main__':
    main()
