import ast
import json
import logging
import os
import os.path as osp
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image

from ..smp import dump, get_cache_path, get_intermediate_file_path, load, modelscope_flag_set
from .video_base import VideoBaseDataset


def _read_split_parquet(root, split_name):
    root = Path(root)
    candidates = []
    patterns = [
        f'data/{split_name}-*.parquet',
        f'{split_name}/test-*.parquet',
        f'{split_name}/*.parquet',
        f'data/*{split_name}*.parquet',
        f'**/*{split_name}*.parquet',
    ]
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    candidates = sorted({x for x in candidates if x.is_file()})
    if not candidates:
        raise FileNotFoundError(f'Cannot find parquet split {split_name!r} under {root}')
    return pd.concat([pd.read_parquet(x) for x in candidates], ignore_index=True)


def _as_text(value, default=''):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        return default
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _segment_label(value):
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                value = parsed
        except Exception:
            return re.sub(r'[^A-Za-z0-9_.-]+', '_', value.strip())
    if isinstance(value, (list, tuple, np.ndarray)):
        return '_'.join(str(x) for x in list(value))
    return str(value)


def _video_path_from_row(row, *keys):
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(value, dict):
            value = value.get('path') or value.get('video_path') or value.get('filename')
        if isinstance(value, bytes):
            continue
        value = _as_text(value)
        if value:
            return value
    return ''


def _video_stem(video_path):
    return Path(str(video_path)).stem


def _safe_video_id(value):
    value = _as_text(value)
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')
    return value or 'video'


class _HFVideoCaptionDataset(VideoBaseDataset):
    TYPE = 'Video-VQA'

    def __init__(self, dataset, nframe=0, fps=-1, adaptive=False):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)

    def _candidate_video_paths(self, line):
        rel_path = _as_text(line.get('video_path', ''))
        if osp.isabs(rel_path):
            yield rel_path
            return

        if rel_path:
            yield osp.join(self.data_root, rel_path)

        video = _as_text(line.get('video', ''))
        if video:
            yield osp.join(self.data_root, video + '.mp4')

    def video_path(self, line):
        candidates = list(self._candidate_video_paths(line))
        for path in candidates:
            if osp.exists(path):
                return path
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f'No video path column found for sample {line}')

    def save_video_frames(self, line):
        video_id = _as_text(line['video'])
        vid_path = self.video_path(line)
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.adaptive:
            indices = self.compute_adaptive_indices(vid)
            frame_paths = self.frame_paths_adaptive(video_id, len(indices))
            strategy = getattr(self, '_last_adaptive_strategy', 'adaptive')
        elif self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_id)
            strategy = f'uniform nframe={self.nframe}'
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = max(1, int(total_duration * self.fps))
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_id, len(indices))
            strategy = f'{self.fps}fps, dur={total_duration:.1f}s'
        else:
            raise ValueError('fps and nframe should be set at least one valid value')

        indices = [min(i, len(vid) - 1) for i in indices]
        self._record_frame_info(video_id, len(indices), strategy)
        if np.all([osp.exists(p) for p in frame_paths]):
            logging.info(f'[frames] {video_id}: {len(frame_paths)} frames ({strategy}) [cached]')
            return frame_paths

        import portalocker
        lock_path = osp.splitext(vid_path)[0] + '.lock'
        with portalocker.Lock(lock_path, 'w', timeout=30):
            if np.all([osp.exists(p) for p in frame_paths]):
                logging.info(f'[frames] {video_id}: {len(frame_paths)} frames ({strategy}) [cached]')
                return frame_paths
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)
        logging.info(f'[frames] {video_id}: {len(frame_paths)} frames ({strategy}) [extracted]')
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_path = self.video_path(line)
        message = []
        if video_llm:
            message.append(self.make_video_struct(video_path, video_id=_as_text(line['video'])))
        else:
            for frame_path in self.save_video_frames(line):
                message.append(dict(type='image', value=frame_path))
        message.append(dict(type='text', value=_as_text(line['question'])))
        return message

    @classmethod
    def _download_dataset(cls, repo_id):
        if modelscope_flag_set():
            from modelscope import dataset_snapshot_download
            return dataset_snapshot_download(dataset_id=repo_id)
        return snapshot_download(repo_id=repo_id, repo_type='dataset')

    def prepare_dataset(self, dataset_name, repo_id=None):
        repo_id = repo_id or self.HF_REPO_ID
        cache_path = get_cache_path(repo_id)
        if cache_path is None:
            dataset_path = self._download_dataset(repo_id)
        else:
            dataset_path = cache_path
        try:
            data_file = self._generate_tsv(dataset_path)
        except FileNotFoundError:
            if cache_path is None:
                raise
            dataset_path = self._download_dataset(repo_id)
            data_file = self._generate_tsv(dataset_path)
        return dict(root=dataset_path, data_file=data_file)


class YouCook2Caption(_HFVideoCaptionDataset):
    HF_REPO_ID = 'lmms-lab/YouCook2'
    PROMPT = 'Provide a one-sentence caption for the provided video.'
    COCO_METRICS = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']

    def __init__(self, dataset='YouCook2', nframe=0, fps=-1, adaptive=False):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)

    @classmethod
    def supported_datasets(cls):
        return ['YouCook2', 'YouCook2_Captioning']

    @classmethod
    def _generate_tsv(cls, dataset_path, force=False):
        data_file = osp.join(dataset_path, 'YouCook2.tsv')
        if osp.exists(data_file) and not force:
            return data_file

        data = _read_split_parquet(dataset_path, 'val')
        rows = []
        for idx, row in data.iterrows():
            row = row.to_dict()
            video_path = _video_path_from_row(row, 'video_path', 'video', 'path')
            youtube_id = _as_text(row.get('youtube_id')) or _video_stem(video_path)
            segment = row.get('segment', '')
            suffix = _segment_label(segment)
            video_id = _safe_video_id(f'{youtube_id}_{suffix}' if suffix else youtube_id)
            rows.append({
                'index': idx,
                'video': video_id,
                'video_path': video_path,
                'question': cls.PROMPT,
                'answer': _as_text(row.get('sentence')),
                'youtube_id': youtube_id,
                'segment': _as_text(segment),
            })
        pd.DataFrame(rows).to_csv(data_file, sep='\t', index=False)
        return data_file

    def _candidate_video_paths(self, line):
        yield from super()._candidate_video_paths(line)
        rel_path = _as_text(line.get('video_path', ''))
        hf_home = os.environ.get('HF_HOME', osp.expanduser('~/.cache/huggingface'))
        for prefix in [
            osp.join(hf_home, 'YouCookIIVideos'),
            osp.join(hf_home, 'YouCookIIVideos', 'YouCookIIVideos'),
        ]:
            if rel_path:
                yield osp.join(prefix, rel_path)
                yield osp.join(prefix, osp.basename(rel_path))

    @classmethod
    def evaluate(cls, eval_file, **kwargs):
        data = load(eval_file)
        refs, gts, vid2capid = {}, {}, {}
        for i, line in enumerate([data.iloc[x] for x in range(len(data))]):
            pred = _as_text(line.get('prediction'))
            answer = _as_text(line.get('answer'))
            video = _as_text(line.get('video', i))
            refs[i] = [{'caption': pred}]
            gts[i] = [{'caption': answer}]
            vid2capid.setdefault(video, []).append(i)

        scores = cls._compute_coco_scores(refs, gts, vid2capid)
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(scores, score_file)
        return scores

    @classmethod
    def _compute_coco_scores(cls, refs, gts, vid2capid):
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        try:
            tokenizer = PTBTokenizer(verbose=False)
        except TypeError:
            tokenizer = PTBTokenizer()
        refs = tokenizer.tokenize(refs)
        gts = tokenizer.tokenize(gts)

        scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Meteor(), 'METEOR'),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
        ]
        all_scores = {k: [] for k in cls.COCO_METRICS}
        for cap_ids in vid2capid.values():
            cur_refs = {idx: refs[idx] for idx in cap_ids}
            cur_gts = {idx: gts[idx] for idx in cap_ids}
            for scorer, method in scorers:
                score, _ = scorer.compute_score(cur_gts, cur_refs)
                if isinstance(method, list):
                    for m, s in zip(method, score):
                        all_scores[m].append(float(s))
                else:
                    all_scores[method].append(float(score))
        return {k: round(float(np.mean(v)) * 100, 4) if v else 0.0 for k, v in all_scores.items()}


class TemporalBenchCaption(_HFVideoCaptionDataset):
    HF_REPO_ID = 'microsoft/TemporalBench'
    SPLIT = 'test_short_caption'

    def __init__(self, dataset='TemporalBench_Captioning', nframe=0, fps=-1, adaptive=False):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)

    @classmethod
    def supported_datasets(cls):
        return ['TemporalBench_Captioning', 'TemporalBench-Cap']

    def _candidate_video_paths(self, line):
        yield from super()._candidate_video_paths(line)
        rel_path = _as_text(line.get('video_path', ''))
        hf_home = os.environ.get('HF_HOME', osp.expanduser('~/.cache/huggingface'))
        for prefix in [osp.join(hf_home, 'temporalbench')]:
            if rel_path:
                yield osp.join(prefix, rel_path)
                yield osp.join(prefix, osp.basename(rel_path))

    @classmethod
    def _generate_tsv(cls, dataset_path, force=False):
        data_file = osp.join(dataset_path, 'TemporalBench_Captioning.tsv')
        if osp.exists(data_file) and not force:
            return data_file

        data = _read_split_parquet(dataset_path, cls.SPLIT)
        rows = []
        for row_idx, row in data.iterrows():
            row = row.to_dict()
            raw_idx = row.get('idx', row_idx)
            video_path = _video_path_from_row(row, 'video_name', 'video_path', 'video')
            rows.append({
                'index': raw_idx,
                'video': _safe_video_id(_video_stem(video_path) or raw_idx),
                'video_path': video_path,
                'question': _as_text(row.get('question')),
                'answer': _as_text(row.get('GT', row.get('answer'))),
                'dataset': _as_text(row.get('dataset')),
                'category': _as_text(row.get('category')),
            })
        pd.DataFrame(rows).to_csv(data_file, sep='\t', index=False)
        return data_file

    @classmethod
    def evaluate(cls, eval_file, **kwargs):
        data = load(eval_file)
        refs = [_as_text(data.iloc[i].get('prediction')) for i in range(len(data))]
        gts = [_as_text(data.iloc[i].get('answer')) for i in range(len(data))]
        score = cls._sent_sim(refs, gts)
        result = {'temporalbench_score': round(score * 100, 4)}
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(result, score_file)
        return result

    @staticmethod
    def _sent_sim(refs, gts):
        if not refs:
            return 0.0
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError as exc:
            raise ImportError('TemporalBench_Captioning evaluation requires sentence_transformers.') from exc

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        combined = refs + gts
        device = None
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
        except Exception:
            device = None
        try:
            embeddings = model.encode(combined, convert_to_tensor=True, device=device)
        except TypeError:
            embeddings = model.encode(combined, convert_to_tensor=True)
        ref_embeddings = embeddings[:len(refs)]
        gt_embeddings = embeddings[len(refs):]
        cosine_scores = util.cos_sim(ref_embeddings, gt_embeddings).diagonal()
        return float(cosine_scores.mean().item())


DATASET_HF_IDS = {
    'YouCook2': YouCook2Caption.HF_REPO_ID,
    'TempCompass-Cap': 'lmms-lab/TempCompass',
    'TemporalBench-Cap': TemporalBenchCaption.HF_REPO_ID,
}
