import os
import re
import json
import os.path as osp

import pandas as pd

from ..smp import *
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

_MANAGED_PROMPT_SENTINEL = {'type': '_managed_prompt'}

# ---------------------------------------------------------------------------
#  Sub-dataset registry
# ---------------------------------------------------------------------------

_TIMELENS_DEFAULT_DIR = '/m2v_intern/xuboshen/zgw/hf_cache_temp/TimeLens-Bench'

_DATASET_CONFIG = {
    'TimeLensBench_Charades': {
        'json_file': 'charades-timelens.json',
        'video_subdir': 'videos/charades',
        'shard_subdir': 'video_shards/charades',
        'unit': 1.0,
    },
    'TimeLensBench_ActivityNet': {
        'json_file': 'activitynet-timelens.json',
        'video_subdir': 'videos/activitynet',
        'shard_subdir': 'video_shards/activitynet',
        'unit': 1.0,
    },
    'TimeLensBench_QVHighlights': {
        'json_file': 'qvhighlights-timelens.json',
        'video_subdir': 'videos/qvhighlights',
        'shard_subdir': 'video_shards/qvhighlights',
        'unit': 1.0,
    },
}

# ---------------------------------------------------------------------------
#  Prompt templates
# ---------------------------------------------------------------------------

# "eval" mode prompt (VLMEvalKit style, with example)
_EVAL_PRE_PROMPT = (
    'Please find the visual event described by a sentence in the video, '
    'determining its starting and ending times. '
    "The format should be: 'The event happens in the start time - end time seconds'. "
    "For example: The event 'person turn a light on' happens in the 24.3 - 30.4 seconds. "
    'Now I will give you the textual sentence: '
)
_EVAL_POST_PROMPT = 'Please return its start time and end time in seconds.'

# "timelens" mode prompt (TimeLens original, no example)
_TIMELENS_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# CoT suffix (shared by both modes)
_GROUNDING_COT_SUFFIX = {
    'direct': '',
    'cot_boxed': ('\nPlease reason step by step, then put your final answer '
                  '(start time - end time in seconds) in \\boxed{} format.'),
    'cot_tags': ('\nPlease think step by step inside <think> tags, '
                 'then provide the start and end times inside <answer> tags.'),
}

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _get_cot_mode():
    env = os.environ.get('USE_COT', '0')
    if env in ('0', ''):
        return 'direct'
    if env == 'tags':
        return 'cot_tags'
    return 'cot_boxed'


def _get_eval_mode():
    """Return 'eval' or 'timelens'. Default is 'eval'."""
    return os.environ.get('TIMELENS_EVAL_MODE', 'eval').lower()


def _parse_query(query):
    """TimeLens-style query cleaning: collapse whitespace, strip trailing periods."""
    return re.sub(r'\s+', ' ', query).strip().strip('.').strip()


# ---------------------------------------------------------------------------
#  Timestamp parsing -- two variants
# ---------------------------------------------------------------------------

def _parse_timestamps_eval(text):
    """Simple regex: 'X - Y' or 'X to Y'. Returns (start, end) or (None, None)."""
    for pat in [
        r'(\d+\.?\d*)\s*[-\u2013]\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            s, e = float(m.group(1)), float(m.group(2))
            if e >= s:
                return s, e
    return None, None


def _extract_time_timelens(paragraph):
    """Full TimeLens parser (ported from TimeLens/timelens/utils.py:extract_time).

    Supports HH:MM:SS, MM:SS, 'X - Y', 'X to Y', and fallback to all number pairs.
    Returns list of (start, end) tuples.
    """
    paragraph = paragraph.lower()
    timestamps = []

    # 1. HH:MM:SS / MM:SS formats
    time_regex = re.compile(
        r'\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b'
    )
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[:len(time_matches) // 2 * 2]
    if time_matches:
        converted = []
        for t in time_matches:
            parts = t.split(':')
            if len(parts) == 3:
                h, m = map(int, parts[:2])
                s = float(parts[2])
                converted.append(float(h * 3600 + m * 60 + s))
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                converted.append(float(m * 60 + s))
        timestamps = [
            (converted[i], converted[i + 1])
            for i in range(0, len(converted), 2)
        ]

    # 2. "X - Y" / "X to Y"
    if not timestamps:
        for pat in [
            r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)',
        ]:
            matches = re.findall(pat, paragraph)
            if matches:
                timestamps = [(float(s), float(e)) for s, e in matches]
                break

    # 3. Fallback: all bare numbers, paired up
    if not timestamps:
        num_regex = re.compile(r'\b(\d+\.\d+|\d+)\b')
        nums = re.findall(num_regex, paragraph)
        nums = nums[:len(nums) // 2 * 2]
        timestamps = [
            (float(nums[i]), float(nums[i + 1]))
            for i in range(0, len(nums), 2)
        ]

    return timestamps


# ---------------------------------------------------------------------------
#  IoU computation -- two variants
# ---------------------------------------------------------------------------

def _compute_iou(pred_start, pred_end, gt_start, gt_end):
    """Temporal IoU with zero-division protection (eval mode)."""
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


def _compute_iou_timelens(a, b):
    """TimeLens-style IoU (matches original TimeLens/timelens/utils.py:iou).

    a, b are (start, end) tuples.
    NOTE: no zero-division guard — faithful to the original code.
    """
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    denom = max1 - min0
    if denom == 0:
        return 0.0
    return max(min1 - max0, 0) / denom


# ---------------------------------------------------------------------------
#  TimeLensBench dataset class
# ---------------------------------------------------------------------------

class TimeLensBench(VideoBaseDataset):
    """Unified TimeLens-Bench temporal grounding benchmark.

    Supports three sub-datasets via `supported_datasets()`:
      - TimeLensBench_Charades
      - TimeLensBench_ActivityNet
      - TimeLensBench_QVHighlights

    Two evaluation modes (env var ``TIMELENS_EVAL_MODE``, default ``eval``):
      - ``eval``:     VLMEvalKit prompt + simple timestamp parser
      - ``timelens``: TimeLens original prompt + aggressive parser + rounding

    Environment variables:
      - ``TIMELENS_DIR``: root directory of TimeLens-Bench data
      - ``TIMELENS_EVAL_MODE``: ``eval`` (default) or ``timelens``
      - ``USE_COT``: ``0`` (default/direct), ``1`` (cot_boxed), ``tags`` (cot_tags)
    """

    TYPE = 'Video-VQA'

    def __init__(self, dataset='TimeLensBench_Charades', nframe=32, fps=-1):
        if fps > 0:
            nframe = 0
        assert dataset in _DATASET_CONFIG, (
            f'Unknown TimeLensBench dataset: {dataset}. '
            f'Supported: {list(_DATASET_CONFIG.keys())}'
        )
        self._cfg = _DATASET_CONFIG[dataset]
        VideoBaseDataset.__init__(self, dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return list(_DATASET_CONFIG.keys())

    # ------------------------------------------------------------------
    #  Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self, dataset_name='TimeLensBench_Charades', **_ignored):
        import glob
        import tarfile

        timelens_dir = os.environ.get('TIMELENS_DIR', _TIMELENS_DEFAULT_DIR)
        cfg = _DATASET_CONFIG[dataset_name]
        json_path = osp.join(timelens_dir, cfg['json_file'])

        if not osp.exists(json_path):
            raise FileNotFoundError(
                f"{cfg['json_file']} not found at {json_path}. "
                f'Set the TIMELENS_DIR env var to the TimeLens-Bench root.'
            )

        # Video directory: first try videos/<sub>, then video_shards/<sub>/videos
        video_dir = osp.join(timelens_dir, cfg['video_subdir'])
        if not osp.isdir(video_dir) or not os.listdir(video_dir):
            shard_dir = osp.join(timelens_dir, cfg['shard_subdir'])
            video_dir = osp.join(shard_dir, 'videos')
            if not osp.isdir(video_dir) or not os.listdir(video_dir):
                os.makedirs(video_dir, exist_ok=True)
                tar_files = sorted(glob.glob(osp.join(shard_dir, '*.tar.gz')))
                if not tar_files:
                    raise FileNotFoundError(
                        f'No *.tar.gz shard found under {shard_dir} and '
                        f'no videos at {osp.join(timelens_dir, cfg["video_subdir"])}.'
                    )
                for tf in tar_files:
                    print(f'TimeLensBench: extracting {tf} -> {video_dir} ...')
                    with tarfile.open(tf, 'r:gz') as t:
                        for member in t.getmembers():
                            if member.name.endswith('.mp4'):
                                member.name = osp.basename(member.name)
                                t.extract(member, path=video_dir)

        # Build annotation TSV (includes duration for timelens mode)
        tsv_file = osp.join(timelens_dir, f'{dataset_name}.tsv')
        if not osp.exists(tsv_file):
            with open(json_path, 'r') as f:
                ann = json.load(f)
            rows = []
            idx = 0
            for video_id, meta in ann.items():
                duration = meta.get('duration', -1)
                for q, span in zip(meta['queries'], meta['spans']):
                    rows.append({
                        'index': idx,
                        'video': video_id,
                        'question': q,
                        'answer': str(list(span)),
                        'duration': duration,
                    })
                    idx += 1
            pd.DataFrame(rows).to_csv(tsv_file, sep='\t', index=False)
            print(f'TimeLensBench({dataset_name}): wrote {idx} rows to {tsv_file}')

        return dict(data_file=tsv_file, root=video_dir)

    # ------------------------------------------------------------------
    #  Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_id = str(line['video'])
        video_path = osp.join(self.data_root, video_id + '.mp4')

        message = []
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            frames = self.save_video_frames(video_id)
            for frame in frames:
                message.append(dict(type='image', value=frame))

        mode = _get_eval_mode()
        query = str(line['question'])

        if mode == 'timelens':
            query_clean = _parse_query(query)
            text = _TIMELENS_PROMPT.format(query_clean)
        else:
            text = f"{_EVAL_PRE_PROMPT}{query}. {_EVAL_POST_PROMPT}"

        text += _GROUNDING_COT_SUFFIX[_get_cot_mode()]
        message.append(dict(type='text', value=text))
        message.append(_MANAGED_PROMPT_SENTINEL)
        return message

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        mode = _get_eval_mode()
        data = load(eval_file)
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            if mode == 'timelens':
                cls._score_timelens(data, eval_file)
            else:
                cls._score_eval(data)
            dump(data, score_file)
        else:
            data = load(score_file)

        if mode == 'timelens':
            return cls._aggregate_timelens(data, eval_file)
        else:
            return cls._aggregate_eval(data, eval_file)

    # --- eval mode scoring ---
    @classmethod
    def _score_eval(cls, data):
        for idx, row in data.iterrows():
            pred = str(row.get('prediction', '')).strip()
            try:
                gt_ts = eval(str(row.get('answer', '')))
                gt_start, gt_end = float(gt_ts[0]), float(gt_ts[1])
            except Exception:
                data.loc[idx, 'iou'] = -1
                continue
            p_start, p_end = _parse_timestamps_eval(pred)
            if p_start is None:
                data.loc[idx, 'iou'] = 0.0
            else:
                data.loc[idx, 'iou'] = _compute_iou(p_start, p_end, gt_start, gt_end)

    @classmethod
    def _aggregate_eval(cls, data, eval_file):
        valid = data[data['iou'] >= 0]
        if len(valid) == 0:
            print('TimeLensBench: no valid predictions found.')
            return {}
        ious = valid['iou'].values
        result = {
            'mIoU': round(float(ious.mean()) * 100, 2),
            'R@1_IoU=0.3': round(float((ious >= 0.3).mean()) * 100, 2),
            'R@1_IoU=0.5': round(float((ious >= 0.5).mean()) * 100, 2),
            'R@1_IoU=0.7': round(float((ious >= 0.7).mean()) * 100, 2),
        }
        acc_file = get_intermediate_file_path(eval_file, '_acc')
        dump(pd.DataFrame([{'metric': k, 'value': v} for k, v in result.items()]), acc_file)
        cls._print_results(result, 'eval')
        return result

    # --- timelens mode scoring ---
    @classmethod
    def _score_timelens(cls, data, eval_file):
        unit = 1.0
        for name, cfg in _DATASET_CONFIG.items():
            if name in eval_file:
                unit = cfg.get('unit', 1.0)
                break

        for idx, row in data.iterrows():
            pred = str(row.get('prediction', '')).strip()
            try:
                gt_ts = eval(str(row.get('answer', '')))
                gt_start, gt_end = float(gt_ts[0]), float(gt_ts[1])
            except Exception:
                data.loc[idx, 'iou'] = 0.0
                continue

            timestamps = _extract_time_timelens(pred)
            if not timestamps:
                duration = float(row.get('duration', 0))
                timestamps = [(duration + 10, duration + 20)]

            p_start, p_end = timestamps[0]
            p_start = round(p_start / unit) * unit
            p_end = round(p_end / unit) * unit

            if p_start >= p_end:
                data.loc[idx, 'iou'] = 0.0
            else:
                data.loc[idx, 'iou'] = _compute_iou_timelens(
                    (gt_start, gt_end), (p_start, p_end)
                )

    @classmethod
    def _aggregate_timelens(cls, data, eval_file):
        num_total = len(data)
        ious = data['iou'].values
        result = {
            'mIoU': round(float(ious.sum()) / num_total * 100, 2),
            'R@1_IoU=0.3': round(float((ious >= 0.3).sum()) / num_total * 100, 2),
            'R@1_IoU=0.5': round(float((ious >= 0.5).sum()) / num_total * 100, 2),
            'R@1_IoU=0.7': round(float((ious >= 0.7).sum()) / num_total * 100, 2),
        }
        acc_file = get_intermediate_file_path(eval_file, '_acc')
        dump(pd.DataFrame([{'metric': k, 'value': v} for k, v in result.items()]), acc_file)
        cls._print_results(result, 'timelens')
        return result

    @staticmethod
    def _print_results(result, mode):
        print(f'TimeLensBench Evaluation Results (mode={mode}):')
        for k, v in result.items():
            print(f'  {k}: {v:.2f}%')
