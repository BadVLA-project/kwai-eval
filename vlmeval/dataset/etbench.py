"""ETBench — E.T. Bench: Towards Open-Ended Event-Level Video-Language Understanding.

Paper: https://arxiv.org/abs/2409.18111
HuggingFace: https://huggingface.co/datasets/PolyU-ChenLab/ETBench

12 tasks across 4 capabilities:
  Referring:   tvg (Temporal Video Grounding), evs (Event-level Visual Search),
               rvs (Referring Video Summarization)
  Grounding:   dvc (Dense Video Captioning), slc (Sequential Location Caption)
  Dense Cap.:  ec  (Event Counting)
  Complex:     rar (Referring Action Recognition), eca (Event Caption Assessment),
               rvq (Referring Video QA), gvq (Grounded Video QA),
               tas (Temporal Action Segmentation), evl (Event-level Localization)

Annotation format (etbench_txt_v1.0.json) — list of dicts:
  {
    "idx":      int,          # sample index
    "task":     str,          # task code (tvg / rar / dvc / ...)
    "source":   str,          # source dataset name
    "video":    str,          # relative path, e.g. "qvhighlights/xxx.mp4"
    "duration": float,        # video length in seconds
    "src":      list[float],  # (optional) input timestamps
    "tgt":      list[list],   # (optional) output spans [[s, e], ...]
    "p":        int,          # (optional) correct option index for MCQ tasks
    "o":        list[str],    # (optional) answer options for MCQ tasks
    "g":        list[str],    # (optional) GT captions for captioning tasks
    "q":        str,          # full prompt to send to model
  }
"""

import ast
import json
import re
import os.path as osp

from ..smp import *
from .video_base import VideoBaseDataset

# ---------------------------------------------------------------------------
# Task categorisation
# ---------------------------------------------------------------------------
# Tasks that need temporal grounding evaluation (predict start-end spans)
_GROUNDING_TASKS = {'tvg', 'evs', 'rvs', 'evl'}
# Multiple-choice tasks (predict a letter option)
_MCQ_TASKS = {'rar', 'eca', 'rvq', 'gvq'}
# Dense captioning tasks (predict multiple spans with captions)
_CAPTIONING_TASKS = {'dvc', 'slc'}

# Server-side data root (preferred); fallback: LMUDataRoot() / ETBench
_SERVER_ROOT = '/m2v_intern/xuboshen/zgw/Benchmarks/ETBench'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_span(text):
    """Extract a single (start, end) float pair from free-form model output.

    Returns (None, None) on failure.
    """
    for pat in [
        r'(\d+\.?\d*)\s*[-\u2013\u2014]\s*(\d+\.?\d*)\s*(?:seconds?)?',
        r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)',
        r'start[^0-9]*(\d+\.?\d*).*?end[^0-9]*(\d+\.?\d*)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            s, e = float(m.group(1)), float(m.group(2))
            if e >= s:
                return s, e
    return None, None


def _parse_spans(text):
    """Extract all (start, end) pairs from model output.

    Used for dense captioning / multi-span grounding tasks.
    """
    spans = []
    for pat in [
        r'(\d+\.?\d*)\s*[-\u2013\u2014]\s*(\d+\.?\d*)\s*(?:seconds?)?',
        r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)',
    ]:
        for m in re.finditer(pat, text, re.IGNORECASE):
            s, e = float(m.group(1)), float(m.group(2))
            if e >= s:
                spans.append([s, e])
    # Deduplicate preserving order
    seen = set()
    unique = []
    for span in spans:
        key = (span[0], span[1])
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return unique


def _temporal_iou(pred_start, pred_end, gt_start, gt_end):
    """Scalar temporal IoU."""
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return inter / union if union > 0 else 0.0


def _best_iou_against_gt_spans(pred_start, pred_end, gt_spans):
    """Maximum IoU of a single predicted span against a list of GT spans."""
    return max(
        (_temporal_iou(pred_start, pred_end, gs, ge) for gs, ge in gt_spans),
        default=0.0,
    )


def _parse_option_letter(text):
    """Extract the choice letter (A/B/C/D) that the model selected."""
    text = text.strip()
    # Explicit "Best Option: (X)"
    m = re.search(r'best\s+option\s*[:\s]*\(?([A-Da-d])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Standalone leading letter
    m = re.match(r'^\(?([A-Da-d])\)?[\s\.\):]', text)
    if m:
        return m.group(1).upper()
    # Any standalone letter in output
    m = re.search(r'\b([A-Da-d])\b', text)
    if m:
        return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

# Official ETBench inference uses load_video(fps=1) on videos_compressed (3fps/224px).
# Set fps=1.0 as the default to match the official configuration.
_OFFICIAL_FPS = 1.0

# Known video source subdirectories produced by extracting the tar.gz archives
_VIDEO_SUBDIRS = [
    'qvhighlights', 'charades_sta', 'hirest', 'qa_ego4d',
    'summe', 'tvsum', 'cross_task', 'ht_step', 'thumos14',
]


class ETBench(VideoBaseDataset):
    """E.T. Bench multi-task video understanding benchmark.

    Official configuration: fps=1.0 (matches ETBench's infer_etbench.py which
    calls load_video with default fps=1 on videos_compressed).

    Supports both full evaluation (7,289 samples) and the 470-sample
    commercial subset used in Table 1 of the paper.

    Videos are auto-extracted from *.tar.gz if not yet unpacked.
    Set env-var ``ETBENCH_DIR`` to override the data root.

    Parameters
    ----------
    dataset : str
        One of 'ETBench' (full) or 'ETBench_subset'.
    nframe : int
        Number of frames to sample uniformly (mutually exclusive with fps).
    fps : float
        Frames per second to sample (mutually exclusive with nframe). Default
        matches the official ETBench evaluation setting (1.0 fps).
    task_filter : list[str] | None
        If given, only include samples whose task code is in this list.
    data_root : str | None
        Override the data root directory.
    """

    TYPE = 'Video-VQA'

    def __init__(
        self,
        dataset='ETBench',
        nframe=0,
        fps=_OFFICIAL_FPS,
        task_filter=None,
        data_root=None,
    ):
        self._task_filter = task_filter
        self._data_root_override = data_root
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['ETBench', 'ETBench_subset']

    # ------------------------------------------------------------------
    #  Data preparation
    # ------------------------------------------------------------------

    def _resolve_root(self):
        if self._data_root_override:
            return self._data_root_override
        env = os.environ.get('ETBENCH_DIR', '')
        if env and osp.isdir(env):
            return env
        if osp.isdir(_SERVER_ROOT):
            return _SERVER_ROOT
        return osp.join(LMUDataRoot(), 'ETBench')

    @staticmethod
    def _ensure_videos_extracted(data_root):
        """Extract any *.tar.gz archives under videos/ that have not yet been unpacked.

        A tar.gz is considered already extracted when a same-named subdirectory
        exists under videos/ with at least one file inside.
        Also checks videos_compressed/ as the preferred source (official config).
        """
        import tarfile
        import glob

        for videos_dir_name in ('videos_compressed', 'videos'):
            videos_dir = osp.join(data_root, videos_dir_name)
            if not osp.isdir(videos_dir):
                continue

            archives = sorted(glob.glob(osp.join(videos_dir, '*.tar.gz')))
            if not archives:
                continue

            for archive in archives:
                stem = osp.basename(archive)[: -len('.tar.gz')]   # e.g. 'qvhighlights'
                expected_dir = osp.join(videos_dir, stem)
                # Skip if already extracted (dir exists and is non-empty)
                if osp.isdir(expected_dir) and len(os.listdir(expected_dir)) > 0:
                    continue
                print(f'ETBench: extracting {archive} → {videos_dir} ...')
                try:
                    with tarfile.open(archive, 'r:gz') as tf:
                        tf.extractall(videos_dir)
                    print(f'ETBench: done extracting {osp.basename(archive)}')
                except Exception as exc:
                    print(f'ETBench: WARNING — failed to extract {archive}: {exc}')

    def prepare_dataset(self, dataset_name='ETBench'):
        data_root = self._resolve_root()

        # ------------------------------------------------------------------
        # Auto-extract tar.gz archives if videos not yet unpacked
        # ------------------------------------------------------------------
        self._ensure_videos_extracted(data_root)

        # Pick annotation file
        is_subset = dataset_name == 'ETBench_subset'
        anno_file = osp.join(data_root, 'annotations', 'etbench_txt_v1.0.json')
        subset_file = osp.join(data_root, 'evaluation', 'subset.json')

        tsv_file = osp.join(data_root, f'{dataset_name}.tsv')
        if self._task_filter:
            task_tag = '_'.join(sorted(self._task_filter))
            tsv_file = osp.join(data_root, f'{dataset_name}_{task_tag}.tsv')

        if not osp.exists(tsv_file):
            assert osp.exists(anno_file), (
                f'ETBench annotation not found: {anno_file}\n'
                f'Set ETBENCH_DIR env-var or place data under {data_root}.'
            )

            with open(anno_file, 'r') as f:
                samples = json.load(f)

            # Apply subset filter
            if is_subset and osp.exists(subset_file):
                with open(subset_file, 'r') as f:
                    subset_ids = set(json.load(f))
                samples = [s for s in samples if s['idx'] in subset_ids]

            # Apply task filter
            if self._task_filter:
                samples = [s for s in samples if s['task'] in self._task_filter]

            rows = []
            for sample in samples:
                video_rel = sample['video']   # "qvhighlights/xxxxx.mp4"
                video_id = osp.splitext(video_rel)[0].replace('/', '__').replace('\\', '__')
                rows.append({
                    'index':      sample['idx'],
                    'video':      video_id,
                    'video_path': video_rel,
                    'task':       sample['task'],
                    'source':     sample.get('source', ''),
                    'duration':   sample.get('duration', -1),
                    'question':   sample['q'],
                    # Serialise complex GT fields as JSON strings
                    'answer':     json.dumps({
                        'tgt':      sample.get('tgt'),
                        'p':        sample.get('p'),
                        'o':        sample.get('o'),
                        'g':        sample.get('g'),
                    }),
                })

            df = pd.DataFrame(rows)
            os.makedirs(data_root, exist_ok=True)
            df.to_csv(tsv_file, sep='\t', index=False)

        # Prefer videos_compressed (official config) over raw videos
        for candidate in ('videos_compressed', 'videos'):
            cand_dir = osp.join(data_root, candidate)
            if osp.isdir(cand_dir):
                # Accept if any known subdirectory exists inside
                for sub in _VIDEO_SUBDIRS:
                    if osp.isdir(osp.join(cand_dir, sub)):
                        video_dir = cand_dir
                        break
                else:
                    continue
                break
        else:
            video_dir = osp.join(data_root, 'videos')   # fallback

        return dict(data_file=tsv_file, root=video_dir)

    # ------------------------------------------------------------------
    #  Frame extraction
    # ------------------------------------------------------------------

    def save_video_frames(self, video):
        """Locate the video file by its stored relative path, then extract frames."""
        import decord

        # Try to find original path stored in annotation
        vid_path = None
        matches = self.data[self.data['video'] == video]
        if len(matches) > 0 and 'video_path' in self.data.columns:
            rel_path = matches.iloc[0]['video_path']
            candidate = osp.join(self.data_root, rel_path)
            if osp.exists(candidate):
                vid_path = candidate
            # Also try compressed variant
            if vid_path is None:
                compressed_root = osp.join(
                    osp.dirname(self.data_root), 'videos_compressed'
                )
                cand2 = osp.join(compressed_root, rel_path)
                if osp.exists(cand2):
                    vid_path = cand2

        if vid_path is None:
            for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                cand = osp.join(self.data_root, video + ext)
                if osp.exists(cand):
                    vid_path = cand
                    break

        if vid_path is None:
            raise FileNotFoundError(
                f'ETBench: cannot find video "{video}" under {self.data_root}'
            )

        vid = decord.VideoReader(vid_path)

        if self.fps > 0:
            total_frames = len(vid)
            video_fps = vid.get_avg_fps()
            total_duration = total_frames / video_fps
            required_frames = max(1, int(total_duration * self.fps))
            step_size = video_fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))
        else:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)

        if np.all([osp.exists(p) for p in frame_paths]):
            return frame_paths

        import portalocker
        lock_path = osp.join(self.frame_root, video + '.lock')
        with portalocker.Lock(lock_path, 'w', timeout=60):
            if np.all([osp.exists(p) for p in frame_paths]):
                return frame_paths
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    # ------------------------------------------------------------------
    #  Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_id = str(line['video'])

        # Prefer video_path column if present in TSV
        video_path = None
        if 'video_path' in line and pd.notna(line['video_path']):
            cand = osp.join(self.data_root, str(line['video_path']))
            if osp.exists(cand):
                video_path = cand
            else:
                # Try compressed
                compressed_root = osp.join(
                    osp.dirname(self.data_root), 'videos_compressed'
                )
                cand2 = osp.join(compressed_root, str(line['video_path']))
                if osp.exists(cand2):
                    video_path = cand2

        if video_path is None:
            for ext in ['.mp4', '.avi', '.mkv', '.mov']:
                cand = osp.join(self.data_root, video_id + ext)
                if osp.exists(cand):
                    video_path = cand
                    break
            if video_path is None:
                video_path = osp.join(self.data_root, video_id + '.mp4')

        message = []
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            frames = self.save_video_frames(video_id)
            for frame in frames:
                message.append(dict(type='image', value=frame))

        message.append(dict(type='text', value=str(line['question'])))
        return message

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        score_file = get_intermediate_file_path(eval_file, '_etbench_score')

        if not osp.exists(score_file):
            records = []
            for _, row in data.iterrows():
                pred = str(row.get('prediction', '')).strip()
                task = str(row.get('task', '')).lower()

                try:
                    gt_dict = json.loads(str(row.get('answer', '{}')))
                except (json.JSONDecodeError, TypeError):
                    gt_dict = {}

                rec = {
                    'index':  row.get('index', _),
                    'task':   task,
                    'source': row.get('source', ''),
                    'pred':   pred,
                }

                # ---- Grounding tasks ----
                if task in _GROUNDING_TASKS:
                    gt_spans = gt_dict.get('tgt') or []
                    # Normalise: tgt may be [[s,e],...] or [s,e]
                    if gt_spans and not isinstance(gt_spans[0], list):
                        gt_spans = [gt_spans]
                    ps, pe = _parse_span(pred)
                    if ps is None:
                        rec['iou'] = 0.0
                        rec['parsed'] = False
                    else:
                        rec['iou'] = _best_iou_against_gt_spans(ps, pe, gt_spans)
                        rec['parsed'] = True

                # ---- MCQ tasks ----
                elif task in _MCQ_TASKS:
                    correct_idx = gt_dict.get('p')
                    options = gt_dict.get('o') or []
                    if correct_idx is not None and len(options) > 0:
                        correct_letter = chr(ord('A') + int(correct_idx))
                        pred_letter = _parse_option_letter(pred)
                        rec['correct_letter'] = correct_letter
                        rec['pred_letter'] = pred_letter
                        rec['correct'] = (pred_letter == correct_letter) if pred_letter else False
                    else:
                        rec['correct'] = False

                # ---- Captioning tasks ----
                elif task in _CAPTIONING_TASKS:
                    gt_spans = gt_dict.get('tgt') or []
                    if gt_spans and not isinstance(gt_spans[0], list):
                        gt_spans = [gt_spans]
                    pred_spans = _parse_spans(pred)
                    # Compute F1 at IoU=0.5 between predicted and GT span sets
                    rec['f1_iou5'] = _span_set_f1(pred_spans, gt_spans, iou_thresh=0.5)
                    rec['gt_captions'] = json.dumps(gt_dict.get('g') or [])
                    rec['pred_spans_count'] = len(pred_spans)

                records.append(rec)

            score_df = pd.DataFrame(records)
            dump(score_df, score_file)
        else:
            score_df = load(score_file)

        results = {}

        # ---- Aggregate grounding metrics ----
        grd = score_df[score_df['task'].isin(_GROUNDING_TASKS)]
        if len(grd) > 0:
            ious = grd['iou'].fillna(0).values
            results['Grounding/mIoU']   = round(float(ious.mean()) * 100, 2)
            results['Grounding/R@0.3']  = round(float((ious >= 0.3).mean()) * 100, 2)
            results['Grounding/R@0.5']  = round(float((ious >= 0.5).mean()) * 100, 2)
            results['Grounding/R@0.7']  = round(float((ious >= 0.7).mean()) * 100, 2)
            # Per-task breakdown
            for task_code in _GROUNDING_TASKS:
                sub = grd[grd['task'] == task_code]
                if len(sub) == 0:
                    continue
                t_ious = sub['iou'].fillna(0).values
                results[f'{task_code.upper()}/mIoU']  = round(float(t_ious.mean()) * 100, 2)
                results[f'{task_code.upper()}/R@0.5'] = round(float((t_ious >= 0.5).mean()) * 100, 2)

        # ---- Aggregate MCQ metrics ----
        mcq = score_df[score_df['task'].isin(_MCQ_TASKS)]
        if len(mcq) > 0:
            acc = mcq['correct'].fillna(False).astype(bool).mean()
            results['MCQ/Accuracy'] = round(float(acc) * 100, 2)
            for task_code in _MCQ_TASKS:
                sub = mcq[mcq['task'] == task_code]
                if len(sub) == 0:
                    continue
                t_acc = sub['correct'].fillna(False).astype(bool).mean()
                results[f'{task_code.upper()}/Accuracy'] = round(float(t_acc) * 100, 2)

        # ---- Aggregate captioning metrics ----
        cap = score_df[score_df['task'].isin(_CAPTIONING_TASKS)]
        if len(cap) > 0:
            f1 = cap['f1_iou5'].fillna(0).mean()
            results['DenseCap/F1@IoU0.5'] = round(float(f1) * 100, 2)
            for task_code in _CAPTIONING_TASKS:
                sub = cap[cap['task'] == task_code]
                if len(sub) == 0:
                    continue
                t_f1 = sub['f1_iou5'].fillna(0).mean()
                results[f'{task_code.upper()}/F1@IoU0.5'] = round(float(t_f1) * 100, 2)

        # ---- Save and print ----
        acc_file = get_intermediate_file_path(eval_file, '_etbench_acc')
        dump(
            pd.DataFrame([{'metric': k, 'value': v} for k, v in results.items()]),
            acc_file,
        )

        print('\nETBench Evaluation Results:')
        for k, v in results.items():
            print(f'  {k}: {v:.2f}')
        return results


# ---------------------------------------------------------------------------
# Helper: span-set F1 at an IoU threshold
# ---------------------------------------------------------------------------

def _span_set_f1(pred_spans, gt_spans, iou_thresh=0.5):
    """Compute F1 between a set of predicted spans and a set of GT spans.

    Each predicted span is matched to at most one GT span (greedy by IoU).
    Returns the F1 score in [0, 1].
    """
    if not gt_spans:
        return 1.0 if not pred_spans else 0.0
    if not pred_spans:
        return 0.0

    matched_gt = set()
    tp = 0
    for ps, pe in pred_spans:
        best_iou, best_j = 0.0, -1
        for j, (gs, ge) in enumerate(gt_spans):
            if j in matched_gt:
                continue
            iou = _temporal_iou(ps, pe, gs, ge)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)

    precision = tp / len(pred_spans) if pred_spans else 0.0
    recall    = tp / len(gt_spans)   if gt_spans    else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
