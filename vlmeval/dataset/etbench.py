"""ETBench — E.T. Bench: Towards Open-Ended Event-Level Video-Language Understanding.

Paper: https://arxiv.org/abs/2409.18111
HuggingFace: https://huggingface.co/datasets/PolyU-ChenLab/ETBench

12 tasks across 4 capabilities (paper columns):
  REF:  rar (Referring Action Recognition), eca (Event Caption Assessment),
        rvq (Referring Video QA)
  GND:  tvg (Temporal Video Grounding), epm (Event Point Marking),
        tal (Temporal Action Localization), evs (Event-level Visual Search),
        vhd (Video Highlight Detection)
  CAP:  dvc (Dense Video Captioning), slc (Sequential Location Caption)
  COM:  tem (Temporal Event Matching), gvq (Grounded Video QA)

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

Evaluation methodology mirrors the official compute_metrics.py (Ye Liu 2024).
"""

import ast
import copy
import json
import os
import random
import re
import string
import os.path as osp

import numpy as np

from ..smp import *
from .video_base import VideoBaseDataset

# ---------------------------------------------------------------------------
# Task categorisation — codes match annotation etbench_txt_v1.0.json
# Paper groupings:
#   REF (Referring):  rar, eca, rvq
#   GND (Grounding):  tvg, epm, tal, evs, vhd
#   CAP (Captioning): dvc, slc
#   COM (Complex):    tem, gvq
# ---------------------------------------------------------------------------
_REF_TASKS        = {'rar', 'eca', 'rvq'}
_GND_TASKS        = {'tvg', 'epm', 'tal', 'evs', 'vhd'}
_CAPTIONING_TASKS = {'dvc', 'slc'}
_COM_TASKS        = {'tem', 'gvq'}

# Legacy aliases kept for format-suffix lookup
_MCQ_TASKS       = _REF_TASKS          # all REF tasks are MCQ
_GROUNDING_TASKS = _GND_TASKS | _COM_TASKS   # broad set for span parsing
_OTHER_TASKS     = set()               # nothing falls through anymore

# Server-side data root (preferred); fallback: LMUDataRoot() / ETBench
_SERVER_ROOT = '/m2v_intern/xuboshen/zgw/Benchmarks/ETBench'

# Sentinel item: when present in the message list, the model should NOT
# append its own post_prompt — the dataset already manages format instructions.
_MANAGED_PROMPT_SENTINEL = {'type': '_managed_prompt'}


def _get_cot_mode():
    """Read USE_COT env var and return one of 'direct', 'cot_boxed', 'cot_tags'."""
    env = os.environ.get('USE_COT', '0')
    if env in ('0', ''):
        return 'direct'
    if env == 'tags':
        return 'cot_tags'
    return 'cot_boxed'


# ---------------------------------------------------------------------------
# Per-task format instructions
# ---------------------------------------------------------------------------
# Each maps cot_mode → suffix string appended after the ETBench question.

_MCQ_FORMAT = {
    'direct':    '\nAnswer with the option letter only.',
    'cot_boxed': '\nPlease put your final answer in \\boxed{} format.',
    'cot_tags':  ('\nPlease think step by step inside <think> tags, '
                  'then provide the final answer inside <answer> tags.'),
}

_GROUNDING_FORMAT = {
    'direct':    '',   # ETBench q already contains format instructions
    'cot_boxed': ('\nPlease reason step by step, then put your final answer '
                  '(start time - end time in seconds) in \\boxed{} format.'),
    'cot_tags':  ('\nPlease think step by step inside <think> tags, '
                  'then provide the start and end times inside <answer> tags.'),
}

_CAPTIONING_FORMAT = {
    'direct':    '',   # ETBench q already contains instructions
    'cot_boxed': ('\nPlease reason step by step, then put your final answer '
                  'in \\boxed{} format.'),
    'cot_tags':  ('\nPlease think step by step inside <think> tags, '
                  'then provide the final answer inside <answer> tags.'),
}

_GENERIC_FORMAT = {
    'direct':    '',
    'cot_boxed': '\nPlease put your final answer in \\boxed{} format.',
    'cot_tags':  ('\nPlease think step by step inside <think> tags, '
                  'then provide the final answer inside <answer> tags.'),
}


def _format_suffix_for_task(task_code):
    """Return the format instruction suffix for a given task code and current COT mode."""
    mode = _get_cot_mode()
    if task_code in _MCQ_TASKS:
        return _MCQ_FORMAT[mode]
    if task_code in _GROUNDING_TASKS:
        return _GROUNDING_FORMAT[mode]
    if task_code in _CAPTIONING_TASKS:
        return _CAPTIONING_FORMAT[mode]
    return _GENERIC_FORMAT[mode]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_subset_filter(samples, subset_file):
    """Filter annotation samples using the official ETBench subset.json.

    subset.json is a nested dict: ``{task: {source: [local_indices]}}``.
    The local indices are positions within each (task, source) group of the
    annotation list.  Task codes in subset.json match the annotation directly.
    """
    with open(subset_file, 'r') as f:
        subset_spec = json.load(f)

    # Flat list of global idx — simple path
    if isinstance(subset_spec, list):
        id_set = {int(x) for x in subset_spec}
        return [s for s in samples if int(s['idx']) in id_set]

    # ── Nested dict: {task: {source: [local_indices]}} ──

    # Group annotation samples by (task, source), preserving order.
    from collections import defaultdict
    groups = defaultdict(list)   # (task, source) → [sample, ...]
    for s in samples:
        groups[(s['task'], s.get('source', ''))].append(s)

    selected = []
    for task, sources in subset_spec.items():
        for src_name, idx_list in sources.items():
            group = groups.get((task, src_name), [])
            for local_idx in idx_list:
                if 0 <= local_idx < len(group):
                    selected.append(group[local_idx])
                else:
                    print(
                        f'ETBench subset: index {local_idx} out of range for '
                        f'({task}, {src_name}) with {len(group)} samples'
                    )

    print(f'ETBench subset: selected {len(selected)} / {len(samples)} samples')
    return selected

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
# Official per-task evaluators (ported from ETBench compute_metrics.py)
# ---------------------------------------------------------------------------

def _remove_nonascii(text):
    return ''.join([c if ord(c) < 128 else ' ' for c in text])


def _random_string(n=15):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))


def _iou_interval(a, b):
    """Segment IoU for two [s, e] pairs (handles the DVC overlap definition)."""
    s_i, e_i = a[0], a[1]
    s, e = b[0], b[1]
    intersection = max(0, min(e, e_i) - max(s, s_i))
    union = min(max(e, e_i) - min(s, s_i), (e - s) + (e_i - s_i))
    return float(intersection) / (union + 1e-8)


def _tvg_eval(samples):
    """tvg / epm — predict a single [s, e] span; eval with mIoU + R@[.1,.3,.5,.7]."""
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    hit = [0] * len(iou_thr)
    cnt, sum_iou = 0, 0
    for sample in samples:
        gt = sample['tgt']
        pred_spans = _parse_spans(sample['pred'])
        if not pred_spans:
            cnt += 1
            continue
        # Use best-matching span as in tvg_format output
        ps, pe = pred_spans[0]
        gt_arr = np.array(gt, dtype=float)
        pred_arr = np.array([[ps, pe]], dtype=float)
        iou = float(_np_temporal_iou(gt_arr, pred_arr))
        sum_iou += iou
        for i, thr in enumerate(iou_thr):
            if iou >= thr:
                hit[i] += 1
    recall = [h / len(samples) for h in hit]
    miou = sum_iou / len(samples)
    out = dict(Total=len(samples), Failed=cnt, mIoU=round(miou, 5))
    for rec, thr in zip(recall, iou_thr):
        out[f'F1@{thr}'] = round(rec, 5)
    out['F1'] = round(sum(recall) / len(recall), 5)
    return out


def _vhd_eval(samples):
    """vhd — predict a single timestamp; check if it falls in any GT highlight window."""
    hit, cnt = 0, 0
    for sample in samples:
        gt = sample['tgt']
        if not isinstance(gt[0][0], (list, tuple)):
            gt = [gt]
        match = re.search(r'[-+]?\d*\.?\d+', sample['pred'])
        if not match:
            cnt += 1
            continue
        pred = float(match.group(0))
        matched = any(pred >= g[0] and pred <= g[1] for annotator in gt for g in annotator)
        if matched:
            hit += 1
    out = dict(Total=len(samples), Failed=cnt)
    out['F1'] = round(hit / len(samples), 5)
    return out


def _tem_eval(samples):
    """tem — predict a single span; eval against multiple GT spans with max-IoU."""
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    hit = [0] * len(iou_thr)
    cnt, sum_iou = 0, 0
    for sample in samples:
        gt = sample['tgt']
        pred_spans = _parse_spans(sample['pred'])
        if not pred_spans:
            cnt += 1
            continue
        ps, pe = pred_spans[0]
        gt_arr = np.array(gt, dtype=float)
        pred_arr = np.array([[ps, pe]], dtype=float)
        iou = float(_np_temporal_iou(gt_arr, pred_arr).max())
        sum_iou += iou
        for i, thr in enumerate(iou_thr):
            if iou >= thr:
                hit[i] += 1
    recall = [h / len(samples) for h in hit]
    miou = sum_iou / len(samples)
    out = dict(Total=len(samples), Failed=cnt, mIoU=round(miou, 5))
    for rec, thr in zip(recall, iou_thr):
        out[f'R@{thr}'] = round(rec, 5)
    out['mRec'] = round(sum(recall) / len(recall), 5)
    return out


def _tal_eval(samples):
    """tal — predict multiple spans; eval with F1@[.1,.3,.5,.7] (precision+recall)."""
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    f1_score = [0.0] * len(iou_thr)
    cnt = 0
    for sample in samples:
        gt = sample['tgt']
        pred_spans = _parse_spans(sample['pred'])
        if not pred_spans:
            cnt += 1
            continue
        gt_arr = np.array(gt, dtype=float)
        pred_arr = np.array(pred_spans, dtype=float)
        iou = _np_temporal_iou(gt_arr, pred_arr)
        for i, thr in enumerate(iou_thr):
            if iou.max() < thr:
                continue
            rec = float((iou.max(axis=1) >= thr).mean())
            prc = float((iou.max(axis=0) >= thr).mean())
            f1_score[i] += 2 * prc * rec / (prc + rec)
    f1_score = [f / len(samples) for f in f1_score]
    out = dict(Total=len(samples), Failed=cnt)
    for f1, thr in zip(f1_score, iou_thr):
        out[f'F1@{thr}'] = round(f1, 5)
    out['F1'] = round(sum(f1_score) / len(f1_score), 5)
    return out


def _evs_eval(samples):
    """evs — predict multiple spans; frame-level F1 over 1000-frame dummy timeline."""
    f1_scores = []
    cnt = 0
    for sample in samples:
        gt = sample['tgt']
        pred_spans = _parse_spans(sample['pred'])
        if not pred_spans:
            cnt += 1
            continue
        gt_map = np.zeros(1000)
        for g in gt:
            gt_map[max(0, round(g[0])):round(g[1])] = 1
        pred_map = np.zeros(1000)
        for p in pred_spans:
            pred_map[max(0, round(p[0])):round(p[1])] = 2
        com = gt_map + pred_map
        tp = int((com == 3).sum())
        fp = int((com == 2).sum())
        fn = int((com == 1).sum())
        if tp == 0:
            f1_scores.append(0.0)
        else:
            rec = tp / (tp + fn)
            prc = tp / (tp + fp)
            f1_scores.append(2 * prc * rec / (prc + rec))
    out = dict(Total=len(samples), Failed=cnt)
    out['F1'] = round(sum(f1_scores) / len(f1_scores), 5) if f1_scores else 0.0
    return out


def _rvq_eval(samples, st):
    """rar / eca / rvq — MCQ with SentSim fallback for unmatched predictions."""
    if not samples:
        return dict(Total=0, Failed=0, Acc=0.0)
    n_opts = len(samples[0]['o'])
    match_map = {chr(ord('a') + i): i for i in range(n_opts)}
    _map = [chr(ord('A') + i) for i in range(n_opts)]
    hit, cnt = 0, 0
    for sample in samples:
        gt = sample['p']
        pred = sample['pred']
        ever_matched = False
        m = re.search(r'\(([A-Za-z])\)', pred)
        if m:
            ever_matched = True
            ch = m.group(1).lower()
            if ch in match_map and gt == match_map[ch]:
                hit += 1
                continue
        pred_lower = pred.lower()
        if pred_lower.startswith('best option:'):
            pred_lower = pred_lower[12:]
        pred_lower = pred_lower.lstrip().lstrip('(').lstrip()
        if len(pred_lower) == 0:
            cnt += 1
            continue
        if len(pred_lower) == 1 or pred_lower[1] in ('.', ',', ' ', ')'):
            ever_matched = True
            if pred_lower[0] in match_map and gt == match_map[pred_lower[0]]:
                hit += 1
                continue
        # SentSim fallback
        best_idx, best_score = 0, float('-inf')
        for idx, option in enumerate(sample['o']):
            if isinstance(option, (list, tuple)):
                opt = f'{option[0]} - {option[1]}'
            else:
                opt = str(option)
            opt = f'({_map[idx]}) {opt}'
            score = st.compute_sim(pred, opt)
            if score > best_score:
                best_idx, best_score = idx, score
        if not ever_matched:
            cnt += 1
        if gt == best_idx:
            hit += 1
    out = dict(Total=len(samples), Failed=cnt, Acc=round(hit / len(samples), 5))
    return out


def _gvq_eval(samples, st):
    """gvq — MCQ accuracy first, then IoU recall only on correctly answered samples."""
    if not samples:
        return dict(Total=0, Failed=0, mIoU=0.0, mRec=0.0, Acc=0.0)
    acc_hit_set, acc_cnt = set(), 0
    _samples = copy.deepcopy(samples)
    for sidx, sample in enumerate(_samples):
        gt = sample['p']
        pred = sample['pred']
        if pred.lower().startswith('best option:'):
            pred = pred[12:]
        pred = pred.lstrip().lstrip('(').lstrip()
        if not pred:
            acc_cnt += 1
            continue
        n_opts = len(sample['o'])
        match_map = {chr(ord('a') + i): i for i in range(n_opts)}
        _map = [chr(ord('A') + i) for i in range(n_opts)]
        if len(pred) == 1 or pred[1] in ('.', ',', ' ', ')'):
            if pred[0].lower() in match_map:
                if gt == match_map[pred[0].lower()]:
                    acc_hit_set.add(sidx)
                continue
        best_idx, best_score = 0, float('-inf')
        for idx, option in enumerate(sample['o']):
            if isinstance(option, (list, tuple)):
                opt = f'{option[0]} - {option[1]}'
            else:
                opt = str(option)
            opt = f'({_map[idx]}) {opt}'
            score = st.compute_sim(pred, opt)
            if score > best_score:
                best_idx, best_score = idx, score
        if best_score == float('-inf'):
            acc_cnt += 1
            continue
        if gt == best_idx:
            acc_hit_set.add(sidx)
    iou_thr = [0.1, 0.3, 0.5, 0.7]
    hit = [0] * len(iou_thr)
    rec_cnt, sum_iou = 0, 0
    for sidx, sample in enumerate(samples):
        if sidx not in acc_hit_set:
            continue
        gt = sample['tgt']
        pred_spans = _parse_spans(sample['pred'])
        if not pred_spans:
            rec_cnt += 1
            continue
        ps, pe = pred_spans[0]
        gt_arr = np.array(gt, dtype=float)
        pred_arr = np.array([[ps, pe]], dtype=float)
        iou = float(_np_temporal_iou(gt_arr, pred_arr))
        sum_iou += iou
        for i, thr in enumerate(iou_thr):
            if iou >= thr:
                hit[i] += 1
    recall = [h / len(samples) for h in hit]
    miou = sum_iou / len(samples)
    out = dict(Total=len(samples), Failed=rec_cnt + acc_cnt, mIoU=round(miou, 5))
    for rec, thr in zip(recall, iou_thr):
        out[f'R@{thr}'] = round(rec, 5)
    out['mRec'] = round(sum(recall) / len(recall), 5)
    out['Acc'] = round(len(acc_hit_set) / len(samples), 5)
    return out


def _dvc_eval(samples, st):
    """dvc / slc — Dense Video Captioning: F1@tIoU + NLP metrics via DVCEval."""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        _has_coco = True
    except ImportError:
        _has_coco = False

    iou_thr = [0.1, 0.3, 0.5, 0.7]
    gt_dict, pred_dict = {}, {'results': {}}
    cnt = 0
    for sample in samples:
        gt_spans = sample['tgt']
        gt_caps = sample['g'] or []
        # Parse predicted timestamps + captions
        pred_spans, pred_caps = _dvc_format(sample['pred'])
        if pred_spans is None or not pred_spans:
            cnt += 1
            continue
        vid = sample['video']
        gt_dict[vid] = dict(timestamps=gt_spans, sentences=gt_caps)
        pred_dict['results'][vid] = [
            dict(sentence=c, timestamp=t) for t, c in zip(pred_spans, pred_caps)
        ]

    scale = len(pred_dict['results']) / len(samples) if samples else 1.0

    nlp_keys = ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SentSim')
    if gt_dict and _has_coco:
        from sentence_transformers.util import dot_score
        import sentence_transformers

        class _ST:
            def __init__(self):
                self.model = sentence_transformers.SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2'
                )
            def compute_sim(self, a, b):
                a_e = self.model.encode([a])
                b_e = self.model.encode([b])
                return float(dot_score(a_e, b_e)[0, 0].cpu())
            def compute_score(self, gts, res):
                keys = list(gts.keys())
                a = [gts[k][0] for k in keys]
                b = [res[k][0] for k in keys]
                a_e = self.model.encode(a)
                b_e = self.model.encode(b)
                sc = dot_score(a_e, b_e).cpu()
                score = sum(sc[i, i].item() for i in range(sc.shape[0])) / sc.shape[0]
                return float(score), None

        _st_inst = _ST()
        try:
            tokenizer = PTBTokenizer(verbose=False)
        except TypeError:
            tokenizer = PTBTokenizer()
        scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Meteor(), 'METEOR'),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
            (_st_inst, 'SentSim'),
        ]
        evaluator = _DVCEval(gt_dict, pred_dict, iou_thr, tokenizer, scorers)
        evaluator.evaluate()
        scores = evaluator.scores
    else:
        scores = {k: [0.0] * len(iou_thr) for k in ('Recall', 'Precision') + nlp_keys}
        scores['Recall'] = [0.0] * len(iou_thr)
        scores['Precision'] = [0.0] * len(iou_thr)

    out = dict(Total=len(samples), Failed=cnt)
    f1_list = []
    for rec, prc, thr in zip(scores['Recall'], scores['Precision'], iou_thr):
        rec = rec * scale
        prc = prc * scale
        f1 = 0.0 if prc + rec == 0 else 2 * prc * rec / (prc + rec)
        out[f'F1@{thr}'] = round(f1, 5)
        f1_list.append(f1)
    out['F1'] = round(sum(f1_list) / len(f1_list), 5)
    for k in ('Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SentSim'):
        out[k] = round(sum(scores.get(k, [0.0])) / max(len(scores.get(k, [1])), 1), 5)
    return out


# ---------------------------------------------------------------------------
# DVCEval helper (ported from official ETBench)
# ---------------------------------------------------------------------------

class _DVCEval:
    def __init__(self, ground_truth, prediction, tious, tokenizer, scorers):
        self.tious = tious
        self.max_proposals = 1000
        self.ground_truths = [ground_truth]
        self.prediction = {
            vid: prediction['results'][vid][:self.max_proposals]
            for vid in prediction['results']
        }
        self.tokenizer = tokenizer
        self.scorers = scorers

    def _gt_vid_ids(self):
        ids = set()
        for gt in self.ground_truths:
            ids |= set(gt.keys())
        return list(ids)

    def _iou(self, a, b):
        s_i, e_i = a[0], a[1]
        s, e = b[0], b[1]
        inter = max(0, min(e, e_i) - max(s, s_i))
        union = min(max(e, e_i) - min(s, s_i), (e - s) + (e_i - s_i))
        return float(inter) / (union + 1e-8)

    def evaluate(self):
        self.scores = {}
        for tiou in self.tious:
            for metric, score in self._eval_tiou(tiou).items():
                self.scores.setdefault(metric, []).append(score)
        self.scores['Recall'] = []
        self.scores['Precision'] = []
        for tiou in self.tious:
            prc, rec = self._eval_detection(tiou)
            self.scores['Recall'].append(rec)
            self.scores['Precision'].append(prc)

    def _eval_detection(self, tiou):
        vid_ids = self._gt_vid_ids()
        recall = [0.0] * len(vid_ids)
        precision = [0.0] * len(vid_ids)
        for vi, vid_id in enumerate(vid_ids):
            best_rec, best_prc = 0.0, 0.0
            for gt in self.ground_truths:
                if vid_id not in gt:
                    continue
                refs = gt[vid_id]
                ref_covered, pred_covered = set(), set()
                if vid_id in self.prediction:
                    for pi, pred in enumerate(self.prediction[vid_id]):
                        for ri, ref_ts in enumerate(refs['timestamps']):
                            if self._iou(pred['timestamp'], ref_ts) > tiou:
                                ref_covered.add(ri)
                                pred_covered.add(pi)
                    new_prc = len(pred_covered) / (pi + 1)
                    best_prc = max(best_prc, new_prc)
                new_rec = len(ref_covered) / len(refs['timestamps'])
                best_rec = max(best_rec, new_rec)
            recall[vi] = best_rec
            precision[vi] = best_prc
        return sum(precision) / len(precision), sum(recall) / len(recall)

    def _eval_tiou(self, tiou):
        vid2capid, res, gts, cur_res, cur_gts = {}, {}, {}, {}, {}
        uid = 0
        for vid_id in self._gt_vid_ids():
            vid2capid[vid_id] = []
            if vid_id not in self.prediction:
                continue
            for pred in self.prediction[vid_id]:
                added = False
                for gt in self.ground_truths:
                    if vid_id not in gt:
                        continue
                    gt_v = gt[vid_id]
                    for cap_idx, cap_ts in enumerate(gt_v['timestamps']):
                        if self._iou(pred['timestamp'], cap_ts) >= tiou:
                            cur_res[uid] = [{'caption': _remove_nonascii(pred['sentence'])}]
                            cur_gts[uid] = [{'caption': _remove_nonascii(gt_v['sentences'][cap_idx])}]
                            vid2capid[vid_id].append(uid)
                            uid += 1
                            added = True
                if not added:
                    cur_res[uid] = [{'caption': _remove_nonascii(pred['sentence'])}]
                    cur_gts[uid] = [{'caption': _random_string()}]
                    vid2capid[vid_id].append(uid)
                    uid += 1

        tok_res = self.tokenizer.tokenize(cur_res)
        tok_gts = self.tokenizer.tokenize(cur_gts)
        for vid in vid2capid:
            res[vid] = {i: tok_res[i] for i in vid2capid[vid]}
            gts[vid] = {i: tok_gts[i] for i in vid2capid[vid]}

        output = {}
        for scorer, method in self.scorers:
            all_scores = {}
            for vid_id in self._gt_vid_ids():
                if not res.get(vid_id) or not gts.get(vid_id):
                    score = [0] * len(method) if isinstance(method, list) else 0
                else:
                    if isinstance(method, list):
                        score, _ = scorer.compute_score(gts[vid_id], res[vid_id], verbose=0)
                    else:
                        score, _ = scorer.compute_score(gts[vid_id], res[vid_id])
                all_scores[vid_id] = score
            if isinstance(method, list):
                scores_arr = np.mean(list(all_scores.values()), axis=0)
                for mi, m in enumerate(method):
                    output[m] = scores_arr[mi]
            else:
                output[method] = float(np.mean(list(all_scores.values())))
        return output


# ---------------------------------------------------------------------------
# DVC format parser (ported from official ETBench)
# ---------------------------------------------------------------------------

def _extract_time_part(time_part):
    radius = 20
    result = re.compile(r'\d+\.*\d*\s*-\s*\d+\.*\d*').findall(time_part)
    if not result:
        if time_part.count(':') == 1:
            t = re.compile(r'\d+\.*\d*:\d+\.*\d*').findall(time_part)
            if t:
                s = int(t[0].split(':')[0]) * 60 + int(t[0].split(':')[1])
                result = [f'{max(0, s-radius)} - {s+radius}']
        elif time_part.count(':') == 2:
            ts = re.compile(r'\d+\.*\d*:\d+\.*\d*').findall(time_part)
            if len(ts) == 2:
                s = int(ts[0].split(':')[0]) * 60 + int(ts[0].split(':')[1])
                e = int(ts[1].split(':')[0]) * 60 + int(ts[1].split(':')[1])
                result = [f'{s} - {e}']
    if not result:
        nums = re.compile(r'\d+\.*\d*(?!\.)').findall(time_part)
        if len(nums) == 1:
            t = float(nums[0])
            result = [f'{max(0, t-radius)} - {t+radius}']
        elif len(nums) == 2:
            result = [f'{nums[0]} - {nums[1]}']
    return result


def _extract_time_from_paragraph(paragraph):
    paragraph = paragraph.lower()
    timestamps, captions = [], []
    for tp, sp in [(r'(\d+\.*\d*)\s*-\s*(\d+\.*\d*)', r'(\d+\.*\d*\s*-\s*\d+\.*\d*)')]:
        time_m = re.findall(tp, paragraph)
        str_m = re.findall(sp, paragraph)
        if time_m:
            timestamps = [[float(s), float(e)] for s, e in time_m]
            rest = paragraph
            for ts in str_m:
                rest = rest.replace(ts, '\n')
            captions = rest.replace('seconds', '').split('\n')
        if timestamps:
            break
    if not timestamps:
        s_pat = r'(?:start(?:ing)?\s+time:\s*(\d+\.*\d*)(?:s|\s+seconds)?)'
        e_pat = r'(?:end(?:ing)?\s+time:\s*(\d+\.*\d*)(?:s|\s+seconds)?)'
        sm = re.findall(s_pat, paragraph, re.IGNORECASE)
        em = re.findall(e_pat, paragraph, re.IGNORECASE)
        if sm and em:
            timestamps = [[float(s), float(e)] for s, e in zip(sm, em)]
            captions = re.findall(r'description:\s*(.*)', paragraph) or re.findall(r'\*\s*(.*)', paragraph)
    if not timestamps:
        se = re.findall(r'start time (\d+\.*\d*), end time (\d+\.*\d*)', paragraph)
        if se:
            timestamps = [[float(s), float(e)] for s, e in se]
            captions = []
    captions = [c.strip().strip(', ').rstrip() for c in captions if len(c) > 5]
    n = min(len(timestamps), len(captions))
    return timestamps[:n], captions[:n]


def _dvc_format(caption):
    """Parse model output into (timestamps, sentences) for DVC evaluation."""
    timestamps, sents = [], []
    try:
        timestamps, sents = _extract_time_from_paragraph(caption)
    except Exception:
        return None, None

    if not timestamps:
        lines = caption.split('\n') if '\n' in caption else caption.split('.')
        lines = [l for l in lines if len(l) > 7]
        if '\n' not in caption:
            lines = [l + '.' for l in lines]
        for line in lines:
            if timestamps:
                break
            try:
                parts = line.split('seconds')
                parts = [p.strip(',') for p in parts]
                tp = _extract_time_part(parts[0])
                if not tp:
                    continue
                stime = round(float(tp[0].split('-')[0].strip()), 2)
                etime = round(float(tp[0].split('-')[1].strip()), 2)
                timestamps.append([stime, etime])
                sents.append(parts[-1].strip())
            except Exception:
                continue

    if not timestamps:
        return None, None

    for i in range(len(timestamps)):
        timestamps[i] = [min(timestamps[i]), max(timestamps[i])]
    return timestamps, sents


def _np_temporal_iou(gt, pred):
    """Vectorised temporal IoU: gt [M,2], pred [N,2] → [M,N] matrix."""
    inter_s = np.maximum(gt[:, 0:1], pred[:, 0])   # [M, N]
    inter_e = np.minimum(gt[:, 1:2], pred[:, 1])
    inter = np.maximum(0.0, inter_e - inter_s)
    gt_len = gt[:, 1] - gt[:, 0]
    pred_len = pred[:, 1] - pred[:, 0]
    union = gt_len[:, None] + pred_len[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


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
    video_source : str
        Controls which video directory to use for frame extraction:
        - ``'auto'``       (default): prefer ``videos_compressed`` when present,
          fall back to ``videos``. Matches official ETBench behaviour.
        - ``'compressed'``: always use ``videos_compressed``. Errors if absent.
        - ``'raw'``       : always use the raw ``videos`` directory.
        - Any absolute/relative path string: use that directory directly.
    """

    TYPE = 'Video-VQA'

    def __init__(
        self,
        dataset='ETBench',
        nframe=0,
        fps=_OFFICIAL_FPS,
        task_filter=None,
        data_root=None,
        video_source='auto',
    ):
        self._task_filter = task_filter
        self._data_root_override = data_root
        self._video_source = video_source
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
                    with tarfile.open(archive, 'r:*') as tf:
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
            if is_subset:
                assert osp.exists(subset_file), (
                    f'ETBench subset file not found: {subset_file}\n'
                    f'Download it from the ETBench HuggingFace repo or use '
                    f'the full ETBench dataset instead of ETBench_subset.'
                )
                samples = _apply_subset_filter(samples, subset_file)

            # Apply task filter
            if self._task_filter:
                samples = [s for s in samples if s['task'] in self._task_filter]

            assert len(samples) > 0, (
                f'ETBench: 0 samples after filtering '
                f'(subset={is_subset}, task_filter={self._task_filter}). '
                f'Check annotation file and subset.json compatibility.'
            )

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

        # Select video directory based on video_source parameter
        vs = self._video_source
        if vs not in ('auto', 'compressed', 'raw') and osp.isdir(vs):
            # Explicit custom path
            video_dir = vs
        elif vs == 'compressed':
            video_dir = osp.join(data_root, 'videos_compressed')
            assert osp.isdir(video_dir), (
                f'ETBench: videos_compressed not found at {video_dir}. '
                f'Download compressed videos or use video_source="auto".'
            )
        elif vs == 'raw':
            video_dir = osp.join(data_root, 'videos')
        else:  # 'auto' — prefer videos_compressed, fall back to raw
            for candidate in ('videos_compressed', 'videos'):
                cand_dir = osp.join(data_root, candidate)
                if osp.isdir(cand_dir):
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
            # Build lookup order based on video_source
            vs = self._video_source
            data_parent = osp.dirname(self.data_root)
            if vs not in ('auto', 'compressed', 'raw') and osp.isdir(vs):
                _lookup_roots = [vs]
            elif vs == 'compressed':
                _lookup_roots = [osp.join(data_parent, 'videos_compressed')]
            elif vs == 'raw':
                _lookup_roots = [self.data_root]
            else:  # auto
                _lookup_roots = [
                    self.data_root,
                    osp.join(data_parent, 'videos_compressed'),
                ]
            for root in _lookup_roots:
                cand = osp.join(root, rel_path)
                if osp.exists(cand):
                    vid_path = cand
                    break

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
            rel_path = str(line['video_path'])
            vs = self._video_source
            data_parent = osp.dirname(self.data_root)
            if vs not in ('auto', 'compressed', 'raw') and osp.isdir(vs):
                _bp_roots = [vs]
            elif vs == 'compressed':
                _bp_roots = [osp.join(data_parent, 'videos_compressed')]
            elif vs == 'raw':
                _bp_roots = [self.data_root]
            else:  # auto
                _bp_roots = [
                    self.data_root,
                    osp.join(data_parent, 'videos_compressed'),
                ]
            for root in _bp_roots:
                cand = osp.join(root, rel_path)
                if osp.exists(cand):
                    video_path = cand
                    break

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

        # Build prompt text with task-appropriate format instruction
        task_code = str(line.get('task', '')).lower() if 'task' in line else ''
        question_text = str(line['question'])
        suffix = _format_suffix_for_task(task_code) if task_code else ''
        prompt = question_text + suffix

        message.append(dict(type='text', value=prompt))
        # Tell the model not to append its own post_prompt
        message.append(_MANAGED_PROMPT_SENTINEL)
        return message

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluate predictions against GT.

        Mirrors the official ETBench compute_metrics.py methodology and
        produces the same REF / GND / CAP / COM four-column summary used
        in Table 1 of the paper.

        Task → evaluator mapping:
          tvg, epm        → _tvg_eval   (mIoU + F1@tIoU)        [GND]
          vhd             → _vhd_eval   (point-in-window F1)     [GND]
          tal             → _tal_eval   (set F1@tIoU)            [GND]
          evs             → _evs_eval   (frame-level F1)         [GND]
          tem             → _tem_eval   (max-IoU mRec)           [COM]
          rar, eca, rvq   → _rvq_eval   (Acc + SentSim fallback) [REF]
          gvq             → _gvq_eval   (mRec + Acc)             [COM]
          dvc, slc        → _dvc_eval   (F1@tIoU + NLP metrics)  [CAP]
        """
        data = load(eval_file)

        # ---------- build per-sample dicts that the evaluators expect ----------
        by_task = {}   # task → list[dict]
        for _, row in data.iterrows():
            task = str(row.get('task', '')).lower()
            try:
                gt = json.loads(str(row.get('answer', '{}')))
            except (json.JSONDecodeError, TypeError):
                gt = {}

            # Normalise tgt nesting depth to [[s,e],...]
            tgt = gt.get('tgt') or []
            if tgt:
                if isinstance(tgt[0], (int, float)):
                    tgt = [tgt]
                elif isinstance(tgt[0], list) and tgt[0] and isinstance(tgt[0][0], list):
                    tgt = tgt[0]

            sample = {
                'pred':   str(row.get('prediction', '')).strip(),
                'task':   task,
                'source': row.get('source', ''),
                'video':  str(row.get('video', '')),
                'tgt':    tgt,
                'p':      gt.get('p'),
                'o':      gt.get('o') or [],
                'g':      gt.get('g') or [],
            }
            by_task.setdefault(task, []).append(sample)

        # ---------- lazy SentSim model (only loaded if REF/COM tasks exist) ----------
        _st = None
        def _get_st():
            nonlocal _st
            if _st is None:
                try:
                    import sentence_transformers
                    from sentence_transformers.util import dot_score as _ds

                    class _STSimple:
                        def __init__(self):
                            self.model = sentence_transformers.SentenceTransformer(
                                'sentence-transformers/all-MiniLM-L6-v2'
                            )
                        def compute_sim(self, a, b):
                            ae = self.model.encode([a])
                            be = self.model.encode([b])
                            return float(_ds(ae, be)[0, 0].cpu())

                    _st = _STSimple()
                except ImportError:
                    _st = None
            return _st

        # ---------- run per-task evaluators ----------
        collected = {}   # task → out_dict

        for task, samples in by_task.items():
            if task in ('tvg', 'epm'):
                collected[task] = _tvg_eval(samples)
            elif task == 'vhd':
                collected[task] = _vhd_eval(samples)
            elif task == 'tem':
                collected[task] = _tem_eval(samples)
            elif task == 'tal':
                collected[task] = _tal_eval(samples)
            elif task == 'evs':
                collected[task] = _evs_eval(samples)
            elif task in ('rar', 'eca', 'rvq'):
                collected[task] = _rvq_eval(samples, _get_st())
            elif task == 'gvq':
                collected[task] = _gvq_eval(samples, _get_st())
            elif task in ('dvc', 'slc'):
                collected[task] = _dvc_eval(samples, _get_st())

        # ---------- aggregate into REF / GND / CAP / COM ----------
        results = {}

        # -- REF: mean Acc(rar, eca, rvq) --
        ref_tasks = [t for t in ('rar', 'eca', 'rvq') if t in collected]
        for t in ref_tasks:
            results[f'{t.upper()}/Acc'] = round(collected[t]['Acc'] * 100, 2)
        if ref_tasks:
            results['REF/Acc'] = round(
                sum(collected[t]['Acc'] for t in ref_tasks) / len(ref_tasks) * 100, 2)

        # -- GND: mean F1(tvg, epm, tal, evs, vhd) --
        gnd_tasks = [t for t in ('tvg', 'epm', 'tal', 'evs', 'vhd') if t in collected]
        for t in gnd_tasks:
            results[f'{t.upper()}/F1'] = round(collected[t].get('F1', 0) * 100, 2)
        if gnd_tasks:
            results['GND/F1'] = round(
                sum(collected[t].get('F1', 0) for t in gnd_tasks) / len(gnd_tasks) * 100, 2)

        # -- CAP: per task F1 + NLP, then mean F1 and mean SentSim --
        cap_tasks = [t for t in ('dvc', 'slc') if t in collected]
        for t in cap_tasks:
            d = collected[t]
            results[f'{t.upper()}/F1'] = round(d.get('F1', 0) * 100, 2)
            results[f'{t.upper()}/SentSim'] = round(d.get('SentSim', 0) * 100, 2)
            for k in ('Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'):
                if k in d:
                    results[f'{t.upper()}/{k}'] = round(d[k] * 100, 2)
        if cap_tasks:
            results['CAP/F1'] = round(
                sum(collected[t].get('F1', 0) for t in cap_tasks) / len(cap_tasks) * 100, 2)
            results['CAP/SentSim'] = round(
                sum(collected[t].get('SentSim', 0) for t in cap_tasks) / len(cap_tasks) * 100, 2)

        # -- COM: tem mRec + gvq mRec → mean Rec --
        com_tasks = [t for t in ('tem', 'gvq') if t in collected]
        for t in com_tasks:
            d = collected[t]
            results[f'{t.upper()}/mRec'] = round(d.get('mRec', 0) * 100, 2)
            if 'Acc' in d:
                results[f'{t.upper()}/Acc'] = round(d['Acc'] * 100, 2)
        if com_tasks:
            results['COM/mRec'] = round(
                sum(collected[t].get('mRec', 0) for t in com_tasks) / len(com_tasks) * 100, 2)

        # -- AVG across four groups --
        group_scores = []
        if 'REF/Acc'    in results: group_scores.append(results['REF/Acc'])
        if 'GND/F1'     in results: group_scores.append(results['GND/F1'])
        if 'CAP/F1'     in results: group_scores.append(results['CAP/F1'])
        if 'COM/mRec'   in results: group_scores.append(results['COM/mRec'])
        if group_scores:
            results['AVG'] = round(sum(group_scores) / len(group_scores), 2)

        # ---------- print paper-style summary ----------
        print('\nETBench Evaluation Results (official metric alignment):')
        for k, v in results.items():
            print(f'  {k}: {v:.2f}')

        # Paper summary row
        cols = ['REF/Acc', 'GND/F1', 'CAP/F1', 'COM/mRec', 'AVG']
        row_vals = [results.get(c, '-') for c in cols]
        header = ' | '.join(f'{c:>10}' for c in ['REF', 'GND', 'CAP', 'COM', 'AVG'])
        values = ' | '.join(
            f'{v:>10.1f}' if isinstance(v, float) else f'{v:>10}' for v in row_vals
        )
        print(f'\n  {"E.T. Bench":>10}')
        print(f'  {header}')
        print(f'  {values}')

        # Save
        acc_file = get_intermediate_file_path(eval_file, '_etbench_acc')
        dump(
            pd.DataFrame([{'metric': k, 'value': v} for k, v in results.items()]),
            acc_file,
        )
        return results

