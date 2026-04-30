import ast
import json
import os
import re

from huggingface_hub import snapshot_download

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset


TASK_CATEGORIES = [
    'Cross-video Anomaly Detection',
    'Cross-video Scene Recognition',
    'Multi-video Key-Action Recognition',
    'Cross-video Event Retrieval',
    'Cross-video Object Recognition',
    'Multi-video Attribute Recognition',
    'Joint-video Counting',
    'Cross-video Entity Matching',
    'Multi-view Scene Understanding',
    'Multi-video Temporal Reasoning',
    'Joint-video Spatial Navigating',
    'Video Difference Caption',
    'Cross-video Counterfactual Reasoning',
    'Joint-video Summarization',
    'Cross-video Procedural Transfer',
]


class CVBenchVideo(VideoBaseDataset):
    """CVBench cross-video reasoning benchmark.

    This is the CVBench/MVR benchmark from Hokhim2/CVBench, not the image
    CV-Bench-2D/3D benchmark already registered as ``CVBench``.
    """

    TYPE = 'Video-MCQ'
    DEFAULT_ROOT_CANDIDATES = (
        '/m2v_intern/xuboshen/zgw/Benchmarks/CVBench',
        '/m2v_intern/xuboshen/zgw/LMUData/datasets/CVBench',
    )

    def __init__(self, dataset='CVBench', nframe=0, fps=-1, adaptive=False):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)

    @classmethod
    def supported_datasets(cls):
        return ['CVBench']

    @staticmethod
    def _first_existing_path(paths):
        for pth in paths:
            if pth and osp.exists(pth):
                return pth
        return None

    @staticmethod
    def _load_hf_dataset(dataset_path):
        from datasets import Dataset, load_from_disk

        if osp.isdir(dataset_path):
            try:
                loaded = load_from_disk(dataset_path)
                if hasattr(loaded, 'keys'):
                    if 'test' in loaded:
                        return loaded['test']
                    first_key = next(iter(loaded.keys()))
                    return loaded[first_key]
                return loaded
            except Exception:
                pass

        arrow_file = osp.join(dataset_path, 'test', 'data-00000-of-00001.arrow')
        if osp.exists(arrow_file):
            return Dataset.from_file(arrow_file)

        raise FileNotFoundError(
            f'[CVBench] Could not load HF-format annotations from {dataset_path}. '
            f'Expected load_from_disk() dataset or {arrow_file}.'
        )

    @staticmethod
    def _normalize_options(options):
        if isinstance(options, str):
            try:
                parsed = ast.literal_eval(options)
                if isinstance(parsed, (list, tuple)):
                    options = parsed
            except Exception:
                options = [x.strip() for x in options.split('\n') if x.strip()]
        return [str(x).strip() for x in options if str(x).strip()]

    @classmethod
    def _extract_answer(cls, prediction):
        if pd.isna(prediction):
            return ''
        text = str(prediction).strip()
        prefixes = [
            'The best answer is',
            'The correct answer is',
            'The answer is',
            'The answer',
            'The best option is',
            'The correct option is',
            'Best answer:',
            'Best option:',
        ]
        for prefix in prefixes:
            text = text.replace(prefix, '')

        if len(text.split()) > 10 and not re.search(r'[ABCD]|YES|NO', text, flags=re.I):
            return ''
        match = re.search(r'(?i)([ABCD]|YES|NO)', text)
        return match.group(0).upper() if match else ''

    def _resolve_video_root(self, data_root):
        env_root = os.environ.get('CVBENCH_VIDEO_DIR', '').strip()
        candidates = [
            env_root,
            osp.join(data_root, 'CVBench'),
            osp.join(data_root, 'videos'),
            data_root,
        ]
        video_root = self._first_existing_path(candidates)
        if video_root is None:
            raise FileNotFoundError(
                '[CVBench] Video root not found. Set CVBENCH_VIDEO_DIR to the directory '
                'containing subfolders such as 102/*.mp4.'
            )
        return video_root

    def prepare_dataset(self, dataset_name='CVBench', repo_id='Dongyh35/CVBench'):
        local_root = os.environ.get('CVBENCH_DIR', '').strip()
        if local_root and osp.exists(local_root):
            data_root = local_root
        else:
            data_root = self._first_existing_path(self.DEFAULT_ROOT_CANDIDATES)

        if data_root is None:
            cache_path = get_cache_path(repo_id)
            if cache_path is None:
                if modelscope_flag_set():
                    raise RuntimeError(
                        '[CVBench] ModelScope download is not configured for CVBench. '
                        'Set CVBENCH_DIR/CVBENCH_VIDEO_DIR or use Hugging Face.'
                    )
                data_root = snapshot_download(repo_id=repo_id, repo_type='dataset')
            else:
                data_root = cache_path

        dataset_path = os.environ.get('CVBENCH_HF_DATASET_DIR', '').strip()
        if not dataset_path:
            dataset_path = osp.join(data_root, 'mvr_dataset')
        if not osp.exists(dataset_path):
            # The official repo stores the HF-format dataset at the repo root.
            dataset_path = data_root

        video_root = self._resolve_video_root(data_root)
        self.video_root = video_root
        tsv_file = osp.join(data_root, f'{dataset_name}.tsv')

        def check_integrity(pth):
            if not osp.exists(pth):
                return False
            data = load(pth)
            required = {'question', 'answer', 'options', 'video', 'video_paths'}
            if not required.issubset(set(data.columns)):
                return False
            for raw_paths in data['video_paths']:
                for video_pth in self._normalize_options(raw_paths):
                    if not osp.exists(osp.join(video_root, video_pth)):
                        return False
            return True

        if not check_integrity(tsv_file):
            hf_data = self._load_hf_dataset(dataset_path)
            rows = []
            for idx, item in enumerate(hf_data):
                video_paths = [
                    item.get(f'video_{i}') for i in range(1, 5)
                    if item.get(f'video_{i}') not in [None, '', 'None']
                ]
                if not video_paths:
                    raise ValueError(f'[CVBench] Sample {idx} has no videos: {item}')

                options = self._normalize_options(item.get('options', []))
                rows.append({
                    'index': item.get('id', idx),
                    'id': item.get('id', idx),
                    'task_type': item.get('task_type', ''),
                    'video': '|'.join([osp.splitext(osp.basename(x))[0] for x in video_paths]),
                    'video_paths': json.dumps(video_paths, ensure_ascii=False),
                    'question': item.get('question', ''),
                    'options': json.dumps(options, ensure_ascii=False),
                    'answer': str(item.get('answer', '')).strip(),
                })

            os.makedirs(data_root, exist_ok=True)
            pd.DataFrame(rows).to_csv(tsv_file, sep='\t', index=False)

        return dict(data_file=tsv_file, root=data_root)

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_paths = self._normalize_options(line['video_paths'])
        options = self._normalize_options(line['options'])
        is_yesno = all(opt.strip().strip('.').lower() in ['yes', 'no'] for opt in options)

        message = [dict(type='_managed_prompt')]
        for i, video_rel in enumerate(video_paths, start=1):
            video_abs = osp.join(self.video_root, video_rel)
            message.append(dict(type='text', value=f'Video {i}:'))
            if video_llm:
                video_struct = self.make_video_struct(video_abs, video_id=f'{line["index"]}_{i}')
                message.append(video_struct)
            else:
                # Current primary path is video-LLM/vLLM. Frame fallback keeps
                # the dataset usable for image-only backends by sampling each
                # video independently under a stable synthetic id.
                frames = self._save_frames_from_path(video_abs, f'{line["index"]}_{i}')
                message.extend([dict(type='image', value=frame) for frame in frames])

        if is_yesno:
            instruction = (
                'Select the best answer to the following yes-no question based on '
                'all listed videos. Respond with only the word (Yes or No) of the correct option.'
            )
            post_prompt = "Answer with the option's word (YES or NO) from the given choices directly."
        else:
            instruction = (
                'Select the best answer to the following multiple-choice question based on '
                'all listed videos. Respond with only the letter (A, B, C, or D) of the correct option.'
            )
            post_prompt = "Answer with the option's letter (A, B, C, or D) from the given choices directly."

        prompt = f'{instruction}\n{line["question"]}\n' + '\n'.join(options) + f'\n{post_prompt}'
        message.append(dict(type='text', value=prompt))
        return message

    def _save_frames_from_path(self, video_path, frame_id):
        import decord

        vid = decord.VideoReader(video_path)
        if self.adaptive:
            indices = self.compute_adaptive_indices(vid)
            frame_paths = self.frame_paths_adaptive(frame_id, len(indices))
        elif self.fps > 0:
            video_fps = vid.get_avg_fps()
            total_duration = len(vid) / video_fps
            required_frames = max(1, int(total_duration * self.fps))
            step_size = video_fps / self.fps
            indices = [min(int(i * step_size), len(vid) - 1) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(frame_id, len(indices))
        else:
            step_size = len(vid) / (self.nframe + 1)
            indices = [min(int(i * step_size), len(vid) - 1) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(frame_id)

        if not np.all([osp.exists(p) for p in frame_paths]):
            lock_path = osp.join(self.frame_root, frame_id + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)
        return frame_paths

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be a supported format (xlsx/json/tsv)'
        )

        score_file = get_intermediate_file_path(eval_file, '_score')
        rating_file = get_intermediate_file_path(eval_file, '_rating', 'csv')

        if not osp.exists(score_file):
            data = load(eval_file)
            for idx, row in data.iterrows():
                pred = cls._extract_answer(row.get('prediction', ''))
                ans = str(row.get('answer', '')).strip().upper()
                data.loc[idx, 'extracted_answer'] = pred
                data.loc[idx, 'score'] = int(pred == ans) if pred else 0
            dump(data, score_file)

        data = load(score_file)
        rows = []
        for task_type in TASK_CATEGORIES:
            sub = data[data['task_type'].astype(str).str.lower() == task_type.lower()]
            if len(sub) == 0:
                continue
            rows.append({
                'task_type': task_type,
                'success': int(sub['score'].sum()),
                'overall': int(len(sub)),
                'acc': round(float(sub['score'].mean()) * 100, 2),
            })

        rows.append({
            'task_type': 'Overall',
            'success': int(data['score'].sum()),
            'overall': int(len(data)),
            'acc': round(float(data['score'].mean()) * 100, 2),
        })
        rating = pd.DataFrame(rows)
        dump(rating, rating_file)
        return rating
