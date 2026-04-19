import json
import os.path as osp
from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

FRAMES_TMPL = (
    'These are the frames of a video. '
    'Answer the following question based on the video.'
)


class Vinoground(VideoBaseDataset):
    """Vinoground benchmark — contrastive video-language understanding.

    Two question types (text-score, video-score) × two variants (pos, neg)
    yield 2000 total entries from 500 base samples.

    Expected data layout under the data root::

        vinoground_videos/vinoground_textscore.json
        vinoground_videos/<video>.mp4
        vinoground_videos_concated/vinoground_videoscore.json
        vinoground_videos_concated/<video>.mp4
    """

    TYPE = 'Video-MCQ'

    _DATA_ROOT = '/m2v_intern/xuboshen/zgw/Benchmarks/Vinoground'

    def __init__(self, dataset: str = 'Vinoground', nframe: int = 8, fps: int = -1, adaptive: bool = False):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)

    @classmethod
    def supported_datasets(cls):
        return ['Vinoground']

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self, dataset_name: str = 'Vinoground'):
        if osp.exists(self._DATA_ROOT):
            data_root = self._DATA_ROOT
        else:
            lmu_root = LMUDataRoot()
            data_root = osp.join(lmu_root, 'Vinoground')

        tsv_file = osp.join(data_root, f'{dataset_name}.tsv')

        if not osp.exists(tsv_file):
            text_json = osp.join(data_root, 'vinoground_videos', 'vinoground_textscore.json')
            video_json = osp.join(data_root, 'vinoground_videos_concated', 'vinoground_videoscore.json')

            assert osp.exists(text_json), (
                f'Vinoground text-score annotation not found: {text_json}\n'
                f'Please place data under {data_root}/'
            )
            assert osp.exists(video_json), (
                f'Vinoground video-score annotation not found: {video_json}\n'
                f'Please place data under {data_root}/'
            )

            with open(text_json, 'r') as fh:
                text_items = json.load(fh)
            with open(video_json, 'r') as fh:
                video_items = json.load(fh)

            # Optionally enrich with major/minor categories from HuggingFace
            meta_map: dict = {}
            try:
                from datasets import load_dataset
                hf_data = load_dataset('HanSolo9682/Vinoground', split='lmmseval')
                for item in hf_data:
                    meta_map[item['index']] = {
                        'major': item.get('major', '') or '',
                        'minor': item.get('minor', '') or '',
                    }
            except Exception as exc:
                logging.warning(
                    f'[Vinoground] Could not load HF metadata (major/minor will be empty): {exc}'
                )

            rows = []
            for items, q_type in [(text_items, 'text'), (video_items, 'video')]:
                for item in items:
                    raw_idx = str(item['idx'])          # e.g. "0_pos"
                    parts = raw_idx.split('_')
                    base_idx = int(parts[0])            # 0-499
                    variant = parts[1]                  # pos / neg

                    hf_index = f'{base_idx}_{variant}_{q_type}'  # "0_pos_text"
                    video_path = item['video_name']     # relative path from data root
                    video_stem = osp.splitext(osp.basename(video_path))[0]

                    meta = meta_map.get(hf_index, {})
                    rows.append({
                        'index': hf_index,
                        'idx': base_idx,
                        'question_type': q_type,
                        'variant': variant,
                        'video': video_stem,
                        'video_path': video_path,
                        'question': item['question'],
                        'answer': item['GT'],
                        'major': meta.get('major', ''),
                        'minor': meta.get('minor', ''),
                    })

            df = pd.DataFrame(rows)
            os.makedirs(data_root, exist_ok=True)
            df.to_csv(tsv_file, sep='\t', index=False)

        return dict(data_file=tsv_file, root=data_root)

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def save_video_frames(self, video: str, video_llm: bool = False):
        """Extract frames for *video* (basename stem) and return (frame_paths, indices, video_info)."""
        import decord

        # Resolve full video path via the video_path column stored in data
        matches = self.data[self.data['video'] == video]
        if len(matches) == 0:
            raise FileNotFoundError(f'[Vinoground] Video stem "{video}" not found in dataset table.')
        video_path_rel = str(matches.iloc[0]['video_path'])
        vid_path = osp.join(self.data_root, video_path_rel)

        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }

        if self.adaptive:
            indices = self.compute_adaptive_indices(vid)
            frame_paths = self.frame_paths_adaptive(video, len(indices))
            _strategy = getattr(self, '_last_adaptive_strategy', 'adaptive')
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))
            _strategy = f'{self.fps}fps (dur={total_duration:.1f}s)'
        else:
            # Uniform nframe sampling (default nframe=8)
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
            _strategy = f'uniform nframe={self.nframe}'

        # In video_llm mode we skip the expensive decode + PNG write
        if video_llm:
            logging.debug(f'[frames] {video}: {len(indices)} frames ({_strategy}) [video_llm, skip decode]')
            return frame_paths, indices, video_info

        flag = np.all([osp.exists(p) for p in frame_paths])
        if not flag:
            lock_path = osp.join(self.frame_root, video + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)
            logging.info(f'[frames] {video}: {len(frame_paths)} frames ({_strategy}) [extracted]')
        else:
            logging.info(f'[frames] {video}: {len(frame_paths)} frames ({_strategy}) [cached]')

        return frame_paths, indices, video_info

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, line, video_llm: bool = False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, _indices, _video_info = self.save_video_frames(line['video'], video_llm)

        message = []
        if video_llm:
            vid_path = osp.join(self.data_root, str(line['video_path']))
            message.append(self.make_video_struct(vid_path, video_id=line['video']))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        text_prompt = (
            f'{FRAMES_TMPL}\n'
            f'{line["question"]}\n'
            'Please only output one English character (A or B).'
        )
        message.append(dict(type='text', value=text_prompt))
        return message

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @classmethod
    def evaluate(cls, eval_file: str, **judge_kwargs):
        from .utils.vinoground import extract_characters_regex, get_vinoground_scores
        from vlmeval.utils.matching_util import extract_answer_from_cot

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be a supported format (xlsx/json/tsv)'
        )

        score_file = get_intermediate_file_path(eval_file, '_score')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')

        if not osp.exists(score_file):
            data = load(eval_file)

            unparsed_count = 0
            for pandas_idx, row in data.iterrows():
                raw_pred = row.get('prediction', '')
                if pd.isna(raw_pred):
                    data.loc[pandas_idx, 'extracted_answer'] = ''
                    data.loc[pandas_idx, 'score'] = 0
                    unparsed_count += 1
                    continue

                pred = str(raw_pred).strip()
                ans = str(row.get('answer', '')).strip().upper()

                if not pred or pred.lower() == 'nan':
                    data.loc[pandas_idx, 'extracted_answer'] = ''
                    data.loc[pandas_idx, 'score'] = 0
                    unparsed_count += 1
                    continue

                extracted = extract_characters_regex(pred)
                if not extracted:
                    extracted = extract_answer_from_cot(pred)
                if not extracted:
                    unparsed_count += 1

                data.loc[pandas_idx, 'extracted_answer'] = extracted
                data.loc[pandas_idx, 'score'] = int(extracted == ans) if extracted else 0

            if unparsed_count > 0:
                print(
                    f'[Vinoground] WARNING: Failed to parse answer for '
                    f'{unparsed_count}/{len(data)} samples'
                )

            dump(data, score_file)
            jsonl_file = score_file.rsplit('.', 1)[0] + '.jsonl'
            dump(data, jsonl_file)

        rating = get_vinoground_scores(score_file)
        dump(rating, tgt_file)

        print(
            f'[Vinoground] text_score={rating["text_score"]:.2f}  '
            f'video_score={rating["video_score"]:.2f}  '
            f'group_score={rating["group_score"]:.2f}'
        )
        return rating
