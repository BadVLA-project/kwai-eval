import json
import math
import warnings

from huggingface_hub import snapshot_download
from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .videomme_v2_utils import (
    resolve_videommev2_paths,
    resolve_videommev2_video_path,
    videommev2_video_relpath,
)

FAIL_MSG = 'Failed to obtain answer via API.'


class VideoMMEv2(VideoBaseDataset):

    SYS = ''

    FRAMES_TMPL_NOSUB = (
        'These are the frames of a video. '
        'Select the best answer to the following multiple-choice question based on the video. '
        'Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.'
    )

    FRAMES_TMPL_SUB = (
        'These are the frames of a video. '
        "This video's subtitles are listed below:\n{}\n"
        'Select the best answer to the following multiple-choice question based on the video. '
        'Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.'
    )

    FRAMES_TMPL_INTERLEAVE = (
        'These are the frames of a video with corresponding subtitles shown between frames. '
        'The subtitles indicate what is being said during the time interval between adjacent frames. '
        'Select the best answer to the following multiple-choice question based on the video. '
        'Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.'
    )

    FRAMES_TMPL_REASONING = (
        'Please perform a detailed reasoning based on the provided video frames to answer the following '
        'multiple-choice question selecting the best option from A through H and providing your final response '
        "strictly in the format: 'Final Answer: <letter>."
    )

    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def __init__(self, dataset='Video-MME-v2', nframe=64, fps=-1,
                 with_subtitle=False, subtitle_interleave=False, reasoning=False,
                 resize_target_area=False, use_subtitle=None, subtitle_mode=None,
                 adaptive=False):
        if use_subtitle is not None:
            with_subtitle = use_subtitle
        if subtitle_mode is not None:
            subtitle_interleave = subtitle_mode == 'interleave'
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)
        self.use_subtitle = with_subtitle
        self.with_subtitle = with_subtitle
        self.subtitle_interleave = subtitle_interleave
        self.subtitle_mode = 'interleave' if subtitle_interleave else 'concat'
        self.reasoning = reasoning
        self.response_prompt = self.FRAMES_TMPL_REASONING if reasoning else ''
        self.resize_target_area = resize_target_area
        self.dataset_name = dataset
        if self.resize_target_area:
            self.frame_root_resize = self.frame_root + f'_resize{self.resize_target_area}'
            os.makedirs(self.frame_root_resize, exist_ok=True)

    @classmethod
    def supported_datasets(cls):
        return ['Video-MME-v2']

    def prepare_dataset(self, dataset_name='Video-MME-v2',
                        repo_id='MME-Benchmarks/Video-MME-v2'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not os.path.exists(data_file):
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        def unzip_videos(pth):
            import zipfile
            target_dir = os.path.join(pth, 'video/')
            if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
                print(f'Video folder {target_dir} exists. Skipping unzip.')
                return

            zip_files = [
                os.path.join(pth, f) for f in os.listdir(pth)
                if f.endswith('.zip') and f.startswith('video')
            ]
            zip_files.sort()

            if zip_files:
                os.makedirs(target_dir, exist_ok=True)
                print(f'Unzipping {len(zip_files)} video zip files...')
                for zip_file in zip_files:
                    try:
                        with zipfile.ZipFile(zip_file, 'r') as zf:
                            for member in zf.namelist():
                                if not member.endswith('/'):
                                    source = zf.open(member)
                                    target = open(os.path.join(target_dir, os.path.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    except Exception as e:
                        print(f'Error unzipping {zip_file}: {e}')
                print('Video files restored.')

        def unzip_subtitle_from_source(paths):
            if not paths or not paths.subtitle_zip:
                return
            if os.path.exists(paths.subtitle_dir) and len(os.listdir(paths.subtitle_dir)) > 0:
                return

            import zipfile
            os.makedirs(paths.subtitle_dir, exist_ok=True)
            with zipfile.ZipFile(paths.subtitle_zip, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.endswith('/'):
                        continue
                    source = zip_ref.open(member)
                    target = open(os.path.join(paths.subtitle_dir, os.path.basename(member)), 'wb')
                    with source, target:
                        target.write(source.read())
            print(f'Subtitle files restored to {paths.subtitle_dir}.')

        def generate_tsv(
            pth,
            data_file=None,
            parquet_file=None,
            source_root=None,
            subtitle_dir='./subtitle',
            overwrite=False,
        ):
            data_file = data_file or osp.join(pth, f'{dataset_name}.tsv')
            if os.path.exists(data_file) and not overwrite:
                return data_file

            if parquet_file is None:
                for candidate in [
                    osp.join(pth, 'test.parquet'),
                    osp.join(pth, 'test-00000-of-00001.parquet'),
                    osp.join(pth, 'videommev2', 'test-00000-of-00001.parquet'),
                    osp.join(pth, 'data', 'test-00000-of-00001.parquet'),
                ]:
                    if os.path.exists(candidate):
                        parquet_file = candidate
                        break

            if parquet_file is None:
                # Try finding any parquet file
                for root, dirs, files in os.walk(pth):
                    for f in files:
                        if f.endswith('.parquet'):
                            parquet_file = os.path.join(root, f)
                            break
                    if parquet_file:
                        break

            if parquet_file is None:
                print(f'Warning: No parquet file found in {pth}, cannot generate TSV.')
                return data_file

            print(f'Generating TSV from {parquet_file}...')
            df = pd.read_parquet(parquet_file)
            df = df.assign(index=range(len(df)))
            df['video'] = df['video_id'].apply(str)
            root_for_video = source_root or pth
            df['video_path'] = df['video_id'].apply(lambda x: videommev2_video_relpath(root_for_video, x))
            if os.path.isabs(subtitle_dir) and source_root is not None:
                df['subtitle_path'] = df['video_id'].apply(
                    lambda x: os.path.join(subtitle_dir, f'{x}.jsonl')
                )
            else:
                subtitle_rel = os.path.relpath(subtitle_dir, source_root or pth)
                df['subtitle_path'] = df['video_id'].apply(
                    lambda x: './' + os.path.join(subtitle_rel, f'{x}.jsonl').replace(os.sep, '/')
                )

            # options may be a numpy array or list; convert to string representation
            if 'options' in df.columns:
                df['options'] = df['options'].apply(
                    lambda x: str(list(x)) if not isinstance(x, str) else x
                )

            keep_cols = [
                'index', 'video', 'video_path', 'question', 'options', 'answer',
                'level', 'group_type', 'group_structure',
                'second_head', 'third_head', 'subtitle_path'
            ]
            df = df[[c for c in keep_cols if c in df.columns]]
            os.makedirs(osp.dirname(data_file), exist_ok=True)
            df.to_csv(data_file, sep='\t', index=False)
            print(f'TSV generated: {data_file}')
            return data_file

        local_paths = resolve_videommev2_paths(dataset_name=dataset_name)

        if local_paths is not None:
            print(f'Loading Video-MME-v2 from local read path: {local_paths.source_root}')
            print(f'Writing Video-MME-v2 artifacts to: {local_paths.artifact_root}')
            dataset_path = local_paths.source_root
            self.artifact_root = local_paths.artifact_root
            self.subtitle_root = local_paths.subtitle_dir
            unzip_subtitle_from_source(local_paths)
            data_file = generate_tsv(
                local_paths.source_root,
                data_file=local_paths.tsv_file,
                parquet_file=local_paths.parquet_file,
                source_root=local_paths.source_root,
                subtitle_dir=local_paths.subtitle_dir,
                overwrite=True,
            )
        else:
            cache_path = get_cache_path(repo_id)
            if cache_path is not None and check_integrity(cache_path):
                dataset_path = cache_path
                data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
            else:
                if modelscope_flag_set():
                    from modelscope import dataset_snapshot_download
                    dataset_path = dataset_snapshot_download(dataset_id=repo_id)
                else:
                    dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
                unzip_videos(dataset_path)
                data_file = generate_tsv(dataset_path)

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):
        vid_path = resolve_videommev2_video_path(self.data_root, video)
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }

        if self.adaptive:
            indices = self.compute_adaptive_indices(vid)
            frame_paths = self._frame_paths_adaptive_resize(video, len(indices)) if self.resize_target_area \
                else self.frame_paths_adaptive(video, len(indices))
        elif self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self._frame_paths_resize(video) if self.resize_target_area else self.frame_paths(video)
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self._frame_paths_fps_resize(video, len(indices)) if self.resize_target_area \
                else self.frame_paths_fps(video, len(indices))

        # video_llm mode: frames are not needed, skip expensive decode + PNG save.
        if video_llm:
            return frame_paths, indices, video_info

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    if self.resize_target_area:
                        images = [self._resize_to_target_area(im, self.resize_target_area) for im in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths, indices, video_info

    @staticmethod
    def _resize_to_target_area(img, target_area, divisor=16):
        """Resize a PIL image keeping aspect ratio and divisor-aligned dimensions."""
        w, h = img.size
        scale = math.sqrt(target_area / (w * h))
        new_w = max(divisor, round(w * scale / divisor) * divisor)
        new_h = max(divisor, round(h * scale / divisor) * divisor)
        if new_w == w and new_h == h:
            return img
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _frame_paths_resize(self, video):
        frame_root = osp.join(self.frame_root_resize, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe)) for i in range(1, self.nframe + 1)]

    def _frame_paths_fps_resize(self, video, num_frames):
        frame_root = osp.join(self.frame_root_resize, video)
        os.makedirs(frame_root, exist_ok=True)
        return [
            osp.join(frame_root, self.frame_tmpl_fps.format(i, num_frames, self.fps))
            for i in range(1, num_frames + 1)
        ]

    def _frame_paths_adaptive_resize(self, video, num_frames):
        frame_root = osp.join(self.frame_root_resize, video)
        os.makedirs(frame_root, exist_ok=True)
        return [
            osp.join(frame_root, self.frame_tmpl_adaptive.format(i, num_frames))
            for i in range(1, num_frames + 1)
        ]

    @staticmethod
    def _resize_kwargs_for_video(video_path, target_area=448 * 448, divisor=16):
        import decord
        vid = decord.VideoReader(video_path)
        h, w = vid[0].shape[:2]
        scale = math.sqrt(target_area / (w * h))
        new_w = max(divisor, round(w * scale / divisor) * divisor)
        new_h = max(divisor, round(h * scale / divisor) * divisor)
        return dict(resized_height=new_h, resized_width=new_w)

    @staticmethod
    def _load_subtitle_jsonl(subtitle_path):
        """Load JSONL subtitle file with word-level timestamps."""
        if not osp.exists(subtitle_path):
            return None
        entries = []
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries if entries else None

    @staticmethod
    def _group_subtitle_segments(entries, gap_threshold=0.5):
        """Group word-level JSONL entries into sentence-level segments."""
        if not entries:
            return []

        segments = []
        current_words = [entries[0]]

        for i in range(1, len(entries)):
            prev = entries[i - 1]
            curr = entries[i]
            time_gap = curr['start_time'] - prev['end_time']
            prev_ends_sentence = prev['text'].rstrip().endswith(('.', '!', '?'))

            if time_gap > gap_threshold or (prev_ends_sentence and time_gap > 0.1):
                segments.append({
                    'text': ' '.join(w['text'] for w in current_words),
                    'start_time': current_words[0]['start_time'],
                    'end_time': current_words[-1]['end_time'],
                })
                current_words = [curr]
            else:
                current_words.append(curr)

        if current_words:
            segments.append({
                'text': ' '.join(w['text'] for w in current_words),
                'start_time': current_words[0]['start_time'],
                'end_time': current_words[-1]['end_time'],
            })

        return segments

    @staticmethod
    def _build_subtitle_concat(entries):
        """Concatenate all subtitle text into one string."""
        if not entries:
            return ''
        return ' '.join(e['text'] for e in entries)

    def _build_subtitle_interleave(self, entries, frame_paths, indices, video_info):
        """Build interleaved frame + subtitle message list."""
        segments = self._group_subtitle_segments(entries)
        vid_fps = video_info['fps']
        frame_timestamps = [idx / vid_fps for idx in indices]

        message_parts = []
        for i, (fp, frame_ts) in enumerate(zip(frame_paths, frame_timestamps)):
            if i < len(frame_timestamps) - 1:
                end_ts = frame_timestamps[i + 1]
            else:
                end_ts = video_info['n_frames'] / vid_fps

            message_parts.append(dict(type='image', value=fp))

            for seg in segments:
                if seg['end_time'] >= frame_ts and seg['start_time'] < end_ts:
                    message_parts.append(dict(
                        type='text',
                        value=f'[Subtitle {seg["start_time"]:.2f}s - {seg["end_time"]:.2f}s]: {seg["text"]}'
                    ))

        return message_parts

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm)

        # Load subtitles if needed
        sub_entries = None
        if self.use_subtitle:
            sub_path = str(line['subtitle_path'])
            if sub_path.startswith('./'):
                sub_path = sub_path[2:]
            if not osp.isabs(sub_path):
                sub_path = osp.join(self.data_root, sub_path)
            if os.path.exists(sub_path):
                sub_entries = self._load_subtitle_jsonl(sub_path)

        message = [dict(type='text', value=self.SYS)]

        if video_llm:
            video_path = resolve_videommev2_video_path(self.data_root, line['video'])
            video_msg = self.make_video_struct(video_path, video_id=line['video'])
            target_area = self.resize_target_area or 448 * 448
            video_msg.update(self._resize_kwargs_for_video(video_path, target_area=target_area))
            if not self.adaptive:
                if self.nframe > 0:
                    video_msg['nframes'] = self.nframe
                if self.fps > 0:
                    video_msg['fps'] = self.fps
            message.append(video_msg)
        else:
            if self.use_subtitle and self.subtitle_mode == 'interleave' and sub_entries:
                interleaved = self._build_subtitle_interleave(
                    sub_entries, frames, indices, video_info)
                message.extend(interleaved)
            else:
                for im in frames:
                    message.append(dict(type='image', value=im))

        # Select text prompt
        if self.reasoning:
            text_prompt = self.FRAMES_TMPL_REASONING
        elif self.use_subtitle and sub_entries:
            if self.subtitle_interleave and not video_llm:
                text_prompt = self.FRAMES_TMPL_INTERLEAVE
            else:
                full_text = self._build_subtitle_concat(sub_entries)
                text_prompt = self.FRAMES_TMPL_SUB.format(full_text)
        else:
            text_prompt = self.FRAMES_TMPL_NOSUB

        message.append(dict(type='text', value=text_prompt))

        # Build question with 8 options (A-H)
        options = eval(line['options']) if isinstance(line['options'], str) else list(line['options'])
        option_lines = [f'{chr(ord("A") + i)}. {opt}' for i, opt in enumerate(options)]
        question_text = str(line['question']) + '\n' + '\n'.join(option_lines)
        prompt = f'Question: {question_text}\nAnswer: '
        message.append(dict(type='text', value=prompt))

        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.videomme_v2 import (
            get_final_rating_v2, extract_characters_regex_v2, extract_option_v2
        )

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be an supported format (xlsx/json/tsv) file'

        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')

            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None

            from vlmeval.utils.matching_util import extract_answer_from_cot

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            unparsed_count = 0
            for idx in data['index']:
                ans = str(data.loc[data['index'] == idx, 'answer'].values[0])
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                extract_pred = extract_characters_regex_v2(pred)
                if extract_pred == '':
                    extract_pred = extract_answer_from_cot(pred, valid_options='ABCDEFGH')
                if extract_pred == '':
                    extract_pred = extract_option_v2(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'Video-MME-v2'
                    )
                    if not extract_pred:
                        unparsed_count += 1

                data.loc[data['index'] == idx, 'extracted_answer'] = extract_pred
                data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans) if extract_pred else -1

            rejected = [x for x in data['score'] if x == -1]

            if unparsed_count > 0:
                print(f'[Video-MME-v2] WARNING: Failed to parse answer for {unparsed_count}/{len(data)} samples')

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)
            jsonl_file = score_file.rsplit('.', 1)[0] + '.jsonl'
            dump(data, jsonl_file)

        rating = get_final_rating_v2(score_file)
        dump(rating, tgt_file)
        return rating
