from abc import abstractmethod
from ..smp import *


class VideoBaseDataset:

    MODALITY = 'VIDEO'

    def __init__(self,
                 dataset='MMBench-Video',
                 pack=False,
                 nframe=0,
                 fps=-1,
                 adaptive=False):
        try:
            import decord
        except Exception as e:
            logging.critical(f'{type(e)}: {e}')
            logging.critical('Please install decord via `pip install decord`.')

        self.dataset_name = dataset
        ret = self.prepare_dataset(dataset)
        assert ret is not None
        lmu_root = LMUDataRoot()
        self.frame_root = osp.join(lmu_root, 'images', dataset)
        os.makedirs(self.frame_root, exist_ok=True)
        self.frame_tmpl = 'frame-{}-of-{}.jpg'
        self.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'
        self.frame_tmpl_adaptive = 'frame-{}-of-{}-adaptive.jpg'

        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)
        if 'index' not in self.data:
            self.data['index'] = np.arange(len(self.data))

        assert 'question' in self.data and 'video' in self.data
        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos
        self.pack = pack
        self.adaptive = adaptive
        if self.adaptive:
            # Adaptive mode: per-sample structs carry fps/nframes;
            # clear dataset-level defaults so inference sync doesn't
            # override model settings.
            self.nframe = 0
            self.fps = -1
        else:
            self.nframe = nframe
            self.fps = fps
        self.frame_info = {}  # video_id -> {'num_frames': int, 'strategy': str}
        if not self.adaptive:
            if self.fps > 0 and self.nframe > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if self.fps <= 0 and self.nframe <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')

    def __len__(self):
        return len(self.videos) if self.pack else len(self.data)

    def __getitem__(self, idx):
        if self.pack:
            assert idx < len(self.videos)
            sub_data = self.data[self.data['video'] == self.videos[idx]]
            return sub_data
        else:
            assert idx < len(self.data)
            return dict(self.data.iloc[idx])

    def frame_paths(self, video):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe)) for i in range(1, self.nframe + 1)]

    def frame_paths_fps(self, video, num_frames):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root,
                         self.frame_tmpl_fps.format(i, num_frames, self.fps)) for i in range(1, num_frames + 1)]

    def frame_paths_adaptive(self, video, num_frames):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root,
                         self.frame_tmpl_adaptive.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def _record_frame_info(self, video_id, num_frames, strategy):
        """Record per-video frame info for output JSONL columns."""
        self.frame_info[video_id] = {'num_frames': num_frames, 'strategy': strategy}

    def compute_adaptive_indices(self, vid):
        """Compute frame indices using adaptive strategy based on video duration.

        Strategy: ≤128s → 2fps, 128–256s → 1fps, >256s → uniform 256 frames.
        """
        total_frames = len(vid)
        video_fps = vid.get_avg_fps()
        duration = total_frames / video_fps

        if duration <= 128:
            fps = 2.0
            strategy = f'2fps (duration={duration:.1f}s)'
            step_size = video_fps / fps
            required = max(1, int(duration * fps))
            indices = [int(i * step_size) for i in range(required)]
        elif duration <= 256:
            fps = 1.0
            strategy = f'1fps (duration={duration:.1f}s)'
            step_size = video_fps / fps
            required = max(1, int(duration * fps))
            indices = [int(i * step_size) for i in range(required)]
        else:
            nframe = 256
            strategy = f'uniform-256 (duration={duration:.1f}s)'
            step_size = total_frames / (nframe + 1)
            indices = [int(i * step_size) for i in range(1, nframe + 1)]

        indices = [min(i, total_frames - 1) for i in indices]
        self._last_adaptive_strategy = strategy
        self._last_adaptive_num_frames = len(indices)
        return indices

    def get_adaptive_video_kwargs(self, video_path, video_id=None):
        """Return dict with fps/nframes for adaptive mode, to embed in video_llm prompt structs.

        For ≤30s videos returns {'fps': 2.0}, for 30-256s {'fps': 1.0},
        for >256s {'nframes': 256}.
        Also records frame info in self.frame_info keyed by video_id.
        """
        import decord
        vid = decord.VideoReader(video_path)
        total_frames = len(vid)
        video_fps = vid.get_avg_fps()
        duration = total_frames / video_fps
        key = video_id or video_path
        if duration <= 128:
            nf = max(1, int(duration * 2.0))
            self.frame_info[key] = {'num_frames': nf, 'strategy': f'2fps (dur={duration:.1f}s)'}
            return {'fps': 2.0}
        elif duration <= 256:
            nf = max(1, int(duration * 1.0))
            self.frame_info[key] = {'num_frames': nf, 'strategy': f'1fps (dur={duration:.1f}s)'}
            return {'fps': 1.0}
        else:
            self.frame_info[key] = {'num_frames': 256, 'strategy': f'uniform-256 (dur={duration:.1f}s)'}
            return {'nframes': 256}

    def make_video_struct(self, video_path, video_id=None):
        """Create a video prompt struct dict, with adaptive kwargs if adaptive mode is enabled."""
        d = dict(type='video', value=video_path)
        if self.adaptive:
            d.update(self.get_adaptive_video_kwargs(video_path, video_id=video_id))
        return d

    def save_video_frames(self, video):
        import decord
        if self.adaptive:
            vid_path = osp.join(self.data_root, video + '.mp4')
            vid = decord.VideoReader(vid_path)
            indices = self.compute_adaptive_indices(vid)
            frame_paths = self.frame_paths_adaptive(video, len(indices))
            strategy = getattr(self, '_last_adaptive_strategy', 'adaptive')

            if np.all([osp.exists(p) for p in frame_paths]):
                logging.info(f'[frames] {video}: {len(frame_paths)} frames ({strategy}) [cached]')
                return frame_paths

            lock_path = osp.join(self.frame_root, video + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    logging.info(f'[frames] {video}: {len(frame_paths)} frames ({strategy}) [cached]')
                    return frame_paths
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            logging.info(f'[frames] {video}: {len(frame_paths)} frames ({strategy}) [extracted]')
            return frame_paths

        if self.fps > 0:
            vid_path = osp.join(self.data_root, video + '.mp4')
            vid = decord.VideoReader(vid_path)

            # 计算视频的总帧数和总时长
            total_frames = len(vid)
            video_fps = vid.get_avg_fps()
            total_duration = total_frames / video_fps

            # 计算需要提取的总帧数
            required_frames = int(total_duration * self.fps)

            # 计算提取帧的间隔
            step_size = video_fps / self.fps

            # 计算提取帧的索引
            indices = [int(i * step_size) for i in range(required_frames)]

            # 提取帧并保存
            frame_paths = self.frame_paths_fps(video, len(indices))
            flag = np.all([osp.exists(p) for p in frame_paths])
            _fps_str = f'{self.fps}fps, dur={total_duration:.1f}s'
            if flag:
                logging.info(f'[frames] {video}: {len(frame_paths)} frames ({_fps_str}) [cached]')
                return frame_paths

            lock_path = osp.join(self.frame_root, video + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    logging.info(f'[frames] {video}: {len(frame_paths)} frames ({_fps_str}) [cached]')
                    return frame_paths
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            logging.info(f'[frames] {video}: {len(frame_paths)} frames ({_fps_str}) [extracted]')
            return frame_paths

        else:
            frame_paths = self.frame_paths(video)
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                logging.info(f'[frames] {video}: {len(frame_paths)} frames (uniform nframe={self.nframe}) [cached]')
                return frame_paths
            lock_path = osp.join(self.frame_root, video + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    logging.info(f'[frames] {video}: {len(frame_paths)} frames (uniform nframe={self.nframe}) [cached]')
                    return frame_paths
                vid_path = osp.join(self.data_root, video + '.mp4')
                vid = decord.VideoReader(vid_path)
                step_size = len(vid) / (self.nframe + 1)
                indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            logging.info(f'[frames] {video}: {len(frame_paths)} frames (uniform nframe={self.nframe}) [extracted]')
            return frame_paths

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return ['MMBench-Video', 'Video-MME', 'MVBench', 'MVBench_MP4',
                'LongVideoBench', 'WorldSense', 'VDC', 'MovieChat1k']

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    @abstractmethod
    def build_prompt(self, idx):
        pass

    @abstractmethod
    def prepare_dataset(self, dataset):
        # The prepare_dataset function should return a dictionary containing:
        # `root` (directory that containing video files)
        # `data_file` (the TSV dataset file)
        pass
