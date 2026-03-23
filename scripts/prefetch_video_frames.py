"""Pre-extract video frames using multiple CPU processes.

Usage:
    python scripts/prefetch_video_frames.py \
        --dataset Video-MME --nframe 16 [--fps -1] \
        [--durations short medium] \
        [--workers 0]   # 0 = use all available CPU cores

This decodes videos and saves frames to the LMUData images cache so that
the actual evaluation run can skip the decoding step entirely.
"""

import argparse
import os
import os.path as osp
import sys
from multiprocessing import Pool

# Ensure the project root is on sys.path
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))


def decode_one_video(args):
    """Worker function: decode one video and save frames to disk."""
    video, dataset_obj = args
    try:
        dataset_obj.save_video_frames(video)
        return video, True, None
    except Exception as e:
        return video, False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Pre-extract video frames in parallel')
    parser.add_argument('--dataset', type=str, default='Video-MME')
    parser.add_argument('--nframe', type=int, default=0)
    parser.add_argument('--fps', type=float, default=-1)
    parser.add_argument('--use-subtitle', action='store_true')
    parser.add_argument('--durations', type=str, nargs='*', default=None,
                        help='Filter by duration, e.g. --durations short medium')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of worker processes. 0 = auto (all CPUs)')
    args = parser.parse_args()

    if args.nframe <= 0 and args.fps <= 0:
        parser.error('At least one of --nframe or --fps must be positive')

    from vlmeval.dataset.videomme import VideoMME

    print(f'Building dataset {args.dataset} (nframe={args.nframe}, fps={args.fps}) ...')
    ds = VideoMME(
        dataset=args.dataset,
        nframe=args.nframe,
        fps=args.fps,
        use_subtitle=args.use_subtitle,
        durations=args.durations,
    )

    videos = ds.videos
    print(f'Total videos to decode: {len(videos)}')

    # Filter out videos whose frames already exist
    pending = []
    for v in videos:
        if args.nframe > 0:
            paths = ds.frame_paths(v)
        else:
            # For fps mode we cannot know the count without decoding,
            # so just check if the frame directory is non-empty
            frame_dir = osp.join(ds.frame_root, v)
            if osp.isdir(frame_dir) and len(os.listdir(frame_dir)) > 0:
                continue
            pending.append(v)
            continue

        if all(osp.exists(p) for p in paths):
            continue
        pending.append(v)

    print(f'Already cached: {len(videos) - len(pending)}, need to decode: {len(pending)}')

    if not pending:
        print('All frames already cached. Nothing to do.')
        return

    nworkers = args.workers if args.workers > 0 else os.cpu_count()
    print(f'Using {nworkers} worker processes ...')

    tasks = [(v, ds) for v in pending]
    success, fail = 0, 0
    with Pool(processes=nworkers) as pool:
        for video, ok, err in pool.imap_unordered(decode_one_video, tasks):
            if ok:
                success += 1
            else:
                fail += 1
                print(f'  FAILED: {video} — {err}')
            total = success + fail
            if total % 50 == 0 or total == len(pending):
                print(f'  Progress: {total}/{len(pending)} (success={success}, fail={fail})')

    print(f'Done. success={success}, fail={fail}')


if __name__ == '__main__':
    main()
