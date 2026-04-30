# Missing Video Benchmarks Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans if this plan is expanded into more code changes. Current status is implementation-ready runbook plus one launcher script.

**Goal:** Add practical adaptive-sampling coverage for VCRBench, CVBench, Video-MME-v2, VideoMMMU, LongVideoBench, and TempCompass MCQ in the current evaluation workflow.

**Architecture:** Use the existing VLMEvalKit-derived framework for benchmarks already registered locally. Integrate the paper's CVBench as `CVBenchVideo` because it is a multi-video reasoning benchmark, while this repo's older `CVBench` class is Cambrian's image CV-Bench.

**Tech Stack:** VLMEvalKit local fork, Qwen3-VL vLLM path, Hugging Face datasets, CVBench/lmms-eval task used only as reference.

---

## Support Matrix

| Benchmark | Current repo support | Registered names | Notes |
| --- | --- | --- | --- |
| VCRBench | Yes | `VCRBench_8frame_nopack`, `VCRBench_16frame_nopack`, `VCRBench_32frame_nopack`, `VCRBench_64frame_nopack`, `VCRBench_1fps_nopack`, `VCRBench_adaptive` | Uses `VLM-Reasoning/VCR-Bench`; evaluation uses GPT judge by default. |
| TempCompass MCQ | Yes | `TempCompass_MCQ_adaptive` plus fixed-frame/fps aggregate names | MCQ-only path from `lmms-lab/TempCompass`; exact matching is the default, so Captioning is not pulled in. |
| LongVideoBench | Yes | `LongVideoBench_8frame`, `LongVideoBench_8frame_subs`, `LongVideoBench_64frame`, `LongVideoBench_1fps`, `LongVideoBench_0.5fps`, `LongVideoBench_0.5fps_subs`, `LongVideoBench_adaptive`, `LongVideoBench_adaptive_subs` | Uses `longvideobench/LongVideoBench`; MCQ exact matching is available. |
| Video-MME-v2 | Yes | Official fixed variants: `Video-MME-v2_64frame`, `Video-MME-v2_1fps`, subtitle/interleave, reasoning, and resize variants; local adaptive variants: `Video-MME-v2_adaptive`, `Video-MME-v2_adaptive_subs` | Reads videos from the read-only local dataset path and writes generated TSV/subtitle artifacts under the writable artifact path. |
| VideoMMMU | Yes | `VideoMMMU_8frame`, `VideoMMMU_64frame`, `VideoMMMU_1fps`, `VideoMMMU_0.5fps`, `VideoMMMU_adaptive` | Uses local HF cache when available. |
| CVBench in screenshots | Yes, added | `CVBench_8frame`, `CVBench_16frame`, `CVBench_32frame`, `CVBench_64frame`, `CVBench_1fps`, `CVBench_adaptive` | Cross-video reasoning CVBench/MVR, implemented from the `Hokhim2/CVBench` lmms-eval task structure. |
| CV-Bench image benchmark | Yes, but not the requested one | `CV-Bench-2D`, `CV-Bench-3D` | Cambrian image MCQ benchmark in `vlmeval/dataset/image_mcq.py`. |

## Implemented Entry Point

Use:

```bash
bash run_missing_video_benchmarks.sh
```

Defaults:

```bash
DATASETS=(
  VCRBench_adaptive
  CVBench_adaptive
  Video-MME-v2_adaptive
  VideoMMMU_adaptive
  LongVideoBench_adaptive
  TempCompass_MCQ_adaptive
)
```

Override as needed:

```bash
MODELS="Qwen3-VL-4B-Instruct Qwen3-VL-8B-Instruct" \
DATA="VCRBench_adaptive CVBench_adaptive Video-MME-v2_adaptive VideoMMMU_adaptive LongVideoBench_adaptive TempCompass_MCQ_adaptive" \
WORK_DIR=/path/to/workdir \
bash run_missing_video_benchmarks.sh
```

The adaptive policy is shared with the rest of this repo: videos up to 60s use 2fps, videos from 60s to 256s use 1fps, and longer videos use uniformly sampled 256 frames.

## CVBench Local Layout

The new `CVBenchVideo` loader accepts either an auto-downloaded Hugging Face snapshot or local paths:

```bash
export CVBENCH_DIR=/path/to/CVBench
export CVBENCH_HF_DATASET_DIR=/path/to/CVBench/mvr_dataset
export CVBENCH_VIDEO_DIR=/path/to/CVBench/Video-R1/src/r1-v/Evaluation/CVBench
```

`CVBENCH_VIDEO_DIR` should contain subfolders like `102/*.mp4`. The annotation dataset follows the lmms-eval `mvr` fields: `video_1..video_4`, `question`, `options`, `answer`, and `task_type`.

## Data Downloads and Storage

| Benchmark | Source | What downloads | Expected storage |
| --- | --- | --- | --- |
| VCRBench | `VLM-Reasoning/VCR-Bench` | TSV plus zipped videos extracted under HF cache | Estimate 10-30 GB after extraction; keep another zip-sized buffer during first unzip. |
| TempCompass | `lmms-lab/TempCompass` | Parquet/json metadata plus `tempcompass_videos.zip`, extracted `videos/` | Existing local cache is preferred; estimate 20-50 GB after extraction if downloading from scratch, with extra zip buffer during first unzip. |
| Video-MME-v2 | `MME-Benchmarks/Video-MME-v2` | `test.parquet`, subtitles, and videos | Existing read-only local path is preferred; storage depends on source media and no duplicate video copy is created by TSV generation. |
| VideoMMMU | `lmms-lab/VideoMMMU` | Parquet annotations, videos, optional image assets | Existing local HF cache is preferred; user-reported cache is about 26 GB. |
| LongVideoBench | `longvideobench/LongVideoBench` | `lvb_val.json`, subtitles, split tar video shards extracted into `videos/` | Largest item: estimate 300-500 GB after extraction; keep up to about 2x peak space while tar parts are concatenated/extracted. |
| CVBench/MVR | `Dongyh35/CVBench` / `Hokhim2/CVBench` reference layout | HF-format QA metadata plus multi-video clips under `CVBench/` | 1,000 QA pairs / 1,315 videos; estimate 50-150 GB depending on encoding and whether only eval assets are pulled. |
| Image CV-Bench, not requested | `nyu-visionx/CV-Bench` via VLMEvalKit TSV mirrors | TSV/images for 2D and 3D image tasks | Usually small compared with video benchmarks, under a few GB. |

Set cache locations before running:

```bash
export HF_HOME=/large_disk/hf_cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export LMUData=/large_disk/LMUData
```

Recommended free space before the first full run: at least 1 TB if LongVideoBench and CVBench are both downloaded on the same volume. For only VCRBench, TempCompass, and LongVideoBench, reserve at least 700 GB to survive compressed-plus-extracted peaks.

## Evaluation Caveats

VCRBench uses judge-based scoring. TempCompass Captioning needs an LLM judge, but the default launcher now uses `TempCompass_MCQ_adaptive`, which evaluates with exact matching by default.

LongVideoBench and TempCompass MCQ can use exact matching. The main runner now defaults `TempCompass_MCQ` to exact matching when `--judge` is not provided.

For final paper numbers, record frame count, fps, subtitle mode, judge model, and whether `USE_COT` is enabled. These settings affect comparability.
