# Missing Video Benchmarks Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans if this plan is expanded into more code changes. Current status is implementation-ready runbook plus one launcher script.

**Goal:** Add practical coverage for VCRBench, CVBench, LongVideoBench, and TempCompass in the current evaluation workflow.

**Architecture:** Use the existing VLMEvalKit-derived framework for benchmarks already registered locally. Integrate the paper's CVBench as `CVBenchVideo` because it is a multi-video reasoning benchmark, while this repo's older `CVBench` class is Cambrian's image CV-Bench.

**Tech Stack:** VLMEvalKit local fork, Qwen3-VL vLLM path, Hugging Face datasets, CVBench/lmms-eval task used only as reference.

---

## Support Matrix

| Benchmark | Current repo support | Registered names | Notes |
| --- | --- | --- | --- |
| VCRBench | Yes | `VCRBench_8frame_nopack`, `VCRBench_16frame_nopack`, `VCRBench_32frame_nopack`, `VCRBench_64frame_nopack`, `VCRBench_1fps_nopack` | Uses `VLM-Reasoning/VCR-Bench`; evaluation uses GPT judge by default. |
| TempCompass | Yes | `TempCompass_8frame`, `TempCompass_64frame`, `TempCompass_1fps`, `TempCompass_0.5fps` | Concat of MCQ, captioning, and yes/no subsets from `lmms-lab/TempCompass`; captioning/yes-no need judge, MCQ can exact-match. |
| LongVideoBench | Yes | `LongVideoBench_8frame`, `LongVideoBench_8frame_subs`, `LongVideoBench_64frame`, `LongVideoBench_1fps`, `LongVideoBench_0.5fps`, `LongVideoBench_0.5fps_subs` | Uses `longvideobench/LongVideoBench`; MCQ exact matching is available. |
| CVBench in screenshots | Yes, added | `CVBench_8frame`, `CVBench_16frame`, `CVBench_32frame`, `CVBench_64frame`, `CVBench_1fps` | Cross-video reasoning CVBench/MVR, implemented from the `Hokhim2/CVBench` lmms-eval task structure. |
| CV-Bench image benchmark | Yes, but not the requested one | `CV-Bench-2D`, `CV-Bench-3D` | Cambrian image MCQ benchmark in `vlmeval/dataset/image_mcq.py`. |

## Implemented Entry Point

Use:

```bash
bash run_missing_video_benchmarks.sh
```

Defaults:

```bash
DATASETS=(
  VCRBench_64frame_nopack
  CVBench_64frame
  TempCompass_64frame
  LongVideoBench_64frame
)
```

Override as needed:

```bash
MODELS="Qwen3-VL-4B-Instruct Qwen3-VL-8B-Instruct" \
DATA="VCRBench_64frame_nopack CVBench_64frame TempCompass_64frame LongVideoBench_64frame" \
WORK_DIR=/path/to/workdir \
bash run_missing_video_benchmarks.sh
```

For the paper tables' 256-frame protocol, the current registrations stop at 64 frames for these four benchmarks. If exact protocol matching matters, add corresponding `*_256frame` entries to `vlmeval/dataset/video_dataset_config.py` and verify Qwen3-VL/vLLM memory before running.

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
| TempCompass | `lmms-lab/TempCompass` | Parquet/json metadata plus `tempcompass_videos.zip`, extracted `videos/` | Estimate 20-50 GB after extraction; keep extra zip buffer during first unzip. |
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

VCRBench, TempCompass captioning, and TempCompass yes/no use judge-based scoring unless exact matching is explicitly supported for the subset. Configure `OPENAI_API_KEY` or your internal `--judge` settings before evaluation.

LongVideoBench is MCQ and can use exact matching. The default wrapper does not force `--judge exact_matching`, because mixed benchmark lists may include judge-based datasets.

For final paper numbers, record frame count, fps, subtitle mode, judge model, and whether `USE_COT` is enabled. These settings affect comparability.
