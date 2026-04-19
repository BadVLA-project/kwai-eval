# Benchmark Integration Design: VideoTT, VideoMMMU, Vinoground

**Date:** 2026-04-19  
**Status:** Draft  
**Author:** Claude

## Overview

Integrate three video benchmarks into VLMEvalKit evaluation pipeline with local path support:
1. **VideoTT** - Adapt existing implementation for local path
2. **VideoMMMU** - Adapt existing implementation for local path  
3. **Vinoground** - New implementation

All datasets are pre-downloaded from lmms-lab to `/m2v_intern/xuboshen/zgw/Benchmarks/`.

## Design Decisions

### Approach
Use **minimal modification approach** following VideoMME's pattern:
- Add local path check at the beginning of `prepare_dataset()`
- Fall back to HuggingFace download if local path doesn't exist
- No changes to data loading or evaluation logic

### Local Data Paths
| Benchmark | Local Path |
|-----------|-----------|
| VideoTT | `/m2v_intern/xuboshen/zgw/Benchmarks/VideoTT` |
| VideoMMMU | `/m2v_intern/xuboshen/zgw/Benchmarks/VideoMMMU` |
| Vinoground | `/m2v_intern/xuboshen/zgw/Benchmarks/Vinoground` |

## Component Designs

### 1. VideoTT Local Path Adaptation

**File:** `vlmeval/dataset/videott.py`

**Changes:**
- Add local path check in `prepare_dataset()` before HuggingFace download logic
- Keep existing TSV generation and video extraction logic

**Expected Data Structure:**
```
VideoTT/
├── video/                    # Extracted video files
│   └── {video_id}.mp4
├── data/
│   └── test-00000-of-00001.parquet
└── Video-TT.tsv              # Generated TSV (auto-created if missing)
```

### 2. VideoMMMU Local Path Adaptation

**File:** `vlmeval/dataset/videommmu.py`

**Changes:**
- Add local path check in `prepare_dataset()` before HuggingFace download logic
- Keep existing ZIP extraction, TSV generation, and image extraction logic

**Expected Data Structure:**
```
VideoMMMU/
├── videos/
│   └── {category}/{id}.mp4
├── images/                   # Extracted images (auto-created)
├── *.parquet                 # Category parquet files
└── VideoMMMU.tsv             # Generated TSV (auto-created if missing)
```

### 3. Vinoground New Implementation

**File:** `vlmeval/dataset/vinoground.py` (new)

**Class:** `Vinoground(VideoBaseDataset)`

**Evaluation Type:** `Video-MCQ`

**Key Design Points:**

#### Data Format (lmms-eval compatible)
```
Vinoground/
├── vinoground_videos/           # For text-score: single videos
│   └── {idx}_pos.mp4 / {idx}_neg.mp4
├── vinoground_videos_concated/  # For video-score: concatenated videos
│   └── {idx}_pos_concated.mp4 / {idx}_neg_concated.mp4
├── vinoground_textscore.json    # Text-score questions
└── vinoground_videoscore.json   # Video-score questions
```

#### Question Types
1. **text-score**: One video, two text captions (A/B choice)
2. **video-score**: One text, two videos concatenated side-by-side (A/B choice)

#### Evaluation Logic
- 500 base samples, each has pos/neg variants
- For each sample `idx`:
  - `text_correct`: both `{idx}_pos_text` and `{idx}_neg_text` answered correctly
  - `video_correct`: both `{idx}_pos_video` and `{idx}_neg_video` answered correctly
  - `group_correct`: both text and video correct

#### Metrics
```python
{
    "text_score": mean(text_correct) * 100,
    "video_score": mean(video_correct) * 100,  
    "group_score": mean(group_correct) * 100,
    # Per-category breakdowns
}
```

#### TSV Schema
| Column | Description |
|--------|-------------|
| index | Unique row ID |
| idx | Base sample index (0-499) |
| question_type | "text" or "video" |
| variant | "pos" or "neg" |
| video_path | Path to video file |
| question | Question text |
| answer | Ground truth (A or B) |
| major | Major category |
| minor | Minor categories (semicolon-separated) |

## Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `vlmeval/dataset/videott.py` | Modify | Add local path check |
| `vlmeval/dataset/videommmu.py` | Modify | Add local path check |
| `vlmeval/dataset/vinoground.py` | Create | New Vinoground implementation |
| `vlmeval/dataset/__init__.py` | Modify | Import and register Vinoground |
| `vlmeval/dataset/utils/vinoground.py` | Create | Helper functions for Vinoground evaluation |

## Testing Plan

1. **Unit Test:** Verify TSV generation for each dataset
2. **Integration Test:** Run single sample inference for each dataset
3. **Full Evaluation:** Run complete evaluation with a small model

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Data structure mismatch | TSV generation handles path resolution dynamically |
| Missing dependencies | lmms-eval utils copied/adapted locally |
| Evaluation logic bugs | Cross-validate against lmms-eval results |
