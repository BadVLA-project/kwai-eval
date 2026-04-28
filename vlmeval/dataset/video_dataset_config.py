from vlmeval.dataset import *
from functools import partial

vcrbench_dataset = {
    'VCRBench_8frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=8, pack=False),
    'VCRBench_16frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=16, pack=False),
    'VCRBench_32frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=32, pack=False),
    'VCRBench_64frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=64, pack=False),
    'VCRBench_1fps_nopack': partial(VCRBench, dataset='VCR-Bench', fps=1.0, pack=False)
}

mmbench_video_dataset = {
    'MMBench_Video_8frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=False),
    'MMBench_Video_8frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=True),
    'MMBench_Video_16frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=16, pack=False),
    'MMBench_Video_64frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=False),
    'MMBench_Video_64frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=True),
    'MMBench_Video_1fps_nopack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=False),
    'MMBench_Video_1fps_pack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=True)
}

mvbench_dataset = {
    'MVBench_8frame': partial(MVBench, dataset='MVBench', nframe=8),
    'MVBench_64frame': partial(MVBench, dataset='MVBench', nframe=64),
    # MVBench not support fps, but MVBench_MP4 does
    'MVBench_MP4_8frame': partial(MVBench_MP4, dataset='MVBench_MP4', nframe=8),
    'MVBench_MP4_1fps': partial(MVBench_MP4, dataset='MVBench_MP4', fps=1.0),
}

tamperbench_dataset = {
    'MVTamperBench_8frame': partial(MVTamperBench, dataset='MVTamperBench', nframe=8),
    'MVTamperBenchStart_8frame': partial(MVTamperBench, dataset='MVTamperBenchStart', nframe=8),
    'MVTamperBenchEnd_8frame': partial(MVTamperBench, dataset='MVTamperBenchEnd', nframe=8),
}

videomme_dataset = {
    'Video-MME_8frame': partial(VideoMME, dataset='Video-MME', nframe=8),
    'Video-MME_64frame': partial(VideoMME, dataset='Video-MME', nframe=64),
    'Video-MME_64frame_short_medium': partial(VideoMME, dataset='Video-MME', nframe=64, durations=['short', 'medium']),
    'Video-MME_8frame_subs': partial(VideoMME, dataset='Video-MME', nframe=8, use_subtitle=True),
    'Video-MME_1fps': partial(VideoMME, dataset='Video-MME', fps=1.0),
    'Video-MME_0.5fps': partial(VideoMME, dataset='Video-MME', fps=0.5),
    'Video-MME_0.5fps_subs': partial(VideoMME, dataset='Video-MME', fps=0.5, use_subtitle=True),
}

videomme_v2_dataset = {
    'Video-MME-v2_64frame': partial(VideoMMEv2, dataset='Video-MME-v2', nframe=64),
    'Video-MME-v2_1fps': partial(VideoMMEv2, dataset='Video-MME-v2', fps=1.0),
    'Video-MME-v2_0.5fps': partial(VideoMMEv2, dataset='Video-MME-v2', fps=0.5),
    'Video-MME-v2_64frame_subs': partial(VideoMMEv2, dataset='Video-MME-v2', nframe=64, use_subtitle=True),
    'Video-MME-v2_0.5fps_subs': partial(VideoMMEv2, dataset='Video-MME-v2', fps=0.5, use_subtitle=True),
    'Video-MME-v2_64frame_subs_interleave': partial(
        VideoMMEv2, dataset='Video-MME-v2', nframe=64, use_subtitle=True, subtitle_mode='interleave'),
}

videommmu_dataset = {
    'VideoMMMU_8frame': partial(VideoMMMU, dataset='VideoMMMU', nframe=8),
    'VideoMMMU_64frame': partial(VideoMMMU, dataset='VideoMMMU', nframe=64),
    'VideoMMMU_1fps': partial(VideoMMMU, dataset='VideoMMMU', fps=1.0),
    'VideoMMMU_0.5fps': partial(VideoMMMU, dataset='VideoMMMU', fps=0.5),
}

longvideobench_dataset = {
    'LongVideoBench_8frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=8),
    'LongVideoBench_8frame_subs': partial(LongVideoBench, dataset='LongVideoBench', nframe=8, use_subtitle=True),
    'LongVideoBench_64frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=64),
    'LongVideoBench_1fps': partial(LongVideoBench, dataset='LongVideoBench', fps=1.0),
    'LongVideoBench_0.5fps': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5),
    'LongVideoBench_0.5fps_subs': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5, use_subtitle=True)
}

mlvu_dataset = {
    'MLVU_8frame': partial(MLVU, dataset='MLVU', nframe=8),
    'MLVU_64frame': partial(MLVU, dataset='MLVU', nframe=64),
    'MLVU_1fps': partial(MLVU, dataset='MLVU', fps=1.0),
    'MLVU_MCQ_64frame': partial(MLVU_MCQ, dataset='MLVU_MCQ', nframe=64),
}

tempcompass_dataset = {
    'TempCompass_8frame': partial(TempCompass, dataset='TempCompass', nframe=8),
    'TempCompass_64frame': partial(TempCompass, dataset='TempCompass', nframe=64),
    'TempCompass_1fps': partial(TempCompass, dataset='TempCompass', fps=1.0),
    'TempCompass_0.5fps': partial(TempCompass, dataset='TempCompass', fps=0.5)
}

# In order to reproduce the experimental results in CGbench paper,
# use_subtitle, use_subtitle_time and use_frame_time need to be set to True.
# When measuring clue-related results, if the number of frames used is greater
# than 32, the frame capture limit will be set to 32.
# We implement the metrics long_acc, clue_acc, miou, CRR, acc@iou and rec@iou
# in the CGBench_MCQ_Grounding_Mini and CGBench_MCQ_Grounding datasets;
# the metric open-ended is implemented in the CGBench_OpenEnded_Mini and CGBench_OpenEnded datasets.
cgbench_dataset = {
    'CGBench_MCQ_Grounding_Mini_8frame_subs_subt': partial(
        CGBench_MCQ_Grounding_Mini,
        dataset='CG-Bench_MCQ_Grounding_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True
    ),
    'CGBench_OpenEnded_Mini_8frame_subs_subt_ft': partial(
        CGBench_OpenEnded_Mini,
        dataset='CG-Bench_OpenEnded_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_MCQ_Grounding_32frame_subs': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=32,
        use_subtitle=True
    ),
    'CGBench_OpenEnded_8frame': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=8
    ),
    'CGBench_MCQ_Grounding_16frame_subs_subt_ft': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_OpenEnded_16frame_subs_subt_ft': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    )
}

megabench_dataset = {
    'MEGABench_core_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="core"),
    'MEGABench_open_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="open"),
    'MEGABench_core_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="core"),
    'MEGABench_open_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="open")
}

moviechat1k_dataset = {
    'moviechat1k_breakpoint_8frame': partial(MovieChat1k, dataset='MovieChat1k', subset='breakpoint', nframe=8),
    'moviechat1k_global_14frame': partial(MovieChat1k, dataset='MovieChat1k', subset='global', nframe=14),
    'moviechat1k_global_8frame_limit0.01': partial(
        MovieChat1k, dataset='MovieChat1k', subset='global', nframe=8, limit=0.01
    )
}

vdc_dataset = {
    'VDC_8frame': partial(VDC, dataset='VDC', nframe=8),
    'VDC_1fps': partial(VDC, dataset='VDC', fps=1.0),
}

worldsense_dataset = {
    'WorldSense_8frame': partial(WorldSense, dataset='WorldSense', nframe=8),
    'WorldSense_8frame_subs': partial(WorldSense, dataset='WorldSense', nframe=8, use_subtitle=True),
    'WorldSense_8frame_audio': partial(WorldSense, dataset='WorldSense', nframe=8, use_audio=True),
    'WorldSense_32frame': partial(WorldSense, dataset='WorldSense', nframe=32),
    'WorldSense_32frame_subs': partial(WorldSense, dataset='WorldSense', nframe=32, use_subtitle=True),
    'WorldSense_32frame_audio': partial(WorldSense, dataset='WorldSense', nframe=32, use_audio=True),
    'WorldSense_1fps': partial(WorldSense, dataset='WorldSense', fps=1.0),
    'WorldSense_1fps_subs': partial(WorldSense, dataset='WorldSense', fps=1.0, use_subtitle=True),
    'WorldSense_1fps_audio': partial(WorldSense, dataset='WorldSense', fps=1.0, use_audio=True),
    'WorldSense_0.5fps': partial(WorldSense, dataset='WorldSense', fps=0.5),
    'WorldSense_0.5fps_subs': partial(WorldSense, dataset='WorldSense', fps=0.5, use_subtitle=True),
    'WorldSense_0.5fps_audio': partial(WorldSense, dataset='WorldSense', fps=0.5, use_audio=True)
}

qbench_video_dataset = {
    'QBench_Video_8frame': partial(QBench_Video, dataset='QBench_Video', nframe=8),
    'QBench_Video_16frame': partial(QBench_Video, dataset='QBench_Video', nframe=16),
}

video_mmlu_dataset = {
    'Video_MMLU_CAP_16frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=16),
    'Video_MMLU_CAP_64frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=64),
    'Video_MMLU_QA_16frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=16),
    'Video_MMLU_QA_64frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=64),
}

video_tt_dataset = {
    'Video_TT_16frame': partial(VideoTT, dataset='Video-TT', nframe=16),
    'Video_TT_32frame': partial(VideoTT, dataset='Video-TT', nframe=32),
    'Video_TT_64frame': partial(VideoTT, dataset='Video-TT', nframe=64),
}

video_holmes_dataset = {
    'Video_Holmes_32frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=32),
    'Video_Holmes_64frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=64),
}

cg_av_counting_dataset = {
    'CG-AV-Counting_32frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=32, use_frame_time=False),
    'CG-AV-Counting_64frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=64, use_frame_time=False)
}

egoexobench_dataset = {
    'EgoExoBench_64frame': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=False),  # noqa: E501
    'EgoExoBench_64frame_skip_EgoExo4D': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=True)  # noqa: E501

}

vsibench_dataset = {
    'vsibench_16frame': partial(VSIBench, dataset='VSIBench', nframe=16),
    'vsibench_32frame': partial(VSIBench, dataset='VSIBench', nframe=32),
    'vsibench_64frame': partial(VSIBench, dataset='VSIBench', nframe=64),
}

dream_1k_dataset = {
    'DREAM-1K_8frame': partial(DREAM, dataset='DREAM-1K', nframe=8),
    'DREAM-1K_64frame': partial(DREAM, dataset='DREAM-1K', nframe=64),
    'DREAM-1K_2fps': partial(DREAM, dataset='DREAM-1K', fps=2.0),
    'DREAM-1K_1fps': partial(DREAM, dataset='DREAM-1K', fps=1.0),
    'DREAM-1K_0.5fps': partial(DREAM, dataset='DREAM-1K', fps=0.5),
}

supported_video_datasets = {}

aotbench_dataset = {
    'AoTBench_ReverseFilm_16frame': partial(AoTBench, dataset='AoTBench_ReverseFilm', nframe=16),
    'AoTBench_UCF101_16frame': partial(AoTBench, dataset='AoTBench_UCF101', nframe=16),
    'AoTBench_Rtime_t2v_16frame': partial(AoTBench, dataset='AoTBench_Rtime_t2v', nframe=16),
    'AoTBench_Rtime_v2t_16frame': partial(AoTBench, dataset='AoTBench_Rtime_v2t', nframe=16),
    'AoTBench_QA_16frame': partial(AoTBench, dataset='AoTBench_QA', nframe=16),
    'AoTBench_ReverseFilm_32frame': partial(AoTBench, dataset='AoTBench_ReverseFilm', nframe=32),
    'AoTBench_UCF101_32frame': partial(AoTBench, dataset='AoTBench_UCF101', nframe=32),
    'AoTBench_Rtime_t2v_32frame': partial(AoTBench, dataset='AoTBench_Rtime_t2v', nframe=32),
    'AoTBench_Rtime_v2t_32frame': partial(AoTBench, dataset='AoTBench_Rtime_v2t', nframe=32),
    'AoTBench_QA_32frame': partial(AoTBench, dataset='AoTBench_QA', nframe=32),
}

futureomni_dataset = {
    'FutureOmni_32frame': partial(FutureOmni, dataset='FutureOmni', nframe=32),
    'FutureOmni_64frame': partial(FutureOmni, dataset='FutureOmni', nframe=64),
    'FutureOmni_128frame': partial(FutureOmni, dataset='FutureOmni', nframe=128),
    'FutureOmni_256frame': partial(FutureOmni, dataset='FutureOmni', nframe=256),
    'FutureOmni_1fps': partial(FutureOmni, dataset='FutureOmni', fps=1.0),
    'FutureOmni_2fps': partial(FutureOmni, dataset='FutureOmni', fps=2.0),
}

charades_sta_dataset = {
    'CharadesSTA_16frame': partial(CharadesSTA, dataset='CharadesSTA', nframe=16),
    'CharadesSTA_32frame': partial(CharadesSTA, dataset='CharadesSTA', nframe=32),
    'CharadesSTA_64frame': partial(CharadesSTA, dataset='CharadesSTA', nframe=64),
    'CharadesSTA_1fps': partial(CharadesSTA, dataset='CharadesSTA', fps=1.0),
    'CharadesSTA_2fps': partial(CharadesSTA, dataset='CharadesSTA', fps=2.0),
}

etbench_dataset = {
    # ── Official setting (matches infer_etbench.py: load_video fps=1 on videos_compressed) ──
    'ETBench_1fps':    partial(ETBench, dataset='ETBench', fps=1.0),      # <── official (auto video source)
    # ── Explicit video_source=compressed: strictly use videos_compressed (mirrors paper setting) ──
    'ETBench_1fps_compressed':    partial(ETBench, dataset='ETBench', fps=1.0, video_source='compressed'),
    'ETBench_subset_1fps_compressed': partial(ETBench, dataset='ETBench_subset', fps=1.0, video_source='compressed'),
    # ── Explicit video_source=raw: use original raw videos ──
    'ETBench_1fps_raw':    partial(ETBench, dataset='ETBench', fps=1.0, video_source='raw'),
    'ETBench_subset_1fps_raw': partial(ETBench, dataset='ETBench_subset', fps=1.0, video_source='raw'),
    # Full benchmark — alternative frame-count variants
    'ETBench_16frame': partial(ETBench, dataset='ETBench', nframe=16, fps=-1),
    'ETBench_32frame': partial(ETBench, dataset='ETBench', nframe=32, fps=-1),
    'ETBench_64frame': partial(ETBench, dataset='ETBench', nframe=64, fps=-1),
    'ETBench_2fps':    partial(ETBench, dataset='ETBench', fps=2.0),
    # 470-sample commercial subset (Table 1 of the paper)
    'ETBench_subset_1fps':    partial(ETBench, dataset='ETBench_subset', fps=1.0),   # <── official
    'ETBench_subset_16frame': partial(ETBench, dataset='ETBench_subset', nframe=16, fps=-1),
    'ETBench_subset_32frame': partial(ETBench, dataset='ETBench_subset', nframe=32, fps=-1),
    # TVG-only slice (quick temporal grounding ablation)
    'ETBench_TVG_1fps':    partial(ETBench, dataset='ETBench', fps=1.0,  task_filter=['tvg']),
    'ETBench_TVG_32frame': partial(ETBench, dataset='ETBench', nframe=32, fps=-1, task_filter=['tvg']),
    'ETBench_TVG_64frame': partial(ETBench, dataset='ETBench', nframe=64, fps=-1, task_filter=['tvg']),
    # MCQ-only slice
    'ETBench_MCQ_1fps':    partial(ETBench, dataset='ETBench', fps=1.0,
                                   task_filter=['rar', 'eca', 'rvq', 'gvq']),
    'ETBench_MCQ_32frame': partial(ETBench, dataset='ETBench', nframe=32, fps=-1,
                                   task_filter=['rar', 'eca', 'rvq', 'gvq']),
}


charades_timelens_dataset = {
    'CharadesTimeLens_8frame': partial(CharadesTimeLens, dataset='CharadesTimeLens', nframe=8),
    'CharadesTimeLens_16frame': partial(CharadesTimeLens, dataset='CharadesTimeLens', nframe=16),
    'CharadesTimeLens_32frame': partial(CharadesTimeLens, dataset='CharadesTimeLens', nframe=32),
    'CharadesTimeLens_64frame': partial(CharadesTimeLens, dataset='CharadesTimeLens', nframe=64),
    'CharadesTimeLens_1fps': partial(CharadesTimeLens, dataset='CharadesTimeLens', fps=1.0),
    'CharadesTimeLens_2fps': partial(CharadesTimeLens, dataset='CharadesTimeLens', fps=2.0),
}

timelens_bench_dataset = {
    # --- Charades ---
    'TimeLensBench_Charades_8frame': partial(TimeLensBench, dataset='TimeLensBench_Charades', nframe=8),
    'TimeLensBench_Charades_16frame': partial(TimeLensBench, dataset='TimeLensBench_Charades', nframe=16),
    'TimeLensBench_Charades_32frame': partial(TimeLensBench, dataset='TimeLensBench_Charades', nframe=32),
    'TimeLensBench_Charades_64frame': partial(TimeLensBench, dataset='TimeLensBench_Charades', nframe=64),
    'TimeLensBench_Charades_1fps': partial(TimeLensBench, dataset='TimeLensBench_Charades', fps=1.0),
    'TimeLensBench_Charades_2fps': partial(TimeLensBench, dataset='TimeLensBench_Charades', fps=2.0),
    # --- ActivityNet ---
    'TimeLensBench_ActivityNet_8frame': partial(TimeLensBench, dataset='TimeLensBench_ActivityNet', nframe=8),
    'TimeLensBench_ActivityNet_16frame': partial(TimeLensBench, dataset='TimeLensBench_ActivityNet', nframe=16),
    'TimeLensBench_ActivityNet_32frame': partial(TimeLensBench, dataset='TimeLensBench_ActivityNet', nframe=32),
    'TimeLensBench_ActivityNet_64frame': partial(TimeLensBench, dataset='TimeLensBench_ActivityNet', nframe=64),
    'TimeLensBench_ActivityNet_1fps': partial(TimeLensBench, dataset='TimeLensBench_ActivityNet', fps=1.0),
    'TimeLensBench_ActivityNet_2fps': partial(TimeLensBench, dataset='TimeLensBench_ActivityNet', fps=2.0),
    # --- QVHighlights ---
    'TimeLensBench_QVHighlights_8frame': partial(TimeLensBench, dataset='TimeLensBench_QVHighlights', nframe=8),
    'TimeLensBench_QVHighlights_16frame': partial(TimeLensBench, dataset='TimeLensBench_QVHighlights', nframe=16),
    'TimeLensBench_QVHighlights_32frame': partial(TimeLensBench, dataset='TimeLensBench_QVHighlights', nframe=32),
    'TimeLensBench_QVHighlights_64frame': partial(TimeLensBench, dataset='TimeLensBench_QVHighlights', nframe=64),
    'TimeLensBench_QVHighlights_1fps': partial(TimeLensBench, dataset='TimeLensBench_QVHighlights', fps=1.0),
    'TimeLensBench_QVHighlights_2fps': partial(TimeLensBench, dataset='TimeLensBench_QVHighlights', fps=2.0),
}

perceptiontest_dataset = {
    'PerceptionTest_val_8frame': partial(PerceptionTest, dataset='PerceptionTest_val', nframe=8),
    'PerceptionTest_val_16frame': partial(PerceptionTest, dataset='PerceptionTest_val', nframe=16),
    'PerceptionTest_val_32frame': partial(PerceptionTest, dataset='PerceptionTest_val', nframe=32),
    'PerceptionTest_val_1fps': partial(PerceptionTest, dataset='PerceptionTest_val', fps=1.0),
    'PerceptionTest_test_8frame': partial(PerceptionTest, dataset='PerceptionTest_test', nframe=8),
    'PerceptionTest_test_16frame': partial(PerceptionTest, dataset='PerceptionTest_test', nframe=16),
    'PerceptionTest_test_32frame': partial(PerceptionTest, dataset='PerceptionTest_test', nframe=32),
    'PerceptionTest_test_1fps': partial(PerceptionTest, dataset='PerceptionTest_test', fps=1.0),
}

# ---------------------------------------------------------------------------
# Adaptive sampling: ≤60s → 2fps, 60–256s → 1fps, >256s → uniform 256 frames
# ---------------------------------------------------------------------------
adaptive_dataset = {
    # AoTBench
    'AoTBench_ReverseFilm_adaptive': partial(AoTBench, dataset='AoTBench_ReverseFilm', adaptive=True),
    'AoTBench_UCF101_adaptive': partial(AoTBench, dataset='AoTBench_UCF101', adaptive=True),
    'AoTBench_Rtime_t2v_adaptive': partial(AoTBench, dataset='AoTBench_Rtime_t2v', adaptive=True),
    'AoTBench_Rtime_v2t_adaptive': partial(AoTBench, dataset='AoTBench_Rtime_v2t', adaptive=True),
    'AoTBench_QA_adaptive': partial(AoTBench, dataset='AoTBench_QA', adaptive=True),
    # CharadesTimeLens
    'CharadesTimeLens_adaptive': partial(CharadesTimeLens, dataset='CharadesTimeLens', adaptive=True),
    'CharadesTimeLensTrainPrompt_adaptive': partial(
        CharadesTimeLens,
        dataset='CharadesTimeLens',
        adaptive=True,
        prompt_style='train',
        dataset_name_alias='CharadesTimeLensTrainPrompt_adaptive',
    ),
    # MVBench_MP4
    'MVBench_MP4_adaptive': partial(MVBench_MP4, dataset='MVBench_MP4', adaptive=True),
    # ETBench
    'ETBench_adaptive': partial(ETBench, dataset='ETBench', adaptive=True),
    'ETBench_adaptive_compressed': partial(ETBench, dataset='ETBench', adaptive=True, video_source='compressed'),
    'ETBench_subset_adaptive': partial(ETBench, dataset='ETBench_subset', adaptive=True),
    'ETBench_subset_adaptive_compressed': partial(
        ETBench, dataset='ETBench_subset', adaptive=True, video_source='compressed'),
    # Video-MME
    'Video-MME_adaptive': partial(VideoMME, dataset='Video-MME', adaptive=True),
    'Video-MME_adaptive_subs': partial(VideoMME, dataset='Video-MME', adaptive=True, use_subtitle=True),
    # Video_Holmes
    'Video_Holmes_adaptive': partial(Video_Holmes, dataset='Video_Holmes', adaptive=True),
    # FutureOmni
    'FutureOmni_adaptive': partial(FutureOmni, dataset='FutureOmni', adaptive=True),
    # PerceptionTest
    'PerceptionTest_val_adaptive': partial(PerceptionTest, dataset='PerceptionTest_val', adaptive=True),
    'PerceptionTest_test_adaptive': partial(PerceptionTest, dataset='PerceptionTest_test', adaptive=True),
    # TimeLensBench
    'TimeLensBench_Charades_adaptive': partial(TimeLensBench, dataset='TimeLensBench_Charades', adaptive=True),
    'TimeLensBench_ActivityNet_adaptive': partial(TimeLensBench, dataset='TimeLensBench_ActivityNet', adaptive=True),
    'TimeLensBench_QVHighlights_adaptive': partial(TimeLensBench, dataset='TimeLensBench_QVHighlights', adaptive=True),
    'TimeLensBenchTrainPrompt_Charades_adaptive': partial(
        TimeLensBench,
        dataset='TimeLensBench_Charades',
        adaptive=True,
        prompt_style='train',
        dataset_name_alias='TimeLensBenchTrainPrompt_Charades_adaptive',
    ),
    'TimeLensBenchTrainPrompt_ActivityNet_adaptive': partial(
        TimeLensBench,
        dataset='TimeLensBench_ActivityNet',
        adaptive=True,
        prompt_style='train',
        dataset_name_alias='TimeLensBenchTrainPrompt_ActivityNet_adaptive',
    ),
    'TimeLensBenchTrainPrompt_QVHighlights_adaptive': partial(
        TimeLensBench,
        dataset='TimeLensBench_QVHighlights',
        adaptive=True,
        prompt_style='train',
        dataset_name_alias='TimeLensBenchTrainPrompt_QVHighlights_adaptive',
    ),
    # MLVU
    'MLVU_MCQ_adaptive': partial(MLVU_MCQ, dataset='MLVU_MCQ', adaptive=True),
    # DREAM-1K
    'DREAM-1K_adaptive': partial(DREAM, dataset='DREAM-1K', adaptive=True),
    # VDC
    'VDC_adaptive': partial(VDC, dataset='VDC', adaptive=True),
    # CharadesSTA
    'CharadesSTA_adaptive': partial(CharadesSTA, dataset='CharadesSTA', adaptive=True),
}

dataset_groups = [
    mmbench_video_dataset, mvbench_dataset, videomme_dataset, videomme_v2_dataset, videommmu_dataset, longvideobench_dataset,
    mlvu_dataset, tempcompass_dataset, cgbench_dataset, worldsense_dataset, tamperbench_dataset,
    megabench_dataset, qbench_video_dataset, moviechat1k_dataset, vdc_dataset, video_holmes_dataset, vcrbench_dataset,
    cg_av_counting_dataset, video_mmlu_dataset, egoexobench_dataset, dream_1k_dataset, video_tt_dataset,
    vsibench_dataset, aotbench_dataset, futureomni_dataset,
    charades_sta_dataset, charades_timelens_dataset, timelens_bench_dataset, perceptiontest_dataset,
    etbench_dataset,
    adaptive_dataset,
]

for grp in dataset_groups:
    supported_video_datasets.update(grp)
