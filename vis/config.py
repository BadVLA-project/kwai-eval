"""Model & dataset registries, colors, and display names."""

from collections import OrderedDict

# ── Model registry ──────────────────────────────────────────────────────
# key = directory name on disk, value = (short_label, group, color)

MODEL_INFO = OrderedDict([
    ('Qwen3-VL-4B-Instruct',
     ('Base', 'base', '#2c3e50')),
    ('Qwen3-VL-4B-Instruct_aot_ablation_exp1_v2t_binary',
     ('Exp1 v2t-bin', 'aot', '#e74c3c')),
    ('Qwen3-VL-4B-Instruct_aot_ablation_exp2_v2t_3way',
     ('Exp2 v2t-3w', 'aot', '#e67e22')),
    ('Qwen3-VL-4B-Instruct_aot_ablation_exp3_t2v_binary',
     ('Exp3 t2v-bin', 'aot', '#f39c12')),
    ('Qwen3-VL-4B-Instruct_aot_ablation_exp4_t2v_3way',
     ('Exp4 t2v-3w', 'aot', '#27ae60')),
    ('Qwen3-VL-4B-Instruct_aot_ablation_exp5_mixed_binary',
     ('Exp5 mix-bin', 'aot', '#1abc9c')),
    ('Qwen3-VL-4B-Instruct_aot_ablation_exp6_mixed_3way',
     ('Exp6 mix-3w', 'aot', '#3498db')),
    ('Qwen3-VL-4B-Instruct_aot_ablation_exp7_v2t_binary_3way_mixed',
     ('Exp7 v2t-b3m', 'aot', '#9b59b6')),
    ('Qwen3-VL-4B-Instruct_aot_ablation_exp8_all_mixed',
     ('Exp8 all-mix', 'aot', '#8e44ad')),
    ('Qwen3-VL-4B-Instruct_aot_ablation_exp9_t2v_binary_3way_mixed',
     ('Exp9 t2v-b3m', 'aot', '#34495e')),
    ('Qwen3-VL-4B-Instruct_tg_ablation_exp1_no_cot_v2',
     ('TG1 no-CoT', 'tg', '#16a085')),
    ('Qwen3-VL-4B-Instruct_tg_ablation_exp2_cot_v2',
     ('TG2 CoT', 'tg', '#2980b9')),
])

MODEL_NAMES = list(MODEL_INFO.keys())
MODEL_LABELS = {k: v[0] for k, v in MODEL_INFO.items()}
MODEL_GROUPS = {k: v[1] for k, v in MODEL_INFO.items()}
MODEL_COLORS = {k: v[2] for k, v in MODEL_INFO.items()}
BASE_MODEL = 'Qwen3-VL-4B-Instruct'

AOT_MODELS = [k for k, v in MODEL_INFO.items() if v[1] == 'aot']
TG_MODELS = [k for k, v in MODEL_INFO.items() if v[1] == 'tg']

# ── Dataset registry ────────────────────────────────────────────────────
# key = dataset_name string (from --data), value = (short_label, score_type)
# score_type determines which loader to use in data_loader.py

DATASET_INFO = OrderedDict([
    ('AoTBench_ReverseFilm_16frame', ('AoT-RF', 'acc_csv')),
    ('AoTBench_UCF101_16frame',      ('AoT-UCF', 'acc_csv')),
    ('AoTBench_Rtime_t2v_16frame',   ('AoT-t2v', 'acc_csv')),
    ('AoTBench_Rtime_v2t_16frame',   ('AoT-v2t', 'acc_csv')),
    ('AoTBench_QA_16frame',          ('AoT-QA', 'acc_csv')),
    ('FutureOmni_64frame',           ('FutureOmni', 'acc_csv')),
    ('CharadesTimeLens_1fps',        ('Charades-TL', 'charades_json')),
    ('MVBench_MP4_1fps',             ('MVBench', 'mvbench_rating')),
    ('PerceptionTest_val_16frame',   ('PTest', 'perception_acc')),
    ('Video-MME_64frame',            ('VideoMME', 'videomme_rating')),
    ('Video_Holmes_64frame',         ('V-Holmes', 'videoholmes_rating')),
])

DATASET_NAMES = list(DATASET_INFO.keys())
DATASET_LABELS = {k: v[0] for k, v in DATASET_INFO.items()}

AOT_DATASETS = [d for d in DATASET_NAMES if d.startswith('AoTBench_')]

# ── Primary metric per dataset (for overall comparison) ─────────────────
# Each returns a single float in 0-100 scale
PRIMARY_METRIC = {
    'AoTBench_ReverseFilm_16frame': 'overall_accuracy',
    'AoTBench_UCF101_16frame':      'overall_accuracy',
    'AoTBench_Rtime_t2v_16frame':   'overall_accuracy',
    'AoTBench_Rtime_v2t_16frame':   'overall_accuracy',
    'AoTBench_QA_16frame':          'overall_accuracy',
    'FutureOmni_64frame':           'overall_accuracy',
    'CharadesTimeLens_1fps':        'mIoU',
    'MVBench_MP4_1fps':             'overall_pct',
    'PerceptionTest_val_16frame':   'overall_accuracy',
    'Video-MME_64frame':            'overall_overall',
    'Video_Holmes_64frame':         'total_acc',
}
