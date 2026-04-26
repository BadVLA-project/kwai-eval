"""Minimal configuration for the vis dashboard.

No hardcoded model or dataset registries — everything is auto-discovered.
"""

# Color palette for auto-assigning colors to models
COLORS = [
    '#2c3e50', '#e74c3c', '#e67e22', '#f39c12', '#27ae60',
    '#1abc9c', '#3498db', '#9b59b6', '#8e44ad', '#34495e',
    '#16a085', '#2980b9', '#c0392b', '#d35400', '#7f8c8d',
    '#6c5ce7', '#00b894', '#fd79a8', '#fdcb6e', '#636e72',
    '#0984e3', '#00cec9', '#e17055', '#fab1a0', '#74b9ff',
    '#a29bfe', '#ffeaa7', '#dfe6e9', '#55efc4', '#81ecec',
]

# Score file patterns: suffix → extension
SCORE_FILE_PATTERNS = [
    ('_acc', 'csv'),
    ('_acc', 'xlsx'),
    ('_rating', 'json'),
    ('_score', 'json'),
]

# Benchmark prefixes whose sub-benchmarks should be merged into one column
MERGE_PREFIXES = ['AoTBench']

# ETBench aggregation groups: group_name -> {key, tasks}
ETBENCH_GROUPS = {
    'REF': {'key': 'REF/Acc', 'tasks': ['RAR/Acc', 'ECA/Acc', 'RVQ/Acc']},
    'GND': {'key': 'GND/F1', 'tasks': ['TVG/F1', 'EPM/F1', 'TAL/F1', 'EVS/F1', 'VHD/F1']},
    'GND@.1': {'key': 'GND/F1@0.1', 'tasks': ['TVG/F1@0.1', 'EPM/F1@0.1', 'TAL/F1@0.1']},
    'GND@.3': {'key': 'GND/F1@0.3', 'tasks': ['TVG/F1@0.3', 'EPM/F1@0.3', 'TAL/F1@0.3']},
    'GND@.5': {'key': 'GND/F1@0.5', 'tasks': ['TVG/F1@0.5', 'EPM/F1@0.5', 'TAL/F1@0.5']},
    'GND@.7': {'key': 'GND/F1@0.7', 'tasks': ['TVG/F1@0.7', 'EPM/F1@0.7', 'TAL/F1@0.7']},
    'CAP(F1)': {'key': 'CAP/F1', 'tasks': ['DVC/F1', 'SLC/F1']},
    'CAP@.1': {'key': 'CAP/F1@0.1', 'tasks': ['DVC/F1@0.1', 'SLC/F1@0.1']},
    'CAP@.3': {'key': 'CAP/F1@0.3', 'tasks': ['DVC/F1@0.3', 'SLC/F1@0.3']},
    'CAP@.5': {'key': 'CAP/F1@0.5', 'tasks': ['DVC/F1@0.5', 'SLC/F1@0.5']},
    'CAP@.7': {'key': 'CAP/F1@0.7', 'tasks': ['DVC/F1@0.7', 'SLC/F1@0.7']},
    'CAP(Sim)': {'key': 'CAP/SentSim', 'tasks': ['DVC/SentSim', 'SLC/SentSim']},
    'COM': {'key': 'COM/mRec', 'tasks': ['TEM/mRec', 'GVQ/mRec']},
}

ETBENCH_TABLE_ORDER = [
    'REF',
    'GND', 'GND@.1', 'GND@.3', 'GND@.5', 'GND@.7',
    'CAP(F1)', 'CAP@.1', 'CAP@.3', 'CAP@.5', 'CAP@.7',
    'CAP(Sim)', 'COM', 'AVG',
]

VINOGROUND_TABLE_COLUMNS = [
    ('text_score', 'Text'),
    ('video_score', 'Video'),
    ('group_score', 'Group'),
]

VIDEO_MME_TABLE_COLUMNS = [
    ('short/overall', 'Short'),
    ('medium/overall', 'Medium'),
    ('long/overall', 'Long'),
    ('overall/overall', 'AVG'),
]

GROUNDING_PRIMARY_KEYS = [
    'mIoU',
    'Average/Average IoU',
    'Average IoU',
]

PRIMARY_METRIC_KEYS = [
    'AVG',
    'group_score',
    'overall/overall',
    'overall/score',
    'overall/acc',
    'overall/accuracy',
    'overall/mIoU',
    'overall',
    'Overall',
    'final_rating/total',
    'M-Avg',
    'Macro Accuracy',
    'Overall Consistency',
    'Final Score',
    'score',
    'accuracy',
    'acc',
]
