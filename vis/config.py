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
    'CAP': {'key': 'CAP/F1', 'tasks': ['DVC/F1', 'DVC/SentSim', 'SLC/F1', 'SLC/SentSim']},
    'COM': {'key': 'COM/mRec', 'tasks': ['TEM/mRec', 'GVQ/mRec']},
}

ETBENCH_TABLE_ORDER = ['REF', 'GND', 'CAP', 'COM', 'AVG']

VINOGROUND_TABLE_COLUMNS = [
    ('text_score', 'Text'),
    ('video_score', 'Video'),
    ('group_score', 'Group'),
]

GROUNDING_PRIMARY_KEYS = [
    'mIoU',
    'Average/Average IoU',
    'Average IoU',
]

PRIMARY_METRIC_KEYS = [
    'AVG',
    'group_score',
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
