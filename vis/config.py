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
