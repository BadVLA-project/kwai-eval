import re
import numpy as np
from ...smp import *


def extract_characters_regex(s: str) -> str:
    """Extract an A or B answer letter from a model prediction string."""
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is',
        'The correct option is',
        'Best answer:',
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, '')

    if len(s.split()) > 10 and not re.search(r'[AB]', s):
        return ''
    matches = re.search(r'[AB]', s)
    if matches is None:
        return ''
    return matches[0]


def get_vinoground_scores(score_file: str) -> dict:
    """Compute Vinoground text/video/group scores from a scored result file.

    The score file must have columns: idx (int 0-499), question_type (text/video),
    variant (pos/neg), score (0 or 1).

    Returns a dict with keys: text_score, video_score, group_score (percentages).
    """
    data = load(score_file)

    n_samples = 500
    # 7 columns:
    #   0: text_pos   1: text_neg   2: text_group  (col0 & col1)
    #   3: video_pos  4: video_neg  5: video_group (col3 & col4)
    #   6: full_group (col2 & col5)
    matrix = np.zeros((n_samples, 7), dtype=int)

    for _, row in data.iterrows():
        base_idx = int(row['idx'])
        q_type = str(row['question_type'])
        variant = str(row['variant'])
        score = int(row.get('score', 0))
        if score < 0:
            score = 0

        if q_type == 'text':
            col = 0 if variant == 'pos' else 1
        else:  # video
            col = 3 if variant == 'pos' else 4

        if 0 <= base_idx < n_samples:
            matrix[base_idx, col] = score

    matrix[:, 2] = matrix[:, 0] & matrix[:, 1]   # text_group
    matrix[:, 5] = matrix[:, 3] & matrix[:, 4]   # video_group
    matrix[:, 6] = matrix[:, 2] & matrix[:, 5]   # full_group

    text_score = float(np.mean(matrix[:, 2])) * 100
    video_score = float(np.mean(matrix[:, 5])) * 100
    group_score = float(np.mean(matrix[:, 6])) * 100

    return {
        'text_score': round(text_score, 2),
        'video_score': round(video_score, 2),
        'group_score': round(group_score, 2),
    }
