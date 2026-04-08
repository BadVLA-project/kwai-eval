import ast
import re
import numpy as np
from ...smp import *
from .multiple_choice import extract_answer_from_item

FAIL_MSG = 'Failed to obtain answer via API.'


def extract_characters_regex_v2(s):
    """Extract answer letter A-H from model response."""
    s = s.strip()
    answer_prefixes = [
        'Final Answer:',
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

    if len(s.split()) > 10 and not re.search('[A-H]', s):
        return ''
    matches = re.search(r'[A-H]', s)
    if matches is None:
        return ''
    return matches[0]


def extract_option_v2(model, input_item, dataset_name):
    """GPT judge fallback for answer extraction (8 options A-H)."""
    options_raw = input_item.get('options', '')
    if isinstance(options_raw, str):
        try:
            options_list = eval(options_raw)
        except Exception:
            options_list = []
    else:
        options_list = list(options_raw)

    for i, opt in enumerate(options_list):
        key = chr(ord('A') + i)
        input_item[key] = str(opt).strip()

    return extract_answer_from_item(model, input_item, dataset_name)['opt']


def cal_relevance(scores):
    """Non-linear relevance scoring for a group of 4 questions.

    Score map: {0: 0.0, 1: 6.25, 2: 25.0, 3: 56.25, 4: 100.0}
    Returns (non_linear_score, linear_score).
    """
    score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    correct_count = sum(scores)
    return score_map.get(correct_count, 0.0), correct_count * 25.0


def cal_logic(scores, group_structure):
    """Non-linear logic scoring with structure-dependent score maps."""
    group_structure_list = ast.literal_eval(group_structure)
    last_correct_idx = -1
    for idx, val in enumerate(scores):
        if val:
            last_correct_idx = idx
        else:
            break

    if group_structure_list == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif group_structure_list == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif group_structure_list == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(f'Unknown group_structure_list: {group_structure_list}')

    return score_map.get(last_correct_idx + 1, 0.0)


def get_final_rating_v2(data_path):
    """Compute per-level, per-category, per-group-type scores with non-linear grouped scoring."""
    data = load(data_path)

    all_groups = [[] for _ in range((len(data) + 1) // 4)]
    final_rating = {
        'level_1': [], 'level_2': [], 'level_3': [],
        'relevance_score': [], 'relevance_linear_score': [],
        'logic_score': [], 'total': [],
    }
    second_head_rating = {}
    third_head_rating = {}

    for i in range(len(data)):
        level = data.iloc[i]['level']
        group_type = data.iloc[i]['group_type']
        group_structure = data.iloc[i]['group_structure']
        score = data.iloc[i]['score']
        second_head = data.iloc[i]['second_head']
        third_head = data.iloc[i]['third_head']
        all_groups[i // 4].append((level, group_type, group_structure, score, second_head, third_head))

    for group in all_groups:
        level = group[-1][0]
        group_type = group[-1][1]
        group_structure = group[-1][2]
        second_head = group[-1][4]
        third_head = group[-1][5]
        scores = [item[3] for item in group]

        if group_type == 'relevance':
            exp_score, linear_score = cal_relevance(scores)
            final_rating['relevance_score'].append(exp_score)
            final_rating['relevance_linear_score'].append(linear_score)
        elif group_type == 'logic':
            exp_score = cal_logic(scores, group_structure)
            final_rating['logic_score'].append(exp_score)
        else:
            raise ValueError(f'Unknown group_type: {group_type}')

        if level is not None and str(level) != 'None':
            final_rating[f'level_{int(level)}'].append(exp_score)
        final_rating['total'].append(exp_score)

        if second_head not in second_head_rating:
            second_head_rating[second_head] = []
        second_head_rating[second_head].append(exp_score)
        if third_head not in third_head_rating:
            third_head_rating[third_head] = []
        third_head_rating[third_head].append(exp_score)

    for key in final_rating:
        vals = final_rating[key]
        final_rating[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    for key in second_head_rating:
        vals = second_head_rating[key]
        second_head_rating[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    for key in third_head_rating:
        vals = third_head_rating[key]
        third_head_rating[key] = round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        'final_rating': final_rating,
        'second_head_rating': second_head_rating,
        'third_head_rating': third_head_rating,
    }
