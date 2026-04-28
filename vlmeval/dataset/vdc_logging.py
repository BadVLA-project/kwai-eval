def _lookup(mapping, key, default=None):
    for candidate in (key, str(key)):
        if candidate in mapping:
            return mapping[candidate]
    try:
        int_key = int(key)
    except (TypeError, ValueError):
        int_key = None
    if int_key is not None and int_key in mapping:
        return mapping[int_key]
    return default


def _row_value(row, key, default=None):
    if key in row:
        value = row[key]
        try:
            if value != value:
                return default
        except Exception:
            pass
        return value
    return default


def build_vdc_judge_io_logs(
        expanded_df,
        indices,
        response_prompts,
        score_prompts,
        response_map,
        score_map,
        response_system_prompt,
        score_system_prompt,
        judge):
    logs = []
    for row_pos, (_, row) in enumerate(expanded_df.iterrows()):
        key = indices[row_pos]
        pred_response = _lookup(response_map, key)
        score_response = _lookup(score_map, key)
        logs.append({
            'index': _row_value(row, 'index', key),
            'judge_key': _row_value(row, 'judge_key', key),
            'video': _row_value(row, 'video'),
            'caption_type': _row_value(row, 'caption_type'),
            'judge': judge,
            'question': _row_value(row, 'question'),
            'answer': _row_value(row, 'answer'),
            'prediction': _row_value(row, 'prediction'),
            'pred_response': pred_response,
            'score': score_response,
            'judge_io': [
                {
                    'stage': 'response_generation',
                    'system_prompt': response_system_prompt,
                    'prompt': response_prompts[row_pos],
                    'response': pred_response,
                },
                {
                    'stage': 'score_evaluation',
                    'system_prompt': score_system_prompt,
                    'prompt': score_prompts[row_pos],
                    'response': score_response,
                },
            ],
        })
    return logs
