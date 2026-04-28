import importlib.util
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'vlmeval' / 'dataset' / 'vdc_logging.py'


def load_vdc_logging():
    spec = importlib.util.spec_from_file_location('vdc_logging_test', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_vdc_judge_io_logs_keeps_prompts_and_outputs():
    vdc_logging = load_vdc_logging()
    expanded_df = pd.DataFrame([
        {
            'index': 7,
            'judge_key': '7::0',
            'video': 'video/7.mp4',
            'caption_type': 'short',
            'question': 'What is the person doing?',
            'answer': 'running',
            'prediction': 'A person runs across a field.',
        }
    ])
    response_prompts = ['response prompt']
    score_prompts = ['score prompt']
    response_map = {'7::0': 'The person is running.'}
    score_map = {'7::0': "{'pred': 'yes', 'score': 5}"}

    logs = vdc_logging.build_vdc_judge_io_logs(
        expanded_df,
        ['7::0'],
        response_prompts,
        score_prompts,
        response_map,
        score_map,
        response_system_prompt='response system',
        score_system_prompt='score system',
        judge='gpt-4o',
    )

    assert logs == [
        {
            'index': 7,
            'judge_key': '7::0',
            'video': 'video/7.mp4',
            'caption_type': 'short',
            'judge': 'gpt-4o',
            'question': 'What is the person doing?',
            'answer': 'running',
            'prediction': 'A person runs across a field.',
            'pred_response': 'The person is running.',
            'score': "{'pred': 'yes', 'score': 5}",
            'judge_io': [
                {
                    'stage': 'response_generation',
                    'system_prompt': 'response system',
                    'prompt': 'response prompt',
                    'response': 'The person is running.',
                },
                {
                    'stage': 'score_evaluation',
                    'system_prompt': 'score system',
                    'prompt': 'score prompt',
                    'response': "{'pred': 'yes', 'score': 5}",
                },
            ],
        }
    ]


def test_build_vdc_judge_io_logs_keeps_duplicate_indices_separate():
    vdc_logging = load_vdc_logging()
    expanded_df = pd.DataFrame([
        {
            'index': 7,
            'judge_key': '7::0',
            'caption_type': 'short',
            'question': 'First question?',
            'answer': 'first answer',
            'prediction': 'A shared model caption.',
        },
        {
            'index': 7,
            'judge_key': '7::1',
            'caption_type': 'short',
            'question': 'Second question?',
            'answer': 'second answer',
            'prediction': 'A shared model caption.',
        },
    ])

    logs = vdc_logging.build_vdc_judge_io_logs(
        expanded_df,
        ['7::0', '7::1'],
        ['response prompt 0', 'response prompt 1'],
        ['score prompt 0', 'score prompt 1'],
        {'7::0': 'first response', '7::1': 'second response'},
        {'7::0': "{'pred': 'yes', 'score': 5}", '7::1': "{'pred': 'no', 'score': 1}"},
        response_system_prompt='response system',
        score_system_prompt='score system',
        judge='gpt-4o',
    )

    assert [record['judge_key'] for record in logs] == ['7::0', '7::1']
    assert [record['pred_response'] for record in logs] == ['first response', 'second response']
    assert [record['score'] for record in logs] == [
        "{'pred': 'yes', 'score': 5}",
        "{'pred': 'no', 'score': 1}",
    ]
