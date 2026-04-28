import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_azure_utils():
    module_path = ROOT / 'vlmeval' / 'api' / 'azure_openai_utils.py'
    spec = importlib.util.spec_from_file_location('azure_openai_utils_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_azure_config_accepts_user_env_aliases():
    utils = load_azure_utils()
    cfg = utils.resolve_azure_openai_config(
        env={
            'ENDPOINT_URL': 'https://example.openai.azure.com/',
            'DEPLOYMENT_NAME': 'mmu-gpt4o',
        }
    )

    assert cfg.endpoint == 'https://example.openai.azure.com'
    assert cfg.deployment_name == 'mmu-gpt4o'
    assert cfg.api_version == '2025-01-01-preview'


def test_resolve_azure_config_prefers_explicit_values():
    utils = load_azure_utils()
    cfg = utils.resolve_azure_openai_config(
        azure_endpoint='https://explicit.azure.com/',
        azure_deployment_name='explicit-deployment',
        api_version='2026-01-01-preview',
        env={
            'ENDPOINT_URL': 'https://env.azure.com',
            'DEPLOYMENT_NAME': 'env-deployment',
            'OPENAI_API_VERSION': '2025-01-01-preview',
        }
    )

    assert cfg.endpoint == 'https://explicit.azure.com'
    assert cfg.deployment_name == 'explicit-deployment'
    assert cfg.api_version == '2026-01-01-preview'


def test_resolve_azure_key_accepts_eval_and_lmms_aliases():
    utils = load_azure_utils()

    assert utils.resolve_azure_openai_key(env={'AZURE_OPENAI_API_KEY': 'eval-key'}) == 'eval-key'
    assert utils.resolve_azure_openai_key(env={'AZURE_API_KEY': 'lmms-key'}) == 'lmms-key'


def test_build_chat_completion_payload_uses_deployment_and_max_completion_tokens():
    utils = load_azure_utils()
    payload = utils.build_chat_completion_payload(
        deployment_name='mmu-gpt4o',
        messages=[{'role': 'developer', 'content': [{'type': 'text', 'text': 'ping'}]}],
        temperature=0,
        max_tokens=2048,
        max_completion_tokens=16384,
        use_max_completion_tokens=False,
        extra_kwargs={'stop': None},
    )

    assert payload['model'] == 'mmu-gpt4o'
    assert payload['max_completion_tokens'] == 16384
    assert 'max_tokens' not in payload
    assert payload['stream'] is False
    assert payload['stop'] is None


def test_o_model_payload_uses_max_completion_tokens_and_drops_temperature():
    utils = load_azure_utils()
    payload = utils.build_chat_completion_payload(
        deployment_name='o4-mini-deployment',
        messages=[{'role': 'user', 'content': 'ping'}],
        temperature=0,
        max_tokens=4096,
        max_completion_tokens=None,
        use_max_completion_tokens=True,
    )

    assert payload['max_completion_tokens'] == 4096
    assert 'max_tokens' not in payload
    assert 'temperature' not in payload


def test_vdc_default_judge_uses_gpt4o_when_llama_is_unavailable():
    run_py = (ROOT / 'run.py').read_text(encoding='utf-8')
    match = re.search(
        r"elif listinstr\(\['VDC'\], dataset_name\):\s*\n\s*judge_kwargs\['model'\] = '([^']+)'",
        run_py,
    )

    assert match is not None
    assert match.group(1) == 'gpt-4o'
