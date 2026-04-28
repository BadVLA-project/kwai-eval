import os
from dataclasses import dataclass


DEFAULT_AZURE_API_VERSION = '2025-01-01-preview'


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    deployment_name: str
    api_version: str


def _first_env(env, names):
    for name in names:
        value = env.get(name)
        if value:
            return value
    return None


def resolve_azure_openai_key(key=None, env=None):
    env = os.environ if env is None else env
    resolved = key or _first_env(env, ('AZURE_OPENAI_API_KEY', 'AZURE_API_KEY'))
    if not resolved:
        raise ValueError('Please set AZURE_OPENAI_API_KEY or AZURE_API_KEY for Azure OpenAI.')
    return resolved


def resolve_azure_openai_config(
    azure_endpoint=None,
    azure_deployment_name=None,
    api_version=None,
    env=None,
):
    env = os.environ if env is None else env
    endpoint = azure_endpoint or _first_env(env, ('AZURE_OPENAI_ENDPOINT', 'ENDPOINT_URL', 'AZURE_ENDPOINT'))
    deployment_name = azure_deployment_name or _first_env(
        env,
        ('AZURE_OPENAI_DEPLOYMENT_NAME', 'DEPLOYMENT_NAME', 'AZURE_DEPLOYMENT_NAME'),
    )
    resolved_api_version = api_version or _first_env(
        env,
        ('OPENAI_API_VERSION', 'AZURE_OPENAI_API_VERSION', 'API_VERSION'),
    ) or DEFAULT_AZURE_API_VERSION

    if not endpoint:
        raise ValueError('Please set AZURE_OPENAI_ENDPOINT, ENDPOINT_URL, or AZURE_ENDPOINT.')
    if not deployment_name:
        raise ValueError('Please set AZURE_OPENAI_DEPLOYMENT_NAME, DEPLOYMENT_NAME, or AZURE_DEPLOYMENT_NAME.')

    return AzureOpenAIConfig(
        endpoint=endpoint.rstrip('/'),
        deployment_name=deployment_name,
        api_version=resolved_api_version,
    )


def build_chat_completion_payload(
    deployment_name,
    messages,
    temperature,
    max_tokens,
    max_completion_tokens=None,
    use_max_completion_tokens=False,
    extra_kwargs=None,
    n=1,
    stream=False,
    include_stream=True,
):
    payload = {
        'model': deployment_name,
        'messages': messages,
        'n': n,
    }
    if temperature is not None:
        payload['temperature'] = temperature
    if include_stream:
        payload['stream'] = stream
    if extra_kwargs:
        payload.update(extra_kwargs)

    if max_completion_tokens is not None or use_max_completion_tokens:
        payload['max_completion_tokens'] = (
            max_completion_tokens if max_completion_tokens is not None else max_tokens
        )
        if use_max_completion_tokens:
            payload.pop('temperature', None)
    else:
        payload['max_tokens'] = max_tokens

    return payload
