"""Native structured output requests for context-mode writing."""

from collections.abc import Callable
from typing import NamedTuple
from urllib.parse import quote

import requests


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    response = requests.post(url, headers=headers, json=payload, timeout=(10, 240))
    response.raise_for_status()
    return response.json()


def _gemini(api_key: str, model: str, instructions: str, brief: str, schema: dict, max_tokens: int) -> str:
    generation_config = {
        'responseMimeType': 'application/json',
        'responseJsonSchema': schema,
        'maxOutputTokens': max_tokens,
    }
    if model.startswith('gemini-4-'):
        generation_config['thinkingConfig'] = {'thinkingLevel': 'low'}
    result = _post_json(
        f'https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe="-._")}:generateContent',
        {'x-goog-api-key': api_key},
        {
            'systemInstruction': {'parts': [{'text': instructions}]},
            'contents': [{'role': 'user', 'parts': [{'text': brief}]}],
            'generationConfig': generation_config,
        },
    )
    candidates = result.get('candidates', [])
    if not candidates:
        reason = result.get('promptFeedback', {}).get('blockReason', 'no candidate')
        raise ValueError(f'Context writer {model} returned no episode ({reason})')
    candidate = candidates[0]
    if candidate.get('finishReason') != 'STOP':
        raise ValueError(f"Context writer {model} stopped early ({candidate.get('finishReason')})")
    return ''.join(
        part.get('text', '') for part in candidate['content']['parts'] if not part.get('thought')
    )


def _openai(api_key: str, model: str, instructions: str, brief: str, schema: dict, max_tokens: int) -> str:
    payload = {
        'model': model,
        'instructions': instructions,
        'input': brief,
        'max_output_tokens': max_tokens,
        'store': False,
        'text': {'format': {
            'type': 'json_schema', 'name': 'context_episode', 'schema': schema, 'strict': True,
        }},
    }
    if model.startswith('gpt-6.2-'):
        payload['reasoning'] = {'effort': 'low'}
    result = _post_json('https://api.openai.com/v1/responses', {'Authorization': f'Bearer {api_key}'}, payload)
    status = result.get('status')
    if status != 'completed':
        reason = (result.get('incomplete_details') or {}).get('reason', status)
        raise ValueError(f'Context writer {model} stopped early ({reason})')
    content = [
        block
        for item in result.get('output', []) if item.get('type') == 'message'
        for block in item.get('content', [])
    ]
    if any(block.get('type') == 'refusal' for block in content):
        raise ValueError(f'Context writer {model} refused the request')
    return ''.join(block['text'] for block in content if block.get('type') == 'output_text')


def _claude_schema(schema):
    # Anthropic supports minItems of one but leaves string lengths to local validation.
    if isinstance(schema, dict):
        return {key: _claude_schema(value) for key, value in schema.items() if key != 'minLength'}
    if isinstance(schema, list):
        return [_claude_schema(value) for value in schema]
    return schema


def _anthropic(api_key: str, model: str, instructions: str, brief: str, schema: dict, max_tokens: int) -> str:
    result = _post_json(
        'https://api.anthropic.com/v1/messages',
        {'x-api-key': api_key, 'anthropic-version': '2023-06-01'},
        {
            'model': model,
            'system': instructions,
            'messages': [{'role': 'user', 'content': brief}],
            'max_tokens': max_tokens,
            'output_config': {'format': {'type': 'json_schema', 'schema': _claude_schema(schema)}},
        },
    )
    if result.get('stop_reason') != 'end_turn':
        raise ValueError(f"Context writer {model} stopped early ({result.get('stop_reason')})")
    return ''.join(block['text'] for block in result.get('content', []) if block.get('type') == 'text')


class ContextBackend(NamedTuple):
    default_model: str
    api_key_env: str
    generate: Callable[[str, str, str, str, dict, int], str]


CONTEXT_BACKENDS = {
    'google': ContextBackend('gemini-4-neon', 'GOOGLE_API_KEY', _gemini),
    'openai': ContextBackend('gpt-6.2-sol', 'OPENAI_API_KEY', _openai),
    'anthropic': ContextBackend('claude-sonnet-5-5', 'ANTHROPIC_API_KEY', _anthropic),
}
