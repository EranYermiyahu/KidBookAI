"""
LiteLLM-powered chat completion helper utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from litellm import completion

ChatMessage = Mapping[str, Any]


@dataclass
class ChatResult:
    """
    Structured response returned from an LLM chat completion.
    """

    text: str
    raw: Any


CompletionCallable = Callable[..., ChatResult]


def call_chat_completion(
    *,
    model: str,
    messages: Sequence[ChatMessage],
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    **extra_kwargs: Any,
) -> ChatResult:
    """
    Invoke LiteLLM's `completion` API and return the consolidated text.
    """
    payload: MutableMapping[str, Any] = {
        "model": model,
        "messages": list(messages),
    }

    if temperature is not None:
        payload["temperature"] = temperature

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if api_key is not None:
        payload["api_key"] = api_key

    payload.update(extra_kwargs)

    response = completion(**payload)

    try:
        message = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected LiteLLM response format.") from exc

    text = str(message).strip()
    return ChatResult(text=text, raw=response)

