"""
Common utilities shared across KidBookAI modules.
"""

from .llm import ChatResult, CompletionCallable, call_chat_completion

__all__ = ["ChatResult", "CompletionCallable", "call_chat_completion"]

