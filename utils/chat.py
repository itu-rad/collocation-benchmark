"""Helpers for safe chat-template invocation across tokenizer families.

Some models (Qwen 3.x, Gemma 3+, DeepSeek-R1 derivatives) include a
'thinking mode' that emits step-by-step chain-of-thought before the
final answer. For structured-output stages — JSON parsers, binary
yes/no graders, single-line rewriters — this CoT preamble blows
through `max_tokens` before the model ever produces the parseable
output, so the downstream router sees empty/junk strings.

`apply_chat_template_safe` disables thinking mode when the tokenizer
supports it (silently no-op for tokenizers that don't, e.g. Llama).
"""

from __future__ import annotations

from typing import Any


def apply_chat_template_safe(
    tokenizer,
    chat: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> str:
    """Apply chat template with `enable_thinking` if the tokenizer accepts it.

    Falls back to the plain call for tokenizers that don't know the kwarg
    (raises TypeError on unexpected keyword).

    Args:
        tokenizer: HuggingFace-style tokenizer with `apply_chat_template`.
        chat: List of `{"role": ..., "content": ...}` dicts.
        enable_thinking: Whether to allow the model's thinking-mode CoT.
            Default False so structured-output stages stay concise.
        **kwargs: Forwarded to `apply_chat_template` (e.g. `tokenize`,
            `add_generation_prompt`).

    Returns:
        The rendered prompt string.
    """
    try:
        return tokenizer.apply_chat_template(
            chat, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        # Tokenizer doesn't know `enable_thinking` (Llama, etc.).
        return tokenizer.apply_chat_template(chat, **kwargs)
