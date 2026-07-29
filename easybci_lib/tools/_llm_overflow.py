"""Shared LLM call wrapper that retries once with bounded input on
context-overflow errors.

Used by callers that previously hard-truncated their LLM input (e.g.
`[:8000]` / `[:2000]`) to avoid exceeding aux-model context windows. The
old approach silently corrupted long inputs even when the model could
have handled them. This helper instead tries the full input first, and
only on a classified context-overflow error does it slice each user-role
message to ``fallback_input_chars`` and retry once.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def call_llm_with_overflow_retry(
    *,
    call_llm: Callable[..., Any],
    classify_api_error: Callable[[Exception], Any] | None = None,
    fallback_input_chars: int = 64_000,
    **call_llm_kwargs: Any,
) -> Any:
    """Call ``call_llm(**kwargs)``; on context-overflow, retry once with
    each user-role message's content sliced to ``fallback_input_chars``.

    Parameters
    ----------
    call_llm:
        The auxiliary client's ``call_llm`` function. Passed in (rather
        than imported) so this helper has zero hard dependencies on
        ``easybci_agent``.
    classify_api_error:
        Optional. Defaults to ``easybci_agent.error_classifier.classify_api_error``
        when not provided. Called with the raised exception; if the
        classification's ``reason`` equals ``FailoverReason.context_overflow``
        we slice and retry.
    fallback_input_chars:
        Per-message char cap applied to user-role messages on retry.
        Defaults to 64_000 — fits comfortably in any modern aux model's
        context window with room for the system prompt and response.
    **call_llm_kwargs:
        Forwarded verbatim to ``call_llm``. Must include ``messages``.

    Raises
    ------
    Re-raises the first call's exception if classification fails or the
    error isn't context-overflow. Re-raises the second call's exception
    if the retry also fails.
    """
    try:
        return call_llm(**call_llm_kwargs)
    except Exception as first_exc:  # noqa: BLE001 — we re-raise after retry
        try:
            if classify_api_error is None:
                from easybci_agent.error_classifier import classify_api_error as _classify
            else:
                _classify = classify_api_error
            classified = _classify(first_exc)
        except Exception:  # noqa: BLE001 — classifier itself failed
            raise first_exc

        from easybci_agent.error_classifier import FailoverReason

        if getattr(classified, "reason", None) != FailoverReason.context_overflow:
            raise

        messages = call_llm_kwargs.get("messages") or []
        new_messages: list[dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                new_messages.append(msg)
                continue
            if msg.get("role") != "user":
                new_messages.append(msg)
                continue
            content = msg.get("content")
            if isinstance(content, str) and len(content) > fallback_input_chars:
                new_messages.append({**msg, "content": content[:fallback_input_chars]})
                logger.info(
                    "call_llm_with_overflow_retry: sliced user message "
                    "from %d to %d chars after context_overflow",
                    len(content), fallback_input_chars,
                )
            else:
                new_messages.append(msg)

        retry_kwargs = {**call_llm_kwargs, "messages": new_messages}
        return call_llm(**retry_kwargs)
