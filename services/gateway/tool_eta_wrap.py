"""Tool handler wrapper that records elapsed-time samples.

Used by the registry instrumentation to capture every "interesting" tool
call's wall-time without modifying each handler file. ``intent_kind == "unknown"``
opts a handler out (passthrough) so stage-scope tools (preprocess_neural,
generate_code, …) don't double-record (they go through the staged tracker).
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

from easybci_lib.tools.neural_processing.progress.history import record_intent_elapsed

logger = logging.getLogger(__name__)


def wrap_handler_with_eta_record(
    handler: Callable[..., Any],
    *,
    intent_kind: str,
) -> Callable[..., Any]:
    """Wrap a tool handler so every dispatch records elapsed seconds.

    Records on both success and exception paths. ``intent_kind="unknown"``
    short-circuits to a passthrough — useful for tools we don't want to
    instrument (preprocess_neural, generate_code, …).
    """
    if intent_kind == "unknown":
        return handler

    @functools.wraps(handler)
    def _wrapped(args, **kw):
        t0 = time.time()
        try:
            return handler(args, **kw)
        finally:
            try:
                record_intent_elapsed(kind=intent_kind, elapsed_s=time.time() - t0)
            except Exception:
                logger.debug("record_intent_elapsed failed in wrap", exc_info=True)

    _wrapped._eta_wrapped = True  # type: ignore[attr-defined]
    return _wrapped
