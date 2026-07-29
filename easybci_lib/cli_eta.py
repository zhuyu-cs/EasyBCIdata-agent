"""CLI helpers for turn-scope ETA — EWMA smoothing + display formatting.

Lives outside cli.py because cli.py is 12k+ LOC and already imports prompt_toolkit
at module top — these helpers stay light and unit-testable.

Display contract is countdown-only: when ``remaining_seconds`` hits zero we sit
at ``+0s`` until the next ETA arrives. No "over by Xs" branch, no ``~?``
low-confidence qualifier — those visual decorations made completed runs look
like they were stuck or buggy.
"""
from __future__ import annotations

from typing import Optional


def apply_ewma(*, new: int, prev: Optional[int], alpha: float = 0.4) -> int:
    """Exponentially weighted moving average — first sample passes through.

    α = 0.4 means new readings replace previous gradually but visibly.
    """
    if prev is None:
        return int(round(new))
    return int(round(alpha * new + (1.0 - alpha) * prev))


def compute_remaining_eta(
    *,
    eta_seconds: Optional[int],
    emitted_at: float,
    now: float,
) -> Optional[int]:
    """Return max(0, eta - elapsed_since_emit). None when input ETA is None."""
    if eta_seconds is None:
        return None
    remaining = eta_seconds - int(now - emitted_at)
    return max(0, remaining)


def format_eta_suffix(
    *,
    remaining_seconds: Optional[int],
    confidence: Optional[str] = None,
) -> str:
    """Format the countdown text ``est +Xs`` (or ``est +Xm Ys``).

    Returns an empty string when ``remaining_seconds`` is ``None``. No leading
    arrow / decoration; the caller composes the surrounding glyphs.
    ``confidence`` is accepted for backwards compatibility but no longer
    changes the rendered text.
    """
    del confidence  # kept for callers; intentionally ignored

    if remaining_seconds is None:
        return ""

    try:
        from easybci_agent.i18n import t  # noqa: PLC0415
    except Exception:
        def t(key, **kw):  # type: ignore[no-redef]
            return key

    m, s = divmod(max(0, int(remaining_seconds)), 60)
    body = f"{m}m {s}s" if m else f"{s}s"
    text = t("progress.eta_remaining", seconds=body)
    if _i18n_miss(text):
        text = f"est +{body}"
    return text


def _i18n_miss(value: str) -> bool:
    return isinstance(value, str) and value.startswith("progress.")
