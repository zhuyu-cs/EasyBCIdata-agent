"""CLI branding constants and helpers.

Kept as a small standalone module so passive UI labels can be unit-tested
without pulling in the full ``easybci_lib.cli`` import chain (prompt_toolkit,
rich, etc).
"""

from __future__ import annotations

import os


BRAND_NAME = "EasyBCI"


def branded_model_label(model_short):
    """Return the model label shown in passive UI surfaces.

    Default: brand name only. Set EASYBCI_SHOW_MODEL=1/true/yes for the real
    backend model name (developer/debug aid). The status-bar snapshot still
    carries the real model name in its ``model_name`` field for diagnostic
    callers — only the rendering layer is brand-locked.
    """
    show = os.environ.get("EASYBCI_SHOW_MODEL", "").strip().lower()
    if show in ("1", "true", "yes"):
        return model_short or BRAND_NAME
    return BRAND_NAME
