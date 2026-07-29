"""Auto-fixer — rule-based code repair for common execution failures.

When generated pipeline.py fails to execute, this module analyzes the
traceback and applies deterministic fixes. Used in pipeline execution for
automatic code repair without LLM involvement.

In Agent mode, the LLM sees the error and fixes code itself (via budget
refund on execute_code). This module is the non-LLM equivalent.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

_MAX_FIX_ATTEMPTS = 10


@dataclass
class FixSuggestion:
    """A suggested code fix."""
    pattern_name: str
    description: str
    line_number: Optional[int] = None
    fix_type: str = "insert"  # insert, replace, remove
    target: str = ""
    replacement: str = ""


# Each pattern: (name, regex on stderr, fix function)
_ERROR_PATTERNS: List[tuple] = [
    (
        "missing_import",
        r"ModuleNotFoundError: No module named '(\w+)'",
        "_fix_missing_import",
    ),
    (
        "name_not_defined",
        r"NameError: name '(\w+)' is not defined",
        "_fix_name_error",
    ),
    (
        "file_not_found",
        r"FileNotFoundError:.*No such file or directory: '([^']+)'",
        "_fix_file_not_found",
    ),
    (
        "unexpected_keyword",
        r"TypeError: \w+\(\) got an unexpected keyword argument '(\w+)'",
        "_fix_unexpected_kwarg",
    ),
    (
        "attribute_none",
        r"AttributeError: 'NoneType' object has no attribute '(\w+)'",
        "_fix_none_attribute",
    ),
    (
        "index_out_of_bounds",
        r"IndexError: index (\d+) is out of bounds",
        "_fix_index_error",
    ),
    (
        "value_error_broadcast",
        r"ValueError:.*could not broadcast",
        "_fix_broadcast_error",
    ),
    (
        "import_from_error",
        r"ImportError: cannot import name '(\w+)' from '([^']+)'",
        "_fix_import_from_error",
    ),
    (
        "sys_path_missing",
        r"ModuleNotFoundError: No module named '(processing|core|agents)'",
        "_fix_sys_path",
    ),
]

# Common module → import mapping for BCI pipelines
_IMPORT_MAP = {
    "mne": "import mne",
    "numpy": "import numpy as np",
    "np": "import numpy as np",
    "scipy": "import scipy",
    "sklearn": "import sklearn",
    "matplotlib": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
    "plt": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
    "pandas": "import pandas as pd",
    "pd": "import pandas as pd",
    "h5py": "import h5py",
    "yaml": "import yaml",
    "json": "import json",
    "os": "import os",
    "sys": "import sys",
    "pickle": "import pickle",
    "pathlib": "from pathlib import Path",
    "Path": "from pathlib import Path",
}


def analyze_error(stderr: str) -> Optional[FixSuggestion]:
    """Analyze execution stderr and suggest a fix.

    Returns None if no pattern matches.
    """
    for name, pattern, fix_func in _ERROR_PATTERNS:
        match = re.search(pattern, stderr)
        if match:
            fix_fn = globals().get(fix_func)
            if fix_fn:
                suggestion = fix_fn(match, stderr)
                if suggestion:
                    suggestion.pattern_name = name
                    return suggestion
    return None


def apply_fix(code: str, fix: FixSuggestion) -> str:
    """Apply a fix suggestion to the source code.

    Returns modified code string.
    """
    if fix.fix_type == "insert":
        return fix.replacement + "\n" + code
    elif fix.fix_type == "replace":
        if fix.target and fix.target in code:
            return code.replace(fix.target, fix.replacement, 1)
        return code
    elif fix.fix_type == "remove":
        if fix.target:
            return code.replace(fix.target, "", 1)
        return code
    return code


def attempt_auto_fix(
    code: str,
    stderr: str,
    attempt: int = 0,
) -> Optional[tuple]:
    """Try to auto-fix code based on error output.

    Returns (fixed_code, fix_description) or None if unfixable.
    """
    if attempt >= _MAX_FIX_ATTEMPTS:
        return None

    fix = analyze_error(stderr)
    if fix is None:
        return None

    fixed_code = apply_fix(code, fix)
    if fixed_code == code:
        return None

    return fixed_code, fix.description


# ── Fix implementations ──────────────────────────────────────────────────

def _fix_missing_import(match: re.Match, stderr: str) -> Optional[FixSuggestion]:
    """Fix ModuleNotFoundError by adding import statement."""
    module = match.group(1)
    import_stmt = _IMPORT_MAP.get(module)
    if not import_stmt:
        import_stmt = f"import {module}"

    return FixSuggestion(
        pattern_name="missing_import",
        description=f"Added missing import: {import_stmt}",
        fix_type="insert",
        replacement=import_stmt,
    )


def _fix_name_error(match: re.Match, stderr: str) -> Optional[FixSuggestion]:
    """Fix NameError — check if it's a common alias."""
    name = match.group(1)
    import_stmt = _IMPORT_MAP.get(name)
    if import_stmt:
        return FixSuggestion(
            pattern_name="name_not_defined",
            description=f"Added import for undefined name '{name}'",
            fix_type="insert",
            replacement=import_stmt,
        )
    return None


def _fix_file_not_found(match: re.Match, stderr: str) -> Optional[FixSuggestion]:
    """Fix FileNotFoundError — add os.makedirs for output directories."""
    filepath = match.group(1)
    if "/" in filepath or "\\" in filepath:
        import os
        dirpath = os.path.dirname(filepath)
        fix_line = f"import os\nos.makedirs('{dirpath}', exist_ok=True)"
        return FixSuggestion(
            pattern_name="file_not_found",
            description=f"Added os.makedirs for missing directory: {dirpath}",
            fix_type="insert",
            replacement=fix_line,
        )
    return None


def _fix_unexpected_kwarg(match: re.Match, stderr: str) -> Optional[FixSuggestion]:
    """Fix unexpected keyword argument by removing it."""
    kwarg = match.group(1)
    return FixSuggestion(
        pattern_name="unexpected_keyword",
        description=f"Removed unexpected keyword argument: {kwarg}",
        fix_type="replace",
        target=f"{kwarg}=",
        replacement="",
    )


def _fix_none_attribute(match: re.Match, stderr: str) -> Optional[FixSuggestion]:
    """Fix NoneType attribute — too context-dependent for auto-fix."""
    return None


def _fix_index_error(match: re.Match, stderr: str) -> Optional[FixSuggestion]:
    """Fix IndexError — too context-dependent for auto-fix."""
    return None


def _fix_broadcast_error(match: re.Match, stderr: str) -> Optional[FixSuggestion]:
    """Fix broadcast ValueError — too context-dependent for auto-fix."""
    return None


def _fix_import_from_error(match: re.Match, stderr: str) -> Optional[FixSuggestion]:
    """Fix ImportError: cannot import name."""
    name = match.group(1)
    module = match.group(2)
    # Remove the offending import line
    return FixSuggestion(
        pattern_name="import_from_error",
        description=f"Removed failed import of '{name}' from '{module}'",
        fix_type="replace",
        target=f"from {module} import {name}",
        replacement=f"# Removed: from {module} import {name}",
    )


def _fix_sys_path(match: re.Match, stderr: str) -> Optional[FixSuggestion]:
    """Fix missing project modules by adding sys.path."""
    return FixSuggestion(
        pattern_name="sys_path_missing",
        description="Added sys.path insertion for project root",
        fix_type="insert",
        replacement="import sys\nimport os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))",
    )
