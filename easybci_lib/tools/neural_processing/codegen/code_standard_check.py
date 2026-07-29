"""Static lint for LLM-emitted pipeline.py.

Walks the AST of a generated ``pipeline.py`` and returns every violation
of the rules in ``easybci_lib/skills/bci/CODE_STANDARD.md``.  Returns an
empty list iff the file is conformant.

The checker is intentionally narrow:

- No third-party deps (only stdlib `ast` + `re`); cheap to run at codegen.
- AST-only — does not import the module under test, so a malformed
  pipeline won't kill the parent process.
- Each violation has ``rule`` / ``line`` / ``message`` / ``blocking`` so
  the agent-visible structured error stays actionable.

Usage::

    from easybci_lib.tools.neural_processing.codegen.code_standard_check import (
        check_pipeline_code_standard,
    )
    violations = check_pipeline_code_standard(Path("code/pipeline.py"))
    if violations:
        # surface to agent for repair
        ...
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# Configuration — rules are versioned via the banner; bump in CODE_STANDARD.md
# and here together.
# --------------------------------------------------------------------------

CODE_STANDARD_VERSION = "1.1.0"

REQUIRED_DOCSTRING_FIELDS: tuple[str, ...] = (
    "Parameters",
    "Returns",
    "Raises",
    "Modality coverage",
    "References",
)

FORBIDDEN_NETWORK_MODULES: frozenset[str] = frozenset({
    "requests", "urllib", "urllib.request", "urllib.parse",
    "httpx", "aiohttp", "socket", "ftplib", "smtplib",
})

# Rule 15 — generated scripts must NOT import from the easybci_* codebase.
# The mini-repo under <work_dir>/code/ is meant to be SHAREABLE and runnable
# without easybci installed; importing easybci_lib / easybci_agent / etc.
# defeats that. Operators may reference easybci as documentation, but the
# generated script must inline mne/scipy/numpy/sklearn directly.
FORBIDDEN_EASYBCI_PREFIXES: tuple[str, ...] = (
    "easybci_lib",
    "easybci_agent",
    "easybci_cli",
    "easybci_web",
    "easybcidata_agent",
    "services.gateway",
    "services.plugins",
    "services.providers",
    "run_agent",
)

FORBIDDEN_FILE_IO_FUNCS: frozenset[str] = frozenset({
    "open", "write_text", "write_bytes",
    "save", "savez", "savez_compressed",
    "dump", "dumps",
})

ALLOWED_FLOAT_DTYPES: frozenset[str] = frozenset({"float32", "float64"})
FORBIDDEN_DTYPES_RE = re.compile(
    r"\b(int8|uint8|int16|uint16|float16)\b"
)

# Operators allowed to do file I/O when they declare it explicitly.
_FILE_IO_ALLOW_RE = re.compile(r"easybci-allow:\s*file-io", re.IGNORECASE)
_RULE_ALLOW_RE = re.compile(r"easybci-allow:\s*([\w-]+)", re.IGNORECASE)


# --------------------------------------------------------------------------
# Public surface
# --------------------------------------------------------------------------


def check_pipeline_code_standard(
    file_path: Path,
    *,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return a list of code-standard violations for ``file_path``.

    Parameters
    ----------
    file_path : Path
        Pipeline script (typically ``code/pipeline.py``) to lint.
    source : Optional[str]
        Override the file contents for testing.  When ``None`` the lint
        reads ``file_path`` from disk.

    Returns
    -------
    list of dict
        Each dict has keys ``rule`` (str), ``line`` (int), ``message`` (str),
        ``blocking`` (bool).  ``blocking=True`` violations stop codegen;
        ``blocking=False`` warnings flow through as advisory issues.
    """
    if source is None:
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return [{
                "rule": "io",
                "line": 0,
                "message": f"cannot read {file_path}: {exc}",
                "blocking": True,
            }]

    violations: List[Dict[str, Any]] = []

    # Rule banner — generated run.py / pipeline.py must declare the standard.
    if "EASYBCI_CODE_STANDARD" not in source:
        violations.append({
            "rule": "banner",
            "line": 1,
            "message": (
                f"missing 'EASYBCI_CODE_STANDARD: {CODE_STANDARD_VERSION}' banner "
                "in the module docstring; codegen always emits one."
            ),
            "blocking": False,  # warning — pipelines hand-edited by users may drop it
        })

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        return [{
            "rule": "syntax",
            "line": exc.lineno or 0,
            "message": f"syntax error: {exc.msg}",
            "blocking": True,
        }]

    # Rule 11 — forbidden imports (network).
    violations.extend(_check_imports(tree))

    # Rules 1–13 per operator function.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("operator_"):
                violations.extend(_check_operator_function(node, source))

    # Rule 13 — forbidden dtype mentions anywhere in the source.
    for match in FORBIDDEN_DTYPES_RE.finditer(source):
        line = source[: match.start()].count("\n") + 1
        violations.append({
            "rule": "dtype",
            "line": line,
            "message": (
                f"forbidden dtype {match.group()!r}; operators must keep "
                "data as float32 or float64."
            ),
            "blocking": True,
        })

    # Rule 6 — bare np.random.* outside seeded RNG.
    violations.extend(_check_bare_random(tree))

    return violations


# --------------------------------------------------------------------------
# Per-rule helpers
# --------------------------------------------------------------------------


def _check_imports(tree: ast.AST) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_NETWORK_MODULES or alias.name in FORBIDDEN_NETWORK_MODULES:
                    out.append({
                        "rule": "no-network",
                        "line": node.lineno,
                        "message": (
                            f"operator bodies must not import network module "
                            f"{alias.name!r}; move network calls to a research tool."
                        ),
                        "blocking": True,
                    })
                if _is_forbidden_easybci_import(alias.name):
                    out.append({
                        "rule": "no-easybci-import",
                        "line": node.lineno,
                        "message": (
                            f"generated scripts must not import {alias.name!r}; "
                            "the mini-repo must run without easybci installed. "
                            "Inline the logic using mne/scipy/numpy/sklearn instead. "
                            "See CODE_STANDARD.md Rule 15."
                        ),
                        "blocking": True,
                    })
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            if top in FORBIDDEN_NETWORK_MODULES or mod in FORBIDDEN_NETWORK_MODULES:
                out.append({
                    "rule": "no-network",
                    "line": node.lineno,
                    "message": (
                        f"operator bodies must not import from network module "
                        f"{mod!r}."
                    ),
                    "blocking": True,
                })
            if _is_forbidden_easybci_import(mod):
                out.append({
                    "rule": "no-easybci-import",
                    "line": node.lineno,
                    "message": (
                        f"generated scripts must not import from {mod!r}; "
                        "the mini-repo must run without easybci installed. "
                        "Inline the logic using mne/scipy/numpy/sklearn instead. "
                        "See CODE_STANDARD.md Rule 15."
                    ),
                    "blocking": True,
                })
    return out


def _is_forbidden_easybci_import(module: str) -> bool:
    """True iff ``module`` (or its dotted prefix) is a forbidden easybci import.

    Matches ``easybci_lib``, ``easybci_lib.tools.foo``, ``easybci_agent`` etc.
    Submodules of ``services`` (gateway/plugins/providers) also count — those
    only exist inside the easybci repo.
    """
    if not module:
        return False
    parts = module.split(".")
    if parts[0] in {
        "easybci_lib", "easybci_agent", "easybci_cli", "easybci_web",
        "easybcidata_agent", "run_agent",
    }:
        return True
    # services.{gateway,plugins,providers}
    if parts[0] == "services" and len(parts) >= 2 and parts[1] in {
        "gateway", "plugins", "providers",
    }:
        return True
    return False


def _check_operator_function(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    name = func.name
    allow_rules = _collect_allow_rules(func, source)

    # Rule 1 — first positional arg named ``data_dict``.
    args = func.args
    if not args.args or args.args[0].arg != "data_dict":
        out.append({
            "rule": "signature",
            "line": func.lineno,
            "message": (
                f"{name!r}: first positional arg must be 'data_dict' "
                f"(got {args.args[0].arg if args.args else 'none'})."
            ),
            "blocking": True,
        })

    # All other args must be keyword-only.
    if len(args.args) > 1:
        # Tail positional args (beyond data_dict) violate the rule.
        for extra in args.args[1:]:
            out.append({
                "rule": "signature",
                "line": func.lineno,
                "message": (
                    f"{name!r}: parameter {extra.arg!r} must be keyword-only "
                    "(move it after a bare ``*``)."
                ),
                "blocking": True,
            })

    # Rule 2 — type annotations.
    if "type-annotation" not in allow_rules:
        if args.args and args.args[0].annotation is None:
            out.append({
                "rule": "type-annotation",
                "line": func.lineno,
                "message": f"{name!r}: 'data_dict' lacks a type annotation.",
                "blocking": True,
            })
        for kw in args.kwonlyargs:
            if kw.annotation is None:
                out.append({
                    "rule": "type-annotation",
                    "line": func.lineno,
                    "message": (
                        f"{name!r}: keyword arg {kw.arg!r} lacks a type annotation."
                    ),
                    "blocking": True,
                })
        if func.returns is None:
            out.append({
                "rule": "type-annotation",
                "line": func.lineno,
                "message": f"{name!r}: return type annotation missing.",
                "blocking": True,
            })

    # Rule 3 — docstring fields.
    docstring = ast.get_docstring(func) or ""
    if "docstring" not in allow_rules:
        missing = [field for field in REQUIRED_DOCSTRING_FIELDS if field not in docstring]
        if missing:
            out.append({
                "rule": "docstring",
                "line": func.lineno,
                "message": (
                    f"{name!r}: docstring missing required field(s) "
                    f"{missing!r}; see CODE_STANDARD.md Rule 3."
                ),
                "blocking": False,  # warning — agent may repair after lint
            })

    # Rule 5 — no in-place mutation of data_dict["data"].
    if "immutability" not in allow_rules:
        out.extend(_check_inplace_data_dict(func, name))

    # Rule 8 — only EasyBCIOperatorError raised explicitly.
    if "error-class" not in allow_rules:
        out.extend(_check_raised_exceptions(func, name))

    # Rule 12 — no file I/O calls in operator body.
    if "file-io" not in allow_rules:
        out.extend(_check_file_io(func, name))

    return out


def _collect_allow_rules(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
) -> frozenset[str]:
    """Read `# easybci-allow: <rule>` markers inside the function body."""
    rules: set[str] = set()
    if func.body:
        body_start = func.body[0].lineno
        body_end = func.end_lineno or body_start
        lines = source.splitlines()
        for line in lines[body_start - 1 : body_end]:
            for m in _RULE_ALLOW_RE.finditer(line):
                rules.add(m.group(1))
    return frozenset(rules)


def _check_inplace_data_dict(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> List[Dict[str, Any]]:
    """Flag ``data_dict["data"][:]`` / ``data_dict["data"][...]`` writes."""
    out: List[Dict[str, Any]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not isinstance(tgt, ast.Subscript):
                continue
            base = tgt.value
            # data_dict["data"][slice] = ...
            if (
                isinstance(base, ast.Subscript)
                and isinstance(base.value, ast.Name)
                and base.value.id == "data_dict"
            ):
                out.append({
                    "rule": "immutability",
                    "line": node.lineno,
                    "message": (
                        f"{name!r}: in-place mutation of data_dict['data'] "
                        "forbidden; return a new dict instead."
                    ),
                    "blocking": True,
                })
    return out


def _check_raised_exceptions(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> List[Dict[str, Any]]:
    """Operators must only raise ``EasyBCIOperatorError`` explicitly.

    Re-raising via bare ``raise`` (no exc) inside ``except`` is allowed —
    that's the canonical chaining pattern.
    """
    out: List[Dict[str, Any]] = []
    allowed = {"EasyBCIOperatorError"}
    for node in ast.walk(func):
        if not isinstance(node, ast.Raise):
            continue
        if node.exc is None:
            continue  # bare `raise` is fine
        exc = node.exc
        # `raise SomeClass(...)` or `raise SomeClass`.
        if isinstance(exc, ast.Call):
            exc = exc.func
        if isinstance(exc, ast.Name) and exc.id not in allowed:
            out.append({
                "rule": "error-class",
                "line": node.lineno,
                "message": (
                    f"{name!r}: only EasyBCIOperatorError may be raised; "
                    f"got {exc.id!r}."
                ),
                "blocking": True,
            })
        elif isinstance(exc, ast.Attribute):
            if exc.attr not in allowed:
                out.append({
                    "rule": "error-class",
                    "line": node.lineno,
                    "message": (
                        f"{name!r}: only EasyBCIOperatorError may be raised; "
                        f"got attribute {exc.attr!r}."
                    ),
                    "blocking": True,
                })
    return out


def _check_file_io(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> List[Dict[str, Any]]:
    """Reject ``open(...)`` / ``Path.write_*`` / ``np.save`` calls."""
    out: List[Dict[str, Any]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Attribute):
            if callee.attr in FORBIDDEN_FILE_IO_FUNCS:
                out.append({
                    "rule": "no-file-io",
                    "line": node.lineno,
                    "message": (
                        f"{name!r}: file I/O call {callee.attr!r} forbidden "
                        "in operator body; move to run.py / qc.py or add "
                        "`# easybci-allow: file-io` with a reason."
                    ),
                    "blocking": True,
                })
        elif isinstance(callee, ast.Name) and callee.id == "open":
            out.append({
                "rule": "no-file-io",
                "line": node.lineno,
                "message": (
                    f"{name!r}: bare open() forbidden in operator body."
                ),
                "blocking": True,
            })
    return out


def _check_bare_random(tree: ast.AST) -> List[Dict[str, Any]]:
    """Reject ``np.random.rand / randn / random`` etc. without explicit seed.

    The seeded path is ``np.random.default_rng(...)`` /
    ``np.random.RandomState(...)``; calls on those instances are fine
    (they sit on a Name, not the ``np.random`` module).
    """
    out: List[Dict[str, Any]] = []
    bare_funcs = {"rand", "randn", "random", "randint", "uniform", "normal"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if (
            isinstance(callee, ast.Attribute)
            and isinstance(callee.value, ast.Attribute)
            and isinstance(callee.value.value, ast.Name)
            and callee.value.value.id in {"np", "numpy"}
            and callee.value.attr == "random"
            and callee.attr in bare_funcs
        ):
            out.append({
                "rule": "rng-seed",
                "line": node.lineno,
                "message": (
                    f"bare np.random.{callee.attr}() forbidden; use "
                    "np.random.default_rng(int(os.environ['EASYBCI_SEED']))."
                ),
                "blocking": True,
            })
    return out


# --------------------------------------------------------------------------
# Structured error helper for codegen
# --------------------------------------------------------------------------


def violations_to_agent_error(
    violations: Sequence[Dict[str, Any]],
    file_path: Path,
) -> Dict[str, Any]:
    """Render the violation list as the structured-error dict shape that
    ``codegen/script_runner`` already uses for traceback surfaces.

    Returns
    -------
    dict
        Keys: ``error_type`` ("code_standard_violation"), ``file``,
        ``violations``, ``hint``, ``suggestion_kind``.
    """
    return {
        "error_type": "code_standard_violation",
        "file": str(file_path),
        "violations": [
            {
                "rule": v["rule"],
                "line": v["line"],
                "message": v["message"],
                "blocking": v["blocking"],
            }
            for v in violations
        ],
        "hint": (
            "Edit the offending operator function to follow "
            "easybci_lib/skills/bci/CODE_STANDARD.md, then re-invoke "
            "generate_code with the same args. Do NOT add "
            "`# easybci-allow:` markers unless the rule explicitly "
            "permits an exception."
        ),
        "suggestion_kind": "code_standard",
    }


def has_blocking_violations(violations: Sequence[Dict[str, Any]]) -> bool:
    return any(v.get("blocking", True) for v in violations)


# --------------------------------------------------------------------------
# Routing-safety check — used by the dispatcher after codegen writes the
# script files. Catches the regression where pipeline.py / qc.py / build_ai_ready.py
# derive ``(subject_id, session_id)`` from ``Path(raw).stem`` instead of the
# routing table. The check is purely textual; it doesn't import the script.
# --------------------------------------------------------------------------

_ROUTING_SAFETY_BANS: tuple[tuple[str, str], ...] = (
    # (regex pattern, human-readable rule name)
    (r"_find_preprocessed\s*\(", "stem-glob fallback _find_preprocessed()"),
    (r"sub_id\s*=\s*stem\b", "sub_id assigned from raw stem"),
    (r"subject_id\s*=\s*Path\([^)]*\)\.stem", "subject_id derived from raw stem"),
    (r"sub_id\s*=\s*Path\([^)]*\)\.stem", "sub_id derived from raw stem"),
)


def run_routing_safety_check(code_dir: Path) -> List[Dict[str, Any]]:
    """Static check that generated scripts route via the routing table only.

    Scans ``pipeline.py``, ``qc.py``, ``build_ai_ready.py`` (each is optional)
    for textual patterns that indicate stem-based ``(subject_id, session_id)``
    derivation. The routing table is the single source of truth — any direct
    stem→sub_id assignment in generated scripts is a regression.

    Returns a list of violation dicts (empty if all checks pass). Each dict
    carries ``script`` / ``rule`` / ``line`` / ``snippet``.
    """
    out: List[Dict[str, Any]] = []
    for name in ("pipeline.py", "qc.py", "build_ai_ready.py", "vis.py"):
        p = Path(code_dir) / name
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            out.append({
                "script": name,
                "rule": "unreadable",
                "line": 0,
                "snippet": str(exc),
            })
            continue
        for pat, rule in _ROUTING_SAFETY_BANS:
            for m in re.finditer(pat, text):
                # Compute 1-indexed line number
                line = text.count("\n", 0, m.start()) + 1
                snippet = text.splitlines()[line - 1] if line - 1 < len(text.splitlines()) else ""
                out.append({
                    "script": name,
                    "rule": rule,
                    "line": line,
                    "snippet": snippet.strip(),
                })
    return out

