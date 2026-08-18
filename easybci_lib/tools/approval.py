"""Dangerous command approval -- detection, prompting, and per-session state.

This module is the single source of truth for the dangerous command system:
- Pattern detection (DANGEROUS_PATTERNS, detect_dangerous_command)
- Per-session approval state (thread-safe, keyed by session_key)
- Approval prompting (CLI interactive + gateway async)
- Smart approval via auxiliary LLM (auto-approve low-risk commands)
- Permanent allowlist persistence (config.yaml)
"""

import contextvars
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from typing import Callable, Optional
from easybci_cli.config import cfg_get

from easybci_lib.utils import is_truthy_value

logger = logging.getLogger(__name__)

# Per-thread/per-task gateway session identity.
# Gateway runs agent turns concurrently in executor threads, so reading a
# process-global env var for session identity is racy. Keep env fallback for
# legacy single-threaded callers, but prefer the context-local value when set.
_approval_session_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_session_key",
    default="",
)


def _fire_approval_hook(hook_name: str, **kwargs) -> None:
    """Invoke a plugin lifecycle hook for the approval system.

    Lazy-imports the plugin manager to avoid circular imports (approval.py is
    imported very early, long before plugins are discovered). Never raises --
    plugin errors are logged and swallowed.

    Only fires for the two approval-specific hooks in VALID_HOOKS:
    pre_approval_request, post_approval_response.
    """
    try:
        from easybci_cli.plugins import invoke_hook
    except Exception:
        # Plugin system not available in this execution context
        # (e.g. bare tool-only imports, minimal test environments).
        return
    try:
        invoke_hook(hook_name, **kwargs)
    except Exception as exc:
        # invoke_hook() already swallows per-callback errors, so reaching here
        # means the dispatch layer itself failed. Log and move on -- approval
        # flow is safety-critical, plugin observability is not.
        logger.debug("Approval hook %s dispatch failed: %s", hook_name, exc)



def set_current_session_key(session_key: str) -> contextvars.Token[str]:
    """Bind the active approval session key to the current context."""
    return _approval_session_key.set(session_key or "")


def reset_current_session_key(token: contextvars.Token[str]) -> None:
    """Restore the prior approval session key context."""
    _approval_session_key.reset(token)


def get_current_session_key(default: str = "default") -> str:
    """Return the active session key, preferring context-local state.

    Resolution order:
    1. approval-specific contextvars (set by gateway before agent.run)
    2. session_context contextvars (set by _set_session_env)
    3. os.environ fallback (CLI, tests)
    """
    session_key = _approval_session_key.get()
    if session_key:
        return session_key
    from services.gateway.session_context import get_session_env
    return get_session_env("EASYBCI_SESSION_KEY", default)


def _get_session_platform() -> str:
    """Return the current gateway platform from contextvars/env fallback."""
    try:
        from services.gateway.session_context import get_session_env

        return get_session_env("EASYBCI_SESSION_PLATFORM", "") or ""
    except Exception:
        return os.getenv("EASYBCI_SESSION_PLATFORM", "") or ""


def _is_gateway_approval_context() -> bool:
    """True when this call is inside a gateway/API session.

    Legacy gateway integrations set EASYBCI_GATEWAY_SESSION in process env.
    Newer concurrent gateway paths bind EASYBCI_SESSION_PLATFORM via
    contextvars so approval mode does not depend on process-global flags.
    """
    if os.getenv("EASYBCI_GATEWAY_SESSION"):
        return True
    return bool(_get_session_platform())

# Sensitive write targets that should trigger approval even when referenced
# via shell expansions like $HOME or $EASYBCI_HOME.
_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|$)'
_EASYBCI_ENV_PATH = (
    r'(?:~\/\.easybci/|'
    r'(?:\$home|\$\{home\})/\.easybci/|'
    r'(?:\$easybci_home|\$\{easybci_home\})/)'
    r'\.env\b'
)
_PROJECT_ENV_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*\.env(?:\.[^/\s"\'`]+)*)'
_PROJECT_CONFIG_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*config\.yaml)'
_SHELL_RC_FILES = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:bashrc|zshrc|profile|bash_profile|zprofile)\b'
)
_CREDENTIAL_FILES = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:netrc|pgpass|npmrc|pypirc)\b'
)
_SENSITIVE_WRITE_TARGET = (
    r'(?:/etc/|/dev/sd|'
    rf'{_SSH_SENSITIVE_PATH}|'
    rf'{_EASYBCI_ENV_PATH}|'
    rf'{_SHELL_RC_FILES}|'
    rf'{_CREDENTIAL_FILES})'
)
_PROJECT_SENSITIVE_WRITE_TARGET = rf'(?:{_PROJECT_ENV_PATH}|{_PROJECT_CONFIG_PATH})'
_COMMAND_TAIL = r'(?:\s*(?:&&|\|\||;).*)?$'

# =========================================================================
# Hardline (unconditional) blocklist
# =========================================================================
#
# Commands so catastrophic they should NEVER run via the agent, regardless
# of --yolo, /yolo, approvals.mode=off.  This is a
# floor below yolo: opting into yolo is the user trusting the agent with
# their files and services, not trusting it to wipe the disk or power the
# box off.
#
# Hardline only applies to environments that can actually damage the host
# (local, ssh).  Containerized backends (docker,
# singularity, modal, daytona) already bypass the dangerous-command layer
# because nothing they do can touch the host, so we leave that behavior
# alone.
#
# The list is deliberately tiny — only things with no recovery path:
# filesystem destruction rooted at /, raw block device overwrites, kernel
# shutdown/reboot, and denial-of-service commands that take the host down.
# Recoverable-but-costly operations (git reset --hard, rm -rf /tmp/x,
# chmod -R 777, curl|sh) stay in DANGEROUS_PATTERNS where yolo can pass
# them through — that's what yolo is for.
#
# Inspired by Mercury Agent's permission-hardened blocklist
# (https://github.com/cosmicstack-labs/mercury-agent).

# Regex fragment matching the *start* of a command (i.e. positions where
# a shell would begin parsing a new command).  Used by shutdown/reboot
# patterns so they don't fire on "echo reboot" or "grep 'shutdown' log".
# Matches: start of string, after command separators (; && || | newline),
# after subshell openers ( `$(` or backtick ), optionally consuming
# leading wrapper commands (sudo, env VAR=VAL, exec, nohup, setsid).
_CMDPOS = (
    r'(?:^|[;&|\n`]|\$\()'         # start position
    r'\s*'                          # optional whitespace
    r'(?:sudo\s+(?:-[^\s]+\s+)*)?'  # optional sudo with flags
    r'(?:env\s+(?:\w+=\S*\s+)*)?'   # optional env with VAR=VAL pairs
    r'(?:(?:exec|nohup|setsid|time)\s+)*'  # optional wrapper commands
    r'\s*'
)

HARDLINE_PATTERNS = [
    # rm recursive targeting the root filesystem or protected roots
    (r'\brm\s+(-[^\s]*\s+)*(/|/\*|/ \*)(\s|$)', "recursive delete of root filesystem"),
    (r'\brm\s+(-[^\s]*\s+)*(/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*)(\s|$)', "recursive delete of system directory"),
    (r'\brm\s+(-[^\s]*\s+)*(~|\$HOME)(/?|/\*)?(\s|$)', "recursive delete of home directory"),
    # Filesystem format
    (r'\bmkfs(\.[a-z0-9]+)?\b', "format filesystem (mkfs)"),
    # Raw block device overwrites (dd + redirection)
    (r'\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*', "dd to raw block device"),
    (r'>\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b', "redirect to raw block device"),
    # Fork bomb (classic shell form)
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Kill every process on the system
    (r'\bkill\s+(-[^\s]+\s+)*-1\b', "kill all processes"),
    # System shutdown / reboot — anchor to command position (start of line,
    # after a command separator, or after sudo/env wrappers) so we don't
    # false-positive on "echo reboot" or "grep 'shutdown' logs".
    # _CMDPOS matches start-of-command positions.
    (_CMDPOS + r'(shutdown|reboot|halt|poweroff)\b', "system shutdown/reboot"),
    (_CMDPOS + r'init\s+[06]\b', "init 0/6 (shutdown/reboot)"),
    (_CMDPOS + r'systemctl\s+(poweroff|reboot|halt|kexec)\b', "systemctl poweroff/reboot"),
    (_CMDPOS + r'telinit\s+[06]\b', "telinit 0/6 (shutdown/reboot)"),
]

# Pre-compiled variant used by the hot-path matcher. Building these at module
# load eliminates the ~2.6 ms cold-cache re.compile fan-out on the first
# terminal() call per process (12 HARDLINE + 47 DANGEROUS patterns, each
# potentially evicted from Python's 512-entry ``re._cache`` by unrelated
# regex work elsewhere in the agent). DANGEROUS_PATTERNS_COMPILED is built
# at the end of this module after DANGEROUS_PATTERNS is defined.
_RE_FLAGS = re.IGNORECASE | re.DOTALL
HARDLINE_PATTERNS_COMPILED = [
    (re.compile(pattern, _RE_FLAGS), description)
    for pattern, description in HARDLINE_PATTERNS
]


# =========================================================================
# Sudo stdin guard — block password guessing via "sudo -S"
# =========================================================================
# When SUDO_PASSWORD is not configured, any explicit "sudo -S" in the
# command is the LLM piping a guessed password via stdin.  This is a
# brute-force attack vector: the model iterates through candidate
# passwords, inspects sudo's "Sorry, try again" output, and refines.
# Treat this as an unconditional block — there is never a legitimate
# reason for the agent to pipe passwords to sudo -S when no password
# has been configured.
_SUDO_STDIN_RE = re.compile(
    r'(?:^|[;&|`\n]|&&|\|\||\$\()\s*sudo\s+-S\b',
    re.IGNORECASE)


def _check_sudo_stdin_guard(command: str) -> tuple:
    """Detect ``sudo -S`` (stdin password) without configured SUDO_PASSWORD.

    When SUDO_PASSWORD is set, ``_transform_sudo_command`` injects ``-S``
    internally — that path is legitimate and handled elsewhere.  This guard
    only fires when SUDO_PASSWORD is *not* set, meaning the LLM explicitly
    wrote ``sudo -S`` to pipe a guessed password.

    Returns:
        (is_blocked: bool, description: str | None)
    """
    if "SUDO_PASSWORD" in os.environ:
        return (False, None)
    normalized = _normalize_command_for_detection(command).lower()
    if _SUDO_STDIN_RE.search(normalized):
        return (True, "sudo password guessing via stdin (sudo -S)")
    return (False, None)


def detect_hardline_command(command: str) -> tuple:
    """Check if a command matches the unconditional hardline blocklist.

    Returns:
        (is_hardline, description) or (False, None)
    """
    normalized = _normalize_command_for_detection(command).lower()
    for pattern_re, description in HARDLINE_PATTERNS_COMPILED:
        if pattern_re.search(normalized):
            return (True, description)
    return (False, None)


def _hardline_block_result(description: str) -> dict:
    """Build the standard block result for a hardline match."""
    return {
        "approved": False,
        "hardline": True,
        "message": (
            f"BLOCKED (hardline): {description}. "
            "This command is on the unconditional blocklist and cannot "
            "be executed via the agent — not even with --yolo, /yolo, "
            "approvals.mode=off. If you genuinely "
            "need to run it, run it yourself in a terminal outside the "
            "agent."
        ),
    }


def _sudo_stdin_block_result(description: str) -> dict:
    """Build the standard block result for sudo stdin guard."""
    return {
        "approved": False,
        "message": (
            f"BLOCKED: {description}. "
            "Do not pipe passwords to 'sudo -S' — this is a brute-force "
            "attack vector. Set SUDO_PASSWORD in your .env file if the "
            "agent needs passwordless sudo, or run the sudo command "
            "manually in your own terminal."
        ),
    }


# =========================================================================
# Source data immutability guard
# =========================================================================
# Detects shell commands that would modify or delete files registered as
# protected source data. Like HARDLINE, this is unconditional — yolo mode,
# approvals.mode=off, cannot bypass it.

_SOURCE_DATA_MUTATING_COMMANDS = re.compile(
    r'(?:^|[;&|\n`]|\$\()\s*'
    r'(?:sudo\s+(?:-[^\s]+\s+)*)?'
    r'(rm|mv|cp|truncate|shred|chmod|chown)\b',
    re.IGNORECASE,
)
_SOURCE_DATA_DD_PATTERN = re.compile(r'\bdd\b[^\n]*\bof=([^\s;|&]+)', re.IGNORECASE)
_SOURCE_DATA_REDIRECT_PATTERN = re.compile(r'>\s*([^\s;|&]+)')

# Redirect targets that discard output rather than write a file. `2>/dev/null`
# is the ubiquitous stderr-discard suffix on read-only commands (ls/find/cat …);
# treating its `>` as a "write" made the mutating-command fallback below
# false-block harmless reads that merely *mention* a protected dir. fd-dups like
# `2>&1` already fail the pattern (the `&` is excluded from the target class).
_DISCARD_REDIRECT_TARGETS = {"/dev/null"}


def _has_write_redirect_to_real_target(command: str) -> bool:
    """True if the command redirects output to a real filesystem path (not
    /dev/null). Used to decide whether a command that defeats structured
    parsing 'looks mutating'. A real target (incl. `2>/some/file`) still
    counts — only pure discards are exempt — so stderr-into-a-source-file is
    NOT let through."""
    for _m in _SOURCE_DATA_REDIRECT_PATTERN.finditer(command):
        _tgt = _m.group(1).strip("'\"")
        if _tgt and _tgt not in _DISCARD_REDIRECT_TARGETS:
            return True
    return False


def check_source_data_command(command: str) -> tuple:
    """Detect shell commands that would modify registered source data files
    or write inside a protected source directory.

    Returns (True, description) if blocked, or (False, None) if safe.
    """
    try:
        from easybci_agent.source_data_guard import (
            is_source_data,
            is_inside_protected_dir,
            get_protected_paths,
            get_protected_dirs,
        )
    except ImportError:
        return (False, None)

    protected_files = get_protected_paths()
    protected_dirs = get_protected_dirs()
    if not protected_files and not protected_dirs:
        return (False, None)

    normalized = command.strip()

    # Structured write-target extraction (covers rm/mv/cp/sed -i/tee/
    # mkdir/touch/chmod/chown/tar -x/unzip and > / >> redirects).
    try:
        from easybci_lib.tools._shell_token_parser import extract_write_targets
        targets = extract_write_targets(normalized)
    except Exception:
        targets = []

    for target in targets:
        if is_source_data(target):
            return (True, f"shell would modify source data file: {target}")
        if is_inside_protected_dir(target):
            return (True, f"shell would write inside protected source dir: {target}")

    # Substring fallback for shell features that defeat parsing (eval, $(...),
    # bash -c, complex pipelines). Per `extract_write_targets`'s own docstring,
    # this fallback only runs when the structured parser returned NO targets —
    # i.e. it could not statically see any write. If the parser DID extract
    # targets and the loop above didn't block, the command is well-understood
    # and the protected path appears only as a read source (e.g.
    # `cp /protected/x /tmp/y` or `cat /protected/x > /tmp/y`), which is
    # allowed — source data is read-only, not no-access.
    #
    # Path-boundary match: the protected path must end at `/`, end-of-string,
    # or a non-filename character. Otherwise sibling paths like
    # /foo/Neuropixel_preprocess_work_dir would falsely match /foo/Neuropixel.
    import re as _re
    write_verb_anywhere = _re.compile(
        r"\b(rm|mv|cp|truncate|shred|chmod|chown|tee|mkdir|touch|tar|unzip|sed)\b",
        _re.IGNORECASE,
    )
    _path_boundary = r"(?=/|$|[^A-Za-z0-9._\-])"
    if (not targets
            and (write_verb_anywhere.search(normalized)
                 or _has_write_redirect_to_real_target(normalized))):
        for d in protected_dirs:
            if _re.search(_re.escape(d) + _path_boundary, normalized):
                return (True, f"shell command references protected source dir: {d}")
        for f in protected_files:
            if _re.search(_re.escape(f) + _path_boundary, normalized):
                return (True, f"shell command references source data file: {f}")

    # Preserve legacy regex coverage for redirect / dd / mutating commands
    # targeting exact source files.
    for match in _SOURCE_DATA_REDIRECT_PATTERN.finditer(normalized):
        target = match.group(1).strip("'\"")
        if _path_matches_source(target, protected_files):
            return (True, f"redirect output to source data file: {target}")

    for match in _SOURCE_DATA_DD_PATTERN.finditer(normalized):
        target = match.group(1).strip("'\"")
        if _path_matches_source(target, protected_files):
            return (True, f"dd write to source data file: {target}")

    if _SOURCE_DATA_MUTATING_COMMANDS.search(normalized):
        for path in _extract_path_arguments(normalized):
            if _path_matches_source(path, protected_files):
                return (True, f"mutating command targets source data: {path}")

    return (False, None)


# =========================================================================
# Preprocess work_dir material protection (confirmable, not a hard block)
# =========================================================================
# The agent occasionally deletes or moves deliverable materials inside a
# *_preprocess_work_dir/ (plan/, code/, preprocessed_output/, pipeline_record
# .json, README.md, …). These are recoverable-but-costly: re-running the
# pipeline can regenerate them, but a silent `rm` mid-batch loses work the
# user cares about. Unlike source data (absolute block), these go through the
# normal approval prompt so the user can confirm. middle_process/ is scratch
# and always exempt — Step 14 cleanup and layout sweeps target it routinely.

_WORKDIR_MUTATING_VERBS = frozenset({"rm", "mv", "shred", "truncate"})
_WORKDIR_SUFFIX = "_preprocess_work_dir"


def _workdir_relpath(target: str) -> Optional[tuple]:
    """If *target* lives inside a ``*_preprocess_work_dir/``, return
    ``(work_dir_name, rel_posix)`` where rel is POSIX-normalised relative to
    the work_dir root (empty string when target IS the work_dir). Otherwise
    None."""
    norm = target.replace("\\", "/").rstrip("/")
    parts = norm.split("/")
    for i, part in enumerate(parts):
        if part.endswith(_WORKDIR_SUFFIX):
            rel = "/".join(parts[i + 1:])
            return (part, rel)
    return None


def _is_middle_process_rel(rel: str) -> bool:
    """True when *rel* (a work_dir-relative POSIX path) is the middle_process
    scratch subtree, which is exempt from the material guard."""
    return rel == "middle_process" or rel.startswith("middle_process/")


# Shell connectors that split a command line into sequentially-run segments.
# Only ``&&`` guarantees the previous command SUCCEEDED before the next runs —
# so only ``&&`` lets a preceding ``cd <abs>`` guarantee the CWD of a following
# ``rm``. Every other connector (``;`` ``&`` ``|`` ``||`` newline) runs the next
# command REGARDLESS of whether the cd succeeded: a failed cd then misdirects a
# relative ``rm`` to whatever the real CWD is ("炸"). Those must be confirmed.
_SEGMENT_SPLIT_RE = None  # lazily compiled in _split_shell_segments


def _split_shell_segments(command: str):
    """Split *command* into ``[(tokens, connector_before), ...]``.

    ``connector_before`` is the connector token that preceded this segment
    (``""`` for the first segment, else one of ``&&`` ``||`` ``;`` ``|`` ``&``
    ``\\n``). Segments that fail to tokenise are skipped. Best-effort: a
    connector char inside a quoted path may mis-split, but that errs toward
    flagging (fail-safe), which is the intended direction here.
    """
    import re as _re
    global _SEGMENT_SPLIT_RE
    if _SEGMENT_SPLIT_RE is None:
        # Order matters: match two-char connectors before their one-char prefix.
        _SEGMENT_SPLIT_RE = _re.compile(r"(&&|\|\||;|\||&|\n)")

    import shlex as _shlex

    raw_parts = _SEGMENT_SPLIT_RE.split(command)
    segments = []
    connector = ""
    for idx, part in enumerate(raw_parts):
        if idx % 2 == 1:
            # This is a captured connector; it precedes the NEXT segment.
            connector = part
            continue
        seg = part.strip()
        if not seg:
            continue
        try:
            tokens = _shlex.split(seg, posix=True)
        except ValueError:
            tokens = seg.split()
        if tokens:
            segments.append((tokens, connector))
        connector = ""
    return segments


def _strip_cmd_quotes(tok: str) -> str:
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ("'", '"'):
        return tok[1:-1]
    return tok


def check_workdir_material_command(command: str) -> tuple:
    """Detect commands that delete/move deliverable materials inside a
    ``*_preprocess_work_dir/`` — including the CWD-dependent ``cd X && rm Y``
    form that static write-target extraction alone cannot resolve.

    Returns ``(True, description)`` when a deletion-class command (rm/mv/shred/
    truncate) touches a non-``middle_process/`` path inside a work_dir, OR when
    a ``cd`` into a work_dir is chained to such a command by a connector that
    does NOT short-circuit on cd-failure (``;`` ``&`` ``|`` ``||``) — because a
    failed cd would then misdirect the delete to an unknown directory. Returns
    ``(False, None)`` otherwise. This is a *confirmable* signal (approval
    prompt), NOT an unconditional block.

    ``middle_process/`` (and its subtree) is exempt ONLY when the target can be
    proven to land there — an absolute middle_process path, or a relative one
    under a ``cd <abs>/middle_process &&`` that guarantees the CWD. A relative
    middle_process delete reached through a non-``&&`` connector is still
    flagged, since a failed cd makes even that dangerous.
    """
    if not command or not command.strip():
        return (False, None)

    # Fast pre-filter: a deletion-class verb must appear somewhere. A benign
    # `cp x work_dir/plan/y` (which only *adds* a file) never prompts.
    import re as _re

    verb_re = _re.compile(
        r"(?:^|[;&|\n`]|\$\()\s*(?:sudo\s+(?:-[^\s]+\s+)*)?(rm|mv|shred|truncate)\b",
        _re.IGNORECASE,
    )
    if not verb_re.search(command):
        return (False, None)

    segments = _split_shell_segments(command)

    # CWD guaranteed by a preceding `cd <abs> &&`. None when unknown.
    guaranteed_cwd: Optional[str] = None
    # A `cd` into a work_dir reached via a NON-`&&` connector: a subsequent
    # relative delete cannot be proven safe (failed cd misdirects it).
    cd_into_workdir_unsafe = False

    for tokens, connector_before in segments:
        # A connector other than `&&` before this segment breaks any CWD
        # guarantee carried from the previous `cd`.
        if connector_before and connector_before != "&&":
            guaranteed_cwd = None

        verb = tokens[0].lstrip("-").lower()
        args = tokens[1:]

        if verb == "cd":
            positional = [
                _strip_cmd_quotes(a) for a in args
                if a and not a.startswith("-")
            ]
            target = positional[0] if positional else ""
            if target and os.path.isabs(target):
                resolved = os.path.normpath(target)
            elif target and guaranteed_cwd:
                resolved = os.path.normpath(os.path.join(guaranteed_cwd, target))
            else:
                resolved = None  # relative cd with unknown base
            continue_unsafe = _workdir_relpath(resolved) is not None if resolved else False
            # The connector AFTER this cd is the connector_before of the NEXT
            # segment; we decide guarantee there. Here we only record intent.
            guaranteed_cwd = resolved
            # If this cd targets a work_dir and it will be followed by a
            # non-&& connector, later relative deletes are unsafe. We can't see
            # the trailing connector from here, so record the workdir-ness and
            # let the delete branch check whether a guarantee survived.
            if continue_unsafe:
                cd_into_workdir_unsafe = True
            continue

        if verb in _WORKDIR_MUTATING_VERBS:
            positional = [
                _strip_cmd_quotes(a) for a in args
                if a and not a.startswith("-")
            ]
            # chmod/chown-style leading mode arg does not apply to these verbs.
            for raw in positional:
                if os.path.isabs(raw):
                    resolved = os.path.normpath(raw)
                    hit = _workdir_relpath(resolved)
                    if hit is None:
                        continue
                    work_dir_name, rel = hit
                    if _is_middle_process_rel(rel):
                        continue
                    shown = rel if rel else work_dir_name
                    return (
                        True,
                        f"delete/move of preprocess work_dir material: {shown} "
                        f"(inside {work_dir_name})",
                    )
                else:
                    # Relative delete target.
                    if guaranteed_cwd is not None:
                        # CWD guaranteed by `cd <abs> &&`: resolve & match.
                        resolved = os.path.normpath(
                            os.path.join(guaranteed_cwd, raw)
                        )
                        hit = _workdir_relpath(resolved)
                        if hit is None:
                            continue
                        work_dir_name, rel = hit
                        if _is_middle_process_rel(rel):
                            continue
                        shown = rel if rel else work_dir_name
                        return (
                            True,
                            f"delete/move of preprocess work_dir material: "
                            f"{shown} (inside {work_dir_name})",
                        )
                    elif cd_into_workdir_unsafe:
                        # A `cd` into a work_dir was chained by a connector that
                        # does not short-circuit on cd-failure. If that cd fails
                        # this relative delete runs in the wrong directory — we
                        # cannot prove it is safe, so require confirmation.
                        return (
                            True,
                            "delete inside a preprocess work_dir via `cd … ; rm` "
                            "(unguarded cd — a failed cd would misdirect the "
                            "delete); confirm the target path",
                        )
                    # else: no work_dir context; leave generic `rm -rf *` to the
                    # recursive-delete DANGEROUS_PATTERN.

    return (False, None)


def _path_matches_source(path: str, protected: frozenset) -> bool:
    """Check if a path argument resolves to a protected source data file."""
    if not path or path.startswith("-"):
        return False
    try:
        resolved = os.path.realpath(os.path.expanduser(path))
    except (OSError, ValueError):
        resolved = path
    if resolved in protected:
        return True
    resolved_sep = resolved + os.sep
    for p in protected:
        if p.startswith(resolved_sep):
            return True
    return False


def _extract_path_arguments(command: str) -> list:
    """Extract likely file path arguments from a shell command.

    Simple heuristic: split on whitespace, keep tokens that look like paths
    (start with /, ./, ~/, or contain no special shell chars and have a / or .).
    Strips quotes.
    """
    paths = []
    parts = command.split()
    skip_next = False
    for i, part in enumerate(parts):
        if skip_next:
            skip_next = False
            continue
        if part.startswith("-"):
            if part in ("-f", "-r", "-rf", "-i", "-v", "-R", "--recursive",
                        "--force", "-n", "--no-clobber"):
                continue
            if "=" not in part and i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                skip_next = True
            continue
        if part in ("rm", "mv", "cp", "truncate", "shred", "chmod", "chown",
                    "sudo", "env", "exec", "nohup"):
            continue
        cleaned = part.strip("'\"")
        if cleaned and (cleaned.startswith("/") or cleaned.startswith("./")
                        or cleaned.startswith("~/") or "." in cleaned):
            paths.append(cleaned)
    return paths


def _source_data_block_result(description: str) -> dict:
    """Build the standard block result for source data violations."""
    return {
        "approved": False,
        "hardline": True,
        "message": (
            f"BLOCKED (source data immutability): {description}. "
            "Original research data files are unconditionally protected — "
            "they cannot be modified or deleted by the agent under any "
            "circumstances, not even with --yolo or user request. "
            "Source data is read-only; write outputs to a separate path."
        ),
    }


# =========================================================================
# Dangerous command patterns
# =========================================================================

DANGEROUS_PATTERNS = [
    (r'\brm\s+(-[^\s]*\s+)*/', "delete in root path"),
    (r'\brm\s+-[^\s]*r', "recursive delete"),
    (r'\brm\s+--recursive\b', "recursive delete (long flag)"),
    (r'\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b', "world/other-writable permissions"),
    (r'\bchmod\s+--recursive\b.*(777|666|o\+[rwx]*w|a\+[rwx]*w)', "recursive world/other-writable (long flag)"),
    (r'\bchown\s+(-[^\s]*)?R\s+root', "recursive chown to root"),
    (r'\bchown\s+--recursive\b.*root', "recursive chown to root (long flag)"),
    (r'\bmkfs\b', "format filesystem"),
    (r'\bdd\s+.*if=', "disk copy"),
    (r'>\s*/dev/sd', "write to block device"),
    (r'\bDROP\s+(TABLE|DATABASE)\b', "SQL DROP"),
    # Use [^\n]* instead of .* so DOTALL mode does not cause a WHERE clause on the
    # *next* line to satisfy the negative lookahead, silently allowing DELETE without WHERE.
    (r'\bDELETE\s+FROM\b(?![^\n]*\bWHERE\b)', "SQL DELETE without WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    (r'>\s*/etc/', "overwrite system config"),
    (r'\bsystemctl\s+(-[^\s]+\s+)*(stop|restart|disable|mask)\b', "stop/restart system service"),
    (r'\bkill\s+-9\s+-1\b', "kill all processes"),
    (r'\bpkill\s+-9\b', "force kill processes"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Any shell invocation via -c or combined flags like -lc, -ic, etc.
    (r'\b(bash|sh|zsh|ksh)\s+-[^\s]*c(\s+|$)', "shell command via -c/-lc flag"),
    (r'\b(python[23]?|perl|ruby|node)\s+-[ec]\s+', "script execution via -e/-c flag"),
    (r'\b(curl|wget)\b.*\|\s*(ba)?sh\b', "pipe remote content to shell"),
    (r'\b(bash|sh|zsh|ksh)\s+<\s*<?\s*\(\s*(curl|wget)\b', "execute remote script via process substitution"),
    (rf'\btee\b.*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via tee"),
    (rf'>>?\s*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via redirection"),
    (rf'\btee\b.*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config via tee"),
    (rf'>>?\s*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config via redirection"),
    (r'\bxargs\s+.*\brm\b', "xargs with rm"),
    (r'\bfind\b.*-exec\s+(/\S*/)?rm\b', "find -exec rm"),
    (r'\bfind\b.*-delete\b', "find -delete"),
    # Gateway lifecycle protection: prevent the agent from killing its own
    # gateway process.  These commands trigger a gateway restart/stop that
    # terminates all running agents mid-work.
    (r'\beasybci\s+gateway\s+(stop|restart)\b', "stop/restart easybci gateway (kills running agents)"),
    (r'\beasybci\s+update\b', "easybci update (restarts gateway, kills running agents)"),
    # Gateway protection: never start gateway outside systemd management
    (r'gateway\s+run\b.*(&\s*$|&\s*;|\bdisown\b|\bsetsid\b)', "start gateway outside systemd (use 'systemctl --user restart easybci-gateway')"),
    (r'\bnohup\b.*gateway\s+run\b', "start gateway outside systemd (use 'systemctl --user restart easybci-gateway')"),
    # Self-termination protection: prevent agent from killing its own process
    (r'\b(pkill|killall)\b.*\b(easybci|gateway|cli\.py)\b', "kill easybci/gateway process (self-termination)"),
    # Self-termination via kill + command substitution (pgrep/pidof).
    # The name-based pattern above catches `pkill easybci` but not
    # `kill -9 $(pgrep.*easybci)` because the substitution is opaque
    # to regex at detection time. Catch the structural pattern instead.
    (r'\bkill\b.*\$\(\s*pgrep\b', "kill process via pgrep expansion (self-termination)"),
    (r'\bkill\b.*`\s*pgrep\b', "kill process via backtick pgrep expansion (self-termination)"),
    # File copy/move/edit into sensitive system paths
    (r'\b(cp|mv|install)\b.*\s/etc/', "copy/move file into /etc/"),
    (rf'\b(cp|mv|install)\b.*\s["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config file"),
    (r'\bsed\s+-[^\s]*i.*\s/etc/', "in-place edit of system config"),
    (r'\bsed\s+--in-place\b.*\s/etc/', "in-place edit of system config (long flag)"),
    # Script execution via heredoc — bypasses the -e/-c flag patterns above.
    # `python3 << 'EOF'` feeds arbitrary code via stdin without -c/-e flags.
    (r'\b(python[23]?|perl|ruby|node)\s+<<', "script execution via heredoc"),
    # Git destructive operations that can lose uncommitted work or rewrite
    # shared history. Not captured by rm/chmod/etc patterns.
    (r'\bgit\s+reset\s+--hard\b', "git reset --hard (destroys uncommitted changes)"),
    (r'\bgit\s+push\b.*--force\b', "git force push (rewrites remote history)"),
    (r'\bgit\s+push\b.*-f\b', "git force push short flag (rewrites remote history)"),
    (r'\bgit\s+clean\s+-[^\s]*f', "git clean with force (deletes untracked files)"),
    (r'\bgit\s+branch\s+-D\b', "git branch force delete"),
    # Script execution after chmod +x — catches the two-step pattern where
    # a script is first made executable then immediately run. The script
    # content may contain dangerous commands that individual patterns miss.
    (r'\bchmod\s+\+x\b.*[;&|]+\s*\./', "chmod +x followed by immediate execution"),
    # Sudo with stdin / askpass / shell / list-privs flags. An LLM-driven
    # agent has no TTY, so sudo invocations that succeed without human
    # interaction are those reading the password from stdin (-S/--stdin)
    # or via an askpass helper (-A/--askpass). The shell-launch (-s) and
    # list-privileges (-a) flags are also gated since they are
    # privilege-relevant invocations the agent can chain after acquiring
    # the password (e.g. read SUDO_PASSWORD from .env -> sudo -S -s ->
    # root shell). Plain `sudo cmd` (no flag) is TTY-bound and excluded.
    # `_normalize_command_for_detection` lowercases input before pattern
    # matching, so case variants of S/s and A/a collapse — both forms
    # are gated below. Lazy `[^;|&\n]*?` allows flag arguments (e.g.
    # `sudo -u root -S whoami`) without spanning command separators.
    (r'\bsudo\b[^;|&\n]*?\s+(?:-s\b|--stdin\b|-a\b|--askpass\b)',
     "sudo with privilege flag (stdin/askpass/shell/list)"),
    # Combined short-flag form: -nS, -ns, -sa, -las — sudo flags packed
    # into a single -X token. Catches the same threat class.
    (r'\bsudo\b[^;|&\n]*?\s+-[a-z]*[sa][a-z]*\b',
     "sudo with combined-flag privilege escalation"),
]


# Pre-compiled variant (same rationale as HARDLINE_PATTERNS_COMPILED above).
DANGEROUS_PATTERNS_COMPILED = [
    (re.compile(pattern, _RE_FLAGS), description)
    for pattern, description in DANGEROUS_PATTERNS
]


def _legacy_pattern_key(pattern: str) -> str:
    """Reproduce the old regex-derived approval key for backwards compatibility."""
    return pattern.split(r'\b')[1] if r'\b' in pattern else pattern[:20]


_PATTERN_KEY_ALIASES: dict[str, set[str]] = {}
for _pattern, _description in DANGEROUS_PATTERNS:
    _legacy_key = _legacy_pattern_key(_pattern)
    _canonical_key = _description
    _PATTERN_KEY_ALIASES.setdefault(_canonical_key, set()).update({_canonical_key, _legacy_key})
    _PATTERN_KEY_ALIASES.setdefault(_legacy_key, set()).update({_legacy_key, _canonical_key})


def _approval_key_aliases(pattern_key: str) -> set[str]:
    """Return all approval keys that should match this pattern.

    New approvals use the human-readable description string, but older
    command_allowlist entries and session approvals may still contain the
    historical regex-derived key.
    """
    return _PATTERN_KEY_ALIASES.get(pattern_key, {pattern_key})


# =========================================================================
# Detection
# =========================================================================

def _normalize_command_for_detection(command: str) -> str:
    """Normalize a command string before dangerous-pattern matching.

    Strips ANSI escape sequences (full ECMA-48 via tools.ansi_strip),
    null bytes, and normalizes Unicode fullwidth characters so that
    obfuscation techniques cannot bypass the pattern-based detection.
    """
    from easybci_lib.tools.ansi_strip import strip_ansi

    # Strip all ANSI escape sequences (CSI, OSC, DCS, 8-bit C1, etc.)
    command = strip_ansi(command)
    # Strip null bytes
    command = command.replace('\x00', '')
    # Normalize Unicode (fullwidth Latin, halfwidth Katakana, etc.)
    command = unicodedata.normalize('NFKC', command)
    return command


def detect_dangerous_command(command: str) -> tuple:
    """Check if a command matches any dangerous patterns.

    Returns:
        (is_dangerous, pattern_key, description) or (False, None, None)
    """
    command_lower = _normalize_command_for_detection(command).lower()
    for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
        if pattern_re.search(command_lower):
            pattern_key = description
            return (True, pattern_key, description)
    return (False, None, None)


# =========================================================================
# Per-session approval state (thread-safe)
# =========================================================================

_lock = threading.Lock()
_pending: dict[str, dict] = {}
_session_approved: dict[str, set] = {}
_session_yolo: set[str] = set()
_permanent_approved: set = set()

# =========================================================================
# Blocking gateway approval (mirrors CLI's synchronous input() flow)
# =========================================================================
# Per-session QUEUE of pending approvals.  Multiple threads (parallel
# subagents, execute_code RPC handlers) can block concurrently — each gets
# its own threading.Event.  /approve resolves the oldest, /approve all
# resolves every pending approval in the session.


class _ApprovalEntry:
    """One pending dangerous-command approval inside a gateway session."""
    __slots__ = ("event", "data", "result")

    def __init__(self, data: dict):
        self.event = threading.Event()
        self.data = data          # command, description, pattern_keys, …
        self.result: Optional[str] = None  # "once"|"session"|"always"|"deny"


_gateway_queues: dict[str, list] = {}        # session_key → [_ApprovalEntry, …]
# session_key → callable(approval_data). Typed as Callable so the call site is
# not flagged; only callables are ever stored (see register_gateway_notify).
_gateway_notify_cbs: dict[str, Callable] = {}


def register_gateway_notify(session_key: str, cb) -> None:
    """Register a per-session callback for sending approval requests to the user.

    The callback signature is ``cb(approval_data: dict) -> None`` where
    *approval_data* contains ``command``, ``description``, and
    ``pattern_keys``.  The callback bridges sync→async (runs in the agent
    thread, must schedule the actual send on the event loop).
    """
    with _lock:
        _gateway_notify_cbs[session_key] = cb


def unregister_gateway_notify(session_key: str) -> None:
    """Unregister the per-session gateway approval callback.

    Signals ALL blocked threads for this session so they don't hang forever
    (e.g. when the agent run finishes or is interrupted).
    """
    with _lock:
        _gateway_notify_cbs.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        entry.event.set()


def resolve_gateway_approval(session_key: str, choice: str,
                             resolve_all: bool = False) -> int:
    """Called by the gateway's /approve or /deny handler to unblock
    waiting agent thread(s).

    When *resolve_all* is True every pending approval in the session is
    resolved at once (``/approve all``).  Otherwise only the oldest one
    is resolved (FIFO).

    Returns the number of approvals resolved (0 means nothing was pending).
    """
    with _lock:
        queue = _gateway_queues.get(session_key)
        if not queue:
            return 0
        if resolve_all:
            targets = list(queue)
            queue.clear()
        else:
            targets = [queue.pop(0)]
        if not queue:
            _gateway_queues.pop(session_key, None)

    for entry in targets:
        entry.result = choice
        entry.event.set()
    return len(targets)


def has_blocking_approval(session_key: str) -> bool:
    """Check if a session has one or more blocking gateway approvals waiting."""
    with _lock:
        return bool(_gateway_queues.get(session_key))


def submit_pending(session_key: str, approval: dict):
    """Store a pending approval request for a session."""
    with _lock:
        _pending[session_key] = approval


def approve_session(session_key: str, pattern_key: str):
    """Approve a pattern for this session only."""
    with _lock:
        _session_approved.setdefault(session_key, set()).add(pattern_key)


def enable_session_yolo(session_key: str) -> None:
    """Enable YOLO bypass for a single session key."""
    if not session_key:
        return
    with _lock:
        _session_yolo.add(session_key)


def disable_session_yolo(session_key: str) -> None:
    """Disable YOLO bypass for a single session key."""
    if not session_key:
        return
    with _lock:
        _session_yolo.discard(session_key)


def clear_session(session_key: str) -> None:
    """Remove all approval and yolo state for a given session."""
    if not session_key:
        return
    with _lock:
        _session_approved.pop(session_key, None)
        _session_yolo.discard(session_key)
        _pending.pop(session_key, None)
        entries = _gateway_queues.pop(session_key, [])
    for entry in entries:
        # Session-boundary cleanup should cancel any blocked approval waits
        # immediately so the old run can unwind instead of idling until timeout.
        entry.result = "deny"
        entry.event.set()


def is_session_yolo_enabled(session_key: str) -> bool:
    """Return True when YOLO bypass is enabled for a specific session."""
    if not session_key:
        return False
    with _lock:
        return session_key in _session_yolo


def is_current_session_yolo_enabled() -> bool:
    """Return True when the active approval session has YOLO bypass enabled."""
    return is_session_yolo_enabled(get_current_session_key(default=""))


def is_approved(session_key: str, pattern_key: str) -> bool:
    """Check if a pattern is approved (session-scoped or permanent).

    Accept both the current canonical key and the legacy regex-derived key so
    existing command_allowlist entries continue to work after key migrations.
    """
    aliases = _approval_key_aliases(pattern_key)
    with _lock:
        if any(alias in _permanent_approved for alias in aliases):
            return True
        session_approvals = _session_approved.get(session_key, set())
        return any(alias in session_approvals for alias in aliases)


def approve_permanent(pattern_key: str):
    """Add a pattern to the permanent allowlist."""
    with _lock:
        _permanent_approved.add(pattern_key)


def load_permanent(patterns: set):
    """Bulk-load permanent allowlist entries from config."""
    with _lock:
        _permanent_approved.update(patterns)



# =========================================================================
# Config persistence for permanent allowlist
# =========================================================================

def load_permanent_allowlist() -> set:
    """Load permanently allowed command patterns from config.

    Also syncs them into the approval module so is_approved() works for
    patterns added via 'always' in a previous session.
    """
    try:
        from easybci_cli.config import load_config
        config = load_config()
        patterns = set(config.get("command_allowlist", []) or [])
        if patterns:
            load_permanent(patterns)
        return patterns
    except Exception as e:
        logger.warning("Failed to load permanent allowlist: %s", e)
        return set()


def save_permanent_allowlist(patterns: set):
    """Save permanently allowed command patterns to config."""
    try:
        from easybci_cli.config import load_config, save_config
        config = load_config()
        config["command_allowlist"] = list(patterns)
        save_config(config)
    except Exception as e:
        logger.warning("Could not save allowlist: %s", e)


# =========================================================================
# Approval prompting + orchestration
# =========================================================================

def prompt_dangerous_approval(command: str, description: str,
                              timeout_seconds: int | None = None,
                              allow_permanent: bool = True,
                              approval_callback=None) -> str:
    """Prompt the user to approve a dangerous command (CLI only).

    Args:
        allow_permanent: When False, hide the [a]lways option (used when
            tirith warnings are present, since broad permanent allowlisting
            is inappropriate for content-level security findings).
        approval_callback: Optional callback registered by the CLI for
            prompt_toolkit integration. Signature:
            (command, description, *, allow_permanent=True) -> str.

    Returns: 'once', 'session', 'always', or 'deny'
    """
    if timeout_seconds is None:
        timeout_seconds = _get_approval_timeout()

    if approval_callback is not None:
        try:
            return approval_callback(command, description,
                                     allow_permanent=allow_permanent)
        except Exception as e:
            logger.error("Approval callback failed: %s", e, exc_info=True)
            return "deny"

    # Fail-closed guard: if prompt_toolkit owns the terminal (interactive
    # CLI session) and no approval callback is registered on this thread,
    # the input() fallback below would spawn a daemon thread whose read
    # can never see Enter -- the user's keystrokes go to prompt_toolkit,
    # not input(), producing an invisible 60s deadlock.
    # Deny fast and log loudly instead so the caller can surface a real
    # error to the agent. Any thread that needs interactive approval must
    # install a callback via tools.terminal_tool.set_approval_callback()
    # before reaching this point (see delegate_tool.py, run_agent.py
    # _execute_tool_calls_concurrent / _spawn_background_review for the
    # established pattern).
    try:
        from prompt_toolkit.application.current import get_app_or_none
        if get_app_or_none() is not None:
            logger.warning(
                "Dangerous-command approval requested on a thread with no "
                "approval callback while prompt_toolkit is active; denying "
                "to avoid stdin deadlock. command=%r description=%r",
                command, description,
            )
            return "deny"
    except Exception:
        # prompt_toolkit not installed, or detection failed -- fall through
        # to the legacy input() path (safe in non-TUI contexts: scripts,
        # tests, sshd, etc.).
        pass

    os.environ["EASYBCI_SPINNER_PAUSE"] = "1"
    try:
        # Resolve the active UI language once per prompt so we don't re-read
        # config/YAML inside the retry loop below.
        from easybci_agent.i18n import t
        while True:
            print()
            print(f"  {t('approval.dangerous_header', description=description)}")
            print(f"      {command}")
            print()
            if allow_permanent:
                print(t("approval.choose_long"))
            else:
                print(t("approval.choose_short"))
            print()
            sys.stdout.flush()

            result = {"choice": ""}

            def get_input():
                try:
                    prompt = t("approval.prompt_long") if allow_permanent else t("approval.prompt_short")
                    result["choice"] = input(prompt).strip().lower()
                except (EOFError, OSError):
                    result["choice"] = ""

            thread = threading.Thread(target=get_input, daemon=True)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                print("\n" + t("approval.timeout"))
                return "deny"

            choice = result["choice"]
            if choice in {'o', 'once'}:
                print(t("approval.allowed_once"))
                return "once"
            elif choice in {'s', 'session'}:
                print(t("approval.allowed_session"))
                return "session"
            elif choice in {'a', 'always'}:
                if not allow_permanent:
                    print(t("approval.allowed_session"))
                    return "session"
                print(t("approval.allowed_always"))
                return "always"
            else:
                print(t("approval.denied"))
                return "deny"

    except (EOFError, KeyboardInterrupt):
        print("\n" + t("approval.cancelled"))
        return "deny"
    finally:
        if "EASYBCI_SPINNER_PAUSE" in os.environ:
            del os.environ["EASYBCI_SPINNER_PAUSE"]
        print()
        sys.stdout.flush()


def _normalize_approval_mode(mode) -> str:
    """Normalize approval mode values loaded from YAML/config.

    YAML 1.1 treats bare words like `off` as booleans, so a config entry like
    `approvals:\n  mode: off` is parsed as False unless quoted. Treat that as the
    intended string mode instead of falling back to manual approvals.
    """
    if isinstance(mode, bool):
        return "off" if mode is False else "manual"
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        return normalized or "manual"
    return "manual"


def _get_approval_config() -> dict:
    """Read the approvals config block. Returns a dict with 'mode', 'timeout', etc."""
    try:
        from easybci_cli.config import load_config
        config = load_config()
        return config.get("approvals", {}) or {}
    except Exception as e:
        logger.warning("Failed to load approval config: %s", e)
        return {}


def _get_approval_mode() -> str:
    """Read the approval mode from config. Returns 'manual', 'smart', or 'off'."""
    mode = _get_approval_config().get("mode", "manual")
    return _normalize_approval_mode(mode)


def _get_approval_timeout() -> int:
    """Read the approval timeout from config. Defaults to 60 seconds."""
    try:
        return int(_get_approval_config().get("timeout", 60))
    except (ValueError, TypeError):
        return 60



def _smart_approve(command: str, description: str) -> str:
    """Use the auxiliary LLM to assess risk and decide approval.

    Returns 'approve' if the LLM determines the command is safe,
    'deny' if genuinely dangerous, or 'escalate' if uncertain.
    """
    try:
        from easybci_agent.auxiliary_client import call_llm

        prompt = f"""You are a security reviewer for an AI coding agent. A terminal command was flagged by pattern matching as potentially dangerous.

Command: {command}
Flagged reason: {description}

Assess the ACTUAL risk of this command. Many flagged commands are false positives — for example, `python -c "print('hello')"` is flagged as "script execution via -c flag" but is completely harmless.

Rules:
- APPROVE if the command is clearly safe (benign script execution, safe file operations, development tools, package installs, git operations, etc.)
- DENY if the command could genuinely damage the system (recursive delete of important paths, overwriting system files, fork bombs, wiping disks, dropping databases, etc.)
- ESCALATE if you're uncertain

Respond with exactly one word: APPROVE, DENY, or ESCALATE"""

        response = call_llm(
            task="approval",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096,
        )

        answer = (response.choices[0].message.content or "").strip().upper()

        if "APPROVE" in answer:
            return "approve"
        elif "DENY" in answer:
            return "deny"
        else:
            return "escalate"

    except Exception as e:
        logger.debug("Smart approvals: LLM call failed (%s), escalating", e)
        return "escalate"


def check_dangerous_command(command: str, env_type: str,
                            approval_callback=None) -> dict:
    """Check if a command is dangerous and handle approval.

    This is the main entry point called by terminal_tool before executing
    any command. It orchestrates detection, session checks, and prompting.

    Args:
        command: The shell command to check.
        env_type: Terminal backend type ('local', 'ssh', 'docker', etc.).
        approval_callback: Optional CLI callback for interactive prompts.

    Returns:
        {"approved": True/False, "message": str or None, ...}
    """
    if env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}:
        return {"approved": True, "message": None}

    # Hardline floor: commands with no recovery path (rm -rf /, mkfs, dd
    # to raw device, shutdown/reboot, fork bomb, kill -1) are blocked
    # unconditionally, BEFORE the yolo bypass.  Opting into yolo is
    # trusting the agent with your files and services, not trusting it
    # to wipe the disk or power the box off.
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc)

    # Source data immutability: block commands that target registered source
    # data files. This fires BEFORE yolo — source data protection is absolute.
    source_blocked, source_desc = check_source_data_command(command)
    if source_blocked:
        logger.warning("Source data block: %s (command: %s)", source_desc, command[:200])
        return _source_data_block_result(source_desc)

    # --yolo / session yolo / approvals.mode=off: bypass approval prompts.
    # Gateway /yolo is session-scoped; CLI --yolo remains process-scoped via
    # the env var for local use. HARDLINE + source-data checks above still
    # fire — those cannot be bypassed by any mode.
    if (
        is_truthy_value(os.getenv("EASYBCI_YOLO_MODE"))
        or is_current_session_yolo_enabled()
        or _get_approval_mode() == "off"
    ):
        return {"approved": True, "message": None}

    is_dangerous, pattern_key, description = detect_dangerous_command(command)
    if not is_dangerous:
        return {"approved": True, "message": None}

    session_key = get_current_session_key()
    if is_approved(session_key, pattern_key):
        return {"approved": True, "message": None}

    is_cli = os.getenv("EASYBCI_INTERACTIVE")
    is_gateway = _is_gateway_approval_context()

    if not is_cli and not is_gateway:
        return {"approved": True, "message": None}

    if is_gateway or os.getenv("EASYBCI_EXEC_ASK"):
        submit_pending(session_key, {
            "command": command,
            "pattern_key": pattern_key,
            "description": description,
        })
        return {
            "approved": False,
            "pattern_key": pattern_key,
            "status": "approval_required",
            "command": command,
            "description": description,
            "message": (
                f"⚠️ This command is potentially dangerous ({description}). "
                f"Asking the user for approval.\n\n**Command:**\n```\n{command}\n```"
            ),
        }

    choice = prompt_dangerous_approval(command, description,
                                       approval_callback=approval_callback)

    if choice == "deny":
        return {
            "approved": False,
            "message": f"BLOCKED: User denied this potentially dangerous command (matched '{description}' pattern). Do NOT retry this command - the user has explicitly rejected it.",
            "pattern_key": pattern_key,
            "description": description,
        }

    if choice == "session":
        approve_session(session_key, pattern_key)
    elif choice == "always":
        approve_session(session_key, pattern_key)
        approve_permanent(pattern_key)
        save_permanent_allowlist(_permanent_approved)

    return {"approved": True, "message": None}


# =========================================================================
# Combined pre-exec guard (tirith + dangerous command detection)
# =========================================================================

def _format_tirith_description(tirith_result: dict) -> str:
    """Build a human-readable description from tirith findings.

    Includes severity, title, and description for each finding so users
    can make an informed approval decision.
    """
    findings = tirith_result.get("findings") or []
    if not findings:
        summary = tirith_result.get("summary") or "security issue detected"
        return f"Security scan: {summary}"

    parts = []
    for f in findings:
        severity = f.get("severity", "")
        title = f.get("title", "")
        desc = f.get("description", "")
        if title and desc:
            parts.append(f"[{severity}] {title}: {desc}" if severity else f"{title}: {desc}")
        elif title:
            parts.append(f"[{severity}] {title}" if severity else title)
    if not parts:
        summary = tirith_result.get("summary") or "security issue detected"
        return f"Security scan: {summary}"

    return "Security scan — " + "; ".join(parts)


def check_all_command_guards(command: str, env_type: str,
                             approval_callback=None) -> dict:
    """Run all pre-exec security checks and return a single approval decision.

    Gathers findings from tirith and dangerous-command detection, then
    presents them as a single combined approval request. This prevents
    a gateway force=True replay from bypassing one check when only the
    other was shown to the user.
    """
    # Skip containers for both checks
    if env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}:
        return {"approved": True, "message": None}

    # Hardline floor: unconditional block for catastrophic commands
    # (rm -rf /, mkfs, dd to raw device, shutdown/reboot, fork bomb,
    # kill -1). Applies BEFORE yolo / mode=off so
    # no session-level setting can bypass it.
    is_hardline, hardline_desc = detect_hardline_command(command)
    if is_hardline:
        logger.warning("Hardline block: %s (command: %s)", hardline_desc, command[:200])
        return _hardline_block_result(hardline_desc)

    # Source data immutability: block commands that target registered source
    # data files. Fires BEFORE yolo — source data protection is absolute.
    source_blocked, source_desc = check_source_data_command(command)
    if source_blocked:
        logger.warning("Source data block: %s (command: %s)", source_desc, command[:200])
        return _source_data_block_result(source_desc)

    # == Sudo stdin guard ==
    # Like the hardline floor above, this is unconditional: there is never a
    # legitimate reason for the agent to pipe passwords to sudo -S when no
    # SUDO_PASSWORD has been configured.  This must fire BEFORE the yolo
    # check so even yolo/smart approval/mode=off cannot bypass it.
    is_sudo_guess, sudo_guess_desc = _check_sudo_stdin_guard(command)
    if is_sudo_guess:
        logger.warning("Sudo stdin guard block: %s (command: %s)",
                       sudo_guess_desc, command[:200])
        return _sudo_stdin_block_result(sudo_guess_desc)

    # --yolo or approvals.mode=off: bypass all approval prompts.
    # Gateway /yolo is session-scoped; CLI --yolo remains process-scoped.
    approval_mode = _get_approval_mode()
    if is_truthy_value(os.getenv("EASYBCI_YOLO_MODE")) or is_current_session_yolo_enabled() or approval_mode == "off":
        return {"approved": True, "message": None}

    is_cli = os.getenv("EASYBCI_INTERACTIVE")
    is_gateway = _is_gateway_approval_context()
    is_ask = os.getenv("EASYBCI_EXEC_ASK")

    # Preserve the existing non-interactive behavior: outside CLI/gateway/ask
    # flows, we do not block on approvals and we skip external guard work.
    if not is_cli and not is_gateway and not is_ask:
        return {"approved": True, "message": None}

    # --- Phase 1: Gather findings from both checks ---

    # Tirith check — wrapper guarantees no raise for expected failures.
    # Only catch ImportError (module not installed).
    tirith_result = {"action": "allow", "findings": [], "summary": ""}
    try:
        from easybci_lib.tools.tirith_security import check_command_security
        tirith_result = check_command_security(command)
    except ImportError:
        pass  # tirith module not installed — allow

    # Dangerous command check (detection only, no approval)
    is_dangerous, pattern_key, description = detect_dangerous_command(command)

    # --- Phase 2: Decide ---

    # Collect warnings that need approval
    warnings = []  # list of (pattern_key, description, is_tirith)

    session_key = get_current_session_key()

    # Tirith block/warn → approvable warning with rich findings.
    # Previously, tirith "block" was a hard block with no approval prompt.
    # Now both block and warn go through the approval flow so users can
    # inspect the explanation and approve if they understand the risk.
    if tirith_result["action"] in {"block", "warn"}:
        findings = tirith_result.get("findings") or []
        # findings come from parsing the tirith binary's JSON stdout, so an
        # element is *usually* a dict but a malformed run could yield a string
        # or other scalar. Guard the .get() so a bad finding degrades rule_id
        # to "unknown" instead of raising AttributeError mid-approval.
        first = findings[0] if findings else None
        rule_id = first.get("rule_id", "unknown") if isinstance(first, dict) else "unknown"
        tirith_key = f"tirith:{rule_id}"
        tirith_desc = _format_tirith_description(tirith_result)
        if not is_approved(session_key, tirith_key):
            warnings.append((tirith_key, tirith_desc, True))

    if is_dangerous:
        if not is_approved(session_key, pattern_key):
            warnings.append((pattern_key, description, False))

    # Preprocess work_dir material protection — deleting/moving deliverable
    # materials (plan/, code/, preprocessed_output/, …) is confirmable.
    # middle_process/ is exempt. This is an approvable warning, not a block.
    material_flagged, material_desc = check_workdir_material_command(command)
    if material_flagged:
        material_key = "workdir-material-delete"
        if not is_approved(session_key, material_key):
            warnings.append((material_key, material_desc, False))

    # Nothing to warn about
    if not warnings:
        return {"approved": True, "message": None}

    # --- Phase 2.5: Smart approval (auxiliary LLM risk assessment) ---
    # When approvals.mode=smart, ask the aux LLM before prompting the user.
    if approval_mode == "smart":
        combined_desc_for_llm = "; ".join(desc for _, desc, _ in warnings)
        verdict = _smart_approve(command, combined_desc_for_llm)
        if verdict == "approve":
            # Auto-approve and grant session-level approval for these patterns
            for key, _, _ in warnings:
                approve_session(session_key, key)
            logger.debug("Smart approval: auto-approved '%s' (%s)",
                         command[:60], combined_desc_for_llm)
            return {"approved": True, "message": None,
                    "smart_approved": True,
                    "description": combined_desc_for_llm}
        elif verdict == "deny":
            combined_desc_for_llm = "; ".join(desc for _, desc, _ in warnings)
            return {
                "approved": False,
                "message": f"BLOCKED by smart approval: {combined_desc_for_llm}. "
                           "The command was assessed as genuinely dangerous. Do NOT retry.",
                "smart_denied": True,
            }
        # verdict == "escalate" → fall through to manual prompt

    # --- Phase 3: Approval ---

    # Combine descriptions for a single approval prompt
    combined_desc = "; ".join(desc for _, desc, _ in warnings)
    primary_key = warnings[0][0]
    all_keys = [key for key, _, _ in warnings]
    has_tirith = any(is_t for _, _, is_t in warnings)

    # Gateway/async approval — block the agent thread until the user
    # responds with /approve or /deny, mirroring the CLI's synchronous
    # input() flow.  The agent never sees "approval_required"; it either
    # gets the command output (approved) or a definitive "BLOCKED" message.
    if is_gateway or is_ask:
        notify_cb = None
        with _lock:
            notify_cb = _gateway_notify_cbs.get(session_key)

        if notify_cb is not None:
            # --- Blocking gateway approval (queue-based) ---
            # Each call gets its own _ApprovalEntry so parallel subagents
            # and execute_code threads can block concurrently.
            approval_data = {
                "command": command,
                "pattern_key": primary_key,
                "pattern_keys": all_keys,
                "description": combined_desc,
            }
            entry = _ApprovalEntry(approval_data)
            with _lock:
                _gateway_queues.setdefault(session_key, []).append(entry)

            # Notify plugins that an approval is being requested. Fires before
            # the gateway notify callback so observers (e.g. macOS notifier
            # plugins, audit logs) get the event in real time.
            _fire_approval_hook(
                "pre_approval_request",
                command=command,
                description=combined_desc,
                pattern_key=primary_key,
                pattern_keys=list(all_keys),
                session_key=session_key,
                surface="gateway",
            )

            # Notify the user (bridges sync agent thread → async gateway)
            try:
                notify_cb(approval_data)
            except Exception as exc:
                logger.warning("Gateway approval notify failed: %s", exc)
                with _lock:
                    queue = _gateway_queues.get(session_key, [])
                    if entry in queue:
                        queue.remove(entry)
                    if not queue:
                        _gateway_queues.pop(session_key, None)
                return {
                    "approved": False,
                    "message": "BLOCKED: Failed to send approval request to user. Do NOT retry.",
                    "pattern_key": primary_key,
                    "description": combined_desc,
                }

            # Block until the user responds or timeout (default 5 min).
            # Poll in short slices so we can fire activity heartbeats every
            # ~10s to the agent's inactivity tracker.  Without this, the
            # blocking event.wait() never touches activity, and the
            # gateway's inactivity watchdog (agent.gateway_timeout, default
            # 1800s) kills the agent while the user is still responding to
            # the approval prompt.  Mirrors the _wait_for_process() cadence
            # in tools/environments/base.py.
            timeout = _get_approval_config().get("gateway_timeout", 300)
            try:
                timeout = int(timeout)
            except (ValueError, TypeError):
                timeout = 300

            try:
                from easybci_lib.tools.environments.base import (
                    touch_activity_if_due as _imported_touch,
                )
                _touch_cb: Optional[Callable] = _imported_touch
            except Exception:  # pragma: no cover
                _touch_cb = None

            _now = time.monotonic()
            _deadline = _now + max(timeout, 0)
            _activity_state = {"last_touch": _now, "start": _now}
            resolved = False
            while True:
                _remaining = _deadline - time.monotonic()
                if _remaining <= 0:
                    break
                # 1s poll slice — the event is set immediately when the
                # user responds, so slice length only controls heartbeat
                # cadence, not user-visible responsiveness.
                if entry.event.wait(timeout=min(1.0, _remaining)):
                    resolved = True
                    break
                if _touch_cb is not None:
                    _touch_cb(
                        _activity_state, "waiting for user approval"
                    )

            # Clean up this entry from the queue
            with _lock:
                queue = _gateway_queues.get(session_key, [])
                if entry in queue:
                    queue.remove(entry)
                if not queue:
                    _gateway_queues.pop(session_key, None)

            choice = entry.result
            # Normalize outcome for the post hook. Unresolved (timeout) and
            # None both mean the user never responded; report that explicitly
            # so plugins can distinguish timeout from explicit deny.
            _outcome = (
                "timeout" if not resolved
                else (choice if choice else "timeout")
            )
            _fire_approval_hook(
                "post_approval_response",
                command=command,
                description=combined_desc,
                pattern_key=primary_key,
                pattern_keys=list(all_keys),
                session_key=session_key,
                surface="gateway",
                choice=_outcome,
            )

            if not resolved or choice is None or choice == "deny":
                reason = "timed out" if not resolved else "denied by user"
                return {
                    "approved": False,
                    "message": f"BLOCKED: Command {reason}. Do NOT retry this command.",
                    "pattern_key": primary_key,
                    "description": combined_desc,
                }

            # User approved — persist based on scope (same logic as CLI)
            for key, _, is_tirith in warnings:
                if choice == "session" or (choice == "always" and is_tirith):
                    approve_session(session_key, key)
                elif choice == "always":
                    approve_session(session_key, key)
                    approve_permanent(key)
                    save_permanent_allowlist(_permanent_approved)
                # choice == "once": no persistence — command allowed this
                # single time only, matching the CLI's behavior.

            return {"approved": True, "message": None,
                    "user_approved": True, "description": combined_desc}

        # Fallback: no gateway callback registered (e.g. batch).
        # Return approval_required for backward compat.
        submit_pending(session_key, {
            "command": command,
            "pattern_key": primary_key,
            "pattern_keys": all_keys,
            "description": combined_desc,
        })
        return {
            "approved": False,
            "pattern_key": primary_key,
            "status": "approval_required",
            "command": command,
            "description": combined_desc,
            "message": (
                f"⚠️ {combined_desc}. Asking the user for approval.\n\n**Command:**\n```\n{command}\n```"
            ),
        }

    # CLI interactive: single combined prompt
    # Hide [a]lways when any tirith warning is present
    _fire_approval_hook(
        "pre_approval_request",
        command=command,
        description=combined_desc,
        pattern_key=primary_key,
        pattern_keys=list(all_keys),
        session_key=session_key,
        surface="cli",
    )
    choice = prompt_dangerous_approval(command, combined_desc,
                                       allow_permanent=not has_tirith,
                                       approval_callback=approval_callback)
    _fire_approval_hook(
        "post_approval_response",
        command=command,
        description=combined_desc,
        pattern_key=primary_key,
        pattern_keys=list(all_keys),
        session_key=session_key,
        surface="cli",
        choice=choice,
    )

    if choice == "deny":
        return {
            "approved": False,
            "message": "BLOCKED: User denied. Do NOT retry.",
            "pattern_key": primary_key,
            "description": combined_desc,
        }

    # Persist approval for each warning individually
    for key, _, is_tirith in warnings:
        if choice == "session" or (choice == "always" and is_tirith):
            # tirith: session only (no permanent broad allowlisting)
            approve_session(session_key, key)
        elif choice == "always":
            # dangerous patterns: permanent allowed
            approve_session(session_key, key)
            approve_permanent(key)
            save_permanent_allowlist(_permanent_approved)

    return {"approved": True, "message": None,
            "user_approved": True, "description": combined_desc}


# Load permanent allowlist from config on module import
load_permanent_allowlist()
