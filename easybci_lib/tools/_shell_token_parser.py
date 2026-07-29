"""Shell command write-target extraction.

Used by source_data_guard / approval to identify which paths a shell command
would write to. Supports the standard write commands (rm/mv/cp/sed -i/tee/
mkdir/touch/chmod/chown/tar -x/unzip) plus > / >> redirects.

For non-recognized commands or shell features (eval, $(...) substitution,
pipelines that embed writes), returns an empty list — the caller should apply
a substring fallback over protected paths in those cases.
"""

from __future__ import annotations

import os
import shlex


_WRITE_COMMANDS_ALL_TARGETS: frozenset[str] = frozenset({
    # `mv` mutates BOTH source and destination: the source path is removed
    # from its original location. For source-data-immutability semantics
    # the source must be classified as a write target, not just the last
    # positional. (`cp` is different — the source is read-only.)
    "rm", "mv", "mkdir", "touch", "chmod", "chown",
})

_WRITE_COMMANDS_LAST_TARGET: frozenset[str] = frozenset({
    "cp",
})

_WRITE_COMMANDS_FLAG_TARGET: dict[str, tuple[str, ...]] = {
    "tar": ("-C", "--directory"),
    "unzip": ("-d",),
}

_WRITE_COMMANDS_FIRST_POSITIONAL: frozenset[str] = frozenset({
    "sed",
    "tee",
})


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _resolve(token: str, cwd: str | None) -> str:
    token = _strip_quotes(token)
    if not token or token.startswith("-"):
        return token
    if os.path.isabs(token):
        return os.path.normpath(token)
    if cwd:
        return os.path.normpath(os.path.join(cwd, token))
    return os.path.normpath(token)


def _is_path_like(token: str) -> bool:
    if not token or token.startswith("-"):
        return False
    return True


def _extract_from_simple_cmd(tokens: list[str], cwd: str | None) -> list[str]:
    if not tokens:
        return []
    cmd = tokens[0].lstrip("-")
    args = tokens[1:]

    if cmd in _WRITE_COMMANDS_ALL_TARGETS:
        targets = [a for a in args if _is_path_like(a)]
        if cmd in ("chmod", "chown") and targets:
            targets = targets[1:]
        return [_resolve(t, cwd) for t in targets]

    if cmd in _WRITE_COMMANDS_LAST_TARGET:
        positional = [a for a in args if _is_path_like(a)]
        if len(positional) >= 1:
            return [_resolve(positional[-1], cwd)]
        return []

    if cmd in _WRITE_COMMANDS_FLAG_TARGET:
        targets: list[str] = []
        flags = _WRITE_COMMANDS_FLAG_TARGET[cmd]
        i = 0
        while i < len(args):
            tok = args[i]
            matched: tuple[str, str] | None = None
            for flag in flags:
                if tok == flag and i + 1 < len(args):
                    matched = (flag, args[i + 1])
                    break
                if tok.startswith(flag + "="):
                    matched = (flag, tok[len(flag) + 1:])
                    break
            if matched is not None:
                targets.append(_resolve(matched[1], cwd))
                if "=" in tok:
                    i += 1
                else:
                    i += 2
                continue
            i += 1
        return targets

    if cmd in _WRITE_COMMANDS_FIRST_POSITIONAL:
        positional = [a for a in args if _is_path_like(a)]
        if cmd == "sed":
            has_inplace = any(a == "-i" or a.startswith("-i") for a in args)
            if has_inplace and positional:
                return [_resolve(positional[-1], cwd)]
            return []
        if cmd == "tee":
            return [_resolve(t, cwd) for t in positional]

    return []


def _split_redirects(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Split a token stream into (cleaned_tokens, redirect_targets)."""
    cleaned: list[str] = []
    redirects: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in (">", ">>"):
            if i + 1 < len(tokens):
                redirects.append(tokens[i + 1])
                i += 2
                continue
            i += 1
            continue
        if t.startswith(">>") and len(t) > 2:
            redirects.append(t[2:])
            i += 1
            continue
        if t.startswith(">") and len(t) > 1 and not t.startswith(">>"):
            redirects.append(t[1:])
            i += 1
            continue
        cleaned.append(t)
        i += 1
    return cleaned, redirects


def extract_write_targets(cmd: str, cwd: str | None = None) -> list[str]:
    """Return resolved write-target paths the shell command would touch.

    Empty list when no recognized write happens, when parsing fails, or when
    the command uses shell features that defeat static analysis (subshells,
    eval, $()). Caller should apply a substring fallback in those cases.
    """
    if not cmd or not cmd.strip():
        return []

    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return []
    if not tokens:
        return []

    cleaned, redirects = _split_redirects(tokens)
    targets: list[str] = [_resolve(r, cwd) for r in redirects if _is_path_like(r)]

    if cleaned:
        targets.extend(_extract_from_simple_cmd(cleaned, cwd))

    seen: set[str] = set()
    deduped: list[str] = []
    for t in targets:
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped
