"""Static safety scan for codegen-generated scripts.

Walks the AST of pipeline.py / qc.py / vis.py / build_ai_ready.py BEFORE
the executor runs them. Rejects writes that target paths inside any
source_data_guard protected directory.

WHY: defense-in-depth on top of file_safety (which blocks the agent's
write_file tool) and approval (which blocks shell write commands). This
catches the case where the agent generates a `pipeline.py` that writes
back into the source data tree via raw.save() / open(mode='w') / etc.
Both literal strings and the literal prefix of f-strings are analyzed;
fully dynamic paths fall through to the executor's mtime audit.
"""

from __future__ import annotations

import ast
import os


class CodegenSafetyViolation(Exception):
    """Raised when AST scan finds a write operation targeting source data."""

    def __init__(self, script_path: str, line: int, target: str, reason: str):
        self.script_path = script_path
        self.line = line
        self.target = target
        self.reason = reason
        super().__init__(
            f"BLOCKED: {script_path}:{line} {reason} target='{target}'. "
            "Choose an output path outside the source data tree."
        )


_WRITE_OPEN_MODES = ("w", "a", "x", "+")

_DESTRUCTIVE_OS_FUNCS = {"remove", "unlink", "rename", "replace"}
_DESTRUCTIVE_SHUTIL_FUNCS = {"copy", "copy2", "copyfile", "move"}
_SAVE_METHOD_NAMES = {"save", "write_text", "write_bytes"}


def _literal_prefix(node: ast.AST) -> str | None:
    """Return the best-effort literal prefix of an AST expression.

    Handles ast.Constant str, ast.JoinedStr (f-string) literal segments,
    ast.BinOp str/+/Div, os.path.join(...), Path("lit").
    Returns None if no determinable literal prefix.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                inner = _literal_prefix(v.value)
                if inner is not None:
                    parts.append(inner)
                    continue
                break
            else:
                break
        return "".join(parts) if parts else None

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_prefix(node.left)
        if left is None:
            return None
        right = _literal_prefix(node.right)
        if right is None:
            return left
        return left + right

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _literal_prefix(node.left)
        right = _literal_prefix(node.right)
        if left is None:
            return None
        if right is None:
            return left
        return os.path.join(left, right)

    if isinstance(node, ast.Call):
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "join"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "path"):
            pieces: list[str] = []
            for a in node.args:
                lp = _literal_prefix(a)
                if lp is None:
                    break
                pieces.append(lp)
            if pieces:
                return os.path.join(*pieces)

        if isinstance(func, ast.Name) and func.id == "Path" and node.args:
            return _literal_prefix(node.args[0])

    return None


def _check_target(path: str, script: str, line: int, reason: str) -> None:
    from easybci_agent.source_data_guard import is_source_data, is_inside_protected_dir

    if is_source_data(path) or is_inside_protected_dir(path):
        raise CodegenSafetyViolation(script, line, path, reason)


class _ScanVisitor(ast.NodeVisitor):
    def __init__(self, script_path: str):
        self.script_path = script_path

    def visit_Call(self, node: ast.Call) -> None:
        self._check_call(node)
        self.generic_visit(node)

    def _check_call(self, node: ast.Call) -> None:
        func = node.func

        if isinstance(func, ast.Name) and func.id == "open":
            self._check_open(node)

        if (isinstance(func, ast.Attribute) and func.attr == "open"
                and isinstance(func.value, ast.Name)
                and func.value.id in {"io", "builtins"}):
            self._check_open(node)

        if isinstance(func, ast.Attribute) and func.attr in {"write_text", "write_bytes"}:
            target = _literal_prefix(func.value)
            if target is not None:
                _check_target(target, self.script_path, node.lineno,
                              f"{func.attr} on pathlib.Path")

        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name) and func.value.id == "shutil"
                and func.attr in _DESTRUCTIVE_SHUTIL_FUNCS):
            if len(node.args) >= 2:
                dst = _literal_prefix(node.args[1])
                if dst is not None:
                    _check_target(dst, self.script_path, node.lineno,
                                  f"shutil.{func.attr} destination")

        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name) and func.value.id == "os"):
            if func.attr in _DESTRUCTIVE_OS_FUNCS and node.args:
                target = _literal_prefix(node.args[0])
                if target is not None:
                    _check_target(target, self.script_path, node.lineno,
                                  f"os.{func.attr}")
            if func.attr == "chdir" and node.args:
                target = _literal_prefix(node.args[0])
                if target is not None:
                    _check_target(target, self.script_path, node.lineno,
                                  "os.chdir into protected dir is forbidden")

        if isinstance(func, ast.Attribute) and func.attr == "save":
            if (isinstance(func.value, ast.Name)
                    and func.value.id in {"nib", "nibabel"}):
                if len(node.args) >= 2:
                    target = _literal_prefix(node.args[1])
                    if target is not None:
                        _check_target(target, self.script_path, node.lineno,
                                      "nibabel.save destination")
            elif node.args:
                target = _literal_prefix(node.args[0])
                if target is not None:
                    _check_target(target, self.script_path, node.lineno,
                                  ".save() write target")

    def _kw_or_pos(self, node: ast.Call, name: str, pos: int) -> ast.AST | None:
        for kw in node.keywords:
            if kw.arg == name:
                return kw.value
        if len(node.args) > pos:
            return node.args[pos]
        return None

    def _check_open(self, node: ast.Call) -> None:
        mode_node = self._kw_or_pos(node, "mode", 1)
        mode_val: str | None = None
        if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
            mode_val = mode_node.value
        elif mode_node is None:
            mode_val = "r"
        else:
            mode_val = "?"

        is_write = any(m in mode_val for m in _WRITE_OPEN_MODES)
        if not is_write:
            return

        if not node.args:
            return
        target = _literal_prefix(node.args[0])
        if target is None:
            return
        _check_target(target, self.script_path, node.lineno,
                      f"open(mode={mode_val!r}) write")


def scan_script(script_path: str, work_dir: str | None = None) -> None:
    """Parse the script and raise CodegenSafetyViolation on any forbidden write.

    Re-raises any IO/parse error — caller should catch and convert into a
    structured error for the agent.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=script_path)
    visitor = _ScanVisitor(script_path)
    visitor.visit(tree)
