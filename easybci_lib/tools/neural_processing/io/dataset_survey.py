"""Dataset directory survey — what a human would look at before planning.

Lists documentation, timing metadata, already-processed copies and every other
file kind next to the recordings, so the planning step can read the dataset's
own description instead of guessing the trial structure. Read-only, bounded
(depth, file count, bytes read, output size). The keyword / suffix lists are a
starting point, not an authority: unmatched files are still counted in
``other_files`` so nothing is invisible.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from easybci_agent.source_data_guard import register_readable_docs, unread_docs

_DOC_SUFFIXES = {".md": "md", ".markdown": "md", ".rst": "rst", ".txt": "txt",
                 ".html": "html", ".htm": "html", ".pdf": "pdf", ".docx": "docx"}
_DOC_FORMAT_RANK = {"md": 0, "rst": 1, "txt": 2, "html": 3, "pdf": 4, "docx": 5}
_DOC_NAME_RE = re.compile(r"readme|说明|使用|description|protocol|codebook|manual|手册", re.I)
_TIMING_KEY_RE = re.compile(
    r"fixation|cue|imagery|task|stim|baseline|rest|phase|timing|onset|offset|"
    r"tmin|tmax|epoch|window|prestim|poststim|duration", re.I)
_TABLE_COL_RE = re.compile(r"phase|trial|cue|condition|label|event|marker|onset|stage", re.I)
_DERIVED_RE = re.compile(r"_(filtered|preprocessed|clean(ed)?|denoised|ica|epo|epochs)(\b|_|\.)", re.I)
_SIGNAL_SUFFIXES = {".edf", ".bdf", ".fif", ".set", ".fdt", ".cnt", ".gdf", ".vhdr", ".eeg",
                    ".vmrk", ".nwb", ".h5", ".hdf5", ".mat", ".21e", ".slp", ".npy", ".npz"}
_SKIP_DIR_SUFFIXES = ("_preprocess_work_dir", "_preprocess_work_dirs")

MAX_DEPTH = 2
MAX_FILES = 200
MAX_DOCS = 10
MAX_HINTS = 8
MAX_JSON_BYTES = 2 * 1024 * 1024
TXT_SNIFF_BYTES = 4096


def _iter_files(root: Path) -> tuple[list[Path], bool]:
    """Files under root, depth <= MAX_DEPTH, work dirs and dot-entries skipped, sorted."""
    out: list[Path] = []
    stack = [(root, 0)]
    while stack and len(out) < MAX_FILES:
        d, depth = stack.pop()
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for p in entries:
            if p.name.startswith("."):
                continue
            if p.is_dir():
                if depth < MAX_DEPTH and not p.name.endswith(_SKIP_DIR_SUFFIXES):
                    stack.append((p, depth + 1))
            elif p.is_file():
                out.append(p)
                if len(out) >= MAX_FILES:
                    break
    out.sort()
    return out, len(out) >= MAX_FILES


def _bids_root(start: Path) -> Path | None:
    p = start
    for _ in range(4):
        if (p / "dataset_description.json").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None


def _decode(raw: bytes) -> str | None:
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _txt_is_doc(path: Path) -> bool:
    """Prose, not a numeric signal / table dump."""
    try:
        with open(path, "rb") as fh:
            text = _decode(fh.read(TXT_SNIFF_BYTES))
    except OSError:
        return False
    if not text or not text.strip():
        return False
    tokens = text.split()
    numeric = sum(1 for t in tokens if re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?[,;]?", t, re.I))
    return numeric < 0.5 * max(len(tokens), 1)


def _doc_entry(path: Path) -> dict[str, Any] | None:
    fmt = _DOC_SUFFIXES.get(path.suffix.lower())
    if fmt is None and path.suffix == "" and _DOC_NAME_RE.search(path.name):
        fmt = "txt"
    if fmt is None:
        return None
    if fmt == "txt" and not _txt_is_doc(path):
        return None
    if fmt in ("html", "docx") and not _DOC_NAME_RE.search(path.stem):
        return None
    entry: dict[str, Any] = {
        "path": str(path), "format": fmt,
        "size_kb": round(path.stat().st_size / 1024, 1),
        "readable": fmt in ("md", "rst", "txt", "html"),
    }
    if fmt == "pdf" and shutil.which("pdftotext"):
        entry["extract_cmd"] = f'pdftotext -layout "{path}" -'
    return entry


def _merge_alt_formats(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stem: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for d in docs:
        p = Path(d["path"])
        by_stem.setdefault((str(p.parent), p.stem), []).append(d)
    merged = []
    for group in by_stem.values():
        group.sort(key=lambda d: _DOC_FORMAT_RANK.get(d["format"], 9))
        head = dict(group[0])
        if len(group) > 1:
            head["alt_formats"] = [g["path"] for g in group[1:]]
        merged.append(head)
    merged.sort(key=lambda d: (0 if _DOC_NAME_RE.search(Path(d["path"]).name) else 1, d["path"]))
    return merged


def _walk_json(obj: Any, prefix: str, depth: int, out: list[tuple[str, Any]]) -> None:
    """Collect (key_path, value) for timing-like keys.

    A flat group of scalars (``timing: {fixationSeconds: 2, ...}``) is reported
    whole; a group holding nested lists / dicts is descended so the key path
    points at the actual value (``baselineSource.windowSeconds``).
    """
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            nested = isinstance(v, dict) and any(isinstance(x, (dict, list)) for x in v.values())
            if _TIMING_KEY_RE.search(str(k)) and _is_small_value(v) and not nested:
                out.append((kp, v))
            elif isinstance(v, (dict, list)):
                _walk_json(v, kp, depth + 1, out)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        _walk_json(obj[0], prefix + "[0]", depth + 1, out)


def _is_small_value(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        return len(v) <= 80
    if isinstance(v, list):
        return 0 < len(v) <= 8 and all(isinstance(x, (int, float, str)) for x in v)
    if isinstance(v, dict):
        return 0 < len(v) <= 8 and all(_is_small_value(x) for x in v.values())
    return False


def _timing_hints(files: list[Path], root: Path) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for p in files:
        suf = p.suffix.lower()
        try:
            if suf in (".json", ".yaml", ".yml") and p.stat().st_size <= MAX_JSON_BYTES:
                text = p.read_text(encoding="utf-8", errors="replace")
                obj = json.loads(text) if suf == ".json" else yaml.safe_load(text)
                found: list[tuple[str, Any]] = []
                _walk_json(obj, "", 0, found)
                for kp, v in found:
                    key = kp + "=" + json.dumps(v, ensure_ascii=False, sort_keys=True)
                    if key in seen:
                        seen[key]["n_files"] += 1
                        continue
                    h = {"file": str(p.relative_to(root)), "key_path": kp, "value": v, "n_files": 1}
                    seen[key] = h
                    hints.append(h)
            elif suf in (".csv", ".tsv"):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    header = fh.readline().strip()
                cols = [c.strip() for c in re.split(r"[,\t;]", header) if c.strip()]
                matched = [c for c in cols if _TABLE_COL_RE.search(c)]
                if matched:
                    key = "cols=" + ",".join(matched)
                    if key in seen:
                        seen[key]["n_files"] += 1
                        continue
                    h = {"file": str(p.relative_to(root)), "columns": matched, "n_files": 1}
                    seen[key] = h
                    hints.append(h)
        except (OSError, ValueError, yaml.YAMLError):
            continue
    return hints


def survey_dataset_dir(path: str) -> dict[str, Any]:
    """Survey the directory holding ``path`` (or ``path`` itself if a dir).

    Returns ``dataset_context``: ``root``, ``docs``, ``timing_hints``,
    ``derived_files``, ``other_files`` (suffix → count), ``n_docs`` /
    ``n_timing_hints`` totals and ``truncated``.
    """
    p = Path(path).expanduser()
    root = p if p.is_dir() else p.parent
    try:
        root = root.resolve()
    except OSError:
        pass
    files, truncated = _iter_files(root)

    bids = _bids_root(root)
    if bids is not None and bids != root:
        for extra in sorted(bids.iterdir()):
            if extra.is_file() and _DOC_NAME_RE.search(extra.name) and extra not in files:
                files.append(extra)

    docs, derived, other = [], [], {}
    doc_paths: set[Path] = set()
    for f in files:
        entry = _doc_entry(f)
        if entry is not None:
            docs.append(entry)
            doc_paths.add(f)
            continue
        if _DERIVED_RE.search(f.name) and f.suffix.lower() in _SIGNAL_SUFFIXES:
            derived.append(str(f))
            continue
        if f.suffix.lower() in _SIGNAL_SUFFIXES:
            continue
        key = f.suffix.lower() or "(none)"
        other[key] = other.get(key, 0) + 1

    docs = _merge_alt_formats(docs)
    hints = _timing_hints([f for f in files if f not in doc_paths], root)
    return {
        "root": str(root),
        "docs": docs[:MAX_DOCS],
        "n_docs": len(docs),
        "timing_hints": hints[:MAX_HINTS],
        "n_timing_hints": len(hints),
        "derived_files": derived[:10],
        "other_files": dict(sorted(other.items())),
        "truncated": truncated,
    }


def register_survey_docs(ctx: dict[str, Any]) -> None:
    """Let the source read barrier pass the docs (and alt formats) found."""
    paths = []
    for d in ctx.get("docs") or []:
        paths.append(d["path"])
        paths.extend(d.get("alt_formats") or [])
    register_readable_docs(paths)


def survey_next_action(ctx: dict[str, Any], *, then: str) -> dict[str, Any] | None:
    """Next step implied by the survey; ``None`` when every doc was read."""
    doc_paths = [d["path"] for d in ctx.get("docs") or []]
    if doc_paths:
        unread = set(unread_docs(doc_paths))
        pending = [p for p in doc_paths if p in unread]
        if not pending:
            return None
        extra = [d["extract_cmd"] for d in ctx["docs"] if d.get("extract_cmd") and d["path"] in pending]
        return {
            "next_tool": "read_file",
            "case": "has_docs",
            "paths": pending,
            "then": then,
            "hint": (
                "This dataset ships its own documentation. Read every listed file "
                "BEFORE planning (read_file is allowed for them) and note: the trial "
                "timeline (which phases exist, their start/end in seconds relative to "
                "the event, which phases differ between classes), bad/open channels, "
                "units, reference, and files that must NOT be used (e.g. already "
                "filtered copies). Pass these as plan_pipeline(dataset_understanding=...)."
                + (" For PDFs without a text twin run: " + "; ".join(extra) if extra else "")
            ),
        }
    if ctx.get("timing_hints"):
        return {
            "next_tool": then,
            "case": "metadata_only",
            "hint": (
                "No documentation file, but timing_hints were found in the dataset "
                "directory. Derive the trial timeline from them and cite the file + "
                "key in plan_pipeline(dataset_understanding=..., "
                "epoch_window_basis={source:'metadata', evidence:...})."
            ),
        }
    return {
        "next_tool": then,
        "case": "no_dataset_info",
        "hint": (
            "No documentation or timing metadata next to the recordings. The epoch "
            "window can only come from the paradigm default or the event durations — "
            "say so in epoch_window_basis (source='paradigm_default' or "
            "'event_duration') and ask the user to confirm it at the proposal step."
        ),
    }
