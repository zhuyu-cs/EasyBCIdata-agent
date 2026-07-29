"""Subject + session identity resolution — single source of truth.

Resolves ``(subject_id, session_id)`` from a recording file's path. Called
ONCE by ``deep_inspect`` and baked into ``inspection_report.json`` under the
``identity`` field. All downstream tools (codegen, repo_builder, figure
writers, README) read from there — they MUST NOT re-derive subject_id from
``Path(data_path).stem``.

Priority chain (highest confidence first):

  1. CLI override (``--subject-id`` / ``--session-id``).
  2. BIDS segment in path (``sub-X`` / ``ses-Y`` directly in the data_path).
  3. Sibling files in the recording's directory carry a consistent ``sub-X``
     prefix — covers the common case where the main data file is non-BIDS
     (vendor format like ``Acq 2026_05_22_1623.cdt``) but companion event/
     channel CSVs are BIDS-named (``sub-S01_ses-001_block-1_...csv``).
  4. ``participants.tsv`` in an ancestor directory + path-prefix lookup.
  5. ``manifest.csv`` in the same directory with a subject column.
  6. Parent directory name (when not a generic word like ``data``, ``raw``,
     ``eeg``…) — the "same directory = same subject" heuristic.
  7. Fallback: ``subject_001`` + sequential / timestamp session.

The ``RecordingIdentity`` returned carries the ``source`` and ``confidence``
of the resolution so downstream consumers can warn the user when the
identity was guessed (``fallback_used=True``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Regex helpers --------------------------------------------------------------

_SUB_RE = re.compile(r"sub-([A-Za-z0-9]+)")
_SES_RE = re.compile(r"ses-([A-Za-z0-9]+)")

# Common timestamp patterns inside file stems. First match wins.
_TIMESTAMP_PATTERNS: tuple[re.Pattern[str], ...] = (
    # YYYY_MM_DD_HHMM, e.g. "Acq 2026_05_22_1623"
    re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{3,4})"),
    # YYYY-MM-DD-HHMM, e.g. "rec-2026-05-22-1623"
    re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{3,4})"),
    # YYYYMMDD_HHMM or YYYYMMDDTHHMM, e.g. "20260522_1623"
    re.compile(r"(\d{8})[T_](\d{4})"),
    # YYYY_MM_DD with no time (single timestamp digit group)
    re.compile(r"(\d{4})_(\d{2})_(\d{2})"),
)

# Directory names too generic to be subject identifiers.
_GENERIC_DIR_WORDS: frozenset[str] = frozenset({
    "data", "datasets", "dataset",
    "raw", "raw_data", "raw-data",
    "preprocessed", "processed",
    "eeg", "meg", "seeg", "ecog", "ieeg", "spike", "spikes", "lfp",
    "recordings", "recording",
    "session", "sessions",
    "input", "inputs",
    "test", "tests", "test_data", "testdata",
    "temp", "tmp",
    "output", "outputs",
    "files", "file",
    "subjects", "subject",
    "experiments", "experiment",
    "downloads",
    "home", "user", "users",
})

# Extensions we consider as primary neural recordings when counting siblings.
_NEURAL_DATA_EXTS: frozenset[str] = frozenset({
    ".cdt", ".bdf", ".edf", ".vhdr", ".eeg", ".set", ".fif",
    ".cnt", ".gdf", ".mff", ".sqd", ".con", ".ds",
    ".nwb", ".h5", ".hdf5",
    ".mat", ".npy", ".npz",
    ".xdf",
    ".pkl", ".pickle",
})


@dataclass
class RecordingIdentity:
    """Result of a subject/session resolution.

    ``subject_id`` / ``session_id`` never carry the ``sub-``/``ses-`` prefix —
    consumers add the prefix themselves when building paths (``sub-{id}/``).
    """

    subject_id: str
    session_id: str
    source: str
    confidence: float
    notes: str
    fallback_used: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RecordingIdentity":
        return cls(
            subject_id=str(d.get("subject_id") or "subject_001"),
            session_id=str(d.get("session_id") or "001"),
            source=str(d.get("source") or "fallback"),
            confidence=float(d.get("confidence", 0.0)),
            notes=str(d.get("notes") or ""),
            fallback_used=bool(d.get("fallback_used", False)),
        )


# Helpers --------------------------------------------------------------------


def _sanitize_id(raw: str) -> str:
    """Collapse whitespace + unsafe chars; preserve readability."""
    if not raw:
        return ""
    out = re.sub(r"\s+", "_", raw.strip())
    out = re.sub(r"[^A-Za-z0-9._-]", "_", out)
    out = re.sub(r"_+", "_", out).strip("_.-")
    return out


def _strip_sub_prefix(s: str) -> str:
    return s[4:] if isinstance(s, str) and s.startswith("sub-") else s


def _strip_ses_prefix(s: str) -> str:
    return s[4:] if isinstance(s, str) and s.startswith("ses-") else s


def _extract_sub_from_path(p: Path) -> Optional[str]:
    """Find first ``sub-XXX`` segment in path parts or stem.

    Matches segments / underscored tokens — not arbitrary substrings, to avoid
    accidentally picking up substrings like 'submarine'.
    """
    for part in p.parts:
        m = _SUB_RE.fullmatch(part)
        if m:
            return m.group(1)
    for tok in p.stem.split("_"):
        m = _SUB_RE.fullmatch(tok)
        if m:
            return m.group(1)
    return None


def _extract_ses_from_path(p: Path) -> Optional[str]:
    for part in p.parts:
        m = _SES_RE.fullmatch(part)
        if m:
            return m.group(1)
    for tok in p.stem.split("_"):
        m = _SES_RE.fullmatch(tok)
        if m:
            return m.group(1)
    return None


def _extract_timestamp_session(stem: str) -> Optional[str]:
    """Try common timestamp patterns; return ``YYYYMMDDTHHMM`` or ``YYYYMMDD``."""
    for rx in _TIMESTAMP_PATTERNS:
        m = rx.search(stem)
        if not m:
            continue
        groups = m.groups()
        if len(groups) == 4:
            y, mo, d, t = groups
            return f"{y}{mo}{d}T{t.zfill(4)}"
        if len(groups) == 2:
            date, t = groups
            return f"{date}T{t.zfill(4)}"
        if len(groups) == 3:
            y, mo, d = groups
            return f"{y}{mo}{d}"
    return None


def _find_bids_root(data_path: Path) -> Optional[Path]:
    p = data_path if data_path.is_dir() else data_path.parent
    for ancestor in [p] + list(p.parents):
        try:
            if (ancestor / "participants.tsv").exists():
                return ancestor
        except OSError:
            continue
    return None


def _scan_sibling_bids(data_dir: Path) -> Optional[tuple[str, Optional[str]]]:
    """Scan sibling filenames for consistent ``sub-X`` (and maybe ``ses-Y``).

    Returns ``(subject_id, session_id_or_None)`` when ONE distinct
    ``subject_id`` is found across all files carrying a BIDS prefix. If
    multiple distinct subject IDs appear, returns None (ambiguous — we
    refuse to guess).
    """
    if not data_dir.is_dir():
        return None
    sub_ids: set[str] = set()
    ses_ids: set[str] = set()
    try:
        entries = list(data_dir.iterdir())
    except OSError:
        return None
    for f in entries:
        try:
            if not f.is_file():
                continue
        except OSError:
            continue
        # Search anywhere in the filename — siblings often interleave the
        # BIDS prefix with other tokens (e.g. ``sub-S01_ses-001_block-1_...``).
        for m in _SUB_RE.finditer(f.name):
            sub_ids.add(m.group(1))
        for m in _SES_RE.finditer(f.name):
            ses_ids.add(m.group(1))
    if len(sub_ids) == 1:
        sub = next(iter(sub_ids))
        ses = next(iter(ses_ids)) if len(ses_ids) == 1 else None
        return (sub, ses)
    return None


def _lookup_participants_tsv(bids_root: Path, data_path: Path) -> Optional[str]:
    """Best-effort: scan participants.tsv for a row that matches data_path.

    BIDS layout puts the data under ``<root>/sub-X/...``, so step 2 already
    catches the canonical case. This is the fallback when participants.tsv
    exists but the data is loose-organized — we try to match by basename.
    """
    tsv = bids_root / "participants.tsv"
    try:
        with open(tsv, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            if "participant_id" not in header:
                return None
            pid_idx = header.index("participant_id")
            stem = data_path.stem
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if pid_idx >= len(cols):
                    continue
                pid = cols[pid_idx]
                if not pid:
                    continue
                # Accept either "sub-S01" or "S01" forms.
                sub_id = _strip_sub_prefix(pid)
                if sub_id and sub_id in stem:
                    return sub_id
    except OSError as exc:
        logger.debug("participants.tsv read failed: %s", exc)
    return None


def _lookup_manifest_csv(data_dir: Path, data_path: Path) -> Optional[str]:
    """Read ``manifest.csv`` in the same directory; look up subject column.

    Recognised subject-column names: ``subject_id``, ``participant_id``,
    ``subject``, ``participant``. The match column is ``base``, ``file``,
    ``filename``, or ``path``, compared against ``data_path.stem`` or
    ``data_path.name``.
    """
    manifest = data_dir / "manifest.csv"
    if not manifest.is_file():
        return None
    try:
        with open(manifest, encoding="utf-8") as f:
            header_line = f.readline().rstrip("\n")
            header = [h.strip().lower() for h in header_line.split(",")]
            id_cols = {"subject_id", "participant_id", "subject", "participant"}
            match_cols = {"base", "file", "filename", "path"}
            id_idx = next((i for i, h in enumerate(header) if h in id_cols), None)
            match_idx = next((i for i, h in enumerate(header) if h in match_cols), None)
            if id_idx is None or match_idx is None:
                return None
            stem = data_path.stem
            name = data_path.name
            for line in f:
                cols = [c.strip() for c in line.rstrip("\n").split(",")]
                if match_idx >= len(cols) or id_idx >= len(cols):
                    continue
                if cols[match_idx] in (stem, name):
                    return cols[id_idx] or None
    except OSError as exc:
        logger.debug("manifest.csv read failed: %s", exc)
    return None


def _directory_subject_id(data_path: Path) -> Optional[str]:
    """Use parent directory name as subject_id when it doesn't look generic.

    Returns None for generic words (``data``, ``raw``, ``eeg``…) so the
    fallback chain continues. Sanitizes whitespace / unsafe chars.
    """
    parent = data_path.parent
    name = parent.name
    if not name or name in (".", "/", ""):
        return None
    if name.lower() in _GENERIC_DIR_WORDS:
        return None
    return _sanitize_id(name)


def _sibling_file_index(data_path: Path) -> int:
    """Position (1-indexed) of data_path among sorted neural-data siblings.

    Used to generate stable session_id numbering when no timestamp is found.
    Returns 1 if data_path is alone in its directory or not present.
    """
    parent = data_path.parent
    if not parent.is_dir():
        return 1
    try:
        siblings = sorted(
            f.name for f in parent.iterdir()
            if f.is_file() and f.suffix.lower() in _NEURAL_DATA_EXTS
        )
    except OSError:
        return 1
    try:
        return siblings.index(data_path.name) + 1
    except ValueError:
        return 1


def _derive_session_id(
    data_path: Path,
    *,
    fallback_seq: bool,
) -> str:
    """Pick a stable session_id for this recording.

    Preference order:
      1. timestamp embedded in the filename → ``YYYYMMDDTHHMM``
      2. ``ses-X`` found anywhere in path → ``X``
      3. when ``fallback_seq=True``: 1-indexed position among siblings → ``00N``
      4. final fallback: ``001``
    """
    ts = _extract_timestamp_session(data_path.stem)
    if ts:
        return ts
    ses_path = _extract_ses_from_path(data_path)
    if ses_path:
        return ses_path
    if fallback_seq:
        idx = _sibling_file_index(data_path)
        return f"{idx:03d}"
    return "001"


# Public entry point ---------------------------------------------------------


def resolve_identity(
    data_path: Path | str,
    *,
    cli_subject_id: Optional[str] = None,
    cli_session_id: Optional[str] = None,
) -> RecordingIdentity:
    """Resolve subject + session identity for one recording.

    See module docstring for the priority chain. Always returns a valid
    ``RecordingIdentity`` — never raises. The ``fallback_used`` flag and
    ``source`` field tell the caller how trustworthy the result is.
    """
    p = Path(data_path)
    data_dir = p.parent if not p.is_dir() else p

    # 1. CLI override --------------------------------------------------------
    if cli_subject_id:
        sub = _sanitize_id(_strip_sub_prefix(cli_subject_id))
        ses_raw = (
            cli_session_id
            or _extract_ses_from_path(p)
            or _extract_timestamp_session(p.stem)
            or "001"
        )
        ses = _sanitize_id(_strip_ses_prefix(ses_raw))
        return RecordingIdentity(
            subject_id=sub or "subject_001",
            session_id=ses or "001",
            source="cli_override",
            confidence=1.0,
            notes=f"--subject-id={cli_subject_id} provided by user.",
            fallback_used=False,
        )

    # 2. BIDS segment in data_path -------------------------------------------
    sub_in_path = _extract_sub_from_path(p)
    if sub_in_path:
        ses = _derive_session_id(p, fallback_seq=False)
        return RecordingIdentity(
            subject_id=_sanitize_id(sub_in_path),
            session_id=_sanitize_id(ses),
            source="bids_path",
            confidence=0.98,
            notes=f"BIDS 'sub-{sub_in_path}' found in data path.",
            fallback_used=False,
        )

    # 3. Sibling files reveal sub-X ------------------------------------------
    sibling = _scan_sibling_bids(data_dir)
    if sibling:
        sub_sib, ses_sib = sibling
        # When siblings ALSO consistently expose one session id, use it only
        # if no per-file timestamp is available (timestamps differentiate
        # multiple recordings sharing the same logical session).
        ts = _extract_timestamp_session(p.stem)
        if ts:
            ses = ts
        elif ses_sib:
            ses = ses_sib
        else:
            ses = _derive_session_id(p, fallback_seq=True)
        return RecordingIdentity(
            subject_id=_sanitize_id(sub_sib),
            session_id=_sanitize_id(ses),
            source="sibling_bids",
            confidence=0.85,
            notes=(
                f"BIDS 'sub-{sub_sib}' found in sibling files of "
                f"'{data_dir.name}/' (main data filename is non-BIDS)."
            ),
            fallback_used=False,
        )

    # 4. participants.tsv lookup --------------------------------------------
    bids_root = _find_bids_root(p)
    if bids_root is not None:
        sub_tsv = _lookup_participants_tsv(bids_root, p)
        if sub_tsv:
            ses = _derive_session_id(p, fallback_seq=True)
            return RecordingIdentity(
                subject_id=_sanitize_id(sub_tsv),
                session_id=_sanitize_id(ses),
                source="participants_tsv",
                confidence=0.80,
                notes=(
                    f"participants.tsv at '{bids_root}/' matched "
                    f"subject id from data_path stem."
                ),
                fallback_used=False,
            )

    # 5. manifest.csv lookup -------------------------------------------------
    sub_mf = _lookup_manifest_csv(data_dir, p)
    if sub_mf:
        ses = _derive_session_id(p, fallback_seq=True)
        return RecordingIdentity(
            subject_id=_sanitize_id(_strip_sub_prefix(sub_mf)),
            session_id=_sanitize_id(ses),
            source="manifest",
            confidence=0.78,
            notes=f"manifest.csv mapped {p.name} → {sub_mf}.",
            fallback_used=False,
        )

    # 6. Parent directory name -----------------------------------------------
    dir_sub = _directory_subject_id(p)
    if dir_sub:
        ses = _derive_session_id(p, fallback_seq=True)
        return RecordingIdentity(
            subject_id=dir_sub,
            session_id=_sanitize_id(ses),
            source="directory",
            confidence=0.55,
            notes=(
                f"Same-directory heuristic: parent dir '{p.parent.name}/' "
                f"treated as one subject; sibling recordings become "
                f"separate sessions."
            ),
            fallback_used=False,
        )

    # 7. Fallback ------------------------------------------------------------
    ses = _derive_session_id(p, fallback_seq=True)
    return RecordingIdentity(
        subject_id="subject_001",
        session_id=_sanitize_id(ses),
        source="fallback",
        confidence=0.20,
        notes=(
            "No BIDS sub-/ses- markers, no participants.tsv, no manifest.csv "
            "with subject column, and parent directory name is too generic. "
            "Defaulting to 'subject_001'. Pass --subject-id <id> to override."
        ),
        fallback_used=True,
    )
