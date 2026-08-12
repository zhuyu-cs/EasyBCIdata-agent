"""NWB writer for the preprocessed/ layer.

Single entry point: :func:`save_nwb`. pynwb is loaded lazily via
``easybci_lib.tools.lazy_deps.ensure("neural.pynwb")`` so this module can be
imported even when pynwb is not installed (the import will trigger pip).

Output contract (Spec § 4.2):

- NWBFile with required identifier/session_description/session_start_time
- One ElectricalSeries named "preprocessed", shape (n_samples, n_channels)
- electrodes table with channel_name + channel_type custom columns
- Subject with subject_id (required upstream)
- Optional Units table for spike modality when payload["spike_times"] is set
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# Lazy-guarded heavy imports. Documented CLAUDE.md exception:
# pynwb is an optional heavy dep, only loaded when this module is used.
try:
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.ecephys import ElectricalSeries
    from pynwb.file import Subject
except ImportError:
    from easybci_lib.tools.lazy_deps import ensure as _ensure_lazy
    _ensure_lazy("neural.pynwb")
    from pynwb import NWBFile, NWBHDF5IO  # noqa: E402
    from pynwb.ecephys import ElectricalSeries  # noqa: E402
    from pynwb.file import Subject  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_SESSION_START = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Conservative alias lists — first non-None/non-empty match wins.
# Kept narrow on purpose: no lowercasing, no substring matching, no fuzzy
# matches, so harmless siblings like ``event_rate`` can't poison ``sfreq``.
_SFREQ_KEYS = ("sfreq", "frequency", "sampling_rate", "fs", "srate", "rate")
_CH_NAMES_KEYS = ("ch_names", "channels", "channel_names")
_CH_TYPES_KEYS = ("ch_types", "channel_types", "types")
_MEAS_DATE_KEYS = ("meas_date", "measurement_date", "date")
_SUBJECT_ID_KEYS = ("subject_id", "subject", "sub_id")


def _pick(d: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """Return ``d[k]`` for the first ``k`` whose value is not None/empty."""
    if not isinstance(d, Mapping):
        return default
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v:
            continue
        return v
    return default


def _info_get(info: Any, name: str) -> Any:
    """``info[name]`` for mne.Info (mapping-like) without crashing on missing keys."""
    if info is None:
        return None
    try:
        return info[name]
    except (KeyError, TypeError, ValueError):
        try:
            return getattr(info, name, None)
        except Exception:
            return None


def _info_channel_types(info: Any) -> Optional[list[str]]:
    if info is None:
        return None
    fn = getattr(info, "get_channel_types", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:
            return None
    return None


def _info_subject_id(info: Any) -> Optional[str]:
    si = _info_get(info, "subject_info")
    if not si:
        return None
    if isinstance(si, Mapping):
        return si.get("his_id") or si.get("id") or si.get("subject_id")
    return getattr(si, "his_id", None) or getattr(si, "id", None)


def _coerce_session_start(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value is None:
        return _DEFAULT_SESSION_START
    if isinstance(value, (tuple, list)) and len(value) == 2:
        # mne historically stored (secs, usecs) tuples in raw.info["meas_date"].
        try:
            secs, usecs = value
            return datetime.fromtimestamp(float(secs) + float(usecs) * 1e-6, tz=timezone.utc)
        except Exception:
            return _DEFAULT_SESSION_START
    return _DEFAULT_SESSION_START


def save_nwb(
    payload: dict[str, Any],
    out_path: Path,
    meta: dict[str, Any],
    mne_info: Any = None,
) -> Path:
    """Write a preprocessed payload to an NWB file.

    Args:
        payload: dict with keys "data" (np.ndarray, shape (n_ch, n_samp)),
            "labels" (optional), "meta" (mirror of meta arg, optional),
            "spike_times" (optional, list of per-unit spike-time iterables).
        out_path: target .nwb path.
        meta: NWB metadata. The writer looks up each field across an alias list
            (e.g. sfreq | frequency | sampling_rate | fs | srate | rate); the
            first non-empty value wins. ``ch_names`` is the only field that
            must be resolvable — everything else has a soft default. Missing
            fields fall back to ``mne_info`` (an mne.Info) when supplied,
            and finally to writer defaults (1970 UTC, ``"unknown"`` channel
            types, ``uuid4`` identifier, ``"unspecified"`` species).
        mne_info: optional mne.Info used as a secondary source when ``meta``
            doesn't carry the field. ``payload`` / ``meta`` may also include
            ``"_mne_info"`` to the same effect.

    Returns:
        The path that was written.
    """
    data = payload["data"]
    if data.ndim != 2:
        raise ValueError(f"payload['data'] must be 2-D, got shape {data.shape}")

    n_ch, _n_samp = data.shape

    # Resolve the secondary info source: explicit arg > payload > meta.
    info = mne_info
    if info is None:
        info = payload.get("_mne_info") if isinstance(payload, Mapping) else None
    if info is None and isinstance(meta, Mapping):
        info = meta.get("_mne_info")

    ch_names = _pick(meta, _CH_NAMES_KEYS)
    if ch_names is None:
        ch_names = _info_get(info, "ch_names")
    if ch_names is None:
        raise ValueError("save_nwb: ch_names not found in meta or mne_info")
    ch_names = list(ch_names)

    ch_types = _pick(meta, _CH_TYPES_KEYS)
    if ch_types is None:
        ch_types = _info_channel_types(info)
    if ch_types is None:
        ch_types = ["unknown"] * n_ch
    else:
        ch_types = list(ch_types)

    if len(ch_names) != n_ch:
        raise ValueError(
            f"ch_names length {len(ch_names)} != n_channels {n_ch}"
        )
    if len(ch_types) != n_ch:
        raise ValueError(
            f"ch_types length {len(ch_types)} != n_channels {n_ch}"
        )

    sfreq_raw = _pick(meta, _SFREQ_KEYS)
    if sfreq_raw is None:
        sfreq_raw = _info_get(info, "sfreq")
    if sfreq_raw is None:
        raise ValueError("save_nwb: sampling rate not found in meta or mne_info")
    sfreq = float(sfreq_raw)

    subject_id = _pick(meta, _SUBJECT_ID_KEYS)
    if subject_id is None:
        subject_id = _info_subject_id(info)
    analysis_goal = meta.get("analysis_goal", "") if isinstance(meta, Mapping) else ""

    meas_date = _pick(meta, _MEAS_DATE_KEYS)
    if meas_date is None:
        meas_date = _info_get(info, "meas_date")
    if meas_date is None:
        # INFO, not WARNING: many raw formats (e.g. Nihon Kohden) legitimately
        # carry no recording date, and this fires once per saved file — a
        # per-file WARNING floods errors.log during batch runs. Downgraded so
        # it stops polluting the error log; the 1970 fallback is expected.
        logger.info(
            "save_nwb: meas_date not found in meta or mne_info — using "
            "1970-01-01 UTC. Set raw.info['meas_date'] upstream (or pass meta["
            "'meas_date']) to get a real timestamp."
        )
    session_start = _coerce_session_start(meas_date)

    session_desc = (
        f"{analysis_goal} — preprocessed by EasyBCIdata"
        if analysis_goal
        else "preprocessed by EasyBCIdata"
    )

    identifier = (
        f"{subject_id}/{out_path.stem}" if subject_id else uuid.uuid4().hex
    )

    nwb = NWBFile(
        session_description=session_desc,
        identifier=identifier,
        session_start_time=session_start,
        experimenter=meta.get("experimenter") if isinstance(meta, Mapping) else None,
        institution=meta.get("institution") if isinstance(meta, Mapping) else None,
        session_id=meta.get("session_id") if isinstance(meta, Mapping) else None,
        subject=Subject(
            subject_id=str(subject_id) if subject_id else "unknown",
            species=(meta.get("species", "unspecified") if isinstance(meta, Mapping) else "unspecified"),
        ),
    )

    device = nwb.create_device(name="EasyBCIdata-recording-device")
    electrode_group = nwb.create_electrode_group(
        name="preprocessed-group",
        description="Channels carried through EasyBCIdata preprocessing.",
        location="unspecified",
        device=device,
    )
    nwb.add_electrode_column(name="channel_name", description="MNE channel name.")
    nwb.add_electrode_column(name="channel_type", description="MNE channel type.")
    for name, ctype in zip(ch_names, ch_types):
        nwb.add_electrode(
            location="unspecified",
            group=electrode_group,
            channel_name=name,
            channel_type=str(ctype),
        )
    electrode_region = nwb.create_electrode_table_region(
        region=list(range(n_ch)),
        description="All preprocessed channels",
    )

    es = ElectricalSeries(
        name="preprocessed",
        data=data.T,
        electrodes=electrode_region,
        rate=sfreq,
        starting_time=0.0,
        description="Continuous post-pipeline signal",
    )
    nwb.add_acquisition(es)

    modality = ""
    if isinstance(meta, Mapping):
        modality = (meta.get("modality") or "").strip().lower()
    spike_times = payload.get("spike_times")
    if modality in ("spike", "spikes", "unit", "units"):
        if spike_times:
            for i, st in enumerate(spike_times):
                nwb.add_unit(spike_times=list(st), id=i)
        else:
            logger.warning(
                "save_nwb: modality is %r but payload['spike_times'] is empty "
                "— writing ElectricalSeries only, skipping Units table.",
                modality,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(out_path), "w") as io:
        io.write(nwb)
    return out_path
