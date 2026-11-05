"""Amplitude thresholds with explicit units (``500uV``, ``0.5mV``, ``2pT``).

Pure stdlib. The generated ``pipeline.py`` inlines this module's source
(between the ``# --- amplitude_units:`` markers) so the mini-repo stays
self-contained; the preprocessing engine imports the same functions.

Loaders record the unit the signal is held in as ``meta["data_unit"]`` (MNE
readers give SI base units: V for electrical channels, T for magnetometers;
some vendor loaders keep µV). A bare threshold is in those data units; a
threshold with a unit suffix is converted to them, so ``clip:500uV`` means the
same on a recording held in V as on one held in µV. Unknown or incompatible
units are never guessed: the value is left unconverted and the caller reports
the status.
"""
# --- amplitude_units: begin ---
import re as _au_re

# unit -> (family, scale to the family's SI base unit)
_AU_UNITS = {
    "v": ("volt", 1.0), "mv": ("volt", 1e-3), "uv": ("volt", 1e-6), "nv": ("volt", 1e-9),
    "t": ("tesla", 1.0), "mt": ("tesla", 1e-3), "ut": ("tesla", 1e-6), "nt": ("tesla", 1e-9),
    "pt": ("tesla", 1e-12), "ft": ("tesla", 1e-15),
}
_AU_ALIASES = {"microvolt": "uv", "microvolts": "uv", "millivolt": "mv", "millivolts": "mv",
               "volt": "v", "volts": "v", "tesla": "t", "femtotesla": "ft", "picotesla": "pt"}
_AU_RX = _au_re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)\s*([a-z]*)$")


def normalize_unit(unit):
    """Canonical lower-case unit key (``"uv"``, ``"v"``, ``"ft"`` …) or ``None``
    when the unit is absent, ``unknown``, ``unitless`` or not a supported
    amplitude unit."""
    if unit is None:
        return None
    u = str(unit).strip().replace("µ", "u").replace("μ", "u").lower()
    u = _AU_ALIASES.get(u, u)
    return u if u in _AU_UNITS else None


def split_amplitude(text):
    """``"500uV"`` -> ``(500.0, "uv")``; ``"500"`` -> ``(500.0, None)``.

    Raises ``ValueError`` for an unparsable number or an unsupported suffix."""
    s = str(text or "").strip().replace("µ", "u").replace("μ", "u").lower()
    m = _AU_RX.match(s)
    if not m:
        raise ValueError("cannot parse amplitude {!r}".format(text))
    value = float(m.group(1))
    if not m.group(2):
        return value, None
    unit = normalize_unit(m.group(2))
    if unit is None:
        raise ValueError("unsupported amplitude unit {!r} in {!r} (use V, mV, uV, nV, T, pT, fT)".format(
            m.group(2), text))
    return value, unit


def to_data_units(value, unit, data_unit):
    """Convert ``value`` given in ``unit`` to the data's unit.

    Returns ``(value_in_data_units, status)``; status is ``data_units`` (no
    suffix: already in data units), ``converted``, ``unit_unknown`` (the data
    unit is not recorded) or ``unit_mismatch`` (e.g. V against T). In the last
    two cases the value is returned unchanged."""
    if unit is None:
        return float(value), "data_units"
    du = normalize_unit(data_unit)
    if du is None:
        return float(value), "unit_unknown"
    fam_u, s_u = _AU_UNITS[unit]
    fam_d, s_d = _AU_UNITS[du]
    if fam_u != fam_d:
        return float(value), "unit_mismatch"
    return float(value) * s_u / s_d, "converted"


def rescaled_unit(data_unit, factor):
    """Data unit after multiplying the signal by ``factor`` (``"V"`` × 1e6 ->
    ``"uV"``); ``"unknown"`` when it does not land on a supported unit."""
    du = normalize_unit(data_unit)
    try:
        f = float(factor)
    except (TypeError, ValueError):
        return "unknown"
    if du is None or not f > 0:
        return "unknown"
    fam, s = _AU_UNITS[du]
    target = s / f
    for key, (fam_k, s_k) in _AU_UNITS.items():
        if fam_k == fam and abs(s_k - target) <= 1e-9 * s_k:
            return {"uv": "uV", "mv": "mV", "nv": "nV", "ut": "uT", "mt": "mT",
                    "nt": "nT", "pt": "pT", "ft": "fT"}.get(key, key.upper())
    return "unknown"
# --- amplitude_units: end ---


def inline_source() -> str:
    """Module source between the inline markers (for codegen embedding)."""
    import inspect
    import sys

    src = inspect.getsource(sys.modules[__name__])
    begin = src.index("# --- amplitude_units: begin ---")
    end = src.index("# --- amplitude_units: end ---") + len("# --- amplitude_units: end ---")
    return src[begin:end] + "\n"
