"""Which channels see the eyes: EOG / blink reference from channel names and types.

Pure stdlib. One resolver for deep_inspect (blink counting), the plan-time
ocular check and the generated pipeline's ``op_ica`` / ``op_regression_eog``
(inlined between the ``# --- eog_reference:`` markers), so planning and
execution agree on the reference.

Recognised conventions are a floor, not a closed set: 10-20 / 10-10 labels
with vendor decoration (``EEG Fp1-REF``, ``Fp1.``, bipolar ``FP1-F7``,
``Fp1:A2``); EOG by channel type or name (``*EOG*``, ``VEO``/``HEO``,
``LOC``/``ROC``, AASM ``E1``/``E2`` unless the net numbers its electrodes
``E1..E256``). A montage whose names are not 10-20-like is reported
(``montage_recognized: False``) so callers can ask for explicit names.
"""
# --- eog_reference: begin ---
import re as _er_re

EOG_FRONTAL_POLE = ("fp1", "fp2", "fpz")
EOG_FRONTAL_NEAR = ("af7", "af8", "af3", "af4", "afz")
_ER_PREFIX = _er_re.compile(r"^(?:eeg|eog|ecg|emg)[\s_:.]+", _er_re.IGNORECASE)
_ER_SPLIT = _er_re.compile(r"[\s\-:/_]")
# 10-20 / 10-10 / 10-5 labels; ear / mastoid only as A1/A2/M1/M2 so that
# BioSemi A1..A32 are not mistaken for ear electrodes.
_ER_1020 = _er_re.compile(
    r"^(?:(?:nz?|fp|aff?|ff|ffc|fcc?|ftt?|f|tt?|tpp?|t|cc|ccp|cpp?|c|ppo?|p|poo?|o|oi|iz?|cb)"
    r"(?:\d{1,2}h?|z)|[am][12])$")
_ER_EOG_LABELS = ("veo", "heo", "veog", "heog", "loc", "roc")


def electrode_label(name):
    """Bare lower-case electrode label: 'EEG Fp1-REF' / 'Fp1.' / 'FP1-F7' -> 'fp1'."""
    s = _ER_PREFIX.sub("", str(name).strip())
    s = _ER_SPLIT.split(s, maxsplit=1)[0]
    return s.strip("._()[]").lower()


def _er_e_numbered(labels):
    """EGI-style nets number their electrodes E1..E256."""
    return sum(1 for lab in labels if _er_re.fullmatch(r"e\d{1,3}", lab)) >= 8


def is_eog_channel(name, ch_type=None, e_numbered=False):
    if ch_type is not None and str(ch_type).lower() == "eog":
        return True
    if "eog" in str(name).lower():
        return True
    lab = electrode_label(name)
    return lab in _ER_EOG_LABELS or (not e_numbered and lab in ("e1", "e2"))


def montage_recognized(channels):
    """True when at least half of the channel names look like 10-20 / 10-10 labels."""
    labs = [electrode_label(c) for c in channels or []]
    return bool(labs) and sum(1 for lab in labs if _ER_1020.match(lab)) * 2 >= len(labs)


def resolve_eog_reference(channels, ch_types=None, explicit=None):
    """Ocular reference: explicit names -> EOG channels -> Fp1/Fp2/Fpz -> AF* -> none.

    Returns ``{source, indices, names, blink_indices, missing,
    montage_recognized}``; ``source`` is ``explicit`` / ``eog_channel`` /
    ``frontal_proxy`` / None. ``blink_indices`` prefers vertical EOG (blinks)
    over horizontal EOG (saccades).
    """
    chans = [str(c) for c in channels or []]
    types = (list(ch_types) if isinstance(ch_types, (list, tuple)) and len(ch_types) == len(chans)
             else [None] * len(chans))
    labels = [electrode_label(c) for c in chans]
    e_num = _er_e_numbered(labels)
    out = {"source": None, "indices": [], "names": [], "blink_indices": [], "missing": [],
           "montage_recognized": montage_recognized(chans)}
    picked = []
    for tok in explicit or []:
        t = str(tok).strip()
        if not t:
            continue
        hit = ([i for i, c in enumerate(chans) if c.lower() == t.lower()]
               or [i for i, lab in enumerate(labels) if lab == electrode_label(t)])
        if hit:
            picked += [i for i in hit if i not in picked]
        else:
            out["missing"].append(t)
    if picked:
        out.update(source="explicit", indices=picked, blink_indices=list(picked))
    if out["source"] is None:
        eog = [i for i, c in enumerate(chans) if is_eog_channel(c, types[i], e_num)]
        if eog:
            vert = [i for i in eog if "veo" in chans[i].lower() or "vert" in chans[i].lower()]
            out.update(source="eog_channel", indices=eog, blink_indices=vert or list(eog))
    if out["source"] is None:
        for group in (EOG_FRONTAL_POLE, EOG_FRONTAL_NEAR):
            fr = sorted((i for i, lab in enumerate(labels) if lab in group),
                        key=lambda i: group.index(labels[i]))
            if fr:
                out.update(source="frontal_proxy", indices=fr, blink_indices=list(fr))
                break
    out["names"] = [chans[i] for i in out["indices"]]
    return out
# --- eog_reference: end ---


def inline_source() -> str:
    """Module source between the inline markers (for codegen embedding)."""
    import inspect
    import sys

    src = inspect.getsource(sys.modules[__name__])
    begin = src.index("# --- eog_reference: begin ---")
    end = src.index("# --- eog_reference: end ---") + len("# --- eog_reference: end ---")
    return src[begin:end] + "\n"
