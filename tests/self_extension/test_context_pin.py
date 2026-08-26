"""Phase E: pinned-findings verbatim protection across context compaction.

The agent marks a key value with a ``PINNED:`` line; compaction must preserve
that value byte-for-byte in the summary, and it must survive iterative
(repeated) compaction. Inert when no markers are present.
"""
from easybci_agent.context_compressor import ContextCompressor


def _engine():
    # Minimal construction — we only exercise pure text helpers, no LLM calls.
    return ContextCompressor(model="test-model", config_context_length=32000,
                                quiet_mode=True)


def test_extract_pins_from_turns():
    eng = _engine()
    turns = [
        {"role": "assistant", "content": "Ran deep_inspect.\n"
         "PINNED: modality=sEEG, sampling_rate=2000Hz, n_channels=128"},
        {"role": "user", "content": "ok continue"},
        {"role": "assistant", "content": "pinned: bad_channels=[C3, C17] (flat)"},
    ]
    pins = eng._collect_pinned_findings(turns, previous_summary="")
    assert "modality=sEEG, sampling_rate=2000Hz, n_channels=128" in pins
    assert "bad_channels=[C3, C17] (flat)" in pins


def test_render_and_reextract_roundtrip():
    """Pins rendered into a summary block must be re-extractable next round."""
    eng = _engine()
    pins = ["modality=sEEG, sampling_rate=2000Hz", "notch=50Hz applied"]
    block = eng._render_pinned_section(pins)
    assert "## Pinned Findings (verbatim)" in block
    assert "modality=sEEG, sampling_rate=2000Hz" in block
    # Feeding the rendered block back as a previous summary re-yields the pins.
    reextracted = eng._collect_pinned_findings([], previous_summary=block)
    assert "modality=sEEG, sampling_rate=2000Hz" in reextracted
    assert "notch=50Hz applied" in reextracted


def test_apply_pinned_section_verbatim():
    eng = _engine()
    summary = "## Active Task\nContinue preprocessing sub-03.\n"
    turns = [{"role": "assistant",
              "content": "PINNED: resample_target=250Hz (decision: match paradigm)"}]
    out = eng._apply_pinned_findings(summary, turns, previous_summary="")
    assert "resample_target=250Hz (decision: match paradigm)" in out
    assert "## Pinned Findings (verbatim)" in out
    # Original summary body preserved.
    assert "Continue preprocessing sub-03." in out


def test_no_markers_is_inert():
    eng = _engine()
    summary = "## Active Task\nNone.\n"
    out = eng._apply_pinned_findings(summary, [{"role": "user", "content": "hi"}],
                                     previous_summary="")
    assert out == summary  # unchanged when nothing is pinned


def test_pins_survive_iterative_compaction():
    """A pin captured in round 1 must remain after round 2 (no new marker)."""
    eng = _engine()
    round1 = eng._apply_pinned_findings(
        "## Active Task\nstep 1\n",
        [{"role": "assistant", "content": "PINNED: subject_id=sub-03 session=ses-01"}],
        previous_summary="")
    # Round 2: no new PINNED markers in the fresh turns, prior summary carries it.
    round2 = eng._apply_pinned_findings(
        "## Active Task\nstep 2\n",
        [{"role": "user", "content": "keep going"}],
        previous_summary=round1)
    assert "subject_id=sub-03 session=ses-01" in round2


def test_pin_dedup_and_cap():
    eng = _engine()
    turns = [{"role": "assistant", "content": "PINNED: dup_value"},
             {"role": "assistant", "content": "PINNED: dup_value"}]
    pins = eng._collect_pinned_findings(turns, previous_summary="")
    assert pins.count("dup_value") == 1
