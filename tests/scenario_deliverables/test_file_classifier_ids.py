"""Q2: _extract_identifiers subject/session regex was too greedy.

`sub[_-]?(\\w+?)` made the separator optional AND captured a lazy word, so:
  - "subaru_rest"     -> subject "aru"      (false BIDS match on a plain word)
  - "subject01_trial" -> subject "ject01"   (docstring's own example, wrong)
  - "S01_session1"    -> session "sion1"    (docstring's own example, wrong)

These IDs feed the light-inspect classification shown to the LLM. BIDS is
always `sub-<label>` / `ses-<label>` with a separator; non-separated forms
(S01 / subject01 / session1) must go through the numeric fallbacks.
"""

from easybci_lib.tools.neural_processing.io.file_classifier import _extract_identifiers


def test_bids_canonical():
    r = _extract_identifiers("sub-01_ses-02_run-01_eeg")
    assert r["subject"] == "01"
    assert r["session"] == "02"
    assert r["run"] == "01"


def test_bids_alphanumeric_label():
    r = _extract_identifiers("sub-control02_task-x")
    assert r["subject"] == "control02"


def test_plain_word_starting_with_sub_is_not_a_subject():
    # "subaru" is a word, not a BIDS subject entity.
    r = _extract_identifiers("subaru_rest")
    assert r["subject"] == "", "must not capture 'aru' from 'subaru'"


def test_subject_word_form_uses_numeric_fallback():
    r = _extract_identifiers("subject01_trial1")
    assert r["subject"] == "01", "subject01 -> 01, not 'ject01'"


def test_session_word_form():
    r = _extract_identifiers("S01_session1_run2")
    assert r["subject"] == "01"
    assert r["session"] == "1", "session1 -> 1, not 'sion1'"
    assert r["run"] == "2"


def test_underscore_bids_variant():
    # Some datasets use sub_01 instead of sub-01.
    r = _extract_identifiers("sub_07_ses_2")
    assert r["subject"] == "07"
    assert r["session"] == "2"
