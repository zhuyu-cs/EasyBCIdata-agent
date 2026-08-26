"""Dangerous-operation confirmation for preprocess work_dir materials.

User requirement (batch-EEG session): the agent must not silently delete or
move deliverable materials inside a ``*_preprocess_work_dir/``. Deleting
``middle_process/`` (scratch) is exempt; every other material (plan/, code/,
preprocessed_output/, pipeline_record.json, README.md, …) must go through the
approval prompt.

This is a *confirmable* warning (ask the user), NOT a hard block — the user
can approve it. Contrast with source-data immutability, which is absolute.
"""

from easybci_lib.tools.approval import check_workdir_material_command


def test_delete_plan_material_is_flagged():
    flagged, desc = check_workdir_material_command(
        "rm /data/subjX_preprocess_work_dir/plan/proposal.json"
    )
    assert flagged is True
    assert "plan/proposal.json" in desc


def test_delete_generated_code_is_flagged():
    flagged, desc = check_workdir_material_command(
        "rm /data/subjX_preprocess_work_dir/code/pipeline.py"
    )
    assert flagged is True


def test_move_preprocessed_output_is_flagged():
    flagged, desc = check_workdir_material_command(
        "mv /data/subjX_preprocess_work_dir/preprocessed_output/AI_ready/01/ses-1/e_epochs.pkl /tmp/z"
    )
    assert flagged is True


def test_recursive_delete_of_whole_workdir_is_flagged():
    flagged, desc = check_workdir_material_command(
        "rm -rf /data/subjX_preprocess_work_dir"
    )
    assert flagged is True


def test_delete_middle_process_is_exempt():
    # middle_process/ is scratch — cleanup here must not prompt.
    flagged, desc = check_workdir_material_command(
        "rm -rf /data/subjX_preprocess_work_dir/middle_process/sweep_20260812"
    )
    assert flagged is False


def test_delete_middle_process_root_is_exempt():
    flagged, _ = check_workdir_material_command(
        "rm -rf /data/subjX_preprocess_work_dir/middle_process"
    )
    assert flagged is False


def test_outside_workdir_not_flagged():
    # Fail-open: paths outside any *_preprocess_work_dir/ never fire here.
    flagged, _ = check_workdir_material_command("rm /tmp/scratch/foo.txt")
    assert flagged is False


def test_read_only_command_not_flagged():
    flagged, _ = check_workdir_material_command(
        "ls /data/subjX_preprocess_work_dir/plan"
    )
    assert flagged is False


def test_cat_material_not_flagged():
    flagged, _ = check_workdir_material_command(
        "cat /data/subjX_preprocess_work_dir/plan/proposal.json"
    )
    assert flagged is False


# ---------------------------------------------------------------------------
# `cd <path> && rm <relative>` — the CWD-dependent delete hole.
#
# User requirement (verbatim): "删除之前还得加个保险，必须匹配上路径字符串才行。
# 有时候是cd到特定路径才删除，但是cd失败之后可没有啥保险机制，这样的话删除可就炸了。"
#
# The shell connector encodes whether a failed `cd` can misdirect the delete:
#   `cd X && rm Y`  -> && short-circuits; if cd fails rm never runs. CWD is
#                      guaranteed X, so we resolve X+Y and match precisely.
#   `cd X ; rm Y`   -> `;` does NOT short-circuit; a failed cd leaves rm running
#   `cd X & rm Y`      in the WRONG cwd ("炸"). Cannot prove safety -> confirm.
# ---------------------------------------------------------------------------


def test_cd_and_delete_plan_material_is_flagged():
    # cd into the plan dir then delete a relative file: && guarantees CWD, so we
    # can resolve to plan/proposal.json and see it is a work_dir material.
    flagged, desc = check_workdir_material_command(
        "cd /data/subjX_preprocess_work_dir/plan && rm -f proposal.json"
    )
    assert flagged is True
    assert "proposal.json" in desc


def test_cd_and_delete_relative_glob_in_workdir_is_flagged():
    flagged, _ = check_workdir_material_command(
        "cd /data/subjX_preprocess_work_dir/code && rm -rf *"
    )
    assert flagged is True


def test_cd_and_delete_middle_process_via_and_is_exempt():
    # `cd X/middle_process && rm -rf *` — && makes it safe against cd failure,
    # and the resolved target lands in middle_process (scratch) -> no prompt.
    flagged, _ = check_workdir_material_command(
        "cd /data/subjX_preprocess_work_dir/middle_process && rm -rf *"
    )
    assert flagged is False


def test_cd_into_workdir_then_semicolon_delete_is_flagged():
    # `;` does not short-circuit: if the cd fails, `rm -rf *` runs in whatever
    # the real CWD is. Cannot prove safety -> must confirm (fail-safe).
    flagged, desc = check_workdir_material_command(
        "cd /data/subjX_preprocess_work_dir/middle_process ; rm -rf *"
    )
    assert flagged is True


def test_cd_into_workdir_then_backgrounded_delete_is_flagged():
    flagged, _ = check_workdir_material_command(
        "cd /data/subjX_preprocess_work_dir & rm -rf plan"
    )
    assert flagged is True


def test_cd_into_workdir_then_semicolon_delete_middle_process_relative_is_flagged():
    # Even a middle_process target is flagged in the `;` form, because a failed
    # cd would misdirect `rm -rf middle_process` to the wrong directory.
    flagged, _ = check_workdir_material_command(
        "cd /data/subjX_preprocess_work_dir ; rm -rf middle_process"
    )
    assert flagged is True


def test_cd_absolute_delete_absolute_still_matches():
    # rm target is absolute -> cd is irrelevant, normal matching applies.
    flagged, _ = check_workdir_material_command(
        "cd /tmp && rm -f /data/subjX_preprocess_work_dir/plan/goal.json"
    )
    assert flagged is True


def test_cd_and_delete_middle_process_abs_target_is_exempt():
    flagged, _ = check_workdir_material_command(
        'cd /tmp && rm -rf "/data/subjX_preprocess_work_dir/middle_process/"'
    )
    assert flagged is False


def test_cd_outside_then_relative_delete_not_flagged():
    # No work_dir mentioned anywhere; generic `rm -rf *` is left to the
    # recursive-delete DANGEROUS_PATTERN, not this work_dir-scoped guard.
    flagged, _ = check_workdir_material_command("cd /tmp/scratch && rm -rf *")
    assert flagged is False


# ---------------------------------------------------------------------------
# Robustness: tirith findings come from parsing an external binary's JSON
# stdout. A malformed run could yield a non-dict finding element; the approval
# flow must degrade rule_id to "unknown" rather than raise AttributeError.
# ---------------------------------------------------------------------------


def test_malformed_tirith_finding_does_not_crash_guards(monkeypatch):
    import easybci_lib.tools.approval as ap
    import easybci_lib.tools.tirith_security as ts

    monkeypatch.setattr(
        ts,
        "check_command_security",
        lambda cmd: {
            "action": "warn",
            "findings": ["not-a-dict"],  # malformed element
            "summary": "malformed",
        },
    )
    monkeypatch.setattr(ap, "_get_approval_mode", lambda: "off")

    # Must not raise AttributeError on findings[0].get(...)
    res = ap.check_all_command_guards("echo hi", "local")
    assert res.get("approved") is True
