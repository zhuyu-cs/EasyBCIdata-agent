"""Single-source-of-truth registry for analysis_goal enum.

Three hardcoded touch-points consume this registry:
- codegen/generator.py — _enforce_clean_output reads inject_* flags
- run_agent.py / handlers — validates goal ∈ REGISTRY
- easybci_cli/web_server.py /api/schema/goal-enum — dynamic enum response

Adding a new goal is one entry here, not a 3-file edit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .analysis_goals_loader import load_and_merge_third_party as _load_tp


@dataclass
class AnalysisGoalSpec:
    name: str
    display_name: Dict[str, str]
    description: str
    inject_drop_bads: bool
    inject_drop_nondata: bool
    allow_aggressive_notch: bool = True
    allow_ica: bool = True
    produces_figures: bool = True
    # This flag is only a default *suggestion*. AI-ready generation is driven
    # by the `deliverables` concept (see preprocess/deliverables.py), decided by
    # the user at confirm time and persisted into the proposal / confirm marker /
    # pipeline_record. The flag is retained as the seed value when the LLM first
    # infers deliverables; codegen/contract_check no longer read it as a gate.
    produces_ai_ready: bool = True
    crystallize_eligible: bool = True
    notes: str = ""


REGISTRY: Dict[str, AnalysisGoalSpec] = {
    "classification": AnalysisGoalSpec(
        name="classification",
        display_name={"en": "Classification", "zh": "分类"},
        description="ML classification (motor imagery, P300, SSVEP, …)",
        inject_drop_bads=True,
        inject_drop_nondata=True,
    ),
    "source_localization": AnalysisGoalSpec(
        name="source_localization",
        display_name={"en": "Source Localization", "zh": "源定位"},
        description="Inverse modelling (sLORETA, beamforming, dipole fitting)",
        inject_drop_bads=False,
        inject_drop_nondata=False,
        notes="Keep EOG / physio for source modelling.",
    ),
    "feature_extraction": AnalysisGoalSpec(
        name="feature_extraction",
        display_name={"en": "Feature Extraction", "zh": "特征提取"},
        description="Bandpower / PSD / time-frequency features for downstream models",
        inject_drop_bads=True,
        inject_drop_nondata=True,
    ),
    "clinical_screening": AnalysisGoalSpec(
        name="clinical_screening",
        display_name={"en": "Clinical Screening", "zh": "临床筛查"},
        description="Reviewing EEG/MEG for clinically-relevant patterns",
        inject_drop_bads=True,
        inject_drop_nondata=True,
    ),
    "exploratory": AnalysisGoalSpec(
        name="exploratory",
        display_name={"en": "Exploratory", "zh": "探索性分析"},
        description="Loose, exploratory inspection — keep maximum information",
        inject_drop_bads=False,
        inject_drop_nondata=False,
        produces_ai_ready=False,
        crystallize_eligible=False,
    ),
    "generic": AnalysisGoalSpec(
        name="generic",
        display_name={"en": "Generic", "zh": "通用"},
        description="No specific downstream task; safest conservative defaults",
        inject_drop_bads=True,
        inject_drop_nondata=True,
        produces_ai_ready=False,
        crystallize_eligible=False,
    ),
    "connectivity": AnalysisGoalSpec(
        name="connectivity",
        display_name={"en": "Functional Connectivity", "zh": "连接性分析"},
        description="Phase-locking, coherence, Granger causality — preserves full channel structure",
        inject_drop_bads=True,
        inject_drop_nondata=False,
        allow_aggressive_notch=False,
        notes="Aggressive notch can distort phase; keep wide-band signal.",
    ),
    "phase_amplitude_coupling": AnalysisGoalSpec(
        name="phase_amplitude_coupling",
        display_name={"en": "Phase-Amplitude Coupling", "zh": "相位—幅值耦合（PAC）"},
        description="Theta-gamma PAC, etc. — phase- and amplitude-preserving processing",
        inject_drop_bads=True,
        inject_drop_nondata=False,
        allow_aggressive_notch=False,
        allow_ica=False,
        notes="ICA may inadvertently strip PAC components.",
    ),
    "online_inference": AnalysisGoalSpec(
        name="online_inference",
        display_name={"en": "Online BCI Inference", "zh": "在线推断"},
        description="Real-time inference; minimal latency, no offline-only steps",
        inject_drop_bads=True,
        inject_drop_nondata=True,
        allow_ica=False,
        produces_figures=False,
        notes="No offline ICA; QC figures not produced. mini-repo contract relaxes for this goal.",
    ),
    "sleep_staging": AnalysisGoalSpec(
        name="sleep_staging",
        display_name={"en": "Sleep Staging (PSG)", "zh": "睡眠分期（多导睡眠）"},
        description=("Polysomnography sleep staging — keep EEG/EOG/EMG plus "
                     "respiratory/SpO2/effort/position auxiliaries; 30 s AASM "
                     "epochs; hypnogram labels when scored."),
        inject_drop_bads=True,
        inject_drop_nondata=False,
        allow_aggressive_notch=True,
        allow_ica=False,
        produces_figures=True,
        crystallize_eligible=True,
        notes=("Do NOT high-pass above 0.5 Hz — slow waves (0.5–2 Hz) are "
               "critical. Low-pass ~35 Hz suffices. Resample default 100 Hz "
               "(AASM). Labels optional: study may be unscored."),
    ),
}


def is_valid_goal(goal: str) -> bool:
    return goal in REGISTRY


def get_spec(goal: str) -> AnalysisGoalSpec:
    """Return spec or raise KeyError."""
    return REGISTRY[goal]


# ── per-operator capability overrides ─────────────────────────────────────
#
# REGISTRY (above) specifies per-goal pipeline-wide flags (inject_drop_bads,
# allow_ica, ...). `op_capability_overrides` complements it with per-operator
# allow/forbid declarations: which goals an operator is permitted to participate
# in. The matrix is consumed by:
#
#   1. `_handle_plan_pipeline` / `propose_pipeline` — when an LLM proposes an
#      operator-goal pair, the dispatcher cross-checks this table and warns / vetoes.
#   2. `_check_consistency.py` (R7 / R8) — frontmatter `analysis_goal_*` lists
#      should not contradict this table; drift is reported.
#   3. CLI `easybci doctor goals` — surfaces unknown / conflicting overrides.
#
# Convention:
#   - "allow_in":  goals where the operator is explicitly recommended.
#   - "forbid_in": goals where the operator MUST NOT appear (hard veto).
#   - Goals not listed in either are implicitly neutral (allowed without recommendation).
#   - A goal appearing in both is an integrity error — `_validate_overrides`
#     raises at module import time.
#
# Add new operator entries here in alphabetical order to keep diffs small.
op_capability_overrides: Dict[str, Dict[str, List[str]]] = {
    "spike_sorting": {
        "allow_in":  ["classification", "feature_extraction", "exploratory", "generic"],
        "forbid_in": ["online_inference", "source_localization", "phase_amplitude_coupling"],
    },
    "threshold_spike": {
        "allow_in":  ["online_inference", "classification", "feature_extraction",
                      "exploratory", "generic"],
        "forbid_in": ["source_localization", "clinical_screening",
                      "connectivity", "phase_amplitude_coupling"],
    },
}


def _validate_overrides(overrides: Dict[str, Dict[str, List[str]]]) -> None:
    """Validate op_capability_overrides at import time.

    Raises
    ------
    ValueError
        On unknown goals or allow/forbid contradictions.
    """
    for op, caps in overrides.items():
        allow = set(caps.get("allow_in", []))
        forbid = set(caps.get("forbid_in", []))
        unknown = (allow | forbid) - set(REGISTRY.keys())
        if unknown:
            raise ValueError(
                f"op_capability_overrides[{op!r}]: unknown goals {sorted(unknown)} "
                f"(REGISTRY has {sorted(REGISTRY.keys())})"
            )
        overlap = allow & forbid
        if overlap:
            raise ValueError(
                f"op_capability_overrides[{op!r}]: goals both allowed and forbidden: "
                f"{sorted(overlap)}"
            )


_validate_overrides(op_capability_overrides)


def goals_for_op(op: str) -> Dict[str, List[str]]:
    """Return ``{"allow_in": [...], "forbid_in": [...]}`` for ``op``.

    Returns empty lists when no override is registered (operator is
    implicitly neutral on all goals).
    """
    caps = op_capability_overrides.get(op, {})
    return {
        "allow_in": list(caps.get("allow_in", [])),
        "forbid_in": list(caps.get("forbid_in", [])),
    }


def is_op_forbidden(op: str, goal: str) -> bool:
    """Return True iff ``op`` is hard-forbidden for ``goal`` per the overrides."""
    return goal in op_capability_overrides.get(op, {}).get("forbid_in", [])


# Auto-load third-party goals (~/.easybci/skills/analysis_goals/*.yaml) on
# module import. Conflicts cached for `easybci doctor goals`.
try:
    _THIRD_PARTY_CONFLICTS = _load_tp(REGISTRY, AnalysisGoalSpec)
except Exception:  # noqa: BLE001
    _THIRD_PARTY_CONFLICTS = []
