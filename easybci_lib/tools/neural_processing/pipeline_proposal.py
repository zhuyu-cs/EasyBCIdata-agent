"""T7 Sub-phase P-F — Typed schema for pipeline proposals.

``PipelineProposal`` is what every ``plan_pipeline`` / ``propose_pipeline``
returns *conceptually* — even though the wire format remains JSON.  The
dataclass exists so:

1. Tools that consume the proposal (``_handle_generate_code`` /
   ``_handle_preprocess_neural``) can call :meth:`validate` and reject
   malformed inputs with a structured error.
2. Step strings get checked against the known-operator allowlist
   (frontmatter ``step_string`` field on every operator skill).  Steps
   the LLM invented are rejected at this gate rather than failing
   silently at execute-time.
3. Test code can build a proposal in one expression.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Step string grammar:
#   "<op>"                            — operator with default params
#   "<op>:<v1>"                        — single positional value
#   "<op>:<v1>,<v2>"                   — comma-separated positional values
#   "<op>:<k1>=<v1>,<k2>=<v2>"         — keyword args
# Examples accepted: "bandpass:1,40", "ica:n_components=20",
# "drop_bads:auto", "scale".
_STEP_STRING_RE = re.compile(
    r"^([a-z][a-z0-9_]*)(?::([^\s]+))?$"
)


@dataclass
class PipelineProposal:
    """Schema returned by ``plan_pipeline`` / ``propose_pipeline``.

    The fields mirror the JSON keys the CLI / WebUI already consume —
    introducing the dataclass does NOT change the wire format, only the
    in-Python typed entry point.
    """
    steps: List[str]
    analysis_goal: str
    rationale: str = ""
    web_evidence: Optional[Dict[str, Any]] = None
    cohort_tag: str = ""
    modality: str = ""
    paradigm: str = "default"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineProposal":
        return cls(
            steps=list(d.get("steps") or []),
            analysis_goal=str(d.get("analysis_goal", "generic")),
            rationale=str(d.get("rationale", "")),
            web_evidence=d.get("web_evidence"),
            cohort_tag=str(d.get("cohort_tag", "") or ""),
            modality=str(d.get("modality", "") or ""),
            paradigm=str(d.get("paradigm", "default") or "default"),
            meta=dict(d.get("meta") or {}),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    _ALLOWED_GOALS: frozenset[str] = frozenset({
        "classification", "source_localization", "feature_extraction",
        "clinical_screening", "exploratory", "generic",
        "connectivity", "phase_amplitude_coupling", "online_inference",
    })

    def validate(
        self,
        *,
        operator_step_strings: Optional[List[str]] = None,
    ) -> List[str]:
        """Return a list of validation errors; empty list = valid.

        Parameters
        ----------
        operator_step_strings : list of str, optional
            Whitelist of valid operator prefixes (the ``step_string``
            frontmatter values from each operator skill).  When not
            provided, the validator falls back to the loader at
            :func:`_load_known_operator_steps`.
        """
        errors: List[str] = []
        if not self.steps:
            errors.append("steps must be non-empty")
        if self.analysis_goal not in self._ALLOWED_GOALS:
            errors.append(
                f"analysis_goal={self.analysis_goal!r} not in {sorted(self._ALLOWED_GOALS)}"
            )

        known = (
            frozenset(operator_step_strings)
            if operator_step_strings is not None
            else _load_known_operator_steps()
        )
        for i, step in enumerate(self.steps):
            if not isinstance(step, str) or not step.strip():
                errors.append(f"steps[{i}] empty / non-string")
                continue
            m = _STEP_STRING_RE.match(step.strip())
            if not m:
                errors.append(
                    f"steps[{i}]={step!r} does not match the step-string grammar; "
                    "expected `<op>` or `<op>:<args>`"
                )
                continue
            op_name = m.group(1)
            if known and op_name not in known:
                errors.append(
                    f"steps[{i}]={step!r}: unknown operator {op_name!r}. "
                    "Run `easybci tools operators` to list registered operators."
                )
        return errors


# --------------------------------------------------------------------------
# Operator step_string allowlist loader
# --------------------------------------------------------------------------


def _load_known_operator_steps() -> frozenset[str]:
    """Read every operator skill's frontmatter ``step_string`` field.

    Returns the set of allowed operator names (the prefix before the
    optional ``:<args>``).  Caches the result for the process lifetime
    so a hot validation path doesn't re-walk the skills directory.
    """
    global _KNOWN_OPS_CACHE
    if _KNOWN_OPS_CACHE is not None:
        return _KNOWN_OPS_CACHE

    ops: set[str] = set()
    try:
        import yaml
        from easybci_lib.constants import get_easybci_home
        # 1) Shipped operator skills (one per directory under operators/).
        repo_ops = (
            Path(__file__).resolve().parent.parent.parent
            / "skills" / "bci" / "operators"
        )
        # 2) User-installed skills.
        user_ops = get_easybci_home() / "skills" / "bci" / "operators"
        for root in (repo_ops, user_ops):
            if not root.is_dir():
                continue
            for skill_md in root.glob("*/SKILL.md"):
                try:
                    text = skill_md.read_text(encoding="utf-8")
                except OSError:
                    continue
                if not text.startswith("---"):
                    continue
                end = text.find("---", 3)
                if end < 0:
                    continue
                try:
                    fm = yaml.safe_load(text[3:end]) or {}
                except yaml.YAMLError:
                    continue
                meta = fm.get("metadata") or {}
                step = (meta.get("step_string") or "").strip()
                if step:
                    # step_string is the operator prefix (no params).
                    ops.add(step.split(":", 1)[0])
                # Fall back to skill name when step_string is absent.
                elif fm.get("name"):
                    ops.add(str(fm["name"]).strip())
    except Exception:  # noqa: BLE001
        # Fail-open: validation passes whatever the LLM produced.  The
        # codegen lint then catches it downstream.
        return frozenset()

    _KNOWN_OPS_CACHE = frozenset(ops)
    return _KNOWN_OPS_CACHE


_KNOWN_OPS_CACHE: Optional[frozenset[str]] = None


def reset_known_operator_steps_cache() -> None:
    """Test helper — drop the cached allowlist so a new skill is picked up."""
    global _KNOWN_OPS_CACHE
    _KNOWN_OPS_CACHE = None
