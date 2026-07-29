"""Code generation skill — produces executable Python from pipeline records.

Public surface (v3, multi-input aware): use ``generate_pipeline_script`` +
``generate_qc_script_v2`` + ``generate_build_ai_ready_script`` +
``generate_run_script_v2``. The legacy ``generate_pipeline_code`` /
``generate_run_script`` / ``generate_qc_script`` were removed in the
multi-session-routing refactor — they hard-coded single-file routing
and could not produce scripts that respect ``inputs_routing.json``.
"""

from easybci_lib.tools.neural_processing.codegen.generator import (
    generate_build_ai_ready_script,
    generate_pipeline_script,
    generate_qc_script_v2,
    generate_run_script_v2,
    generate_vis_script,
)
