"""Central plot style for QC visualizations — single source of truth for font
sizes and save DPI so every QC figure (before/after comparison, per-subject QC,
and the generated vis.py figures) looks consistent.

Two DPIs on purpose:
- ``QC_DPI_SAVE`` (300) for figures written to disk (the artifacts a human
  reviews).
- ``QC_DPI_PREVIEW`` (100) for base64 thumbnails embedded in SSE/JSON for the
  WebUI — kept low so payload size does not balloon (~36x at 600).

NOTE: the generated ``vis.py`` is a standalone script (runs on a bare
``pip install`` venv and CANNOT import this package), so it inlines an
equivalent rcParams block in its template header — keep the two in sync.
See codegen/generator.py's vis templates.
"""

# Save DPIs — disk artifacts vs. web preview thumbnails.
QC_DPI_SAVE = 300
QC_DPI_PREVIEW = 100

# Unified font sizes for every QC figure. One definition → no per-plot drift.
QC_FONT = {
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
}


def apply_qc_style() -> None:
    """Apply the unified QC font sizes to matplotlib's global rcParams.

    Idempotent; call once before building a figure. Only touches font sizes —
    colors/themes stay under each caller's control.
    """
    import matplotlib as mpl

    mpl.rcParams.update(QC_FONT)
