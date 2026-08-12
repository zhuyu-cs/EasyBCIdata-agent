"""Reference-import: ingest a structured gold-standard preprocessing project
into an enhanced proven-pipeline skill (skeleton steps + adaptation_slots +
qc_baselines). Source data is read-only (Rule 5)."""

from easybci_lib.tools.neural_processing.reference.ingest import ingest_reference

__all__ = ["ingest_reference"]
