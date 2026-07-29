"""Single source of truth for the reproducibility seed.

Importing this from anywhere in the codebase (runtime operators, codegen,
record builders) guarantees there is exactly one canonical value for the
fixed seed used to make EasyBCI mini-repos byte-reproducible.
"""

EASYBCI_SEED: int = 42
