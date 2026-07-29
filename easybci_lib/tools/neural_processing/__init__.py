"""BCI Data Processing Skills — Agent-Callable Neural Data Pipeline.

Lightweight, numpy-based skills for multi-modal BCI data processing.
Each skill exposes simple functions that the easybci-agent can invoke
directly with dict/primitive parameters.

Output format: pkl dict + meta.json
No torch dependency — pure numpy for maximum flexibility.
"""

__version__ = "0.1.0"
