"""Agent internals -- extracted modules from run_agent.py.

Submodules:
  iteration_budget  — Thread-safe iteration counter
  parallelism       — Parallel tool execution constants & helpers
  sanitization      — Message/payload sanitization (surrogates, JSON repair)
  stream_filter     — Streaming reasoning-tag suppression
  system_prompt     — Three-tier system prompt assembly
  tool_permissions  — Per-tool permission tier controller
"""
