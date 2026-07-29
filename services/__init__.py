"""services — external integrations and HTTP service layer.

Holds three sub-packages:
- plugins/   : optional integration backends (memory/web/model-providers).
- providers/ : LLM provider adapters (OpenAI / Anthropic / Bedrock / ...).
- gateway/   : HTTP/SSE server that fronts the agent for the WebUI / external clients.
"""
