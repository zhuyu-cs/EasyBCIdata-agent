"""EasyBCI execution environment backends.

Each backend provides the same interface (BaseEnvironment ABC) for running
shell commands in a specific execution context: local, Docker, SSH,
Singularity, Modal, Daytona, or Vercel Sandbox. (Modal additionally has
direct modes, selected via terminal.modal_mode.)

The terminal_tool.py factory (_create_environment) selects the backend
based on the TERMINAL_ENV configuration.
"""

from easybci_lib.tools.environments.base import BaseEnvironment

__all__ = ["BaseEnvironment"]
