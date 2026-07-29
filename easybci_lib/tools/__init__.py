#!/usr/bin/env python3
"""Tools package namespace.

Keep package import side effects minimal. Importing ``tools`` should not
eagerly import the full tool stack, because several subsystems load tools while
``easybci_cli.config`` is still initializing.

Callers should import concrete submodules directly, for example:

    import easybci_lib.tools.web_tools
    from easybci_lib.tools import terminal_tool

Python will resolve those submodules via the package path without needing them
to be re-exported here.
"""


def check_file_requirements():
    """File tools only require terminal backend availability."""
    from .terminal_tool import check_terminal_requirements

    return check_terminal_requirements()


__all__ = ["check_file_requirements"]
