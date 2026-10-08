"""
OpenEvolve Native — MAP-Elites + island-based evolutionary search.

A faithful port of the OpenEvolve search algorithm adapted to
Coding Agent's base classes.  Named ``openevolve_native`` to avoid
any confusion with the external ``openevolve`` package.
"""

from coding_agent.search.openevolve_native.database import OpenEvolveNativeDatabase

__all__ = ["OpenEvolveNativeDatabase"]
