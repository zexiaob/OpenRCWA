"""OpenRCWA user-facing API facade.

This package re-exports the primary user API from the internal package `rcwa`,
so users can simply do `import OpenRCWA as orcwa`.
"""

from rcwa import *  # noqa: F401,F403 - intentionally re-export everything

try:  # propagate explicit export list if present
    from rcwa import __all__ as _rcwa_all  # type: ignore
    __all__ = list(_rcwa_all)  # re-expose same surface
except Exception:
    # Fallback: best effort export of common names
    __all__ = [name for name in globals().keys() if not name.startswith("_")]
