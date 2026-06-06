"""Per-typeclass-family MANIFEST partitions.

Each module exports `ENTRIES: dict[str, Entry]` covering the FFIs
bound to one typeclass slice (`UserExecutor<Family>`), plus the
`internal` module for unbound lifecycle helpers.

The package `__init__.py` merges these into the public `MANIFEST`.
"""
