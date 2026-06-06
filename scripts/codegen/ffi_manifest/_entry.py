"""The `Entry` dataclass — one row in the manifest, plus the classifier
alphabet that describes a row's `args` / `ret` fields.

Type abbreviations for arg/return classification:

  T  — wrapped Tensor handle (Idris AnyPtr, vector-ref to unwrap)
  R  — raw AnyPtr (pass through; not a wrapped Tensor)
  i  — Int
  d  — Double
  s  — String
  v  — void / unit (return only)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    """One FFI primitive's full description.

    Manifest is keyed by base C function name (no _tape/_torch/_mlx
    backend suffix; those are stripped before lookup). Multiple instance
    methods can share a C symbol via the `c_symbol` field — the manifest
    key stays distinct (one per Idris-level method), but `c_symbol` points
    to the actual C function called by the wrap-template.
    """
    args: tuple              # arg-class tuple ("T", "i", "d", "s", "R", or "v")
    ret: str                 # ret-class
    slice: str = None        # None when the FFI is not bound to any instance method
    idris_method: str = None # typeclass method name (`primX`); set iff slice is set
    c_symbol: str = None     # canonical C name; None = use the manifest key
    tape: str = "direct"     # direct | bespoke
    torch: str = "direct"    # direct | bespoke
    mlx: str = "streamed"    # streamed | direct | bespoke
