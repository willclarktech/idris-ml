#!/usr/bin/env python3
"""
One-off migration: extend the guardian lazy-init carried by the INIT_FFI
create wrappers so it also primes `idris-drain-once` (the guardian drain
helper). Before this, the drain helper was installed only by
`initManagedHandles` — called nowhere in production — so the drain
epilogues in `native_train_step_*` and `withNoGrad` were dormant and mlx
husks never reached rc==0 (the long-grad-mode handle leak).

The wrapper bodies are already `%foreign "scheme:..."`, so
`ffi-convert-to-scheme.py` (which only rewrites `C:` → `scheme:`) doesn't
touch them. The guardian-only lazy-init appears verbatim in each create
wrapper; swap it for the extended `GUARDIAN_LAZY_INIT` (guardian + drain).

Idempotent: re-running finds zero occurrences of the old string. Kept in
the tree (like migrate-wrap-v2.py) as the regeneration record.

Usage:  python3 scripts/lifecycle/prime-drain-lazy-init.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ffi_manifest import GUARDIAN_ONLY_INIT, GUARDIAN_LAZY_INIT, WRAP_HANDLE_FILES

REPO = Path(__file__).resolve().parents[2]


def main():
    total = 0
    for rel in WRAP_HANDLE_FILES:
        path = REPO / rel
        text = path.read_text()
        n = text.count(GUARDIAN_ONLY_INIT)
        if n == 0:
            continue
        # Guard against double-application: if the extended form is already
        # present, the old string only appears as its prefix — count would
        # still be n, but replacing is a no-op since NEW starts with OLD.
        # Detect by checking whether OLD is already followed by the drain part.
        if GUARDIAN_LAZY_INIT in text:
            print(f"{rel}: already primed, skipping")
            continue
        text = text.replace(GUARDIAN_ONLY_INIT, GUARDIAN_LAZY_INIT)
        path.write_text(text)
        total += n
        print(f"{rel}: primed {n} create wrapper(s)")
    print(f"done: {total} wrappers updated")


if __name__ == "__main__":
    main()
