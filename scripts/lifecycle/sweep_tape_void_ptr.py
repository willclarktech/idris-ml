#!/usr/bin/env python3
"""Sweep tape backend: wrap Tensor->data/grad reads with ((double*)...) casts
so they remain valid after the struct change to void* data/grad.

Tensor accesses to rewrite:
  <chain>->data[...]      -> ((double*)<chain>->data)[...]
  <chain>->grad[...]      -> ((double*)<chain>->grad)[...]
  <chain>->data + X       -> ((double*)<chain>->data) + X
  <chain>->grad + X       -> ((double*)<chain>->grad) + X
  <chain>->data - X       -> ((double*)<chain>->data) - X
  <chain>->grad - X       -> ((double*)<chain>->grad) - X

<chain> is one of:
  bare identifier:  e.g. `t`
  multi-arrow:      e.g. `meta->gamma`
  Tensor cast:      e.g. `((Tensor*)h)`  /  `((Tensor*)tensors[i])`

We EXCLUDE ArenaChunk.data accesses (it's char*, valid arithmetic without cast):
  any chain ending in `->head->data` / `->tail->data` / `->current->data`
  any `arena_head` / `arena_current` / arena chunk var `c->data` inside arena fns
"""
import re, pathlib, sys

path = pathlib.Path('packages/backends/backend_tape.c')
src = path.read_text()

# Build a chain pattern that matches:
#   bare:        \w+
#   cast:        \(\(Tensor\*\)[^)]*\)
#   multi-arrow: extend either above with (->\w+)*
chain = r'(?:\(\(Tensor\s*\*\)[^()]*\)|[A-Za-z_][A-Za-z_0-9]*)(?:->[A-Za-z_][A-Za-z_0-9]*)*'

# Pattern matches `<chain>->(data|grad)` followed by `[` or `+`/`-` (not `+=`/`-=`/`->`).
# Negative lookahead avoids matching `data[` inside an already-wrapped `((double*)X->data)[`.
read_pat = re.compile(
    r'(?<![*])(' + chain + r')->(data|grad)(\s*)(\[|[+-](?![=+-]))'
)

# Arena-chain prefixes to skip (the only multi-step chains in the file that
# terminate in ->data are LayerNorm meta (gamma/bias/beta — these ARE Tensor*),
# Linear meta (bias — Tensor*), and arena chunks (head/tail/current — char*)).
arena_terminators = ('head', 'tail', 'current')

def repl(m: re.Match) -> str:
    chain_text = m.group(1)
    field = m.group(2)
    ws = m.group(3)
    nxt = m.group(4)
    # Skip arena chunks (char* data field, not Tensor's void*).
    # The chain ends with one of the arena-chunk struct field names.
    last_step = chain_text.rsplit('->', 1)[-1]
    if last_step in arena_terminators:
        return m.group(0)
    # Skip when the chain ITSELF starts with `arena_head`/`arena_current` —
    # those are the static ArenaChunk* globals.
    if chain_text.startswith('arena_'):
        return m.group(0)
    return f'((double*){chain_text}->{field}){ws}{nxt}'

new = read_pat.sub(repl, src)

# Sanity: count of unmatched (non-cast) `->data[`/`->grad[` after sweep
remaining = re.findall(r'(?<!\)\))[A-Za-z_][A-Za-z_0-9]*(?:->[A-Za-z_][A-Za-z_0-9]*)*->(data|grad)\[', new)
print(f"remaining ->data/grad[ after sweep (should be only arena/casts handled separately): {len(remaining)}")

path.write_text(new)
