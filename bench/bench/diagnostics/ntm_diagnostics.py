"""NTM diagnostic extraction and summary metrics.

Matches idris-ml's Debug.idr: per-timestep state extraction, NtmSummary
computation, and formatted output for train/test comparison.
"""

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from bench.ntm.addressing import content_address


@dataclass
class NtmTimestep:
    """Per-timestep NTM internal state snapshot."""

    timestep: int
    input_symbol: int  # argmax of input
    target_symbol: int  # argmax of target
    read_addr: Tensor  # (n,) addressing weights
    write_addr: Tensor  # (n,)
    read_key: Tensor  # (w,)
    read_beta: float
    read_g: float
    read_gamma: float
    read_shift: Tensor  # (3,)
    write_key: Tensor  # (w,)
    write_beta: float
    write_g: float
    write_gamma: float
    write_shift: Tensor  # (3,)
    write_erase: Tensor  # (w,)
    write_add: Tensor  # (w,)
    memory: Tensor  # (n, w)
    read_output: Tensor  # (w,)
    content_read_weights: Tensor  # (n,) pre-interpolation
    content_write_weights: Tensor  # (n,) pre-interpolation


@dataclass
class NtmSummary:
    """Aggregated NTM metrics matching Debug.idr NtmSummary."""

    write_g_input: float
    write_g_output: float
    read_g_input: float
    read_g_output: float
    avg_write_beta: float
    avg_read_beta: float
    avg_write_gamma: float
    avg_read_gamma: float
    write_addr_entropy: float
    read_addr_entropy: float
    write_addr_peak_mass: float
    read_addr_peak_mass: float
    write_monotonic: bool
    read_monotonic: bool
    write_argmaxes: list[int] = field(default_factory=list)
    read_argmaxes: list[int] = field(default_factory=list)
    slots_used: int = 0
    num_slots: int = 0
    seq_len: int = 0
    content_match_rate: float | None = None


# ---------------------------------------------------------------------------
# Head parameter extraction from stashed controller output
# ---------------------------------------------------------------------------


def _read_head_input_width(w: int) -> int:
    return w + 3 + 3  # key + shift_kernel + (beta, g, gamma)


def _write_head_input_width(w: int) -> int:
    return _read_head_input_width(w) + 2 * w  # + erase + add


def _extract_read_params(raw: Tensor, w: int) -> tuple[Tensor, float, float, float, Tensor]:
    """Extract (key, beta, g, gamma, shift) from read head input."""
    shift_kernel_size = 3
    main = raw[: w + shift_kernel_size]
    params = raw[w + shift_kernel_size :]
    key = main[:w]
    shift_vec = main[w : w + shift_kernel_size]
    beta = torch.log(1 + torch.exp(params[0])).item()
    g = torch.sigmoid(params[1]).item()
    gamma = (1 + torch.log(1 + torch.exp(params[2]))).item()
    return key, beta, g, gamma, shift_vec


def _extract_write_params(
    raw: Tensor, w: int
) -> tuple[Tensor, float, float, float, Tensor, Tensor, Tensor]:
    """Extract (key, beta, g, gamma, shift, erase, add) from write head input."""
    rh_width = _read_head_input_width(w)
    rh_input = raw[:rh_width]
    key, beta, g, gamma, shift_vec = _extract_read_params(rh_input, w)
    raw_erase = raw[rh_width : rh_width + w]
    raw_add = raw[rh_width + w : rh_width + 2 * w]
    erase = torch.sigmoid(raw_erase)
    add = raw_add
    return key, beta, g, gamma, shift_vec, erase, add


# ---------------------------------------------------------------------------
# Instrumented forward
# ---------------------------------------------------------------------------


def instrumented_forward(
    model: torch.nn.Module,
    inputs: list[Tensor],
    targets: list[Tensor],
) -> list[NtmTimestep]:
    """Run model timestep-by-timestep, extracting NTM diagnostics.

    The model must have a .ntm attribute (NTMLayer with _diag dict).
    Calls model.reset_state() first.
    """
    model.reset_state()  # type: ignore[attr-defined]
    w: int = model.ntm.w  # type: ignore[attr-defined]
    rh_width = _read_head_input_width(w)
    wh_width = _write_head_input_width(w)
    timesteps: list[NtmTimestep] = []

    with torch.no_grad():
        for t, (x, y) in enumerate(zip(inputs, targets, strict=True)):
            _ = model(x)  # type: ignore[operator]
            # pyright can't infer dict type through dynamic attr
            diag: dict[str, Tensor] = model.ntm._diag  # type: ignore[attr-defined]
            ctrl_out = diag["controller_output"]

            # Split controller output same as NTMLayer.forward
            read_raw = ctrl_out[:rh_width]
            write_raw = ctrl_out[rh_width : rh_width + wh_width]

            # Extract head parameters
            r_key, r_beta, r_g, r_gamma, r_shift = _extract_read_params(read_raw, w)
            w_key, w_beta, w_g, w_gamma, w_shift, w_erase, w_add = _extract_write_params(
                write_raw, w
            )

            # Compute content weights (pre-interpolation)
            memory = diag["memory"]
            content_read_w = content_address(torch.tensor(r_beta), memory, r_key)
            content_write_w = content_address(torch.tensor(w_beta), memory, w_key)

            timesteps.append(
                NtmTimestep(
                    timestep=t,
                    input_symbol=int(x.argmax().item()),
                    target_symbol=int(y.argmax().item()),
                    read_addr=diag["read_addr"],
                    write_addr=diag["write_addr"],
                    read_key=r_key,
                    read_beta=r_beta,
                    read_g=r_g,
                    read_gamma=r_gamma,
                    read_shift=r_shift,
                    write_key=w_key,
                    write_beta=w_beta,
                    write_g=w_g,
                    write_gamma=w_gamma,
                    write_shift=w_shift,
                    write_erase=w_erase,
                    write_add=w_add,
                    memory=memory,
                    read_output=diag["read_output"],
                    content_read_weights=content_read_w,
                    content_write_weights=content_write_w,
                )
            )

    return timesteps


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _entropy(weights: Tensor) -> float:
    """Shannon entropy of a distribution."""
    w = weights.clamp(min=1e-12)
    return -(w * w.log()).sum().item()


def _peak_mass(weights: Tensor) -> float:
    return weights.max().item()


def _is_strictly_increasing(xs: list[int]) -> bool:
    return all(a < b for a, b in zip(xs, xs[1:], strict=False))


def _count_slots_used(memory: Tensor, threshold: float = 0.01) -> int:
    """Count memory rows with norm > threshold."""
    norms = memory.norm(dim=-1)
    return int((norms > threshold).sum().item())


def compute_summary(timesteps: list[NtmTimestep], seq_len: int) -> NtmSummary:
    """Aggregate per-timestep data into NtmSummary.

    seq_len: number of input-phase timesteps (first half).
    Matches Debug.idr computeSummary.
    """
    n_total = len(timesteps)

    write_gs = [ts.write_g for ts in timesteps]
    read_gs = [ts.read_g for ts in timesteps]
    write_betas = [ts.write_beta for ts in timesteps]
    read_betas = [ts.read_beta for ts in timesteps]
    write_gammas = [ts.write_gamma for ts in timesteps]
    read_gammas = [ts.read_gamma for ts in timesteps]
    write_addrs = [ts.write_addr for ts in timesteps]
    read_addrs = [ts.read_addr for ts in timesteps]

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    write_g_in = avg(write_gs[:seq_len])
    write_g_out = avg(write_gs[seq_len:])
    read_g_in = avg(read_gs[:seq_len])
    read_g_out = avg(read_gs[seq_len:])

    w_argmaxes = [int(w.argmax().item()) for w in write_addrs]
    r_argmaxes = [int(r.argmax().item()) for r in read_addrs]

    # Memory state at midpoint
    mid_idx = min(seq_len, n_total - 1)
    mid_memory = timesteps[mid_idx].memory
    num_slots = mid_memory.shape[0]

    return NtmSummary(
        write_g_input=write_g_in,
        write_g_output=write_g_out,
        read_g_input=read_g_in,
        read_g_output=read_g_out,
        avg_write_beta=avg(write_betas),
        avg_read_beta=avg(read_betas),
        avg_write_gamma=avg(write_gammas),
        avg_read_gamma=avg(read_gammas),
        write_addr_entropy=avg([_entropy(w) for w in write_addrs]),
        read_addr_entropy=avg([_entropy(r) for r in read_addrs]),
        write_addr_peak_mass=avg([_peak_mass(w) for w in write_addrs]),
        read_addr_peak_mass=avg([_peak_mass(r) for r in read_addrs]),
        write_monotonic=_is_strictly_increasing(w_argmaxes[:seq_len]),
        read_monotonic=_is_strictly_increasing(r_argmaxes[seq_len:]),
        write_argmaxes=w_argmaxes,
        read_argmaxes=r_argmaxes,
        slots_used=_count_slots_used(mid_memory),
        num_slots=num_slots,
        seq_len=seq_len,
    )


# ---------------------------------------------------------------------------
# Content match analysis (recall-specific)
# ---------------------------------------------------------------------------


def compute_content_match(
    timesteps: list[NtmTimestep],
    store_write_slots: list[int],
    query_timestep_indices: list[int],
    query_to_store_map: list[int],
) -> float:
    """Measure content-based addressing accuracy during recall queries.

    store_write_slots: write argmax at each store timestep (where keys were written)
    query_timestep_indices: timestep indices where model should recall values
    query_to_store_map: for each query, which store index it corresponds to

    Returns fraction of queries where content_read_weights.argmax() matches
    the memory slot where the corresponding key was stored.
    """
    if not query_timestep_indices:
        return 0.0

    matches = 0
    for qi, store_idx in zip(query_timestep_indices, query_to_store_map, strict=True):
        ts = timesteps[qi]
        content_argmax = int(ts.content_read_weights.argmax().item())
        expected_slot = store_write_slots[store_idx]
        if content_argmax == expected_slot:
            matches += 1

    return matches / len(query_timestep_indices)


# ---------------------------------------------------------------------------
# Formatting (matches Debug.idr output)
# ---------------------------------------------------------------------------


def _show_f(x: float) -> str:
    """Format to 4 decimal places matching Debug.idr showF."""
    if math.isnan(x):
        return "NaN"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    whole_and_frac = int(ax * 10000 + 0.5)
    w = whole_and_frac // 10000
    f = whole_and_frac % 10000
    return f"{sign}{w}.{f:04d}"


def _show_bool(b: bool) -> str:
    return "YES" if b else "NO"


def _show_delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    flag = " !" if abs(d) > 0.15 else ""
    return f"  ({sign}{_show_f(d)}{flag})"


def _slot_grid(n_slots: int, argmax_idx: int) -> str:
    slots = ["#" if i == argmax_idx else "." for i in range(n_slots)]
    return "[" + "".join(slots) + "]"


def _show_timesteps(n_slots: int, start_t: int, argmaxes: list[int]) -> str:
    parts = [f"t{start_t + i}{_slot_grid(n_slots, a)}" for i, a in enumerate(argmaxes)]
    return " ".join(parts)


def _show_argmax_list(xs: list[int]) -> str:
    return "[" + ",".join(str(x) for x in xs) + "]"


def print_summary(label: str, s: NtmSummary) -> None:
    """Print compact NTM summary matching Debug.idr printSummary."""
    print(f"=== NTM Summary: {label} ===")
    print("  Gate (g: 1=content, 0=location):")
    print(f"    Write:  input={_show_f(s.write_g_input)}  output={_show_f(s.write_g_output)}")
    print(f"    Read:   input={_show_f(s.read_g_input)}  output={_show_f(s.read_g_output)}")
    print(f"  Beta:  write={_show_f(s.avg_write_beta)}  read={_show_f(s.avg_read_beta)}")
    print(f"  Gamma: write={_show_f(s.avg_write_gamma)}  read={_show_f(s.avg_read_gamma)}")
    print(
        f"  Focus: write entropy={_show_f(s.write_addr_entropy)}"
        f" peak={_show_f(s.write_addr_peak_mass)}"
        f" | read entropy={_show_f(s.read_addr_entropy)}"
        f" peak={_show_f(s.read_addr_peak_mass)}"
    )
    print(f"  Memory: {s.slots_used}/{s.num_slots} slots used")
    sl = s.seq_len
    w_in = _show_argmax_list(s.write_argmaxes[:sl])
    r_out = _show_argmax_list(s.read_argmaxes[sl:])
    print(
        f"  Sequential: write={w_in} {_show_bool(s.write_monotonic)}"
        f" | read={r_out} {_show_bool(s.read_monotonic)}"
    )
    if s.content_match_rate is not None:
        print(f"  Content match rate: {_show_f(s.content_match_rate)}")


def print_addr_grid(s: NtmSummary) -> None:
    """Print addressing grid matching Debug.idr printAddrGrid."""
    ns = s.num_slots
    sl = s.seq_len
    w_in = s.write_argmaxes[:sl]
    w_out = s.write_argmaxes[sl:]
    r_in = s.read_argmaxes[:sl]
    r_out = s.read_argmaxes[sl:]
    print("  Addressing grid:")
    print(f"    Write: {_show_timesteps(ns, 0, w_in)}  (input)")
    print(f"           {_show_timesteps(ns, sl, w_out)}  (output)")
    print(f"    Read:  {_show_timesteps(ns, 0, r_in)}  (input)")
    print(f"           {_show_timesteps(ns, sl, r_out)}  (output)")


def print_comparison(train: NtmSummary, test: NtmSummary) -> None:
    """Print train vs test comparison matching Debug.idr printComparison."""
    print("=== Train vs Test Comparison (averaged) ===")
    print("                          Train    Test     Delta")

    def row(label: str, tv: float, testv: float) -> None:
        print(f"  {label:<22}{_show_f(tv)}   {_show_f(testv)}{_show_delta(testv - tv)}")

    row("Gate g (write/in):", train.write_g_input, test.write_g_input)
    row("Gate g (write/out):", train.write_g_output, test.write_g_output)
    row("Gate g (read/in):", train.read_g_input, test.read_g_input)
    row("Gate g (read/out):", train.read_g_output, test.read_g_output)
    row("Beta (write):", train.avg_write_beta, test.avg_write_beta)
    row("Beta (read):", train.avg_read_beta, test.avg_read_beta)
    row("Gamma (write):", train.avg_write_gamma, test.avg_write_gamma)
    row("Gamma (read):", train.avg_read_gamma, test.avg_read_gamma)
    row("Entropy (write):", train.write_addr_entropy, test.write_addr_entropy)
    row("Entropy (read):", train.read_addr_entropy, test.read_addr_entropy)
    row(
        "Peak mass (write):",
        train.write_addr_peak_mass,
        test.write_addr_peak_mass,
    )
    row(
        "Peak mass (read):",
        train.read_addr_peak_mass,
        test.read_addr_peak_mass,
    )
    print(
        f"  {'Write monotonic:':<22}"
        f"{_show_bool(train.write_monotonic):<9}"
        f"{_show_bool(test.write_monotonic)}"
    )
    print(
        f"  {'Read monotonic:':<22}"
        f"{_show_bool(train.read_monotonic):<9}"
        f"{_show_bool(test.read_monotonic)}"
    )
    if train.content_match_rate is not None and test.content_match_rate is not None:
        row(
            "Content match:",
            train.content_match_rate,
            test.content_match_rate,
        )
