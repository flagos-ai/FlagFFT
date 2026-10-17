"""Vector FFT exchange expressed through tensor split/join and permutations."""

from dataclasses import asdict
import hashlib
import json
import math


def build_split_tensor_fft(config):
    n, pack = config.length, config.rows_per_program
    if n & (n - 1):
        raise ValueError("tensor split exchange requires a power-of-two length")
    identity = hashlib.sha256(json.dumps({"version": 2, **asdict(config)}, sort_keys=True).encode()).hexdigest()[:16]
    name = f"_tensor_split_n{n}_{config.mode}_{identity}"
    ilen = n // 2 + 1 if config.mode == "c2r" else n
    olen = n // 2 + 1 if config.mode == "r2c" else n
    extent = config.row_tile_extent
    lines = ["@triton.jit", f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch, program_offset):",
        "    program = program_offset + tl.program_id(0).to(tl.int64)",
        f"    lane = tl.arange(0, {pack})"]
    if extent:
        tiles = (extent + pack - 1) // pack
        lines += [f"    inner_row = (program % {tiles}) * {pack} + lane",
            f"    group_row = (program // {tiles}) * {extent}",
            "    row = group_row + inner_row",
            f"    valid_row = (row < nbatch) & (inner_row < {extent})"]
    else:
        lines += [f"    row = program * {pack} + lane", "    valid_row = row < nbatch"]
    lines += [f"    frequency = tl.arange(0, {n})",
        f"    input_base = (row // {config.input_stride}) * {ilen * config.input_stride} + row % {config.input_stride}"]
    if config.mode == "c2r":
        padded = 1 << (ilen - 1).bit_length()
        lines += [f"    compact = tl.arange(0, {padded})",
            f"    source = input_base[:, None] + compact[None, :] * {config.input_stride}",
            f"    valid = valid_row[:, None] & (compact[None, :] < {ilen})",
            "    point = tl.load(in_ptr + source[:, :, None] * 2 + tl.arange(0, 2)[None, None, :], valid[:, :, None], 0.0)",
            "    compact_r, compact_i = tl.split(point)",
            f"    reflected = frequency > {n // 2}",
            f"    source_frequency = tl.where(reflected, {n} - frequency, frequency)",
            f"    indices = tl.broadcast_to(source_frequency[None, :], ({pack}, {n}))",
            "    r = tl.gather(compact_r, indices, 1)",
            "    i = tl.gather(compact_i, indices, 1)",
            "    i = tl.where(reflected[None, :], -i, i)",
            f"    i = tl.where((frequency[None, :] == 0) | (frequency[None, :] == {n // 2}), 0.0, i)"]
    else:
        lines += [f"    source = input_base[:, None] + frequency[None, :] * {config.input_stride}"]
        if config.mode == "r2c":
            lines += ["    r = tl.load(in_ptr + source, valid_row[:, None], 0.0)",
                f"    i = tl.full(({pack}, {n}), 0.0, tl.float32)"]
        else:
            lines += ["    r = tl.load(in_ptr + source * 2, valid_row[:, None], 0.0)",
                "    i = tl.load(in_ptr + source * 2 + 1, valid_row[:, None], 0.0)"]
    span = n
    while span > 1:
        half, groups = span // 2, n // span
        shape = (pack * groups, 2, half)
        lines += [f"    r0, r1 = tl.split(tl.trans(tl.reshape(r, {shape}), (0, 2, 1)))",
            f"    i0, i1 = tl.split(tl.trans(tl.reshape(i, {shape}), (0, 2, 1)))",
            "    plus_r, plus_i = r0 + r1, i0 + i1",
            "    minus_r, minus_i = r0 - r1, i0 - i1"]
        if half > 1:
            angle = (-1 if config.direction == "forward" else 1) * 2 * math.pi / span
            lines += [f"    phase = tl.arange(0, {half}) * {angle!r}",
                "    wr, wi = tl.cos(phase)[None, :], tl.sin(phase)[None, :]",
                "    minus_r, minus_i = minus_r * wr - minus_i * wi, minus_r * wi + minus_i * wr"]
        lines += [f"    r = tl.reshape(tl.trans(tl.join(plus_r, minus_r), (0, 2, 1)), ({pack}, {n}))",
            f"    i = tl.reshape(tl.trans(tl.join(plus_i, minus_i), (0, 2, 1)), ({pack}, {n}))"]
        span = half
    bits = n.bit_length() - 1
    if bits > 1:
        reversed_index = " | ".join(f"(((frequency >> {bit}) & 1) << {bits - 1 - bit})" for bit in range(bits))
        lines += [f"    reversed_index = {reversed_index}",
            f"    indices = tl.broadcast_to(reversed_index[None, :], ({pack}, {n}))",
            "    r = tl.gather(r, indices, 1)",
            "    i = tl.gather(i, indices, 1)"]
    step = config.output_rows_per_batch or config.output_stride
    lines += [f"    output_base = (row // {step}) * {olen * step} + row % {step}",
        f"    target = output_base[:, None] + frequency[None, :] * {step}",
        f"    valid = valid_row[:, None] & (frequency[None, :] < {olen})"]
    if config.mode == "c2r":
        lines += ["    tl.store(out_ptr + target, r, valid)"]
    else:
        lines += ["    tl.store(out_ptr + target * 2, r, valid)",
            "    tl.store(out_ptr + target * 2 + 1, i, valid)"]
    return name, "\n".join(lines) + "\n"
