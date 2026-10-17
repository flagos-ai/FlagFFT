"""Vector FFT exchange expressed through tensor split/join and permutations."""

from dataclasses import asdict
import hashlib
import json
import math


def gather_twiddle_values(config):
    """Dense stage roots; each vector is loaded once before local broadcast."""
    values = []
    span = config.length
    sign = -1 if config.direction == "forward" else 1
    while span > 1:
        half = span // 2
        for frequency in range(config.length):
            angle = sign * 2 * math.pi * (frequency % half) / span
            values.extend((math.cos(angle), math.sin(angle)))
        span = half
    return values


def build_gather_tensor_fft(config):
    n, pack = config.length, config.rows_per_program
    if n < 2 or n & (n - 1):
        raise ValueError("gather tensor exchange requires a power-of-two length >=2")
    identity = hashlib.sha256(json.dumps({"version": 2, **asdict(config)}, sort_keys=True).encode()).hexdigest()[:16]
    name = f"_tensor_gather_n{n}_{config.mode}_{identity}"
    ilen = n // 2 + 1 if config.mode == "c2r" else n
    olen = n // 2 + 1 if config.mode == "r2c" else n
    extent, bits = config.row_tile_extent, n.bit_length() - 1
    lines = ["@triton.jit", f"def {name}(in_ptr, out_ptr, twiddle_ptr, nbatch, program_offset):",
        "    program = program_offset + tl.program_id(0).to(tl.int64)",
        f"    offset = tl.arange(0, {pack * n})",
        f"    lane, frequency = offset // {n}, offset % {n}"]
    if extent:
        tiles = (extent + pack - 1) // pack
        lines += [f"    inner_row = (program % {tiles}) * {pack} + lane",
            f"    row = (program // {tiles}) * {extent} + inner_row",
            f"    valid_row = (row < nbatch) & (inner_row < {extent})"]
    else:
        lines += [f"    row = program * {pack} + lane", "    valid_row = row < nbatch"]
    lines += [f"    input_base = (row // {config.input_stride}) * {ilen * config.input_stride} + row % {config.input_stride}"]
    if config.mode == "c2r":
        lines += [f"    reflected = frequency > {n // 2}",
            f"    source_frequency = tl.where(reflected, {n} - frequency, frequency)"]
    else:
        lines += ["    source_frequency = frequency"]
    lines += [f"    source = input_base + source_frequency * {config.input_stride}"]
    if config.mode == "r2c":
        lines += ["    r = tl.load(in_ptr + source, valid_row, 0.0)",
            f"    i = tl.full(({pack * n},), 0.0, tl.float32)"]
    else:
        lines += ["    point = tl.load(in_ptr + source[:, None] * 2 + tl.arange(0, 2)[None, :], valid_row[:, None], 0.0)",
            "    r, i = tl.split(point)"]
        if config.mode == "c2r":
            lines += ["    i = tl.where(reflected, -i, i)",
                f"    i = tl.where((source_frequency == 0) | (source_frequency == {n // 2}), 0.0, i)"]
    lines += [f"    root_frequency = tl.arange(0, {n})"]
    for stage in range(bits):
        half = 1 << (bits - 1 - stage)
        lines += [f"    partner = offset ^ {half}",
            "    pr, pi = tl.gather(r, partner, 0), tl.gather(i, partner, 0)",
            f"    lower = (frequency & {half}) == 0",
            "    sum_r, sum_i = r + pr, i + pi",
            "    difference_r = tl.where(lower, r - pr, pr - r)",
            "    difference_i = tl.where(lower, i - pi, pi - i)",
            f"    root = ({stage * n} + root_frequency) * 2",
            "    point = tl.load(twiddle_ptr + root[:, None] + tl.arange(0, 2)[None, :])",
            "    root_r, root_i = tl.split(point)",
            "    wr, wi = tl.gather(root_r, frequency, 0), tl.gather(root_i, frequency, 0)",
            "    product_r = difference_r * wr - difference_i * wi",
            "    product_i = difference_r * wi + difference_i * wr",
            "    r, i = tl.where(lower, sum_r, product_r), tl.where(lower, sum_i, product_i)"]
    reversed_index = " | ".join(f"(((frequency >> {bit}) & 1) << {bits - 1 - bit})" for bit in range(bits))
    lines += [f"    reversed_index = {reversed_index}",
        f"    indices = lane * {n} + reversed_index",
        f"    r = tl.reshape(tl.gather(r, indices, 0), ({pack}, {n}))",
        f"    i = tl.reshape(tl.gather(i, indices, 0), ({pack}, {n}))"]
    step = config.output_rows_per_batch or config.output_stride
    lines += [f"    output_base = (row // {step}) * {olen * step} + row % {step}",
        f"    target = tl.reshape(output_base + frequency * {step}, ({pack}, {n}))",
        f"    valid = tl.reshape(valid_row & (frequency < {olen}), ({pack}, {n}))"]
    if config.mode == "c2r":
        lines += ["    tl.store(out_ptr + target, r, valid)"]
    else:
        lines += ["    tl.store(out_ptr + target * 2, r, valid)",
            "    tl.store(out_ptr + target * 2 + 1, i, valid)"]
    return name, "\n".join(lines) + "\n"


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
