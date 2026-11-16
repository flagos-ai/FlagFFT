# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Execute generated shared accesses and enforce the overwrite happens-before rule."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flagfft_codegen.backend_profile import BackendProfile, reset_profile, set_profile
from flagfft_codegen.emit import emit_jit_kernel
from flagfft_codegen.kernels_common import (LeafPlan, cooperative_stage_lanes_for,
                                           hcu_convolution_single_buffer)


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype)


def tensor(value):
    return np.asarray(value).view(Tensor)


class Shared:
    def __init__(self):
        self.written = set()
        self.read = set()
        self.reading = False
        self.fenced = False
        self.overwrites = 0

    def load(self, indices):
        selected = set(map(int, indices))
        assert selected <= self.written, "read of an uninitialized shared value"
        self.read |= selected
        self.reading = True

    def store(self, indices):
        if self.reading:
            assert self.fenced and self.written <= self.read, "shared overwrite without a complete fenced read"
            self.reading = False
            self.fenced = False
            self.read.clear()
            self.written.clear()
            self.overwrites += 1
        self.written |= set(map(int, indices))

    def barrier(self):
        if self.reading and self.written <= self.read:
            self.fenced = True


class Pointer:
    def __init__(self, data, offset=0, shared=None):
        self.data = np.asarray(data).reshape(-1)
        self.offset = offset
        self.shared = shared

    def __add__(self, offset):
        return Pointer(self.data, self.offset + offset, self.shared)

    def __sub__(self, offset):
        return self + (-offset)

    def to(self, dtype):
        return Pointer(self.data.view(dtype), self.offset, self.shared)


class TL:
    float32, float64 = np.float32, np.float64
    int32, int64, uint32, uint64 = np.int32, np.int64, np.uint32, np.uint64
    constexpr = int
    static_range = staticmethod(range)
    range = staticmethod(lambda *args, **unused: range(*args))
    arange = staticmethod(lambda *a: tensor(np.arange(*a)))
    full = staticmethod(lambda *a, **k: tensor(np.full(*a, **k)))
    zeros_like = staticmethod(lambda x: tensor(np.zeros_like(x)))
    reshape = staticmethod(lambda x, s: tensor(np.reshape(x, s)))
    trans = staticmethod(lambda x: tensor(np.transpose(x)))
    join = staticmethod(lambda a, b: tensor(np.stack((a, b), axis=-1)))
    gather = staticmethod(lambda a, i, axis=0: tensor(np.take_along_axis(a, i, axis)))
    where = staticmethod(lambda c, a, b: tensor(np.where(c, a, b)))
    minimum = staticmethod(lambda a, b: tensor(np.minimum(a, b)))
    maximum = staticmethod(lambda a, b: tensor(np.maximum(a, b)))
    pointer_type = staticmethod(lambda dtype: dtype)
    program_id = staticmethod(lambda axis: tensor(0))
    multiple_of = staticmethod(lambda value, hint: value)
    max_contiguous = multiple_of

    @staticmethod
    def cast(value, dtype, bitcast=False):
        if isinstance(value, Pointer):
            return Pointer(value.data.view(dtype), value.offset, value.shared)
        a = np.asarray(value)
        return tensor(a.view(dtype) if bitcast else a.astype(dtype))


class Trace:
    def __init__(self):
        self.buffers = []

    def alloc(self, shape, dtype, **unused):
        state = Shared()
        self.buffers.append(state)
        return Pointer(np.zeros(shape, dtype=dtype), shared=state)

    def barrier(self):
        for buffer in self.buffers:
            buffer.barrier()

    @staticmethod
    def load(ptr, mask=True, other=0):
        index, valid = np.broadcast_arrays(ptr.offset, mask)
        selected = index[valid].astype(np.int64)
        if ptr.shared:
            ptr.shared.load(selected.ravel())
        result = np.full(index.shape, other, dtype=ptr.data.dtype)
        result[valid] = ptr.data[selected]
        return tensor(result)

    @staticmethod
    def store(ptr, value, mask=True):
        index, values, valid = np.broadcast_arrays(ptr.offset, value, mask)
        selected = index[valid].astype(np.int64)
        if ptr.shared:
            ptr.shared.store(selected.ravel())
        ptr.data[selected] = values[valid]


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("n,factors,lanes", [(64, (4, 4, 4), 16),
    (256, (16, 16), 16), (2048, (16, 8, 16), 256)])
def test_generated_full_convolution_fences_every_shared_overwrite(tmp_path, monkeypatch, dtype, n, factors, lanes):
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_CONV_SINGLE_BUFFER", "1")
    monkeypatch.setenv("FLAGFFT_HCU_VECTOR_IO", "0")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "0")
    try:
        plan = LeafPlan(n, factors, 1, lanes, 8, (), n, dtype=dtype)
        assert hcu_convolution_single_buffer(plan, "bluestein_full_leaf")
        widths = cooperative_stage_lanes_for(plan)
        meta = emit_jit_kernel(kernel="leaf_bluestein", length=n, factors=factors, lanes=lanes,
            num_warps=8, generic_radices=(), smem_size=n, direction="forward", dtype=dtype,
            prime_n=n//2-3, four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
    finally:
        reset_profile(token)
    source = Path(meta["module_path"]).read_text()
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    main = next(fn for fn in functions if fn.name == meta["kernel_name"])
    assert not any(isinstance(node, ast.Name) and node.id.startswith("smem_a") for node in ast.walk(main))
    scalar = np.float64 if dtype == "complex128" else np.float32
    complex_dtype = np.complex128 if dtype == "complex128" else np.complex64
    prime = n//2-3
    rng = np.random.default_rng(29)
    data = (rng.normal(size=prime)+1j*rng.normal(size=prime)).astype(complex_dtype)
    positions = np.arange(prime, dtype=np.int64)
    # Bound the phase before conversion, as the host's modular chirp does.
    phase = np.pi*((positions*positions) % (2*prime))/prime
    chirp = np.exp(-1j*phase).astype(complex_dtype)
    b_time = np.zeros(n, dtype=complex_dtype)
    b_time[:prime] = np.conj(chirp)
    b_time[n-positions[1:]] = np.conj(chirp[1:])
    b_fft = np.fft.fft(b_time).astype(complex_dtype)
    tables = {}
    prefix = factors[0]
    for stage in range(1, len(factors)):
        width, radix = widths[stage], factors[stage]
        physical = np.arange(n)
        lane, element = physical % width, physical // width
        codelet = lane+width*(element//radix)
        angle = -2*np.pi*(codelet % prefix)*(element % radix)/(prefix*radix)
        tables[f"tw{stage}_r_ptr"] = np.cos(angle).astype(scalar)
        tables[f"tw{stage}_i_ptr"] = np.sin(angle).astype(scalar)
        prefix *= radix

    def execute(remove_barriers=False):
        trace = Trace()
        runtime = type("Runtime", (TL,), {"load": staticmethod(trace.load),
            "store": staticmethod(trace.store), "debug_barrier": staticmethod(trace.barrier)})
        gpu = SimpleNamespace(alloc=trace.alloc, smem="shared",
                              local_ptr=lambda ptr, indices: ptr + indices[0])
        tree = ast.parse(ast.unparse(ast.Module(body=functions, type_ignores=[])))
        if remove_barriers:
            class RemoveBarriers(ast.NodeTransformer):
                def visit_Expr(self, node):
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "debug_barrier":
                        return ast.copy_location(ast.Pass(), node)
                    return node
            tree = RemoveBarriers().visit(tree)
        namespace = {"tl": runtime, "tle": SimpleNamespace(gpu=gpu)}
        exec(compile(tree, "<native-full-convolution-shared-trace>", "exec"), namespace)
        output = np.full(prime, np.nan, dtype=complex_dtype)
        values = {"in_ptr": data.view(scalar), "out_ptr": output.view(scalar),
                  "chirp_ptr": chirp.view(scalar), "b_fft_ptr": b_fft.view(scalar), **tables}
        argv = [1 if name == "nbatch" else Pointer(values[name]) for name in meta["arg_names"]]
        namespace[meta["kernel_name"]](*argv)
        return trace, output

    trace, output = execute()
    assert len(trace.buffers) == 2
    assert all(buffer.overwrites == 2*len(factors)-2 for buffer in trace.buffers)
    expected = np.fft.fft(data)
    relative = np.linalg.norm(output-expected)/np.linalg.norm(expected)
    assert relative < (2e-14 if dtype == "complex128" else 5e-7)
    with pytest.raises(AssertionError, match="shared overwrite without a complete fenced read"):
        execute(remove_barriers=True)


def test_partial_convolution_keeps_ping_pong_storage(tmp_path, monkeypatch):
    """A small-radix stage cannot overwrite values another group still needs."""
    token = set_profile(BackendProfile(backend="hcu", device_arch="gfx936", warp_size=64,
                                     max_dynamic_shared_memory=65536, policy="native"))
    monkeypatch.setenv("FLAGFFT_HCU_CONV_SINGLE_BUFFER", "1")
    monkeypatch.setenv("FLAGFFT_HCU_VECTOR_IO", "0")
    monkeypatch.setenv("FLAGFFT_HCU_DENSE_WAVE", "0")
    try:
        plan = LeafPlan(2048, (32, 32, 2), 1, 128, 4, (), 2048, dtype="complex128")
        assert not hcu_convolution_single_buffer(plan, "bluestein_full_leaf")
        meta = emit_jit_kernel(kernel="leaf_bluestein", length=2048, factors=plan.factors,
            lanes=128, num_warps=4, generic_radices=(), smem_size=2048,
            direction="forward", dtype="complex128", prime_n=997,
            four_step_n1=0, four_step_n2=0, out_dir=tmp_path)
    finally:
        reset_profile(token)
    functions = [node for node in ast.parse(Path(meta["module_path"]).read_text()).body
                 if isinstance(node, ast.FunctionDef)]
    main = next(fn for fn in functions if fn.name == meta["kernel_name"])
    allocations = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "alloc"]
    assert len(allocations) == 4
