"""Bounded C API validation for the Ascend Direct DFT baseline.

Run in the documented CANN 9 container. JSON is written to stdout, so callers
can archive results outside the source tree. No performance claims are made.
"""

import argparse
import ctypes
import json

import numpy as np
import torch
import torch_npu  # noqa: F401

from ascend_plain_mvp_smoke import _load_api


def check(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation}: status={code}")


def signals(n, batch):
    rng = np.random.default_rng(20260916 + n + batch)
    random = (rng.standard_normal((batch, n)) +
              1j * rng.standard_normal((batch, n))).astype(np.complex64)
    impulse = np.zeros((batch, n), np.complex64)
    impulse[:, 0] = np.arange(1, batch + 1) / batch
    tone = np.broadcast_to(np.exp(2j * np.pi * np.arange(n) / n),
                           (batch, n)).astype(np.complex64).copy()
    return {"random": random, "impulse": impulse,
            "constant": np.ones((batch, n), np.complex64), "tone": tone}


def run_case(api, handle_type, n, batch, stream, repeats):
    plan = handle_type()
    check(api.flagfftPlan1d(ctypes.byref(plan), n, 0x29, batch), "plan")
    records = []
    try:
        check(api.flagfftSetStream(plan, ctypes.c_void_p(stream.npu_stream)
                                  if stream is not None else None), "set_stream")
        for pattern, source in signals(n, batch).items():
            source_tensor = torch.from_numpy(source.view(np.float32).copy()).to("npu")
            # Aligned guard zones; NaNs expose omitted stores as well as bad results.
            storage = torch.full((batch * n * 2 + 32,), float("nan"), device="npu")
            output = storage[16:-16].view(batch, n * 2)
            torch.npu.synchronize()  # Explicit producer dependency for raw ACL streams.
            for direction in (-1, 1):
                expected = (np.fft.fft(source.astype(np.complex128), axis=1)
                            if direction == -1 else
                            n * np.fft.ifft(source.astype(np.complex128), axis=1))
                for _ in range(repeats):
                    check(api.flagfftExecC2C(plan, source_tensor.data_ptr(),
                                            output.data_ptr(), direction), "execute")
                if stream is None:
                    torch.npu.synchronize()
                else:
                    stream.synchronize()
                actual = output.cpu().numpy().copy().view(np.complex64)
                error = np.abs(actual - expected)
                relative_l2 = float(np.linalg.norm(error) /
                                    max(np.linalg.norm(expected), 1e-30))
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
                if relative_l2 > 3e-5:
                    raise AssertionError(f"relative L2={relative_l2}")
                guards = storage.cpu().numpy()
                assert np.isnan(guards[:16]).all() and np.isnan(guards[-16:]).all()
                np.testing.assert_array_equal(source_tensor.cpu().numpy(),
                                              source.view(np.float32))
                records.append(dict(n=n, batch=batch, pattern=pattern,
                                    direction=direction, stream="user" if stream else "default",
                                    repeats=repeats, max_abs=float(error.max()),
                                    relative_l2=relative_l2))
        return records
    finally:
        torch.npu.synchronize()
        check(api.flagfftDestroy(plan), "destroy")


def rejected_cases(api, handle_type):
    records = []

    def reject(name, call):
        plan = handle_type()
        code = call(ctypes.byref(plan))
        if code == 0:
            api.flagfftDestroy(plan)
        if code != 14 or plan.value is not None:
            raise AssertionError(f"{name}: status={code}, handle={plan.value}")
        records.append(name)

    for n in (129, 256, 65536):
        reject(f"length_{n}", lambda p: api.flagfftPlan1d(p, n, 0x29, 1))
    for dtype in (0x2a, 0x2c, 0x69):
        reject(f"type_{dtype}", lambda p: api.flagfftPlan1d(p, 8, dtype, 1))
    reject("rank2", lambda p: api.flagfftPlan2d(p, 8, 8, 0x29))
    dim = (ctypes.c_int * 1)(8)
    reject("stride2", lambda p: api.flagfftPlanMany(
        p, 1, dim, dim, 2, 16, dim, 2, 16, 0x29, 1))
    reject("padded_distance", lambda p: api.flagfftPlanMany(
        p, 1, dim, dim, 1, 16, dim, 1, 16, 0x29, 1))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 3, 65, 257])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.lengths + args.batches) < 1:
        parser.error("lengths, batches and repeats must be positive")
    torch.npu.set_device(args.device)
    api, handle_type = _load_api(args.library)
    api.flagfftSetStream.argtypes = [handle_type, ctypes.c_void_p]
    api.flagfftSetStream.restype = ctypes.c_int
    int_ptr = ctypes.POINTER(ctypes.c_int)
    api.flagfftPlanMany.argtypes = [ctypes.POINTER(handle_type), ctypes.c_int,
                                   int_ptr, int_ptr, ctypes.c_int, ctypes.c_int,
                                   int_ptr, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int]
    api.flagfftPlanMany.restype = ctypes.c_int
    import triton
    print(json.dumps(dict(event="environment", torch=torch.__version__,
                          torch_npu=torch_npu.__version__, triton=triton.__version__,
                          device=torch.npu.get_device_name(args.device))), flush=True)
    count = 0
    for n in args.lengths:
        for batch in args.batches:
            for stream in (None, torch.npu.Stream(device=args.device)):
                records = run_case(api, handle_type, n, batch, stream, args.repeats)
                for record in records:
                    print(json.dumps(record), flush=True)
                count += len(records)
    rejected = rejected_cases(api, handle_type)
    print(json.dumps(dict(event="summary", status="PASS", accuracy_checks=count,
                          rejected=rejected)), flush=True)


if __name__ == "__main__":
    main()
