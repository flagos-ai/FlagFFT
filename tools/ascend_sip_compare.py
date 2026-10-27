"""Paired complete-FFT execution comparison with SiP (resident device buffers).

Plan/table creation, compilation and H2D/D2H are outside all timed intervals.
Both normal API submission and equal-size device graphs are measured.
"""
import argparse
import ctypes
import hashlib
import json
import importlib.util
import sys
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
import triton
from ascend_fourstep_probe import fourstep_fused, axis_twiddles
from ascend_local_fft_codegen import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['baseline','local-first','local-both'], default='baseline')
    parser.add_argument('--pack',type=int,default=1)
    parser.add_argument('--factor',type=int,default=32)
    parser.add_argument('--radix',type=int,choices=[2,4],default=4)
    parser.add_argument('--single-buffer',action='store_true')
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n', type=int, default=2048)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--iters', type=int, default=20)
    args = parser.parse_args()
    n, batch, a = args.n, args.batch, args.factor
    b = n // a
    os.environ['TRITON_ALL_BLOCKS_PARALLEL'] = '1'
    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    cpu = torch.randn(batch, n, 2)
    x, y, z = cpu.npu(), torch.empty_like(cpu, device='npu'), torch.empty_like(cpu, device='npu')
    wa, wb = axis_twiddles(n, a, b), axis_twiddles(n, b, a)
    k = torch.arange(n, dtype=torch.int64)
    angle = -2 * math.pi * (k//b) * (k%b) / n
    cross = torch.cat((angle.cos(), angle.sin())).float().npu()
    options = {'enable_auto_blockify': True}
    if args.single_buffer:
        options['multibuffer'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_sha = None
    if args.variant == 'baseline':
        ck = fourstep_fused[(batch,)](x, y, wa, wb, cross, n, a, b,
                a.bit_length()-1, b.bit_length()-1, False, False, **options)
        runner = ck[(batch, 1, 1)]
        def fft():
            runner(x, y, wa, wb, cross, n, a, b, a.bit_length()-1, b.bit_length()-1, False, False)
    else:
        name, source = generate(n=n, factor=a, pack=args.pack,
                                both=args.variant=='local-both', radix=args.radix)
        path = args.output.with_suffix('.generated.py')
        path.write_text(source)
        generated_sha = hashlib.sha256(source.encode()).hexdigest()
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        kernel = getattr(module, name)
        if args.variant=='local-first':
            wb = axis_twiddles(n*args.pack,b,args.pack*a)
        grid = triton.cdiv(batch,args.pack)
        ck = kernel[(grid,)](x,y,cross,wb,batch,**options)
        runner = ck[(grid,1,1)]
        def fft():
            runner(x,y,cross,wb,batch)
    for key,data in ck.asm.items():
        path=args.output.with_suffix('.'+key)
        if isinstance(data,str): path.write_text(data)
        elif isinstance(data,(bytes,bytearray)): path.write_bytes(data)

    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.sip_error.restype = ctypes.c_char_p
    lib.sip_create.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    lib.sip_create.restype = ctypes.c_void_p
    lib.sip_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.sip_run.restype = ctypes.c_int
    lib.sip_destroy.argtypes = [ctypes.c_void_p]
    handle = lib.sip_create(n, batch, x.data_ptr(), z.data_ptr())
    if not handle: raise RuntimeError(lib.sip_error().decode())
    def sip():
        if lib.sip_run(handle, torch.npu.current_stream().npu_stream):
            raise RuntimeError(lib.sip_error().decode())
    ref = torch.fft.fft(torch.view_as_complex(cpu), dim=-1)
    def check(output):
        got = torch.view_as_complex(output.cpu())
        rel = ((got-ref).abs().square().sum()/ref.abs().square().sum()).sqrt().item()
        assert torch.allclose(got, ref, atol=3e-4, rtol=3e-5), rel
        return rel
    fft(); sip(); torch.npu.synchronize()
    errors = {'fourstep': check(y), 'sip': check(z)}
    assert torch.equal(x.cpu(), cpu), 'SiP modified the input'
    graphs = {}
    for name, launch, output in [('fourstep', fft, y), ('sip', sip, z)]:
        for _ in range(10): launch()
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            for _ in range(args.steps): launch()
        output.fill_(float('nan'))
        g.replay(); torch.npu.synchronize(); check(output)
        graphs[name] = g

    launches = {'fourstep': fft, 'sip': sip}
    results = {}
    for mode in ('normal', 'graph'):
        samples = {name: [] for name in launches}
        wall = {name: [] for name in launches}
        for r in range(args.rounds):
            order = ('fourstep', 'sip') if r % 2 == 0 else ('sip', 'fourstep')
            for name in order:
                launch = launches[name] if mode == 'normal' else graphs[name].replay
                factor = 1 if mode == 'normal' else args.steps
                for _ in range(5): launch()
                torch.npu.synchronize()
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                t0 = time.perf_counter()
                start.record()
                for _ in range(args.iters): launch()
                end.record(); end.synchronize()
                wall[name].append((time.perf_counter()-t0)*1e6/(args.iters*factor))
                samples[name].append(start.elapsed_time(end)*1000/(args.iters*factor))
        med = {k: statistics.median(v) for k,v in samples.items()}
        wm = {k: statistics.median(v) for k,v in wall.items()}
        results[mode] = dict(device_us=med, host_and_device_us=wm,
                speedup_sip_over_fourstep=med['sip']/med['fourstep'],
                wall_speedup_sip_over_fourstep=wm['sip']/wm['fourstep'],
                device_trials_us=samples, wall_trials_us=wall)
    errors.update(fourstep_after=check(y), sip_after=check(z))
    assert torch.equal(x.cpu(), cpu), 'input changed during replay'
    lib.sip_destroy(handle)
    result = dict(variant=args.variant,pack=args.pack,factor=a,radix=args.radix,single_buffer=args.single_buffer,generated_sha256=generated_sha,status='pass', n=n, batch=batch, seed=args.seed,
                  graph_steps=args.steps, rounds=args.rounds, iters=args.iters,
                  results=results, rel_l2=errors, replay_verified=True,
                  scope='complete FFT execution on resident device buffers; plans/tables/JIT/copies excluded',
                  product_api_integrated=False, triton=triton.__file__, kernel_hash=ck.hash,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
