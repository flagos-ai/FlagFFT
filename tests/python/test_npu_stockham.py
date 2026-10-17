"""The table stage keeps native grouping, sum order, boundaries and ABI."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from flagfft_codegen.kernels_npu_stockham import (
    build_npu_table_stockham_source as kernel_source,
    npu_stockham_layout as parameters,
)
from test_npu_grouped_fft64 import Language, Pointer, insert_slice


def tables(n, r, span, inverse=False):
    p = parameters(n, r, span)
    sign = 1 if inverse else -1
    roots = np.zeros((2, r, 24), np.float32)
    angle = sign * 2 * np.pi * np.arange(r)[:, None] * np.arange(r)[None, :] / r
    roots[0, :, :r] = np.cos(angle)
    roots[1, :, :r] = np.sin(angle)
    tw = np.zeros((r, p['twiddle_stride'], 2), np.float32)
    angle = sign * 2 * np.pi * np.arange(r)[:, None] * (np.arange(p['twiddle_stride']) % span)[None, :] / (r * span)
    tw[:, :, 0] = np.cos(angle)
    tw[:, :, 1] = np.sin(angle)
    indices = np.zeros(5 * 128, np.uint32)
    indices[:128] = 2 * np.arange(128) * 4
    indices[128:256] = (2 * np.arange(128) + 1) * 4
    indices[256:512:2] = np.arange(128) * 4
    indices[257:512:2] = (128 + np.arange(128)) * 4
    indices[512:531] = np.arange(19) * 24 * 4
    lanes = p['store_lanes']
    stride = (2 * r * lanes + 7) & ~7
    store = np.zeros(p['batch_group'] * stride if p['contiguous'] else 1, np.uint32)
    if p['contiguous']:
        for b in range(p['batch_group']):
            for q in range(r):
                for lane in range(lanes):
                    dst = lane * r + q if span == 1 else q * lanes + lane
                    src = q * 128 + b * lanes + lane
                    store[b * stride + 2 * dst] = src * 4
                    store[b * stride + 2 * dst + 1] = (r * 128 + src) * 4
    return tw.reshape(-1), roots.reshape(-1), indices, store


def expected(data, r, span, tw, roots):
    batch, n, _ = data.shape
    nd = n // r
    k = np.arange(nd)
    j = k % span
    tw = tw.reshape(r, -1, 2)
    roots = roots.reshape(2, r, 24)
    x = data.reshape(batch, r, nd, 2).copy()
    if span > 1:
        for digit in range(1, r):
            a, b = x[:, digit, :, 0].copy(), x[:, digit, :, 1].copy()
            cr, ci = tw[digit, j, 0], tw[digit, j, 1]
            x[:, digit, :, 0] = a * cr - b * ci
            x[:, digit, :, 1] = a * ci + b * cr
    result = np.empty_like(data)
    for q in range(r):
        ar = np.zeros((batch, nd), np.float32)
        ai = np.zeros_like(ar)
        for digit in range(r):
            xr, xi = x[:, digit, :, 0], x[:, digit, :, 1]
            if digit:
                cr, ci = roots[0, q, digit], roots[1, q, digit]
                ar = ar + (xr * cr - xi * ci)
                ai = ai + (xr * ci + xi * cr)
            else:
                ar = ar + xr
                ai = ai + xi
        dst = r * k - (r - 1) * j + q * span
        result[:, dst, 0] = ar
        result[:, dst, 1] = ai
    return result


class Grid(Language):
    minimum = staticmethod(np.minimum)
    where = staticmethod(np.where)
    blocks = 1

    def num_programs(self, axis):
        assert axis == 0
        return self.blocks


@pytest.mark.parametrize('n,r,span,batch', [
    (48,12,1,35),(48,2,12,35),(48,2,24,7),
    (46189,11,1,1),(46189,13,11,1),(46189,17,143,1),(46189,19,2431,1),
    (512,4,8,17),(1024,8,16,3),(8192,4,256,2),
    (131072,8,1,1),(131072,4,8,3),(131072,8,32,3),
    (131072,4,256,1),(131072,8,1024,1),(131072,4,8192,1),(131072,4,32768,1),
    (46189,19,1,1),(46189,17,19,1),(46189,13,323,1),(46189,11,4199,1),
])
@pytest.mark.parametrize('inverse',[False,True])
@pytest.mark.parametrize('compact_loop',[False,True])
def test_table_stockham_keeps_native_bits_and_grid_stride(n,r,span,batch,inverse,compact_loop):
    params = parameters(n,r,span)
    rng = np.random.default_rng(20261018)
    data = rng.standard_normal((batch,n,2)).astype(np.float32)
    tw,roots,indices,store = tables(n,r,span,inverse)
    reference = expected(data,r,span,tw,roots)
    name,body = kernel_source(n,r,span,compact_loop=compact_loop)
    tree = ast.parse(body); tree.body[0].decorator_list=[]
    language = Grid()
    namespace = dict(tl=language,tle=SimpleNamespace(dsa=SimpleNamespace(insert_slice=insert_slice)),
        _cmul=lambda ar,ai,br,bi:(ar*br-ai*bi,ar*bi+ai*br))
    exec(compile(tree,name,'exec'),namespace)
    guard,marker = 64,np.float32(31415.25)
    output = np.full(data.size+2*guard,marker,np.float32)
    language.blocks = min(3,(batch+params['batch_group']-1)//params['batch_group']*params['tiles'])
    for pid in range(language.blocks):
        language.pid=pid
        namespace[name](Pointer(data.reshape(-1)),Pointer(output,guard),Pointer(tw),Pointer(roots),
            Pointer(indices),Pointer(store),n,r,span,batch)
    np.testing.assert_array_equal(output[guard:-guard].view(np.uint32),reference.reshape(-1).view(np.uint32))
    assert np.all(output[:guard]==marker) and np.all(output[-guard:]==marker)


@pytest.mark.parametrize('n,r,span',[(0,2,1),(48,1,1),(48,20,1),(48,2,0),(49,2,1),(48,2,5),(2**32,2,1)])
def test_table_stockham_rejects_invalid_parameters(n,r,span):
    with pytest.raises(ValueError):
        kernel_source(n,r,span)


@pytest.mark.parametrize('backend,dtype',[('npu','complex64'),('cuda','complex64'),
    ('ix','complex64'),('maca','complex64'),('npu','complex128')])
@pytest.mark.parametrize('compact_loop',[False,True])
def test_table_stockham_cli_scope_and_abi(tmp_path,backend,dtype,compact_loop):
    repo=Path(__file__).resolve().parents[2]
    command=[sys.executable,'-m','flagfft_codegen.jit_source','--kernel','npu_table_stockham',
        '--length','131072','--factors','4,8','--direction','forward','--dtype',dtype,
        '--target',f'{backend}:source-contract:32','--out-dir',str(tmp_path)]
    if compact_loop:
        command.append('--npu-stockham-loop')
    result=subprocess.run(command,text=True,capture_output=True,
        env=dict(os.environ,PYTHONPATH=str(repo/'python'),PYTHONDONTWRITEBYTECODE='1'))
    if backend!='npu' or dtype!='complex64':
        assert result.returncode!=0 and 'NPU FP32 target' in result.stderr
        return
    assert result.returncode==0,result.stderr
    metadata=json.loads(result.stdout)
    assert metadata.get('compact_loop',False)==compact_loop
    assert metadata['signature']==','.join(['*fp32:16']*4+['*u32:16']*2+['i32']*4)
    assert metadata['tile']==128 and metadata['batch_group']==16 and metadata['tiles_per_transform']==4096
    assert metadata['num_warps']==4 and metadata['num_stages']==1
    source=Path(metadata['module_path']).read_text()
    assert 'tle.dsa.insert_slice' in source and 'tl.num_programs(0)' in source
