"""Dense batch prefix: full FFT oracles, unchanged suffixes and emitter ABI.

Small execution batches deliberately exercise the public-batch generated
schedule's tail masks. They do not test C++ planner admission; the native
policy tests cover that separately. CPU dot is a correctly rounded layout
oracle, not a substitute for the native IEEE FP32 accuracy requirement.
"""
import ast
import itertools
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from flagfft_codegen.kernels_npu_dif import emit_vector_dif
from flagfft_codegen.kernels_npu_dif_cube import build_cube_prefix_dif
from flagfft_codegen.npu_batch_finish import replace_finish
from flagfft_codegen.npu_vector_basis import optimize_vector_basis
from test_npu_dif_cube import _tables
from test_npu_structured_fft import Pointer, execute_module


@pytest.fixture(autouse=True)
def selected_options(monkeypatch):
    for name in ('FUSED_FINISH', 'GROUP_BRANCHES', 'COMPONENT17', 'CODELET_REUSE',
                 'VECTOR_BASIS', 'PREFIX_PIPELINE'):
        monkeypatch.setenv('FLAGFFT_NPU_DIF_' + name, '1')


def _kwargs(direction, dense, columns, frequencies, global_inverse):
    return dict(vector_first17=True, batch_pack=True, vector_second15=True,
                global_inverse=global_inverse, matrix_first17=True,
                prefix_pitch=32, codelet_reuse=True, dense_prefix=dense,
                dense_prefix_columns=columns, dense_prefix_frequencies=frequencies)


def _build(factors, direction, dense=True, columns=64, frequencies=64, finish4=False, global_inverse=None):
    global_inverse = direction == 'inverse' if global_inverse is None else global_inverse
    source, specs = build_cube_prefix_dif(factors, direction,
                                         **_kwargs(direction, dense, columns, frequencies, global_inverse))
    if finish4:
        source, specs = replace_finish(source, specs, 'forward' if global_inverse else direction, global_inverse)
    return optimize_vector_basis(source, specs)


def _weights(factors, specs, direction, global_inverse=None):
    # The selected global-inverse schedule keeps its late butterflies forward.
    global_inverse = direction == 'inverse' if global_inverse is None else global_inverse
    tables = _tables(factors, specs, -1 if direction == 'forward' or global_inverse else 1)
    first = specs[0]
    if first.get('dense_prefix'):
        length, padded, _, frequencies, tiles = first['dense_prefix']
        values = []
        for tile in range(tiles):
            freq = tile * frequencies + torch.arange(frequencies, dtype=torch.float64)[:, None]
            digit = torch.arange(padded)[None, :]
            angle = (-1 if direction == 'forward' else 1) * 2 * math.pi * freq * digit / length
            valid = (freq < length // 2 + 1) & (digit < length)
            values.append(torch.cat((torch.where(valid, angle.cos(), 0).float(),
                                     torch.where(valid, angle.sin(), 0).float())).flatten())
        tables[0] = torch.cat(values)
    else:
        angle = (-1 if direction == 'forward' else 1) * 2 * math.pi * torch.arange(
            1, 9, dtype=torch.float64)[:, None] * torch.arange(1, 17)[None, :] / 17
        tables[0] = torch.cat((angle.cos().float(), angle.sin().float()))
    last = specs[-1]
    if last.get('finish_map'):
        rows, padded, initial = last['finish_rows'], last['padded_prefix'], last['initial_prefix']
        points = 4 * rows * padded
        word = torch.arange(2 * rows * padded)
        take = torch.where(word < 2 * rows * initial,
                           (word & 1) * points + word // (2 * initial) * padded
                           + (word // 2) % initial, 0).float()
        tables[-1] = torch.cat((tables[-1].flatten(), take))
    return tables


def _relative_per_batch(actual, expected):
    actual, expected = actual.cdouble().reshape(actual.shape[0], -1), expected.cdouble().reshape(expected.shape[0], -1)
    return torch.linalg.vector_norm(actual - expected, dim=1) / torch.linalg.vector_norm(expected, dim=1)


# Each row tests a distinct shape or ABI property without running a large
# Cartesian product. m=520 lies in the actual batch-dense selector's range.
_EXECUTION_CASES = [
    ((17, 15, 13, 2), 1, 64, 64, False, 'basis'),
    ((17, 15, 13, 3), 3, 64, 64, False, 'tone'),
    ((17, 15, 7, 8), 16, 64, 64, False, 'random'),
    ((17, 15, 13, 4), 64, 64, 64, False, 'random'),
    ((17, 15, 13, 8), 3, 64, 64, True, 'cancellation'),
    ((17, 15, 13, 10, 4), 1, 64, 64, False, 'random'),
    ((17, 15, 13, 3), 3, 32, 64, False, 'random'),
    ((17, 15, 13, 3), 3, 32, 128, False, 'random'),
]


@pytest.mark.parametrize('factors,batch,columns,frequencies,finish4,pattern', _EXECUTION_CASES)
@pytest.mark.parametrize('direction', ['forward', 'inverse'])
def test_complete_batch_dense_fft(tmp_path, factors, batch, columns, frequencies, finish4, pattern, direction, global_inverse=None):
    global_inverse = direction == 'inverse' if global_inverse is None else global_inverse
    source, specs = _build(factors, direction, columns=columns,
                           frequencies=frequencies, finish4=finish4, global_inverse=global_inverse)
    tables = _weights(factors, specs, direction, global_inverse=global_inverse)
    path = tmp_path / 'dense.py'
    path.write_text(source)
    namespace, pid = execute_module(path)
    pid.append(0)
    namespace['tl'].dot = lambda a, b, acc=None, **kw: (
        a.double() @ b.double() + (acc.double() if acc is not None else 0)).float()
    n, prefix = math.prod(factors), 255
    torch.manual_seed(20261110)
    x = torch.randn(batch, n, 2)
    if pattern == 'basis':
        x.zero_()
        x[:, n // 3, 1] = 1.
    elif pattern == 'tone':
        phase = 2 * math.pi * 37 * torch.arange(n, dtype=torch.float64) / n
        x = torch.view_as_real(torch.polar(torch.ones_like(phase), phase)).float()[None].expand(batch, n, 2).clone()
    elif pattern == 'cancellation':
        x[:, n // 2:2 * (n // 2)] = -x[:, :n // 2].flip(1)
        x[:, -1] *= 1e-6
    scales = torch.tensor([1., 1e-6, 1e6, -.25])[torch.arange(batch) % 4]
    x *= scales[:, None, None]
    saved = x.clone()
    z = torch.view_as_complex(x).cdouble()
    reference = torch.fft.fft(z) if direction == 'forward' else torch.fft.ifft(z) * n
    prefix_input = z.reshape(batch, prefix, n // prefix)
    prefix_reference = torch.fft.fft(prefix_input, dim=1) if direction == 'forward' else torch.fft.ifft(prefix_input, dim=1) * prefix
    if global_inverse:
        prefix_reference = prefix_reference.conj()
    prefix_reference = prefix_reference.transpose(1, 2)
    pn = max(spec.get('physical_n', n) for spec in specs)
    banks = [torch.full((2 * batch * pn + 256,), 21828.5) for _ in range(2)]
    output = torch.full((2 * batch * n + 256,), 31415.25)
    current = x
    for index, spec in enumerate(specs):
        final = index + 1 == len(specs)
        buf, canary = (output, 31415.25) if final else (banks[index % 2], 21828.5)
        buf.fill_(canary)
        launch = tuple(spec['grid']) + (batch,) if spec['phase'] == 'transpose' else (
            spec['grid'][0], math.ceil(batch / spec.get('batch_pack', 1)))
        for coordinate in itertools.product(*(range(value) for value in launch)):
            pid[:] = list(coordinate) + [0] * (3 - len(coordinate))
            arguments = [Pointer(current), Pointer(tables[index]), Pointer(buf, 128)]
            if spec.get('batch_pack', 1) > 1 or spec.get('finish_map'):
                arguments.append(batch)
            namespace[spec['name']](*arguments)
        assert (buf[:128] == canary).all() and (buf[-128:] == canary).all()
        current = buf[128:-128]
        if index == 0:
            prefix_pn = spec['physical_n']
            planes = current[:2 * batch * prefix_pn].reshape(batch, 2, n // prefix, 256)
            actual = torch.complex(planes[:, 0], planes[:, 1])
            assert (_relative_per_batch(actual[:, :, :prefix], prefix_reference) < 5e-7).all()
            assert (actual[:, :, prefix:] == 0).all()
    got = torch.view_as_complex(output[128:-128].reshape(batch, n, 2))
    assert torch.isfinite(got).all()
    assert (_relative_per_batch(got, reference) < 5e-7).all()
    assert torch.equal(x, saved)


def test_inverse_without_global_conjugation(tmp_path):
    test_complete_batch_dense_fft(tmp_path, (17, 15, 13, 10, 4), 1,
                                  64, 64, False, 'random', 'inverse', global_inverse=False)


@pytest.mark.parametrize('factors,finish4', [((17, 15, 13, 7, 8), True), ((17, 15, 13, 10, 10, 2), False)])
@pytest.mark.parametrize('direction', ['forward', 'inverse'])
def test_replacement_preserves_suffix_and_deferred_tables(factors, finish4, direction):
    control_source, control_specs = _build(factors, direction, dense=False, finish4=finish4)
    dense_source, dense_specs = _build(factors, direction, finish4=finish4)
    switch = next(i for i, spec in enumerate(control_specs) if spec['phase'] == 'transpose')
    assert dense_specs[1:] == control_specs[switch + 1:]
    assert dense_specs[1]['deferred_twiddle'] and dense_specs[1]['stage'] == 2
    definitions = lambda source: {node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    old, new = definitions(control_source), definitions(dense_source)
    for spec in dense_specs[1:]:
        assert new[spec['name']] == old[spec['name']]
    control_tables = _weights(factors, control_specs, direction)
    dense_tables = _weights(factors, dense_specs, direction)
    assert all(torch.equal(a, b) for a, b in zip(dense_tables[1:], control_tables[switch + 1:]))
    if finish4:
        assert dense_specs[-1]['finish_map'] and dense_specs[-1]['batch_pack'] == 4


@pytest.mark.parametrize('direction', ['forward', 'inverse'])
def test_dense_tiles_have_distinct_identity_and_no_prefix_pipeline(tmp_path, direction):
    factors = (17, 15, 13, 10, 10, 2)
    common = dict(aligned=True, cube_prefix=True, vector_first17=True, batch_pack=True,
                  vector_second15=True, matrix_first17=True, prefix_pitch=32,
                  codelet_reuse=True, vector_basis=True, global_inverse=direction == 'inverse')
    paths = []
    for columns, frequencies in ((32, 64), (64, 64), (32, 128)):
        meta = emit_vector_dif(factors, direction, 0, tmp_path, dense_prefix=True,
            dense_prefix_columns=columns, dense_prefix_frequencies=frequencies, **common)
        assert meta['kernel_name'] == 'dense_prefix_half'
        assert meta['num_stages'] == 1 and meta['batch_per_block'] == 1
        assert meta['signature'] == '*fp32:16,*fp32:16,*fp32:16'
        assert (meta['dense_prefix_columns'], meta['dense_prefix_frequencies']) == (columns, frequencies)
        assert json.loads(Path(meta['module_path']).with_suffix('.json').read_text()) == meta
        paths.append(meta['module_path'])
    assert len(set(paths)) == 3
    original = emit_vector_dif(factors, direction, 0, tmp_path, **common)
    assert original['kernel_name'] == 'vector_stage_0'
    assert original['num_stages'] == (2 if direction == 'forward' else 1)


def test_dense_finish_metadata_keeps_batch_abi(tmp_path):
    meta = emit_vector_dif((17, 15, 13, 7, 8), 'forward', 3, tmp_path,
        aligned=True, cube_prefix=True, vector_first17=True, batch_pack=True,
        vector_second15=True, matrix_first17=True, prefix_pitch=32,
        codelet_reuse=True, vector_basis=True, finish_batch4=True, dense_prefix=True,
        dense_prefix_columns=64, dense_prefix_frequencies=64)
    assert meta['kernel_name'] == 'fused_finish'
    assert meta['batch_per_block'] == 4 and meta['arg_names'][-1] == 'NB'
    assert meta['signature'].endswith(',i32')


@pytest.mark.parametrize('direction', ['forward', 'inverse'])
def test_scalar_dense_default_stays_32_by_32(direction):
    kwargs = dict(vector_first17=True, dense_prefix=True, global_inverse=direction == 'inverse')
    original = build_cube_prefix_dif((17, 15, 13, 2), direction, **kwargs)
    explicit = build_cube_prefix_dif((17, 15, 13, 2), direction,
        dense_prefix_columns=32, dense_prefix_frequencies=32, **kwargs)
    assert original == explicit
    assert original[1][0]['dense_prefix'][2:4] == (32, 32)


@pytest.mark.parametrize('columns,frequencies', [(31, 64), (32, 48), (64, 128)])
def test_unsupported_dense_tile_is_rejected(columns, frequencies):
    with pytest.raises(ValueError, match='Unsupported dense-prefix tile'):
        _build((17, 15, 13, 2), 'forward', columns=columns, frequencies=frequencies)
