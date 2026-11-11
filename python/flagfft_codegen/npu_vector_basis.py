# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded AIV butterfly alternatives for public batch complex DIF plans.

Use direct quadratic products in radix13 and a mixed cosine/sine radix7.
Keep the latter in either a full spatial vector or four contiguous j rows;
other batch layouts retain the established codelet.
"""
import ast
import math
from .kernels_leaf import _emit_natural_order_codelet_call

_RADIX7_SOURCE = '@triton.jit\ndef _npu_vector_rad7(r0, r1, r2, r3, r4, r5, r6, i0, i1, i2, i3, i4, i5, i6):\n    p1r = r1 + r6\n    m1r = r1 - r6\n    p2r = r2 + r5\n    m2r = r2 - r5\n    p3r = r3 + r4\n    m3r = r3 - r4\n    totalr = p1r + p2r + p3r\n    y0r = r0 + totalr\n    baser = r0 - totalr * (1.0 / 6.0)\n    ur = 0.7901564685254008 * (p1r - p3r)\n    vr = 0.05585426728964815 * (p3r - p2r)\n    wr = 0.7343022012357524 * (p2r - p1r)\n    c1r = baser + ur + vr\n    c2r = baser - ur - wr\n    c3r = baser + wr - vr\n    s1r = 0.7818314824680298 * m1r + 0.9749279121818236 * m2r + 0.43388373911755823 * m3r\n    s2r = 0.9749279121818236 * m1r + -0.433883739117558 * m2r + -0.7818314824680299 * m3r\n    s3r = 0.43388373911755823 * m1r + -0.7818314824680299 * m2r + 0.9749279121818237 * m3r\n    p1i = i1 + i6\n    m1i = i1 - i6\n    p2i = i2 + i5\n    m2i = i2 - i5\n    p3i = i3 + i4\n    m3i = i3 - i4\n    totali = p1i + p2i + p3i\n    y0i = i0 + totali\n    basei = i0 - totali * (1.0 / 6.0)\n    ui = 0.7901564685254008 * (p1i - p3i)\n    vi = 0.05585426728964815 * (p3i - p2i)\n    wi = 0.7343022012357524 * (p2i - p1i)\n    c1i = basei + ui + vi\n    c2i = basei - ui - wi\n    c3i = basei + wi - vi\n    s1i = 0.7818314824680298 * m1i + 0.9749279121818236 * m2i + 0.43388373911755823 * m3i\n    s2i = 0.9749279121818236 * m1i + -0.433883739117558 * m2i + -0.7818314824680299 * m3i\n    s3i = 0.43388373911755823 * m1i + -0.7818314824680299 * m2i + 0.9749279121818237 * m3i\n    y1r, y6r = (c1r + s1i, c1r - s1i)\n    y1i, y6i = (c1i - s1r, c1i + s1r)\n    y2r, y5r = (c2r + s2i, c2r - s2i)\n    y2i, y5i = (c2i - s2r, c2i + s2r)\n    y3r, y4r = (c3r + s3i, c3r - s3i)\n    y3i, y4i = (c3i - s3r, c3i + s3r)\n    return (y0r, y1r, y2r, y3r, y4r, y5r, y6r, y0i, y1i, y2i, y3i, y4i, y5i, y6i)\n'

def _quadratic_products(fn):
    out = []
    i = 0
    while i < len(fn.body):
        a = fn.body[i]
        if isinstance(a, ast.Assign) and isinstance(a.targets[0], ast.Name) and a.targets[0].id.endswith('m') and isinstance(a.value, ast.BinOp) and isinstance(a.value.op, ast.Mult) and isinstance(a.value.right, ast.BinOp) and isinstance(a.value.right.op, ast.Sub):
            b = ast.literal_eval(a.value.left)
            u = ast.unparse(a.value.right.left)
            v = ast.unparse(a.value.right.right)
            zu, zv = fn.body[i + 1:i + 3]
            cu = ast.literal_eval(zu.value.left.left)
            cv = ast.literal_eval(zv.value.left.left)
            text = f'{zu.targets[0].id}={cu + b!r}*{u}-{b!r}*{v}\n{zv.targets[0].id}={b!r}*{u}+{cv - b!r}*{v}'
            out.extend(ast.parse(text).body)
            i += 3
            continue
        out.append(a)
        i += 1
    fn.body = out
    return fn

def _spatial_radix7(s, direction):
    jpack = 4
    pp, pn, m = (s['physical_prefix'], s['physical_n'], s['m'])
    points = jpack * pp
    lines = ['@triton.jit', f"def {s['name']}(X,TW,Y):", '    batch=tl.program_id(1).to(tl.int64)', f'    q=(tl.program_id(0)*{points}+tl.arange(0,{points}))[None,:]', f'    valid=q<{m * pp}']
    groups = []
    start = 0
    while start < 7:
        g = 1 << (7 - start).bit_length() - 1
        groups.append((start, g))
        count = g * points
        lines += [f'    d=(tl.arange(0,{g})+{start})[:,None]', f'    off=d*{m * pp}+q', f'    xr=tl.load(X+batch*{2 * pn}+off,valid,0.)', f'    xi=tl.load(X+batch*{2 * pn}+{pn}+off,valid,0.)', f'    wr=tl.load(TW+off,valid,0.)', f'    wi=tl.load(TW+{pn}+off,valid,0.)', '    re,im=xr*wr-xi*wi,xr*wi+xi*wr', f'    re=tl.reshape(re,({count},))', f'    im=tl.reshape(im,({count},))']
        for d in range(g):
            lines += [f'    r{start + d}=al.extract_slice(re,[{d * points}],[{points}],[1])', f'    i{start + d}=al.extract_slice(im,[{d * points}],[{points}],[1])']
        start += g
    lines += _emit_natural_order_codelet_call('    ', 7, direction)
    for start, g in groups:
        count = g * points
        lines += [f'    re=tl.zeros(({count},),tl.float32)', f'    im=tl.zeros(({count},),tl.float32)']
        for d in range(g):
            lines += [f'    re=al.insert_slice(re,r{start + d},[{d * points}],[{points}],[1])', f'    im=al.insert_slice(im,i{start + d},[{d * points}],[{points}],[1])']
        lines += [f'    j=(tl.program_id(0)*{jpack}+tl.arange(0,{jpack}))[:,None]', f'    word=tl.arange(0,{g * pp})[None,:]', f'    out=j*{7 * pp}+{start * pp}+word', f'    re=tl.reshape(tl.permute(tl.reshape(re,({g},{jpack},{pp})),1,0,2),({jpack},{g * pp}))', f'    im=tl.reshape(tl.permute(tl.reshape(im,({g},{jpack},{pp})),1,0,2),({jpack},{g * pp}))', f'    tl.store(Y+batch*{2 * pn}+out,re,j<{m})', f'    tl.store(Y+batch*{2 * pn}+{pn}+out,im,j<{m})']
    s.update(grid=(math.ceil(m / jpack),), batch_pack=1)
    return ast.parse('\n'.join(lines)).body[0]

def optimize_vector_basis(source, specs, direction='forward'):
    """Return a complete module and matching grids; keep other layouts intact."""
    specs = [dict(s) for s in specs]
    stages = {s['name']: s for s in specs}
    spatial = {s['name'] for s in specs if s.get('phase') == 'late'
               and s.get('r') == 7 and s.get('batch_pack') == 4
               and s.get('physical_prefix') == 128 and s['name'].startswith('coarse_stage_')}
    wide = {s['name'] for s in specs if s.get('phase') == 'late'
            and s.get('r') == 7 and s.get('batch_pack', 1) == 1
            and s.get('physical_prefix', 0) >= 512 and s['name'] != 'fused_finish'}
    field = any(s.get('phase') == 'late' and s.get('r') == 13 for s in specs)
    if not (spatial or wide or field):
        return source, specs
    tree = ast.parse(source)
    class CallSite(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == '_fwd_rad7_b1':
                node.func.id = '_npu_vector_rad7'
            return node
    for i, fn in enumerate(tree.body):
        if not isinstance(fn, ast.FunctionDef):
            continue
        if field and fn.name in ('_npu_cos6', '_npu_sin6'):
            _quadratic_products(fn)
        if fn.name in spatial:
            fn = _spatial_radix7(stages[fn.name], direction)
            tree.body[i] = fn
        if fn.name in spatial or fn.name in wide:
            CallSite().visit(fn)
    if spatial or wide:
        tree.body.extend(ast.parse(_RADIX7_SOURCE).body)
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n', specs
