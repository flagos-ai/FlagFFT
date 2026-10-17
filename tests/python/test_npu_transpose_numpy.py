"""The NPU pair IO path preserves every payload bit and masked tail."""
import ast

import numpy as np
import pytest

from flagfft_codegen.kernels_layout import (
    _build_complex_pair_transpose_kernel_source,
    _build_tiled_transpose3d_tile_kernel_source,
)
from test_tensor_fft_numpy import Language, Pointer


class TransposeLanguage(Language):
    ids = (0, 0, 0)

    def program_id(self, axis):
        return self.ids[axis]


@pytest.mark.parametrize('shape,order', [((17,33),'10'), ((65,17),'10'),
    ((17,33,65),'021'), ((17,33,65),'210'),
    ((17,33,65),'201'), ((17,33,65),'120')])
def test_complex_pair_transpose_bits(shape,order):
    batch=3
    rng=np.random.default_rng(20261017)
    bits=rng.integers(0,2**32,size=(batch,*shape,2),dtype=np.uint32)
    expected=np.transpose(bits,(0,*(int(axis)+1 for axis in order),len(shape)+1)).copy()
    if len(shape)==2:
        name,source,_=_build_complex_pair_transpose_kernel_source(*shape)
        grid=((shape[1]+31)//32,(shape[0]+31)//32,batch)
    else:
        name,source,_,grid_x=_build_tiled_transpose3d_tile_kernel_source(
            *shape,order,'complex64',pair_load=True,pair_store=True,clamp_offsets=False)
        grid=(grid_x,1,batch)
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):node.decorator_list=[]
    language=TransposeLanguage();namespace={'tl':language}
    exec(compile(tree,'generated_pair_transpose','exec'),namespace)
    guard=128;marker=np.uint32(0x3fadff12)
    output=np.full(bits.size+2*guard,marker,dtype=np.uint32)
    for z in range(grid[2]):
        for y in range(grid[1]):
            for x in range(grid[0]):
                language.ids=(x,y,z)
                namespace[name](Pointer(bits.reshape(-1).view(np.float32)),
                    Pointer(output.view(np.float32),guard),batch)
    np.testing.assert_array_equal(output[guard:-guard],expected.reshape(-1))
    assert np.all(output[:guard]==marker) and np.all(output[-guard:]==marker)
