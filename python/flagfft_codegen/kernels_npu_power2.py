"""One Triton implementation for grouped/strided complex and real power-of-two FFTs."""
from .kernels_npu_structured import FFT_HELPERS, recursive_axis_body


def build_npu_power2_source(length, group, stride, output_stride, mode, direction, recursive=False):
    if length not in (16, 32, 64, 128, 256, 512, 1024, 2048):
        raise ValueError('Unsupported power-of-two FFT length')
    if group not in (1, 2, 4, 8, 16, 32) or stride < 1 or output_stride < 0 or mode not in (0, 1, 2):
        raise ValueError('Invalid grouped FFT layout')
    if mode and (stride != 1 or output_stride != 1):
        raise ValueError('Real boundaries require contiguous transforms')
    if direction not in ('forward', 'inverse') or (mode == 1 and direction != 'forward') or (mode == 2 and direction != 'inverse'):
        raise ValueError('FFT direction disagrees with the real boundary')
    n, g, h = length, group, length//2+1
    count = n*g
    inverse = direction == 'inverse'
    name = f'npu_power2_n{n}_g{g}_s{stride}_o{output_stride}_m{mode}_{direction}'+('_recursive' if recursive else '')
    source = 'import triton.language.extra.cann.extension as al\n'+FFT_HELPERS
    source += f'''@triton.jit
def {name}(in_ptr,out_ptr,idx_ptr,roots_ptr,nbatch,output_row_stride,transform_offset):
    base=tl.program_id(0).to(tl.int64)*{g}+transform_offset
    transforms=base+tl.arange(0,{g})
    k=tl.arange(0,{n})
'''
    if mode == 0 and stride == 1:
        source += f'''    word=tl.arange(0,{2*n})
    data=tl.load(in_ptr+transforms[:,None]*{2*n}+word[None,:],transforms[:,None]<nbatch,0)
    r,i=tl.split(tl.reshape(data,({count},2)))
    r=tl.reshape(tl.trans(tl.reshape(r,({g},{n}))),({count},))
    i=tl.reshape(tl.trans(tl.reshape(i,({g},{n}))),({count},))
'''
    elif mode == 0 and stride % g == 0:
        source += f'''    word=tl.arange(0,{2*g})[None,:]
    row=tl.arange(0,{n})[:,None]
    matrix_base=(base//{stride})*{n*stride}+base%{stride}
    offsets=matrix_base*2+row*{2*stride}+word
    data=tl.load(in_ptr+offsets,base+word//2<nbatch,0)
    r,i=tl.split(tl.reshape(data,({count},2)))
'''
    elif mode == 0:
        source += f'''    lane=tl.arange(0,{2*count})
    transform=base+(lane//2)%{g}
    offsets=((transform//{stride})*{n*stride}+(lane//{2*g})*{stride}+transform%{stride})*2+lane%2
    data=tl.load(in_ptr+offsets,transform<nbatch,0)
    r,i=tl.split(tl.reshape(data,({count},2)))
'''
    elif mode == 1:
        source += f'''    r=tl.load(in_ptr+transforms[:,None]*{n}+k[None,:],transforms[:,None]<nbatch,0)
    r=tl.reshape(tl.trans(r),({count},))
    i=tl.zeros(({count},),tl.float32)
'''
    else:
        padded = 1 << (2*g*h-1).bit_length()
        source += f'''    lane=tl.arange(0,{padded})
    data=tl.load(in_ptr+base*{2*h}+lane,(lane<{2*g*h})&(base*{2*h}+lane<nbatch*{2*h}),0)
    element=tl.arange(0,{count})
    freq=element%{n}
    reflected=tl.minimum(freq,{n}-freq)
    index=(element//{n})*{2*h}+reflected*2
    r=tl.gather(data,index,0)
    i=tl.gather(data,index+1,0)*tl.where(freq>{n//2},-1.0,1.0)
    r=tl.reshape(tl.trans(tl.reshape(r,({g},{n}))),({count},))
    i=tl.reshape(tl.trans(tl.reshape(i,({g},{n}))),({count},))
'''
    source += f'''    if {inverse}: i=-i
'''
    source += recursive_axis_body(n,g) if recursive and n>=256 else f'    r,i=fft_axis_sliced(r,i,roots_ptr,{count},{n},{g},{n.bit_length()-1})\n'
    source += f'''    if {inverse}: i=-i
    k=tl.arange(0,{n})
    r=tl.reshape(r,({n},{g}));i=tl.reshape(i,({n},{g}))
'''
    if mode == 2:
        source += f'''    tl.store(out_ptr+transforms[:,None]*{n}+k[None,:],tl.trans(r),transforms[:,None]<nbatch)
'''
    elif mode == 1:
        padded = 1 << (2*g*h-1).bit_length()
        source += f'''    packed=tl.reshape(tl.join(tl.reshape(tl.trans(r),({count},)),tl.reshape(tl.trans(i),({count},))),({2*count},))
    lane=tl.arange(0,{padded})
    index=(lane//{2*h})*{2*n}+((lane//2)%{h})*2+lane%2
    value=tl.gather(packed,tl.minimum(index,{2*count-1}),0)
    tl.store(out_ptr+base*{2*h}+lane,value,(lane<{2*g*h})&(base*{2*h}+lane<nbatch*{2*h}))
'''
    elif output_stride == 1 and count>2048:
        # Large joins would acquire padded intermediate rows on CANN.
        # Bound every interleave and store to a contiguous frequency tile.
        source += f'    zr=tl.reshape(r,({count},));zi=tl.reshape(i,({count},))\n'
        for transform in range(g):
            for start in range(0,n,512):
                size=min(512,n-start)
                source += f'''    q=tl.arange(0,{size})
    ix=(q+{start})*{g}+{transform}
    pr=tl.gather(zr,ix,0);pi=tl.gather(zi,ix,0)
    packed=tl.reshape(tl.join(pr,pi),({2*size},))
    word=tl.arange(0,{2*size})
    tl.store(out_ptr+(base+{transform})*{2*n}+{2*start}+word,packed,base+{transform}<nbatch)
'''
    elif output_stride == 1:
        source += f'''    word=tl.arange(0,{2*n})
    packed=tl.reshape(tl.join(tl.reshape(tl.trans(r),({count},)),tl.reshape(tl.trans(i),({count},))),({g},{2*n}))
    tl.store(out_ptr+transforms[:,None]*{2*n}+word[None,:],packed,transforms[:,None]<nbatch)
'''
    elif output_stride > 1 and output_stride % g == 0 and (g<4 or count>2048):
        # Keep aligned physical rows while bounding each private interleave.
        # Mask the padding columns so adjacent transform groups remain intact.
        store_g=max(4,g);rows=min(n,512//store_g);tile=rows*store_g
        source += f'''    zr=tl.reshape(r,({count},));zi=tl.reshape(i,({count},))
    matrix_base=(base//{output_stride})*{n*output_stride}+base%{output_stride}
    word=tl.arange(0,{2*store_g})[None,:]
'''
        for start in range(0,n,rows):
            source += f'''    q=tl.arange(0,{tile})
    ix=({start}+q//{store_g})*{g}+q%{store_g}
    ix=tl.minimum(ix,{count-1})
    pr=tl.gather(zr,ix,0);pi=tl.gather(zi,ix,0)
    packed=tl.reshape(tl.join(pr,pi),({rows},{2*store_g}))
    row={start}+tl.arange(0,{rows})[:,None]
    tl.store(out_ptr+matrix_base*2+row*{2*output_stride}+word,packed,(word<{2*g})&(base+word//2<nbatch))
'''
    elif output_stride > 1 and output_stride % g == 0:
        source += f'''    packed=tl.reshape(tl.join(tl.reshape(r,({count},)),tl.reshape(i,({count},))),({n},{2*g}))
    row=tl.arange(0,{n})[:,None]
    word=tl.arange(0,{2*g})[None,:]
    matrix_base=(base//{output_stride})*{n*output_stride}+base%{output_stride}
    tl.store(out_ptr+matrix_base*2+row*{2*output_stride}+word,packed,base+word//2<nbatch)
'''
    else:
        stride_expr = str(output_stride) if output_stride else 'output_row_stride'
        source += f'''    packed=tl.reshape(tl.join(tl.reshape(r,({count},)),tl.reshape(i,({count},))),({2*count},))
    lane=tl.arange(0,{2*count})
    transform=base+(lane//2)%{g}
    offsets=((transform//({stride_expr}))*{n}*({stride_expr})+(lane//{2*g})*({stride_expr})+transform%({stride_expr}))*2+lane%2
    tl.store(out_ptr+offsets,packed,transform<nbatch)
'''
    return name, source
