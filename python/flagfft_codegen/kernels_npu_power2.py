"""One Triton implementation for grouped/strided complex and real power-of-two FFTs."""
from .kernels_npu_structured import FFT_HELPERS


def build_npu_power2_source(length, group, stride, output_stride, mode, direction):
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
    name = f'npu_power2_n{n}_g{g}_s{stride}_o{output_stride}_m{mode}_{direction}'
    source = 'import triton.language.extra.cann.extension as al\n'+FFT_HELPERS
    source += f'''@triton.jit
def {name}(in_ptr,out_ptr,idx_ptr,roots_ptr,nbatch,output_row_stride,transform_offset):
    base=tl.program_id(0).to(tl.int64)*{g}+transform_offset
    transforms=base+tl.arange(0,{g})
    k=tl.arange(0,{n})
'''
    if mode == 0 and stride == 1:
        source += f'''    pair=tl.arange(0,2)
    data=tl.load(in_ptr+transforms[:,None,None]*{2*n}+2*k[None,:,None]+pair[None,None,:],transforms[:,None,None]<nbatch,0)
    r,i=tl.split(tl.reshape(data,({count},2)))
    r=tl.reshape(tl.trans(tl.reshape(r,({g},{n}))),({count},))
    i=tl.reshape(tl.trans(tl.reshape(i,({g},{n}))),({count},))
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
    r,i=fft_axis_sliced(r,i,roots_ptr,{count},{n},{g},{n.bit_length()-1})
    if {inverse}: i=-i
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
    elif output_stride == 1:
        source += f'''    pair=tl.arange(0,2)
    packed=tl.reshape(tl.join(tl.reshape(tl.trans(r),({count},)),tl.reshape(tl.trans(i),({count},))),({g},{n},2))
    tl.store(out_ptr+transforms[:,None,None]*{2*n}+2*k[None,:,None]+pair[None,None,:],packed,transforms[:,None,None]<nbatch)
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
