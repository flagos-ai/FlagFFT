"""Resource-bounded fusion of short power-of-two tensor FFT axes."""
import math
from .kernels_npu_structured import FFT_HELPERS


def tensor_fft_layout(shape):
    shape=tuple(shape)
    count=math.prod(shape)
    if len(shape) not in (2,3) or count>4096 or any(n<8 or n>64 or n&(n-1) for n in shape):
        raise ValueError('Tensor FFT requires 2/3 power-of-two axes of length 8..64 and at most 4096 points')
    offsets=[];size=0
    for length in shape:
        offsets.append(size)
        size+=(length.bit_length()-1)//2*6*(count//4)
    return count,offsets


def build_npu_tensor_source(shape,mode,direction):
    n,offsets=tensor_fft_layout(shape)
    if mode not in (0,1,2) or direction not in ('forward','inverse'):
        raise ValueError('Invalid tensor FFT boundary or direction')
    if (mode==1 and direction!='forward') or (mode==2 and direction!='inverse'):
        raise ValueError('Tensor real boundary disagrees with its direction')
    rank=len(shape);last=shape[-1];half=last//2+1;outer=n//last;compact=outer*half
    inverse=direction=='inverse'
    name='npu_tensor_'+'x'.join(map(str,shape))+f'_m{mode}_{direction}'
    source='import triton.language.extra.cann.extension as al\n'+FFT_HELPERS
    source+=f'''@triton.jit
def {name}(in_ptr,out_ptr,idx_ptr,roots_ptr,nbatch,output_row_stride,transform_offset):
    transform=tl.program_id(0).to(tl.int64)+transform_offset
    if transform>=nbatch: return
    element=tl.arange(0,{n})
'''
    if mode==0:
        source+=f'''    raw=tl.load(in_ptr+transform*{2*n}+tl.arange(0,{2*n}))
    r,i=tl.split(tl.reshape(raw,({n},2)))
'''
    elif mode==1:
        source+=f'''    r=tl.load(in_ptr+transform*{n}+element)
    i=tl.zeros(({n},),tl.float32)
'''
    else:
        padded=1<<(2*compact-1).bit_length()
        source+=f'''    lane=tl.arange(0,{padded})
    raw=tl.load(in_ptr+transform*{2*compact}+lane,lane<{2*compact},0)
    mirror=element%{last}>{last//2}
'''
        strides=[math.prod(shape[a+1:]) for a in range(rank)]
        terms=[]
        for axis,(length,stride) in enumerate(zip(shape,strides)):
            source+=f'    coordinate{axis}=(element//{stride})%{length}\n'
            if axis<rank-1:
                source+=f'    coordinate{axis}=tl.where(mirror,({length}-coordinate{axis})%{length},coordinate{axis})\n'
                terms.append(f'coordinate{axis}*{math.prod(shape[axis+1:-1])*half}')
            else:
                source+=f'    coordinate{axis}=tl.where(mirror,{length}-coordinate{axis},coordinate{axis})\n'
                terms.append(f'coordinate{axis}')
        source+='    compact_index='+'+'.join(terms)+'\n'
        source+=f'''    r=tl.gather(raw,compact_index*2,0)
    i=tl.gather(raw,compact_index*2+1,0)
    i=tl.where(mirror,-i,i)
'''
    if inverse:source+='    i=-i\n'
    order=list(range(rank))
    for axis,length in enumerate(shape):
        permutation=[order.index(axis)]+[j for j,k in enumerate(order) if k!=axis]
        old_shape=tuple(shape[k] for k in order)
        if permutation!=list(range(rank)):
            for component in ('r','i'):
                source+=f'    {component}=tl.reshape(tl.permute(tl.reshape({component},{old_shape}),'+','.join(map(str,permutation))+f'),({n},))\n'
        order=[axis]+[k for k in order if k!=axis]
        source+=f'    r,i=fft_axis_sliced(r,i,roots_ptr+{offsets[axis]},{n},{length},{n//length},{length.bit_length()-1})\n'
    permutation=[order.index(k) for k in range(rank)]
    for component in ('r','i'):
        source+=f'    {component}=tl.reshape(tl.permute(tl.reshape({component},{tuple(shape[k] for k in order)}),'+','.join(map(str,permutation))+f'),({n},))\n'
    if inverse:source+='    i=-i\n'
    if mode==2:
        source+=f'    tl.store(out_ptr+transform*{n}+element,r)\n'
    else:
        source+=f'    packed=tl.reshape(tl.join(r,i),({2*n},))\n'
        if mode==0:
            source+=f'    tl.store(out_ptr+transform*{2*n}+tl.arange(0,{2*n}),packed)\n'
        else:
            for label,size,start in (('main',n,0),('tail',2*outer,n)):
                source+=f'''    lane_{label}=tl.arange(0,{size})+{start}
    index_{label}=((lane_{label}//{2*half})*{last}+(lane_{label}//2)%{half})*2+lane_{label}%2
    value_{label}=tl.gather(packed,index_{label},0)
    tl.store(out_ptr+transform*{2*compact}+lane_{label},value_{label})
'''
    return name,source
