"""Resource-bounded fusion of short power-of-two tensor FFT axes."""
import math
import json
from pathlib import Path
from .artifacts import write_text_atomic
from .metadata import _module_source
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
    if inverse:source+='    i=-i\n'
    output_count=n if mode in (0,2) else compact
    # Gather the final permutation directly into small contiguous stores.
    # Avoid allocating full transposed real/imaginary tensors and a full AoS
    # interleave at the same time as the tensor FFT's working set.
    chunk=min(512,n)
    intervals=[(start,chunk) for start in range(0,(output_count//chunk)*chunk,chunk)]
    remainder=output_count%chunk
    if remainder:intervals.append((output_count-remainder,remainder))
    strides=[math.prod(shape[axis+1:]) for axis in range(rank)]
    for number,(start,size) in enumerate(intervals):
        source+=f'    destination{number}=tl.arange(0,{size})+{start}\n'
        if mode==1:
            source+=f'    natural{number}=(destination{number}//{half})*{last}+destination{number}%{half}\n'
        else:source+=f'    natural{number}=destination{number}\n'
        terms=[]
        for position,axis in enumerate(order):
            source+=f'    coordinate{number}_{axis}=(natural{number}//{strides[axis]})%{shape[axis]}\n'
            terms.append(f'coordinate{number}_{axis}*{math.prod(shape[k] for k in order[position+1:])}')
        source+=f'    gather_index{number}='+ '+'.join(terms)+'\n'
        source+=f'    part_r{number}=tl.gather(r,gather_index{number},0)\n'
        if mode==2:
            source+=f'    tl.store(out_ptr+transform*{n}+destination{number},part_r{number})\n'
        else:
            source+=f'    part_i{number}=tl.gather(i,gather_index{number},0)\n'
            source+=f'    packed{number}=tl.reshape(tl.join(part_r{number},part_i{number}),({2*size},))\n'
            source+=f'    tl.store(out_ptr+transform*{2*output_count}+tl.arange(0,{2*size})+{2*start},packed{number})\n'
    return name,source


def emit_npu_tensor(shape, mode, direction, out_dir):
    name, body = build_npu_tensor_source(shape, mode, direction)
    path = Path(out_dir) / (name + '.py')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, _module_source(body))
    meta = dict(module_path=str(path), kernel_name=name,
        signature='*fp32:16,*fp32:16,*i32:16,*fp32:16,i32,i64,i64',
        arg_names=['in_ptr','out_ptr','idx_ptr','roots_ptr','nbatch',
                   'output_row_stride','transform_offset'],
        num_warps=4, num_stages=1, batch_per_block=1,
        kernel_type='npu_grouped_fft64', dtype='complex64', direction=direction,
        length=math.prod(shape), tensor_shape=list(shape), mode=mode)
    write_text_atomic(path.with_suffix('.json'), json.dumps(meta, sort_keys=True))
    return meta
