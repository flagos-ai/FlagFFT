# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IX warp-local radix-4 x FFT64 prototype; source generator only."""
import math

def build_ix_warp256_source(pack: int = 8, mode: str = "outer", inverse: bool = False) -> tuple[str, str]:
    return build_ix_warp_source(256, pack, mode, inverse)


def build_ix_warp_source(n: int, pack: int = 8, mode: str = "outer", inverse: bool = False,
                         *, recurrence: bool = False, fast_math: bool = False,
                         local_codelet: bool = False, manual_store: bool = False,
                         output_radix_chunk: int = 0, lane_width: int = 64,
                         packed_gather: bool = False,
                         four_step_other: int = 0, wide: bool = False,
                         table_roots: bool = False) -> tuple[str, str]:
    """Screen register FFTs followed by a lane-local FFT64; no native dispatch."""
    if n not in {16,32,64,128,256,512,1024,2048}:
        raise ValueError("unsupported IX warp FFT length")
    if n<64 and lane_width==64:lane_width=n
    if lane_width<16 or lane_width>n or lane_width&(lane_width-1) or n%lane_width:
        raise ValueError("invalid IX FFT lane width")
    if manual_store and lane_width!=64:
        raise ValueError("manual output experiments require 64-lane FFTs")
    radix=n//lane_width
    chunk=output_radix_chunk or radix
    if chunk<1 or chunk>radix or chunk&(chunk-1) or (manual_store and output_radix_chunk):
        raise ValueError("invalid IX warp output chunk")
    if pack not in {1,2,4,8,16,32,64}:
        raise ValueError("IX warp FFT pack must be 1, 2, 4, 8, 16, 32 or 64")
    row_modes={'four_step_row','four_step_real_row','four_step_hermitian_row'}
    col_modes={'four_step_col','four_step_r2c_col','four_step_c2r_col'}
    if mode not in {"contiguous","outer","inner","r2c","contiguous_r2c","c2r","packed_r2c","packed_c2r"}|row_modes|col_modes:
        raise ValueError("unsupported IX warp FFT I/O mode")
    four_step=mode in row_modes|col_modes
    if wide and (mode not in {"contiguous", "contiguous_r2c", "c2r", "packed_r2c", "packed_c2r", "outer", "inner", "r2c"}|row_modes|col_modes
                 or recurrence or fast_math or manual_store or packed_gather or local_codelet):
        raise ValueError("wide wave FFT requires supplied FP64 roots and ordinary math")
    if four_step and (four_step_other<1 or lane_width!=64 or manual_store):
        raise ValueError("invalid IX warp FourStep geometry")
    if mode in {'packed_r2c','packed_c2r'} and lane_width!=64:
        raise ValueError('packed real warp FFT requires a 64-lane FFT')
    if (mode in {"r2c","contiguous_r2c","packed_r2c","four_step_real_row","four_step_r2c_col"} and inverse) or (mode in {"c2r","packed_c2r","four_step_hermitian_row","four_step_c2r_col"} and not inverse):
        raise ValueError("real transform direction disagrees with the I/O mode")
    sign=1 if inverse else -1
    name=f'warp{n}_{mode}_p{pack}_{"inv" if inverse else "fwd"}'
    if four_step: name+=f'_other{four_step_other}'
    if recurrence: name+='_rec'
    if fast_math: name+='_fast'
    if local_codelet: name+='_codelet'
    if manual_store: name+='_manual'
    if output_radix_chunk: name+=f'_chunk{chunk}'
    if lane_width!=64: name+=f'_lanes{lane_width}'
    if packed_gather: name+='_pairgather'
    if wide: name+='_f64'
    if table_roots: name+='_roots'
    a=['import triton','import triton.language as tl','@triton.jit',
       f'def {name}(inp,out,span,nbatch,input_distance,output_distance):',
       '    pid=tl.program_id(0)',f'    slot=tl.arange(0,{pack})',
       f'    lane=tl.arange(0,{lane_width})[None,:]']
    if mode=='inner':
        a+= [f'    row_base=(pid//span)*{pack}*span+pid%span',
             '    row=row_base+slot*span',
             f'    gbase=(pid%span)*(nbatch//span)+(pid//span)*{pack}+slot']
    elif mode=='outer':
        a += [f'    groups=(span+{pack-1})//{pack}',
              f'    row_base=(pid//groups)*span+(pid%groups)*{pack}',
              '    row=row_base+slot',
              f'    gbase=(pid//groups)*({n}*span)+(pid%groups)*{pack}+slot']
    else:
        a+= [f'    row_base=pid*{pack}','    row=row_base+slot']
        if mode in {'outer','r2c'}:
            outn=n//2+1 if mode=='r2c' else n
            a+=[f'    gbase=(row_base//span)*({outn}*span)+(row_base%span)+slot']
    a+= ['    valid=row[:,None]<nbatch']
    if four_step:
        a += [f'    valid=row[:,None]<{four_step_other}',
              f'    matrix_base=tl.program_id(1)*{n*four_step_other}']
    if mode=='outer': a += [f'    valid=valid&(((pid%groups)*{pack}+slot[:,None])<span)']
    for j in range(radix):
        a+=[f'    pos=lane+{j*lane_width}']
        if four_step:
            # Read adjacent columns together before routing each FFT to a warp.
            a += [f'    position=(tl.arange(0,64)[:,None]+{j*64})*{four_step_other}+row[None,:]']
            if mode=='four_step_real_row':
                a += ['    offset=tl.program_id(1)*input_distance+position',
                      f'    x{j}r=tl.trans(tl.load(inp+offset,row[None,:]<{four_step_other},0.0))',
                      f'    x{j}i=tl.full(({pack},64),0.0,tl.float64)' if wide else
                      f'    x{j}i=tl.full(({pack},64),0.0,tl.float32)']
                continue
            if mode=='four_step_hermitian_row':
                a += [f'    compact=tl.minimum(position,{n*four_step_other}-position)',
                      '    offset=tl.program_id(1)*input_distance+compact']
            else:a += ['    offset=matrix_base+position']
            if wide:
                a += [f'    x{j}r=tl.trans(tl.load(inp+offset*2,row[None,:]<{four_step_other},0.0))',
                      f'    x{j}i=tl.trans(tl.load(inp+offset*2+1,row[None,:]<{four_step_other},0.0))']
            else:a += [
                  '    ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                  f'    pair=tl.load(ip+offset,row[None,:]<{four_step_other},0)',
                  '    pair=tl.trans(pair)',
                  f'    x{j}r=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                  f'    x{j}i=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
            if mode=='four_step_hermitian_row':
                a += [f'    x{j}i=tl.where(tl.trans(position)<={n*four_step_other//2},x{j}i,-x{j}i)',
                      f'    x{j}i=tl.where((tl.trans(position)==0)|(tl.trans(position)=={n*four_step_other//2}),0.0,x{j}i)']
        elif mode=='packed_c2r':
            if wide:
                a += ['    offset=row[:,None]*input_distance+pos',
                      '    xr=tl.load(inp+offset*2,valid,0.0)',
                      '    xi=tl.load(inp+offset*2+1,valid,0.0)',
                      f'    offset=row[:,None]*input_distance+{n}-pos',
                      '    qr=tl.load(inp+offset*2,valid,0.0)',
                      '    qi=-tl.load(inp+offset*2+1,valid,0.0)',
                      '    xi=tl.where(pos==0,0.0,xi)', '    qi=tl.where(pos==0,0.0,qi)',
                      '    wr=tl.load(packed_twiddle_ptr+pos*2)',
                      '    wi=tl.load(packed_twiddle_ptr+pos*2+1)',
                      '    dr=xr-qr', '    di=xi-qi',
                      f'    x{j}r=xr+qr-(di*wr-dr*wi)',
                      f'    x{j}i=xi+qi+(dr*wr+di*wi)']
                continue
            a += ['    ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                  '    pair=tl.load(ip+row[:,None]*input_distance+pos,valid,0)',
                  f'    xr=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                  f'    xi=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)',
                  f'    pair=tl.load(ip+row[:,None]*input_distance+{n}-pos,valid,0)',
                  '    qr=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                  '    qi=-tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)',
                  '    xi=tl.where(pos==0,0.0,xi)','    qi=tl.where(pos==0,0.0,qi)']
            if table_roots:
                a += ['    wr=tl.load(packed_twiddle_ptr+pos*2)',
                      '    wi=tl.load(packed_twiddle_ptr+pos*2+1)']
            else:a += [f'    angle=pos*{-math.pi/n:.17g}',
                       '    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            a += ['    dr=xr-qr','    di=xi-qi',
                  f'    x{j}r=xr+qr-(di*wr-dr*wi)',
                  f'    x{j}i=xi+qi+(dr*wr+di*wi)']
        elif mode=='packed_r2c':
            if wide:
                a += ['    offset=row[:,None]*input_distance+pos*2',
                      f'    x{j}r=tl.load(inp+offset,valid,0.0)',
                      f'    x{j}i=tl.load(inp+offset+1,valid,0.0)']
                continue
            # Real distances may be odd for a custom batched descriptor.
            a += ['    if input_distance%2==0:',
                  '        ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                  '        pair=tl.load(ip+row[:,None]*(input_distance//2)+pos,valid,0)',
                  f'        x{j}r=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                  f'        x{j}i=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)',
                  '    else:',
                  f'        x{j}r=tl.load(inp+row[:,None]*input_distance+pos*2,valid,0.0)',
                  f'        x{j}i=tl.load(inp+row[:,None]*input_distance+pos*2+1,valid,0.0)']
        elif mode=='c2r':
            a+= [f'    hpos=tl.where(pos<={n//2},pos,{n}-pos)',
                 '    offset=row[:,None]*input_distance+hpos',
                 f'    x{j}r=tl.load(inp+offset*2,valid,0.0)',
                 f'    x{j}i=tl.load(inp+offset*2+1,valid,0.0)',
                 f'    x{j}i=tl.where(pos<={n//2},x{j}i,-x{j}i)']
            if wide:a += [f'    x{j}i=tl.where((pos==0)|(pos=={n//2}),0.0,x{j}i)']
        elif mode in {'r2c','contiguous_r2c'}:
            a+=[f'    x{j}r=tl.load(inp+row[:,None]*input_distance+pos,valid,0.0)',
                f'    x{j}i=tl.full(({pack},{lane_width}),0.0,tl.float64)' if wide else
                f'    x{j}i=tl.full(({pack},{lane_width}),0.0,tl.float32)']
        else:
            a+=['    offset=row[:,None]*input_distance+pos',
                 f'    offset=tl.max_contiguous(tl.multiple_of(offset,(1,{lane_width})),(1,{lane_width}))']
            if wide:
                a += [f'    x{j}r=tl.load(inp+offset*2,valid,0.0)',
                      f'    x{j}i=tl.load(inp+offset*2+1,valid,0.0)']
            else:a += ['    ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
                 '    pair=tl.load(ip+offset,valid,0)',
                 f'    x{j}r=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                 f'    x{j}i=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
    if radix==4:
        for c in 'ri': a += [f'    a{c}=x0{c}+x2{c}',f'    b{c}=x1{c}+x3{c}',f'    c{c}=x0{c}-x2{c}',f'    d{c}=x1{c}-x3{c}']
        a+= ['    r0=ar+br','    i0=ai+bi','    r2=ar-br','    i2=ai-bi',
             f'    r1=cr-({sign})*di',f'    i1=ci+({sign})*dr',
             f'    r3=cr+({sign})*di',f'    i3=ci-({sign})*dr']
    elif local_codelet and radix in {16,32}:
        from .kernels_leaf import _emit_radix16_codelet_call, _emit_natural_order_radix32_codelet_call
        for j in range(radix): a += [f'    r{j}=x{j}r',f'    i{j}=x{j}i']
        emit = _emit_radix16_codelet_call if radix==16 else _emit_natural_order_radix32_codelet_call
        a += emit('    ','inverse' if inverse else 'forward')
    else:
        bits=radix.bit_length()-1
        for j in range(radix):
            rev=int(f'{j:0{bits}b}'[::-1],2)
            a += [f'    r{j}=x{rev}r',f'    i{j}=x{rev}i']
        for stage in range(bits):
            half=1<<stage
            for group in range(0,radix,half*2):
                for k in range(half):
                    lo,hi=group+k,group+k+half
                    wr=math.cos(sign*math.pi*k/half)
                    wi=math.sin(sign*math.pi*k/half)
                    if k==0:
                        a += [f'    tr=r{hi}',f'    ti=i{hi}']
                    elif k*2==half:
                        a += [f'    tr={-sign}.0*i{hi}',f'    ti={sign}.0*r{hi}']
                    else:
                        a += [f'    tr=r{hi}*{wr:.17g}-i{hi}*{wi:.17g}',
                              f'    ti=r{hi}*{wi:.17g}+i{hi}*{wr:.17g}']
                    a += [f'    ar=r{lo}',f'    ai=i{lo}',f'    r{lo}=ar+tr',f'    i{lo}=ai+ti',
                          f'    r{hi}=ar-tr',f'    i{hi}=ai-ti']
    for j in range(1,radix):
        if wide or table_roots:
            a += [f'    root_index=(lane*{j})%{n}',
                  '    wr=tl.load(wave_tw_r_ptr+root_index)',
                  '    wi=tl.load(wave_tw_i_ptr+root_index)']
        elif recurrence and j>1:
            a+=['    next_wr=wr*base_wr-wi*base_wi','    wi=wr*base_wi+wi*base_wr','    wr=next_wr']
        else:
            a+=[f'    angle=lane*{sign*2*math.pi*j/n:.17g}',
                '    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            if recurrence: a+=['    base_wr=wr','    base_wi=wi']
        a+=[f'    tr=r{j}*wr-i{j}*wi',f'    ti=r{j}*wi+i{j}*wr',f'    r{j}=tr',f'    i{j}=ti']
    if lane_width==64:
        a+=['    rev=((lane&1)<<5)|((lane&2)<<3)|((lane&4)<<1)|((lane&8)>>1)|((lane&16)>>3)|((lane&32)>>5)']
    else:
        bits=lane_width.bit_length()-1
        a+=['    rev='+'|'.join(f'(((lane>>{i})&1)<<{bits-1-i})' for i in range(bits))]
    a += [f'    idx=tl.broadcast_to(rev,({pack},{lane_width}))']
    def pair_source(j):
        return f'tl.cast(tl.cast(r{j},tl.uint32,bitcast=True),tl.uint64)|(tl.cast(tl.cast(i{j},tl.uint32,bitcast=True),tl.uint64)<<32)'
    for j in range(radix):
        if packed_gather:
            a += [f'    pair={pair_source(j)}','    pair=tl.gather(pair,idx,axis=1)',
                  f'    r{j}=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                  f'    i{j}=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
        else: a += [f'    r{j}=tl.gather(r{j},idx,axis=1)',f'    i{j}=tl.gather(i{j},idx,axis=1)']
    for stage in range(lane_width.bit_length()-1):
        half=1<<stage
        a += [f'    idx=tl.broadcast_to(lane^{half},({pack},{lane_width}))',f'    high=(lane&{half})!=0']
        if half==1: a += ['    wr=1.0','    wi=0.0']
        elif half==2: a += ['    wr=tl.where((lane&1)==0,1.0,0.0)',f'    wi=tl.where((lane&1)==0,0.0,{float(sign)})']
        elif wide or table_roots:
            a += [f'    root_index=(lane&{half-1})*{n//(2*half)}',
                  '    wr=tl.load(wave_tw_r_ptr+root_index)',
                  '    wi=tl.load(wave_tw_i_ptr+root_index)']
        else: a += [f'    angle=(lane&{half-1})*{sign*math.pi/half:.17g}','    wr=tl.cos(angle)','    wi=tl.sin(angle)']
        for j in range(radix):
            if packed_gather:
                a += [f'    pair={pair_source(j)}','    pair=tl.gather(pair,idx,axis=1)',
                      '    pr=tl.cast(tl.cast(pair,tl.uint32),tl.float32,bitcast=True)',
                      '    pi=tl.cast(tl.cast(pair>>32,tl.uint32),tl.float32,bitcast=True)']
            else: a += [f'    pr=tl.gather(r{j},idx,axis=1)',f'    pi=tl.gather(i{j},idx,axis=1)']
            a += [f'    br=tl.where(high,r{j},pr)',f'    bi=tl.where(high,i{j},pi)',
                  '    tr=br*wr-bi*wi','    ti=br*wi+bi*wr',
                  f'    ar=tl.where(high,pr,r{j})',f'    ai=tl.where(high,pi,i{j})',
                  f'    r{j}=tl.where(high,ar-tr,ar+tr)',f'    i{j}=tl.where(high,ai-ti,ai+ti)']
    if mode=='packed_r2c':
        a += ['    nyquist_r=r0-i0']
        for j in range(radix):
            partner=(-j)%radix
            a += [f'    idx=tl.broadcast_to((-lane-{int(j>0)})&63,({pack},64))',
                  f'    br=tl.gather(r{partner},idx,1)',f'    bi=tl.gather(i{partner},idx,1)']
            if wide or table_roots:
                a += [f'    packed_k=lane*{radix}+{j}',
                      '    wr=tl.load(packed_twiddle_ptr+packed_k*2)',
                      '    wi=tl.load(packed_twiddle_ptr+packed_k*2+1)']
            else:a += [f'    angle=(lane*{radix}+{j})*{-math.pi/n:.17g}',
                       '    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            a += [f'    dr=r{j}-br',f'    di=i{j}+bi',
                  f'    post_r{j}=0.5*(r{j}+br+wi*dr+wr*di)',
                  f'    post_i{j}=0.5*(i{j}-bi-wr*dr+wi*di)']
        for j in range(radix):a += [f'    r{j}=post_r{j}',f'    i{j}=post_i{j}']
    if mode in row_modes:
        for j in range(radix):
            if wide or table_roots:
                a += [f'    root_index=(row[:,None]*{n}+lane*{radix}+{j})*2',
                      '    wr=tl.load(twiddle_ptr+root_index,valid,0.0)',
                      '    wi=tl.load(twiddle_ptr+root_index+1,valid,0.0)']
            else:a += [f'    angle=row[:,None]*(lane*{radix}+{j})*{sign*2*math.pi/(n*four_step_other):.17g}',
                       '    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            a += [f'    tr=r{j}*wr-i{j}*wi',f'    i{j}=r{j}*wi+i{j}*wr',f'    r{j}=tr']
    if mode in {'c2r','four_step_c2r_col'}:
        for j in range(radix): a += [f'    z{j}=r{j}']
    elif not wide:
        for j in range(radix): a += [f'    z{j}=tl.cast(tl.cast(r{j},tl.uint32,bitcast=True),tl.uint64)|(tl.cast(tl.cast(i{j},tl.uint32,bitcast=True),tl.uint64)<<32)']
    if manual_store and mode in {'outer','inner','r2c'}:
        shift=(radix*pack).bit_length()-1
        a += [f'    shared=tle.gpu.alloc([{n*pack}],dtype=tl.uint64,layout=None,scope=tle.gpu.smem,nv_mma_shared_layout=False)']
        for j in range(radix):
            a += [f'    index=(lane*{radix}+{j})*{pack}+slot[:,None]',
                  f'    index=index^((index>>{shift})&15)',
                  '    index=ld.brev(ld.brev(index.to(tl.int32)))',
                  f'    tl.store(tle.gpu.local_ptr(shared,(index,)),z{j})']
        a += ['    tl.debug_barrier()',f'    flat=tl.arange(0,{n*pack})',
              f'    index=flat^((flat>>{shift})&15)',
              '    index=ld.brev(ld.brev(index.to(tl.int32)))',
              '    value=tl.load(tle.gpu.local_ptr(shared,(index,)))',
              f'    output_slot=flat%{pack}',f'    freq=flat//{pack}']
        if mode=='inner':
            a += [f'    gbase=(pid%span)*(nbatch//span)+(pid//span)*{pack}+output_slot',
                  '    mask=(row_base+output_slot*span)<nbatch','    stride=nbatch']
        elif mode=='outer':
            a += [f'    gbase=(pid//groups)*({n}*span)+(pid%groups)*{pack}+output_slot',
                  f'    mask=((row_base+output_slot)<nbatch)&((pid%groups)*{pack}+output_slot<span)',
                  '    stride=span']
        else:
            a += [f'    gbase=(row_base//span)*({n//2+1}*span)+(row_base%span)+output_slot',
                  f'    mask=((row_base+output_slot)<nbatch)&(freq<{n//2+1})','    stride=span']
        a += ['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+freq*stride+gbase,value,mask)']
        return _finish_warp_source(name,a,fast_math,radix if local_codelet else 0,True)
    def join(indices, component='z'):
        if len(indices)==1: return f'{component}{indices[0]}'
        name=('j_' if component=='z' else f'j{component}_')+'_'.join(map(str,indices))
        lhs,rhs=join(indices[::2],component),join(indices[1::2],component)
        a.append(f'    {name}=tl.join({lhs},{rhs})')
        return name
    for start in range(0,radix,chunk):
        indices=list(range(start,start+chunk))
        wide_complex=wide and mode not in {'c2r','four_step_c2r_col'}
        if wide_complex:
            zr,zi=join(indices,'r'),join(indices,'i')
            a += [f'    zr=tl.reshape({zr},({pack},{lane_width*chunk}))',
                  f'    zi=tl.reshape({zi},({pack},{lane_width*chunk}))']
        else:
            values=join(indices)
            a += [f'    z=tl.reshape({values},({pack},{lane_width*chunk}))']
        a += [f'    freq=tl.arange(0,{lane_width*chunk})']
        if chunk!=radix: a += [f'    freq=(freq//{chunk})*{radix}+freq%{chunk}+{start}']
        if four_step:
            a += ['    ptr=out' if wide or mode=='four_step_c2r_col' else '    ptr=tl.cast(out,tl.pointer_type(tl.uint64))']
            if mode in row_modes:
                a += [f'    address=matrix_base+row[:,None]*{n}+freq[None,:]']
                if wide_complex:
                    a += ['    z=tl.join(zr,zi)',
                          f'    tl.store(ptr+address[:,:,None]*2+tl.arange(0,2)[None,None,:],z,(row[:,None]<{four_step_other})[:,:,None])']
                else:a += [f'    tl.store(ptr+address,z,row[:,None]<{four_step_other})']
            else:
                a += [f'    position=freq[:,None]*{four_step_other}+row[None,:]',
                      '    base=tl.program_id(1)*output_distance' if mode!='four_step_col' else '    base=matrix_base',
                      '    address=base+position',f'    mask=row[None,:]<{four_step_other}']
                if mode=='four_step_r2c_col':a += [f'    mask=mask&(position<{n*four_step_other//2+1})']
                if wide_complex:
                    a += ['    z=tl.join(tl.trans(zr),tl.trans(zi))',
                          '    tl.store(ptr+address[:,:,None]*2+tl.arange(0,2)[None,None,:],z,mask[:,:,None])']
                else:a += ['    tl.store(ptr+address,tl.trans(z),mask)']
        elif mode in {'contiguous','c2r','contiguous_r2c','packed_r2c','packed_c2r'}:
            a+=['    address=row[:,None]*output_distance+freq[None,:]',
                 '    mask=row[:,None]<nbatch']
            if mode=='contiguous_r2c': a += [f'    mask=mask&(freq[None,:]<{n//2+1})']
            if mode=='c2r': a+=['    tl.store(out+address,z,mask)']
            elif wide_complex:
                a += ['    z=tl.join(zr,zi)',
                      '    tl.store(out+address[:,:,None]*2+tl.arange(0,2)[None,None,:],z,mask[:,:,None])']
            else: a+=['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+address,z,mask)']
            if mode=='packed_r2c' and start==0:
                a += [f'    pos=lane*{radix}',f'    address=row[:,None]*output_distance+{n}',
                      f'    address=tl.broadcast_to(address,({pack},64))']
                if wide:
                    a += ['    tl.store(out+address*2,nyquist_r,valid&(pos==0))',
                          '    tl.store(out+address*2+1,0.0,valid&(pos==0))']
                else:a += [
                      '    nyquist=tl.cast(tl.cast(nyquist_r,tl.uint32,bitcast=True),tl.uint64)',
                      '    tl.store(ptr+address,nyquist,valid&(pos==0))']
        else:
            a+=['    gbase=tl.max_contiguous(tl.multiple_of(gbase,'+str(pack)+'),'+str(pack)+')',
                 '    stride=nbatch' if mode=='inner' else '    stride=span',
                 '    address=freq[:,None]*stride+gbase[None,:]',
                 '    mask=(row[None,:]<nbatch)']
            if mode=='outer': a += [f'    mask=mask&(((pid%groups)*{pack}+slot[None,:])<span)']
            if mode=='r2c': a+=[f'    mask=mask&(freq[:,None]<{n//2+1})']
            if wide_complex:
                a += ['    z=tl.join(tl.trans(zr),tl.trans(zi))',
                      '    tl.store(out+address[:,:,None]*2+tl.arange(0,2)[None,None,:],z,mask[:,:,None])']
            else:a+=['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+address,tl.trans(z),mask)']
    return _finish_warp_source(name,a,fast_math,radix if local_codelet else 0,False)


def _finish_warp_source(name, lines, fast_math, codelet_radix, manual_store):
    source='\n'.join(lines)+'\n'
    if fast_math:
        source=source.replace('import triton.language as tl',
                              'import triton.language as tl\nfrom triton.language.extra import libdevice as ld')
        source=source.replace('tl.cos(', 'ld.fast_cosf(').replace('tl.sin(', 'ld.fast_sinf(')
    if codelet_radix in {16,32}:
        from .metadata import _module_source
        source=_module_source(source,(16,))
    if manual_store:
        source=('import triton.experimental.tle.language as tle\n'
                'from triton.language.extra import libdevice as ld\n')+source
    return name,source


def build_ix_warp256_leaf_source(plan, io_mode: str, perm_form: str, params: list[str],
                                 pack: int, permutation_span: int = 0) -> tuple[str,str]:
    """Retain the raw-leaf argument list, including unused twiddle pointers."""
    mode = {
        "contiguous":"contiguous", "contiguous_c2r":"c2r",
        "contiguous_r2c":"contiguous_r2c",
        "packed_r2c":"packed_r2c", "packed_c2r":"packed_c2r",
        "permuted_r2c":"r2c", "permuted_store_r2c":"r2c",
    }.get(io_mode, perm_form)
    n=plan.length
    chunk=(2 if n==256 else 8 if n>=1024 else 0) if mode in {'outer','inner','r2c'} else 0
    if mode in {'packed_r2c','packed_c2r'}:chunk=1
    name,source=build_ix_warp_source(n,pack,mode,plan.direction=='inverse',fast_math=True,
                                    output_radix_chunk=chunk)
    leaf_name='ix_'+name+'_leaf'
    aliases=['    inp=in_ptr','    out=out_ptr']
    if permutation_span:
        aliases.append(f'    span=tl.full((),{permutation_span},tl.int64)')
    else:
        aliases.append('    span=perm_span' if 'perm_span' in params else '    span=tl.full((),1,tl.int64)')
    if 'input_distance' not in params: aliases.append(f'    input_distance=tl.full((),{n},tl.int64)')
    if 'output_distance' not in params:
        aliases.append(f'    output_distance=tl.full((),{n//2+1 if mode in {"r2c","contiguous_r2c"} else n+1 if mode=="packed_r2c" else n},tl.int64)')
    header=f'def {name}(inp,out,span,nbatch,input_distance,output_distance):'
    assert header in source
    source=source.replace(header,f'def {leaf_name}('+','.join(params)+'):\n'+'\n'.join(aliases),1)
    return leaf_name,source


def build_ix_warp_four_step_source(plan, io_mode, n1, n2, params, pack):
    row=io_mode in {'four_step_row','four_step_real_row','four_step_hermitian_row'}
    other=n2 if row else n1
    name,source=build_ix_warp_source(plan.length,pack,io_mode,
        plan.direction=='inverse',fast_math=True,four_step_other=other)
    leaf_name='ix_'+name+'_leaf'
    header=f'def {name}(inp,out,span,nbatch,input_distance,output_distance):'
    aliases='    inp=in_ptr\n    out=out_ptr'
    source=source.replace(header,f'def {leaf_name}('+','.join(params)+'):\n'+aliases,1)
    return leaf_name,source
