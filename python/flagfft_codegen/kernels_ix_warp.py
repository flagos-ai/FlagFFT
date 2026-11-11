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
                         output_radix_chunk: int = 0) -> tuple[str, str]:
    """Screen register FFTs followed by a lane-local FFT64; no native dispatch."""
    if n not in {128,256,512,1024,2048}:
        raise ValueError("IX warp FFT length must be 128, 256, 512, 1024 or 2048")
    radix=n//64
    chunk=output_radix_chunk or radix
    if chunk<1 or chunk>radix or chunk&(chunk-1) or (manual_store and output_radix_chunk):
        raise ValueError("invalid IX warp output chunk")
    if pack not in {4,8,16,32,64}:
        raise ValueError("IX warp FFT pack must be 4, 8, 16, 32 or 64")
    if mode not in {"contiguous","outer","inner","r2c","contiguous_r2c","c2r"}:
        raise ValueError("unsupported IX warp FFT I/O mode")
    if (mode in {"r2c","contiguous_r2c"} and inverse) or (mode == "c2r" and not inverse):
        raise ValueError("real transform direction disagrees with the I/O mode")
    sign=1 if inverse else -1
    name=f'warp{n}_{mode}_p{pack}_{"inv" if inverse else "fwd"}'
    if recurrence: name+='_rec'
    if fast_math: name+='_fast'
    if local_codelet: name+='_codelet'
    if manual_store: name+='_manual'
    if output_radix_chunk: name+=f'_chunk{chunk}'
    a=['import triton','import triton.language as tl','@triton.jit',
       f'def {name}(inp,out,span,nbatch,input_distance,output_distance):',
       '    pid=tl.program_id(0)',f'    slot=tl.arange(0,{pack})',
       '    lane=tl.arange(0,64)[None,:]']
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
    if mode=='outer': a += [f'    valid=valid&(((pid%groups)*{pack}+slot[:,None])<span)']
    for j in range(radix):
        a+=[f'    pos=lane+{j*64}']
        if mode=='c2r':
            a+= [f'    hpos=tl.where(pos<={n//2},pos,{n}-pos)',
                 '    offset=row[:,None]*input_distance+hpos',
                 f'    x{j}r=tl.load(inp+offset*2,valid,0.0)',
                 f'    x{j}i=tl.load(inp+offset*2+1,valid,0.0)',
                 f'    x{j}i=tl.where(pos<={n//2},x{j}i,-x{j}i)']
        elif mode in {'r2c','contiguous_r2c'}:
            a+=[f'    x{j}r=tl.load(inp+row[:,None]*input_distance+pos,valid,0.0)',f'    x{j}i=tl.full(({pack},64),0.0,tl.float32)']
        else:
            a+=['    offset=row[:,None]*input_distance+pos',
                 '    offset=tl.max_contiguous(tl.multiple_of(offset,(1,64)),(1,64))',
                 '    ip=tl.cast(inp,tl.pointer_type(tl.uint64))',
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
        if recurrence and j>1:
            a+=['    next_wr=wr*base_wr-wi*base_wi','    wi=wr*base_wi+wi*base_wr','    wr=next_wr']
        else:
            a+=[f'    angle=lane*{sign*2*math.pi*j/n:.17g}',
                '    wr=tl.cos(angle)','    wi=tl.sin(angle)']
            if recurrence: a+=['    base_wr=wr','    base_wi=wi']
        a+=[f'    tr=r{j}*wr-i{j}*wi',f'    ti=r{j}*wi+i{j}*wr',f'    r{j}=tr',f'    i{j}=ti']
    a+=['    rev=((lane&1)<<5)|((lane&2)<<3)|((lane&4)<<1)|((lane&8)>>1)|((lane&16)>>3)|((lane&32)>>5)',
         f'    idx=tl.broadcast_to(rev,({pack},64))']
    for j in range(radix): a += [f'    r{j}=tl.gather(r{j},idx,axis=1)',f'    i{j}=tl.gather(i{j},idx,axis=1)']
    for half in (1,2,4,8,16,32):
        a += [f'    idx=tl.broadcast_to(lane^{half},({pack},64))',f'    high=(lane&{half})!=0']
        if half==1: a += ['    wr=1.0','    wi=0.0']
        elif half==2: a += ['    wr=tl.where((lane&1)==0,1.0,0.0)',f'    wi=tl.where((lane&1)==0,0.0,{float(sign)})']
        else: a += [f'    angle=(lane&{half-1})*{sign*math.pi/half:.17g}','    wr=tl.cos(angle)','    wi=tl.sin(angle)']
        for j in range(radix):
            a += [f'    pr=tl.gather(r{j},idx,axis=1)',f'    pi=tl.gather(i{j},idx,axis=1)',
                  f'    br=tl.where(high,r{j},pr)',f'    bi=tl.where(high,i{j},pi)',
                  '    tr=br*wr-bi*wi','    ti=br*wi+bi*wr',
                  f'    ar=tl.where(high,pr,r{j})',f'    ai=tl.where(high,pi,i{j})',
                  f'    r{j}=tl.where(high,ar-tr,ar+tr)',f'    i{j}=tl.where(high,ai-ti,ai+ti)']
    if mode=='c2r':
        for j in range(radix): a += [f'    z{j}=r{j}']
    else:
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
    def join(indices):
        if len(indices)==1: return f'z{indices[0]}'
        name='j_'+'_'.join(map(str,indices))
        lhs,rhs=join(indices[::2]),join(indices[1::2])
        a.append(f'    {name}=tl.join({lhs},{rhs})')
        return name
    for start in range(0,radix,chunk):
        values=join(list(range(start,start+chunk)))
        a += [f'    z=tl.reshape({values},({pack},{64*chunk}))',f'    freq=tl.arange(0,{64*chunk})']
        if chunk!=radix: a += [f'    freq=(freq//{chunk})*{radix}+freq%{chunk}+{start}']
        if mode in {'contiguous','c2r','contiguous_r2c'}:
            a+=['    address=row[:,None]*output_distance+freq[None,:]',
                 '    mask=row[:,None]<nbatch']
            if mode=='contiguous_r2c': a += [f'    mask=mask&(freq[None,:]<{n//2+1})']
            if mode=='c2r': a+=['    tl.store(out+address,z,mask)']
            else: a+=['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+address,z,mask)']
        else:
            a+=['    gbase=tl.max_contiguous(tl.multiple_of(gbase,'+str(pack)+'),'+str(pack)+')',
                 '    stride=nbatch' if mode=='inner' else '    stride=span',
                 '    address=freq[:,None]*stride+gbase[None,:]',
                 '    mask=(row[None,:]<nbatch)']
            if mode=='outer': a += [f'    mask=mask&(((pid%groups)*{pack}+slot[None,:])<span)']
            if mode=='r2c': a+=[f'    mask=mask&(freq[:,None]<{n//2+1})']
            a+=['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+address,tl.trans(z),mask)']
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
        "permuted_r2c":"r2c", "permuted_store_r2c":"r2c",
    }.get(io_mode, perm_form)
    n=plan.length
    name,source=build_ix_warp_source(n,pack,mode,plan.direction=='inverse',fast_math=True)
    leaf_name='ix_'+name+'_leaf'
    aliases=['    inp=in_ptr','    out=out_ptr']
    if permutation_span:
        aliases.append(f'    span=tl.full((),{permutation_span},tl.int64)')
    else:
        aliases.append('    span=perm_span' if 'perm_span' in params else '    span=tl.full((),1,tl.int64)')
    if 'input_distance' not in params: aliases.append(f'    input_distance=tl.full((),{n},tl.int64)')
    if 'output_distance' not in params:
        aliases.append(f'    output_distance=tl.full((),{n//2+1 if mode in {"r2c","contiguous_r2c"} else n},tl.int64)')
    header=f'def {name}(inp,out,span,nbatch,input_distance,output_distance):'
    assert header in source
    source=source.replace(header,f'def {leaf_name}('+','.join(params)+'):\n'+'\n'.join(aliases),1)
    return leaf_name,source
