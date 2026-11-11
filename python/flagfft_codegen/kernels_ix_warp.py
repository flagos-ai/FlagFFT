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


def build_ix_warp_source(n: int, pack: int = 8, mode: str = "outer", inverse: bool = False) -> tuple[str, str]:
    """Screen register FFTs followed by a lane-local FFT64; no native dispatch."""
    if n not in {128,256,512,1024,2048}:
        raise ValueError("IX warp FFT length must be 128, 256, 512, 1024 or 2048")
    radix=n//64
    if pack not in {4,8,16,32,64}:
        raise ValueError("IX warp FFT pack must be 4, 8, 16, 32 or 64")
    if mode not in {"contiguous","outer","inner","r2c","c2r"}:
        raise ValueError("unsupported IX warp FFT I/O mode")
    if (mode == "r2c" and inverse) or (mode == "c2r" and not inverse):
        raise ValueError("real transform direction disagrees with the I/O mode")
    sign=1 if inverse else -1
    name=f'warp{n}_{mode}_p{pack}_{"inv" if inverse else "fwd"}'
    a=['import triton','import triton.language as tl','@triton.jit',
       f'def {name}(inp,out,span,nbatch,input_distance,output_distance):',
       '    pid=tl.program_id(0)',f'    slot=tl.arange(0,{pack})',
       '    lane=tl.arange(0,64)[None,:]']
    if mode=='inner':
        a+= [f'    row_base=(pid//span)*{pack}*span+pid%span',
             '    row=row_base+slot*span',
             f'    gbase=(pid%span)*(nbatch//span)+(pid//span)*{pack}+slot']
    else:
        a+= [f'    row_base=pid*{pack}','    row=row_base+slot']
        if mode in {'outer','r2c'}:
            outn=n//2+1 if mode=='r2c' else n
            a+=[f'    gbase=(row_base//span)*({outn}*span)+(row_base%span)+slot']
    a+= ['    valid=row[:,None]<nbatch']
    for j in range(radix):
        a+=[f'    pos=lane+{j*64}']
        if mode=='c2r':
            a+= [f'    hpos=tl.where(pos<={n//2},pos,{n}-pos)',
                 '    offset=row[:,None]*input_distance+hpos',
                 f'    x{j}r=tl.load(inp+offset*2,valid,0.0)',
                 f'    x{j}i=tl.load(inp+offset*2+1,valid,0.0)',
                 f'    x{j}i=tl.where(pos<={n//2},x{j}i,-x{j}i)']
        elif mode=='r2c':
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
        a+=[f'    angle=lane*{sign*2*math.pi*j/n:.17g}',
             '    wr=tl.cos(angle)','    wi=tl.sin(angle)',
             f'    tr=r{j}*wr-i{j}*wi',f'    ti=r{j}*wi+i{j}*wr',f'    r{j}=tr',f'    i{j}=ti']
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
    def join(indices):
        if len(indices)==1: return f'z{indices[0]}'
        name='j_'+'_'.join(map(str,indices))
        lhs,rhs=join(indices[::2]),join(indices[1::2])
        a.append(f'    {name}=tl.join({lhs},{rhs})')
        return name
    values=join(list(range(radix)))
    a += [f'    z=tl.reshape({values},({pack},{n}))',f'    freq=tl.arange(0,{n})']
    if mode in {'contiguous','c2r'}:
        a+=['    address=row[:,None]*output_distance+freq[None,:]',
             '    mask=row[:,None]<nbatch']
        if mode=='c2r': a+=['    tl.store(out+address,z,mask)']
        else: a+=['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+address,z,mask)']
    else:
        a+=['    gbase=tl.max_contiguous(tl.multiple_of(gbase,'+str(pack)+'),'+str(pack)+')',
             '    stride=nbatch' if mode=='inner' else '    stride=span',
             '    address=freq[:,None]*stride+gbase[None,:]',
             '    mask=(row[None,:]<nbatch)']
        if mode=='r2c': a+=[f'    mask=mask&(freq[:,None]<{n//2+1})']
        a+=['    ptr=tl.cast(out,tl.pointer_type(tl.uint64))','    tl.store(ptr+address,tl.trans(z),mask)']
    return name,'\n'.join(a)+'\n'


def build_ix_warp256_leaf_source(plan, io_mode: str, perm_form: str, params: list[str],
                                 pack: int, permutation_span: int = 0) -> tuple[str,str]:
    """Retain the raw-leaf argument list, including unused twiddle pointers."""
    mode = {
        "contiguous":"contiguous", "contiguous_c2r":"c2r",
        "permuted_r2c":"r2c", "permuted_store_r2c":"r2c",
    }.get(io_mode, perm_form)
    name,source=build_ix_warp256_source(pack,mode,plan.direction=='inverse')
    leaf_name='ix_'+name+'_leaf'
    aliases=['    inp=in_ptr','    out=out_ptr']
    if permutation_span:
        aliases.append(f'    span=tl.full((),{permutation_span},tl.int64)')
    else:
        aliases.append('    span=perm_span' if 'perm_span' in params else '    span=tl.full((),1,tl.int64)')
    if 'input_distance' not in params: aliases.append('    input_distance=tl.full((),256,tl.int64)')
    if 'output_distance' not in params:
        aliases.append('    output_distance=tl.full((),'+('129' if mode=='r2c' else '256')+',tl.int64)')
    header=f'def {name}(inp,out,span,nbatch,input_distance,output_distance):'
    assert header in source
    source=source.replace(header,f'def {leaf_name}('+','.join(params)+'):\n'+'\n'.join(aliases),1)
    return leaf_name,source
