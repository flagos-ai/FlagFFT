"""Packed real FFTs around the shared two-axis complex primitive.

A program owns a group of complete compact rows. Its first store has a
power-of-two extent and its second store contains only the group tail.
Groups of four keep adjacent programs on separate 32-byte output ranges.
"""
from .kernels_npu_structured import FFT_HELPERS


def packed_real_layout(length, group):
    if length not in (128,256,512,1024,2048) or group not in (1,4):
        raise ValueError('Packed real FFT requires length 128..2048 and group 1/4')
    m=length//2
    a=1<<((m.bit_length()-1+1)//2)
    b=m//a
    count=m*group
    wa=(a.bit_length()-1)//2*6*(count//4)
    wb=(b.bit_length()-1)//2*6*(count//4)
    return a,b,wa,wb


def build_npu_packed_real_source(length,group,mode):
    a,b,wa,wb=packed_real_layout(length,group)
    if mode not in (1,2):
        raise ValueError('Packed real boundary mode must be forward or inverse')
    n,g,m,h=length,group,length//2,length//2+1
    count=m*g
    cross=wa+wb
    boundary=cross+2*count
    name=f'npu_packed_real_n{n}_g{g}_m{mode}'
    source='import triton.language.extra.cann.extension as al\n'+FFT_HELPERS
    source+='''@triton.jit
def packed_real_output(r,i,W,j,M:tl.constexpr,H:tl.constexpr):
    row=j//H
    k=j%H
    forward=row*M+tl.where(k==M,0,k)
    reverse=row*M+tl.where((k==0)|(k==M),0,M-k)
    ar=tl.gather(r,forward,0);ai=tl.gather(i,forward,0)
    br=tl.gather(r,reverse,0);bi=tl.gather(i,reverse,0)
    wr=tl.load(W+k);wi=tl.load(W+H+k)
    dr=ar-br;di=ai+bi
    return 0.5*(ar+br+wr*di+wi*dr),0.5*(ai-bi-wr*dr+wi*di)

'''
    source+=f'''@triton.jit
def {name}(in_ptr,out_ptr,idx_ptr,roots_ptr,nbatch,output_row_stride,transform_offset):
    base=tl.program_id(0).to(tl.int64)*{g}+transform_offset
    element=tl.arange(0,{count})
    transforms=base+element//{m}
    k=element%{m}
'''
    if mode==1:
        source+=f'''    lane=tl.arange(0,{2*count})
    data=tl.load(in_ptr+base*{n}+lane,base*{n}+lane<nbatch*{n},0)
    r,i=tl.split(tl.reshape(data,({count},2)))
'''
    else:
        padded=1<<(2*g*h-1).bit_length()
        source+=f'''    lane=tl.arange(0,{padded})
    data=tl.load(in_ptr+base*{2*h}+lane,(lane<{2*g*h})&(base*{2*h}+lane<nbatch*{2*h}),0)
    forward=(element//{m})*{2*h}+2*k
    reverse=(element//{m})*{2*h}+2*({m}-k)
    ar=tl.gather(data,forward,0);ai=tl.gather(data,forward+1,0)
    br=tl.gather(data,reverse,0);bi=-tl.gather(data,reverse+1,0)
    wr=tl.load(roots_ptr+{boundary}+k);wi=tl.load(roots_ptr+{boundary+h}+k)
    dr=ar-br;di=ai-bi
    r=ar+br-di*wr+dr*wi
    i=-(ai+bi+dr*wr+di*wi)
'''
    source+=f'''    r=tl.reshape(tl.permute(tl.reshape(r,({g},{a},{b})),1,2,0),({count},))
    i=tl.reshape(tl.permute(tl.reshape(i,({g},{a},{b})),1,2,0),({count},))
    r,i=fft_axis_sliced(r,i,roots_ptr,{count},{a},{b*g},{a.bit_length()-1})
    wr=tl.load(roots_ptr+{cross}+element);wi=tl.load(roots_ptr+{cross+count}+element)
    rr=r*wr-i*wi;ii=r*wi+i*wr
    r=tl.reshape(tl.permute(tl.reshape(rr,({a},{b},{g})),1,0,2),({count},))
    i=tl.reshape(tl.permute(tl.reshape(ii,({a},{b},{g})),1,0,2),({count},))
    r,i=fft_axis_sliced(r,i,roots_ptr+{wa},{count},{b},{a*g},{b.bit_length()-1})
    r=tl.reshape(tl.permute(tl.reshape(r,({b},{a},{g})),2,0,1),({count},))
    i=tl.reshape(tl.permute(tl.reshape(i,({b},{a},{g})),2,0,1),({count},))
'''
    if mode==2:
        source+=f'''    data=tl.reshape(tl.join(r,-i),({2*count},))
    lane=tl.arange(0,{2*count})
    tl.store(out_ptr+base*{n}+lane,data,base*{n}+lane<nbatch*{n})
'''
    else:
        for suffix,size,start in (('main',count,0),('tail',g,count)):
            source+=f'''    j_{suffix}=tl.arange(0,{size})+{start}
    fr_{suffix},fi_{suffix}=packed_real_output(r,i,roots_ptr+{boundary},j_{suffix},{m},{h})
    data_{suffix}=tl.reshape(tl.join(fr_{suffix},fi_{suffix}),({2*size},))
    lane_{suffix}=tl.arange(0,{2*size})+{2*start}
    tl.store(out_ptr+base*{2*h}+lane_{suffix},data_{suffix},base*{2*h}+lane_{suffix}<nbatch*{2*h})
'''
    return name,source
