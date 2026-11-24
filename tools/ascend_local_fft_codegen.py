"""Generate lane-local FFT codelets, with explicit UB transport between axes.

Each scalar FFT value is a vector across independent columns/transforms.
The Python generator expands radix-2/4 codelets and constant roots. Only the
cross-axis twiddle remains an HBM table. No high-level gather is used.
"""
import cmath
import math


def generate(*, n=2048, factor=32, pack=1, both=True, radix=4):
    a, b = factor, n//factor
    if n != a*b or any(v < 1 or v & (v-1) for v in (a,b,pack)):
        raise ValueError('power-of-two factorization and packing required')
    if radix not in (2,4):
        raise ValueError('radix must be 2 or 4')
    name=f'local_fft_n{n}_a{a}_p{pack}_both{int(both)}_r{radix}'
    lines=['import triton','import triton.language as tl',
           'import triton.language.extra.cann.extension as al',
           'from ascend_fourstep_probe import fft_axis_sliced', '',
           '@triton.jit', f'def {name}(X,Y,CROSS,WB,NB:tl.constexpr):',
           f'    bid=tl.program_id(0)', f'    lane=tl.arange(0,{2*n})',
           f'    batch=bid*{pack}+tl.arange(0,{pack})',
           f'    raw=tl.load(X+batch[:,None]*{2*n}+lane[None,:],batch[:,None]<NB,0)',
           f'    raw=tl.reshape(raw,({pack},{a},{2*b}))',
           f'    c=tl.load(CROSS+tl.arange(0,{n}))',
           f'    d=tl.load(CROSS+{n}+tl.arange(0,{n}))']
    serial=0
    def emit(expr):
        nonlocal serial
        key=f'v{serial}';serial+=1
        lines.append(f'    {key}={expr}')
        return key
    def plus(x,y): return (emit(f'{x[0]}+{y[0]}'),emit(f'{x[1]}+{y[1]}'))
    def minus(x,y): return (emit(f'{x[0]}-{y[0]}'),emit(f'{x[1]}-{y[1]}'))
    def root(x,k,m):
        k%=m
        if k==0:return x
        if k*4==m:return (x[1],emit(f'-{x[0]}'))
        if k*2==m:return (emit(f'-{x[0]}'),emit(f'-{x[1]}'))
        if k*4==3*m:return (emit(f'-{x[1]}'),x[0])
        w=cmath.exp(-2j*math.pi*k/m)
        return (emit(f'{x[0]}*({w.real!r})-{x[1]}*({w.imag!r})'),
                emit(f'{x[0]}*({w.imag!r})+{x[1]}*({w.real!r})'))
    def fft(values):
        length=len(values)
        if length==1:return values
        if length==2:
            return [plus(*values),minus(*values)]
        if radix==2:
            e,o=fft(values[::2]),fft(values[1::2]); out=[None]*length
            for k in range(length//2):
                t=root(o[k],k,length)
                out[k],out[k+length//2]=plus(e[k],t),minus(e[k],t)
            return out
        quarter=length//4
        children=[fft(values[j::4]) for j in range(4)]
        out=[None]*length
        for k in range(quarter):
            x0=children[0][k]
            x1,x2,x3=[root(children[j][k],j*k,length) for j in (1,2,3)]
            t0,t1,t2,t3=plus(x0,x2),minus(x0,x2),plus(x1,x3),minus(x1,x3)
            out[k]=plus(t0,t2)
            out[k+quarter]=(emit(f'{t1[0]}+{t3[1]}'),emit(f'{t1[1]}-{t3[0]}'))
            out[k+2*quarter]=minus(t0,t2)
            out[k+3*quarter]=(emit(f'{t1[0]}-{t3[1]}'),emit(f'{t1[1]}+{t3[0]}'))
        return out
    values=[]
    for row in range(a):
        v=emit(f'tl.reshape(al.extract_slice(raw,[0,{row},0],[{pack},1,{2*b}],[1,1,1]),({pack*b},2))')
        r,i=f'xr{row}',f'xi{row}'
        lines.append(f'    {r},{i}=tl.split({v})')
        values.append((r,i))
    outputs=fft(values)
    rr,ii=emit(f'tl.zeros(({pack*n},),tl.float32)'),emit(f'tl.zeros(({pack*n},),tl.float32)')
    for row,(r,i) in enumerate(outputs):
        cr=emit(f'tl.reshape(al.extract_slice(c,[{row*b}],[{b}],[1]),(1,{b}))')
        ci=emit(f'tl.reshape(al.extract_slice(d,[{row*b}],[{b}],[1]),(1,{b}))')
        r=emit(f'tl.reshape({r},({pack},{b}))');i=emit(f'tl.reshape({i},({pack},{b}))')
        re,im=emit(f'{r}*{cr}-{i}*{ci}'),emit(f'{r}*{ci}+{i}*{cr}')
        re,im=emit(f'tl.reshape({re},({pack*b},))'),emit(f'tl.reshape({im},({pack*b},))')
        rr=emit(f'al.insert_slice({rr},{re},[{row*pack*b}],[{pack*b}],[1])')
        ii=emit(f'al.insert_slice({ii},{im},[{row*pack*b}],[{pack*b}],[1])')
    # Row -> column exchange; lane order is now (batch, first frequency).
    rr=emit(f'tl.reshape(tl.permute(tl.reshape({rr},({a},{pack},{b})),2,1,0),({b},{pack*a}))')
    ii=emit(f'tl.reshape(tl.permute(tl.reshape({ii},({a},{pack},{b})),2,1,0),({b},{pack*a}))')
    if both:
        values=[]
        for row in range(b):
            r=emit(f'tl.reshape(al.extract_slice({rr},[{row},0],[1,{pack*a}],[1,1]),({pack*a},))')
            i=emit(f'tl.reshape(al.extract_slice({ii},[{row},0],[1,{pack*a}],[1,1]),({pack*a},))')
            values.append((r,i))
        outputs=fft(values)
        rr,ii=emit(f'tl.zeros(({pack*n},),tl.float32)'),emit(f'tl.zeros(({pack*n},),tl.float32)')
        for row,(r,i) in enumerate(outputs):
            rr=emit(f'al.insert_slice({rr},{r},[{row*pack*a}],[{pack*a}],[1])')
            ii=emit(f'al.insert_slice({ii},{i},[{row*pack*a}],[{pack*a}],[1])')
    else:
        rr=emit(f'tl.reshape({rr},({pack*n},))');ii=emit(f'tl.reshape({ii},({pack*n},))')
        # Sliced path processes B across leading axis, preserving P*A lanes.
        lines.append(f'    rtemp,itemp=fft_axis_sliced({rr},{ii},WB,{pack*n},{b},{pack*a},{b.bit_length()-1})')
        rr,ii='rtemp','itemp'
    rr=emit(f'tl.permute(tl.reshape({rr},({b},{pack},{a})),1,0,2)')
    ii=emit(f'tl.permute(tl.reshape({ii},({b},{pack},{a})),1,0,2)')
    rr=emit(f'tl.reshape({rr},({pack},{n}))');ii=emit(f'tl.reshape({ii},({pack},{n}))')
    for p in range(pack):
        r=emit(f'tl.reshape(al.extract_slice({rr},[{p},0],[1,{n}],[1,1]),({n},))')
        i=emit(f'tl.reshape(al.extract_slice({ii},[{p},0],[1,{n}],[1,1]),({n},))')
        v=emit(f'tl.reshape(tl.join({r},{i}),({2*n},))')
        lines.append(f'    tl.store(Y+(bid*{pack}+{p})*{2*n}+lane,{v},bid*{pack}+{p}<NB)')
    return name,'\n'.join(lines)+'\n'
