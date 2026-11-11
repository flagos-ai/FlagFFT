# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Register Rader codelets for explicitly screened IX mixed-radix leaves."""
import cmath
import math


def emit_prime_codelet(indent, prime, direction):
    """Emit a natural-order, unnormalized DFT using a short cyclic convolution."""
    if prime not in (13, 17) or direction not in ('forward', 'inverse'):
        raise ValueError('IX register Rader supports radix 13 or 17')
    size=prime-1
    generator=next(g for g in range(2,prime)
                   if len({pow(g,j,prime) for j in range(size)})==size)
    order=[pow(generator,j,prime) for j in range(size)]
    output_order=[pow(generator,-j,prime) for j in range(size)]
    sign=-1 if direction=='forward' else 1
    chirp=[cmath.exp(sign*2j*math.pi*k/prime) for k in output_order]
    spectrum=[sum(x*cmath.exp(-2j*math.pi*k*j/size) for j,x in enumerate(chirp))
              for k in range(size)]
    lines=[]
    serial=0
    def put(code): lines.append(indent+code)
    def multiply(real,imag,value,prefix):
        wr,wi=value.real,value.imag
        if abs(wi)<1e-14:
            rr,ii=f'({real})*{wr:.17g}',f'({imag})*{wr:.17g}'
        elif abs(wr)<1e-14:
            rr,ii=f'-({imag})*{wi:.17g}',f'({real})*{wi:.17g}'
        else:
            rr=f'({real})*{wr:.17g}-({imag})*{wi:.17g}'
            ii=f'({real})*{wi:.17g}+({imag})*{wr:.17g}'
        put(f'{prefix}r={rr}');put(f'{prefix}i={ii}')
        return prefix+'r',prefix+'i'
    def fft(values,inverse=False):
        nonlocal serial
        if len(values)==1:return values
        radix=2 if len(values)%2==0 else 3
        width=len(values)//radix
        children=[fft(values[j::radix],inverse) for j in range(radix)]
        result=[]
        for k in range(len(values)):
            serial+=1;prefix=f'ixr_{serial}_'
            terms=[children[0][k%width]]
            for j in range(1,radix):
                value=cmath.exp((1 if inverse else -1)*2j*math.pi*j*k/len(values))
                terms.append(multiply(*children[j][k%width],value,prefix+str(j)+'_'))
            rr=prefix+'r';ii=prefix+'i'
            put(rr+'='+'+'.join(v[0] for v in terms))
            put(ii+'='+'+'.join(v[1] for v in terms))
            result.append((rr,ii))
        return result
    put('ixr_zero_r=r0');put('ixr_zero_i=i0')
    put('ixr_dc_r='+'+'.join('r'+str(j) for j in range(prime)))
    put('ixr_dc_i='+'+'.join('i'+str(j) for j in range(prime)))
    transformed=fft([(f'r{k}',f'i{k}') for k in order])
    products=[multiply(*value,spectrum[j],f'ixr_product{j}_') for j,value in enumerate(transformed)]
    convolution=fft(products,True)
    # Assign only after every original input has been consumed.
    put('r0=ixr_dc_r');put('i0=ixr_dc_i')
    for k,(real,imag) in zip(output_order,convolution):
        put(f'r{k}=ixr_zero_r+({real})/{size}.0')
        put(f'i{k}=ixr_zero_i+({imag})/{size}.0')
    return lines
