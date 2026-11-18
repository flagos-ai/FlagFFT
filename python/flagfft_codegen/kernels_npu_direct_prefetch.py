# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiny direct DFTs with contiguous input and coefficient windows."""


def build_single_direct_dft(n, direction):
    """Reduce contiguous, padded coefficient rows for one complex transform."""
    if not 2 <= n <= 32 or direction not in ('forward', 'inverse'):
        raise ValueError('Single direct DFT requires length 2..32 and a valid direction')
    size = 1 << (n - 1).bit_length()
    name = f'single_direct_dft_n{n}_{direction}'
    source = f'''@triton.jit
def {name}(in_ptr, out_ptr, dft_r_ptr, dft_i_ptr, nbatch):
    word = tl.arange(0, {2 * size})
    value = tl.load(in_ptr + word, word < {2 * n}, 0.)
    xr, xi = tl.split(tl.reshape(value, ({size}, 2)))
    lane = tl.arange(0, {size})
    offset = lane[:, None] * {size} + lane[None, :]
    wr = tl.load(dft_r_ptr + offset)
    wi = tl.load(dft_i_ptr + offset)
    real = tl.sum(xr[None, :] * wr - xi[None, :] * wi, axis=1)
    imag = tl.sum(xr[None, :] * wi + xi[None, :] * wr, axis=1)
    result = tl.reshape(tl.join(real, imag), ({2 * size},))
    tl.store(out_ptr + word, result, word < {2 * n})
'''
    return name, source, ['in_ptr', 'out_ptr', 'dft_r_ptr', 'dft_i_ptr', 'nbatch']


def build_prefetched_direct_dft(n, mode, direction, batch_tile, pair=False):
    if not 2 <= n <= 32 or mode not in ('c2c', 'r2c', 'c2r'):
        raise ValueError('Prefetched direct DFT requires length 2..32')
    inverse = direction == 'inverse'
    if direction not in ('forward', 'inverse') or (mode != 'c2c' and (mode == 'c2r') != inverse):
        raise ValueError('Real direct DFT direction disagrees with its boundary')
    # Unit batch tiles have a native UB failure in this gather/reduction form.
    # Callers retain the existing generator for narrower tiles.
    if batch_tile not in (8, 16, 32):
        raise ValueError("Prefetched DFT requires a batch tile of 8, 16 or 32")
    bt = batch_tile
    half = n // 2 + 1
    ins = n if mode == 'r2c' else 2 * half if mode == 'c2r' else 2 * n
    outs = n if mode == 'c2r' else 2 * half if mode == 'r2c' else 2 * n
    frequencies = half if mode == 'r2c' else n
    p = 1 << (frequencies - 1).bit_length()
    words = 1 << (bt * ins - 1).bit_length()
    table_elements = n * (n//2+1) if pair else n*n
    table_words = 1 << (table_elements - 1).bit_length()
    prefix = ('direct_idft' if inverse else 'direct_dft') if mode == 'c2c' else f'direct_dft_{mode}'
    name = f'{prefix}_kernel_n{n}_f32_b{p}_prefetch'+('_pair' if pair else '')+f'_bt{bt}'
    args = ['in_ptr', 'out_ptr', 'dft_r_ptr', 'dft_i_ptr', 'nbatch']
    lines = ['@triton.jit', f'def {name}({",".join(args)}):',
        f'    base=tl.program_id(0)*{bt}', f'    word=tl.arange(0,{words})',
        f'    data=tl.load(in_ptr+base*{ins}+word,(word<{bt*ins})&(word<(nbatch-base)*{ins}),0.)',
        f'    b=base+tl.arange(0,{bt})', f'    idx=tl.arange(0,{bt})',
        f'    k=tl.arange(0,{p})', f'    wt=tl.arange(0,{table_words})',
        f'    wr_all=tl.load(dft_r_ptr+wt,wt<{table_elements},0.)',
        f'    wi_all=tl.load(dft_i_ptr+wt,wt<{table_elements},0.)',
        f'    ar=tl.zeros(({bt},{p}),tl.float32)']
    if mode != 'c2r': lines.append(f'    ai=tl.zeros(({bt},{p}),tl.float32)')
    if pair:
        lines.append(f'    xr=tl.gather(data,idx*{ins},0)')
        lines.append('    ar=ar+xr[:,None]')
        if mode == 'c2c':
            lines += [f'    xi=tl.gather(data,idx*{ins}+1,0)', '    ai=ai+xi[:,None]']
        for j in range(1,(n+1)//2):
            offset=j if mode=='r2c' else 2*j
            lines += [f'    xr=tl.gather(data,idx*{ins}+{offset},0)',
                f'    wr=tl.gather(wr_all,{j*n}+k,0)', f'    wi=tl.gather(wi_all,{j*n}+k,0)']
            if mode == 'r2c':
                lines += [f'    qr=tl.gather(data,idx*{ins}+{n-j},0)',
                    '    ar=ar+(xr+qr)[:,None]*wr[None,:]',
                    '    ai=ai+(xr-qr)[:,None]*wi[None,:]']
            else:
                lines.append(f'    xi=tl.gather(data,idx*{ins}+{2*j+1},0)')
                if mode == 'c2r':
                    lines.append('    ar=ar+2.*(xr[:,None]*wr[None,:]-xi[:,None]*wi[None,:])')
                else:
                    lines += [f'    qr=tl.gather(data,idx*{ins}+{2*(n-j)},0)',
                        f'    qi=tl.gather(data,idx*{ins}+{2*(n-j)+1},0)',
                        '    ar=ar+((xr+qr)[:,None]*wr[None,:]-(xi-qi)[:,None]*wi[None,:])',
                        '    ai=ai+((xi+qi)[:,None]*wr[None,:]+(xr-qr)[:,None]*wi[None,:])']
        if n%2 == 0:
            offset=n//2 if mode=='r2c' else n
            lines += [f'    xr=tl.gather(data,idx*{ins}+{offset},0)',
                f'    wr=tl.gather(wr_all,{n*n//2}+k,0)', '    ar=ar+xr[:,None]*wr[None,:]']
            if mode == 'c2c':
                lines += [f'    xi=tl.gather(data,idx*{ins}+{n+1},0)', '    ai=ai+xi[:,None]*wr[None,:]']
    else:
        for j in range(n):
            sample = j if mode != 'c2r' or j < half else n - j
            offset = j if mode == 'r2c' else 2 * sample
            lines += [f'    xr=tl.gather(data,idx*{ins}+{offset},0)',
                f'    wr=tl.gather(wr_all,{j*n}+k,0)', f'    wi=tl.gather(wi_all,{j*n}+k,0)']
            if mode == 'r2c':
                lines += ['    ar=ar+xr[:,None]*wr[None,:]', '    ai=ai+xr[:,None]*wi[None,:]']
            else:
                imag = 'tl.zeros_like(xr)' if mode == 'c2r' and (j == 0 or n % 2 == 0 and j == n // 2) else f'tl.gather(data,idx*{ins}+{2*sample+1},0)'
                if mode == 'c2r' and j >= half: imag = '-' + imag
                lines += [f'    xi={imag}', '    ar=ar+(xr[:,None]*wr[None,:]-xi[:,None]*wi[None,:])']
                if mode != 'c2r': lines.append('    ai=ai+(xr[:,None]*wi[None,:]+xi[:,None]*wr[None,:])')
    if mode == 'r2c':
        endpoint = f'(k==0)|(k=={n//2})' if n%2==0 else '(k==0)'
        lines.append(f'    ai=tl.where(({endpoint})[None,:],0.,ai)')
    if mode == 'c2r':
        lines.append(f'    tl.store(out_ptr+b[:,None]*{n}+k[None,:],ar,(b[:,None]<nbatch)&(k[None,:]<{n}))')
    else:
        lines += [f'    tl.store(out_ptr+b[:,None]*{outs}+2*k[None,:],ar,(b[:,None]<nbatch)&(k[None,:]<{frequencies}))',
            f'    tl.store(out_ptr+b[:,None]*{outs}+2*k[None,:]+1,ai,(b[:,None]<nbatch)&(k[None,:]<{frequencies}))']
    return name, '\n'.join(lines) + '\n', args
