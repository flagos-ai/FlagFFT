#!/usr/bin/env python3
"""Isolated IX CT-single screening; acceptance accuracy is a separate gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--shapes', default='2048,1048576')
    p.add_argument('--apis', default='c2c')
    p.add_argument('--variants', default='baseline,p1w4,p2w4,p4w4,p4w8,p8w8')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--leaf-factors', action='append', default=[], help='length=radix,radix,...')
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--exchange', default='direct_all')
    p.add_argument('--direction', default='forward')
    p.add_argument('--warps', choices=('1', '2', '4', '8'))
    p.add_argument('--stockham-radix', choices=('8', '16', '32'))
    p.add_argument('--vector-leaf', action='store_true')
    p.add_argument('--stockham-store-join', action='store_true')
    p.add_argument('--thread-local', action='store_true')
    p.add_argument('--recurrence', action='store_true')
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(a.repeats):
        for variant in a.variants.split(','):
            env = {k: v for k, v in os.environ.items() if not k.startswith('FLAGFFT_IX_')}
            env.update(CUDA_VISIBLE_DEVICES='2', IX_VISIBLE_DEVICES='2', FLAGFFT_TUNE_DISABLE='1')
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'python') + os.pathsep + env.get('PYTHONPATH', '')
            if variant != 'baseline':
                pack, warps = variant.removeprefix('p').split('w')
                env.update(FLAGFFT_IX_PORTABLE_LEAF='1', FLAGFFT_IX_INNER_PACK=pack,
                           FLAGFFT_IX_MAX_WARPS=warps, FLAGFFT_IX_EXCHANGE=a.exchange)
            for setting in a.leaf_factors:
                length, factors = setting.split('=', 1)
                env['FLAGFFT_IX_LEAF_FACTORS_' + length] = factors
            if a.warps:
                env['FLAGFFT_IX_WARPS'] = a.warps
            if a.stockham_radix:
                env['FLAGFFT_IX_STOCKHAM_RADIX'] = a.stockham_radix
            if a.vector_leaf:
                env['FLAGFFT_IX_VECTOR_LEAF'] = '1'
            if a.stockham_store_join:
                env['FLAGFFT_IX_STOCKHAM_STORE_JOIN'] = '1'
            if a.thread_local:
                env['FLAGFFT_IX_THREAD_LOCAL'] = '1'
            if a.recurrence:
                env['FLAGFFT_IX_RECURRENCE'] = '1'
            for n in a.shapes.split(','):
                for api in a.apis.split(','):
                    name = f'{variant}_{n}_{api}_{repeat}'
                    cmd = [a.binary, 'bench', '--api', api, '--rank', '1', '--shape', n,
                           '--batch', '1', '--warmup', '20', '--iters', '200', '--json']
                    if api == 'c2c':
                        cmd += ['--direction', a.direction]
                    try:
                        proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        row = dict(variant=variant, n=int(n), api=api, repeat=repeat, timeout=a.timeout)
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        (out / 'sweep.json').write_text(json.dumps(rows, indent=2))
                        continue
                    (out / f'{name}.json').write_text(proc.stdout)
                    (out / f'{name}.err').write_text(proc.stderr)
                    row = dict(variant=variant, n=int(n), api=api, repeat=repeat, exit_code=proc.returncode)
                    if proc.returncode == 0:
                        row.update(json.loads(proc.stdout)['cases'][0]['timing'])
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    (out / 'sweep.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
