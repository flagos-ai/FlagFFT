#!/usr/bin/env python3
"""Serial IX CT-single baseline/default comparison on physical GPU 2.

This is performance screening only. Use run_tests.py for the accuracy gate.
Environment overrides are fixed at process startup; each trial gets a fresh
process so native and filesystem kernel caches cannot cross policy settings.
"""
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
    p.add_argument('--apis', default='c2c,c2r,r2c')
    p.add_argument('--variants', default='baseline,default')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--directions', default='forward,inverse')
    a = p.parse_args()
    variants = a.variants.split(',')
    if any(v not in {'baseline', 'default'} for v in variants):
        p.error('--variants accepts baseline,default')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    rows = []
    for repeat in range(a.repeats):
        # Alternate order across repetitions to expose temporal drift.
        for variant in variants if repeat % 2 == 0 else variants[::-1]:
            env = {k: v for k, v in os.environ.items() if not k.startswith('FLAGFFT_IX_')}
            env.update(CUDA_VISIBLE_DEVICES='2', IX_VISIBLE_DEVICES='2', FLAGFFT_TUNE_DISABLE='1',
                       FLAGFFT_IX_CT_SINGLE='0' if variant == 'baseline' else '1')
            env['PYTHONPATH'] = str(root / 'python') + os.pathsep + env.get('PYTHONPATH', '')
            for n in a.shapes.split(','):
                for api in a.apis.split(','):
                    directions = a.directions.split(',') if api == 'c2c' else ['inverse' if api == 'c2r' else 'forward']
                    for direction in directions:
                        name = f'{variant}_{n}_{api}_{direction}_{repeat}'
                        cmd = [a.binary, 'bench', '--api', api, '--rank', '1', '--shape', n,
                               '--batch', '1', '--direction', direction, '--warmup', '30',
                               '--iters', '300', '--json', '--print-path']
                        row = dict(variant=variant, n=int(n), api=api, direction=direction,
                                   repeat=repeat, git_commit=commit, command=cmd)
                        try:
                            proc = subprocess.run(cmd, env=env, text=True, capture_output=True,
                                                  timeout=a.timeout)
                            (out / f'{name}.json').write_text(proc.stdout)
                            (out / f'{name}.err').write_text(proc.stderr)
                            row['exit_code'] = proc.returncode
                            if proc.returncode == 0:
                                row.update(json.loads(proc.stdout)['cases'][0]['timing'])
                        except subprocess.TimeoutExpired:
                            row['timeout'] = a.timeout
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        (out / 'sweep.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
